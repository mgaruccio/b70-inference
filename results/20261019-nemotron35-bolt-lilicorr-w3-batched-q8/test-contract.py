#!/usr/bin/env python3
"""CPU-only source/dispatch/launch contract, NOT a SYCL compile or arithmetic test.

Compile actual extracted host code against recording SYCL stubs. The device body
is checked byte-for-byte against r2, not emulated. Reuse the existing guard fixture.
"""
import argparse
import runpy
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYCL = Path('ggml/src/ggml-sycl')




RECORDING_SYCL = r'''
#include <array>
#include <set>
#include <tuple>
#define SYCL_CHECK(x) x
#define CHECK_TRY_ERROR(x) x
#define GGML_UNUSED(x) (void)(x)
namespace sycl {
template<int D> struct range {
    std::array<size_t,D> v;
    template<class... T> explicit range(T... x) : v{size_t(x)...} {}
};
template<int D> struct nd_range { range<D> global, local; nd_range(range<D> g, range<D> l):global(g),local(l){} };
template<int D> struct nd_item {
    std::array<size_t,D> group, local;
    size_t get_group(int d) const { return group.at(d); }
    size_t get_local_id(int d) const { return local.at(d); }
};
struct handler {
    size_t slm=0;
    template<class F> void parallel_for(nd_range<2> r, F f) {
        assert(r.local.v[1]==1 && r.global.v[0]%r.local.v[0]==0);
        for(size_t row=0;row<r.global.v[0]/r.local.v[0];++row)
            for(size_t col=0;col<r.global.v[1];++col)
                for(size_t tid=0;tid<r.local.v[0];++tid) f(nd_item<2>{{row,col},{tid,0}});
    }
};
template<class T,int D> struct local_accessor {
    size_t size;
    local_accessor(range<D> r, handler &h):size(r.v[0]) { h.slm=size; }
};
struct queue {
    int submissions=0;
    template<class F> void submit(F f) { ++submissions; handler h; f(h); }
};
}
namespace dpct { using queue_ptr=sycl::queue*; }
struct invocation { const void *x; const float *y; float *d; int k,m,wg; size_t row,col,tid,slm; };
std::vector<invocation> invocations;
template<int WG,class Item> void dequantize_mul_mat_vec_q8_0_reorder_esimd(
    const void*x,const float*y,float*d,int k,int m,sycl::local_accessor<float,1> lmem,const Item &it) {
    invocations.push_back({x,y,d,k,m,WG,it.get_group(0),it.get_group(1),it.get_local_id(0),lmem.size});
}
struct old_call { const float*y; float*d; int64_t low,high,n; };
std::vector<old_call> old_calls;
void ggml_sycl_op_dequantize_mul_mat_vec(ggml_backend_sycl_context &,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,
    const char*,const float*y,const char*,float*d,int64_t low,int64_t high,int64_t n,int64_t,const dpct::queue_ptr &) {
    old_calls.push_back({y,d,low,high,n});
}
void verify_mapping(const void*x,const float*y,float*d,int k,int m,int n,int64_t ys,int64_t ds) {
    const int blocks=k/32, wg=blocks>=64?8:blocks>=32?4:blocks>=16?2:1;
    assert(invocations.size()==size_t((m+1)/2*n*wg));
    std::set<std::tuple<size_t,size_t,size_t>> seen;
    for(const auto &v:invocations) {
        assert(v.x==x && v.k==k && v.m==m && v.wg==wg && v.slm==size_t(wg*2));
        assert(v.row<size_t((m+1)/2) && v.col<size_t(n) && v.tid<size_t(wg));
        assert(v.y==y+v.col*ys && v.d==d+v.col*ds);
        assert(seen.emplace(v.row,v.col,v.tid).second);
    }
}
'''

DISPATCH_PREFIX = r'''
void dispatch(fixture &f, sycl::queue &q, int64_t submitted=-1) {
    auto &ctx=f.ctx;
    const ggml_tensor *src0=&f.a, *src1=&f.b;
    ggml_tensor *dst=&f.d;
    auto *src0_extra=(ggml_tensor_extra_gpu*)src0->extra;
    src0_extra->optimized_feature.reorder=true; // Existing opt_for_reorder before this callsite.
    auto op=ggml_sycl_op_dequantize_mul_mat_vec;
    const int64_t src1_ncols=submitted<0?src1->ne[1]:submitted;
    const int64_t ne00=src0->ne[0],ne10=src1->ne[0],ne0=dst->ne[0],row_diff=src0->ne[1];
    const int64_t src1_padded_col_size=ne10;
    const char *src0_dd_i=(const char*)src0->data, *src1_ddq_i=nullptr;
    const float *src1_ddf_i=(const float*)src1->data;
    float *dst_dd_i=(float*)dst->data;
    const bool dst_on_device=f.buffers[3].sycl;
    struct { int64_t row_low,row_high; } dev[]={{0,row_diff}};
    const int i=0;
    dpct::queue_ptr stream=&q;
    (void)ne00; // Used only by the ESIMD-enabled branch.
'''

TESTS = r'''
int main() {
    int dispatch_cases=0,mapping_cases=0;
    for(int n=1;n<=9;++n) for(bool f32:{false,true}) for(int32_t mark:{0,0x4e535631,0x4e535630}) {
        fixture f(false,n,f32);
        f.d.op_params[15]=mark;
        sycl::queue q;
        invocations.clear(); old_calls.clear();
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
        const bool eligible=n==2 || (n>=3 && n<=8 && f32 && mark==0x4e535631);
#else
        const bool eligible=false;
#endif
        assert(f.eligible()==eligible && f.eligible()==f.eligible(true));
        dispatch(f,q);
        if(eligible && n>2) {
            assert(q.submissions==1 && old_calls.empty());
            verify_mapping(f.a.data,(float*)f.b.data,(float*)f.d.data,256,32,n,256,32);
        } else {
            assert(q.submissions==0 && invocations.empty());
            assert(old_calls.size()==size_t(eligible?2:1));
            for(size_t col=0;col<old_calls.size();++col) {
                assert(old_calls[col].y==(float*)f.b.data+col*256);
                assert(old_calls[col].d==(float*)f.d.data+col*32);
                assert(old_calls[col].n==(eligible?1:n));
                assert(old_calls[col].low==0 && old_calls[col].high==32);
            }
        }
        ++dispatch_cases;
    }
    std::vector<std::function<void(fixture&)>> reject={
        [](fixture&f){f.buffers[0].split=true;}, [](fixture&f){f.buffers[1].sycl=false;},
        [](fixture&f){f.buffers[3].sycl=false;}, [](fixture&f){f.buffers[3].split=true;},
        [](fixture&f){f.device[3].device=1;}, [](fixture&f){f.b.nb[1]+=4;},
        [](fixture&f){f.d.nb[1]+=4;}, [](fixture&f){f.d.buffer=f.b.buffer; f.d.data=f.b.data;},
        [](fixture&f){f.buffers[3].size=ggml_nbytes(&f.d)-1;},
        [](fixture&f){f.a.type=GGML_TYPE_IQ2_XXS;}, [](fixture&f){f.d.src[1]=&f.a;},
        [](fixture&f){f.ctx.opt_feature.reorder=false;}, [](fixture&f){f.a.ne[0]=INT64_MAX;}
    };
    for(int n:{3,8}) for(auto &mutate:reject) {
        fixture f(false,n,true); mutate(f);
        assert(!f.eligible() && f.eligible()==f.eligible(true));
        sycl::queue q; invocations.clear(); old_calls.clear(); dispatch(f,q);
        assert(q.submissions==0 && invocations.empty() && old_calls.size()==1);
        ++dispatch_cases;
    }
    for(bool *flag:{&g_ggml_sycl_enable_esimd,&g_ggml_sycl_enable_optimize,&f16_ok,&reorder_ok}) {
        fixture f(false,4,true); *flag=false;
        assert(!f.eligible()); sycl::queue q; old_calls.clear(); dispatch(f,q);
        assert(q.submissions==0 && old_calls.size()==1); *flag=true; ++dispatch_cases;
    }
    { // A column chunk is not a full marked tensor; preserve fallback.
        fixture f(false,4,true); sycl::queue q; old_calls.clear(); dispatch(f,q,2);
        assert(q.submissions==0 && old_calls.size()==1 && old_calls[0].n==2); ++dispatch_cases;
    }
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    for(int blocks:{1,7,8,15,16,17,31,32,33,63,64,65,67,128})
        for(int rows:{1,3,5,64,65}) for(int n=3;n<=8;++n) for(int padding:{0,4}) {
            const int k=blocks*32,ys=k+padding,ds=rows+padding;
            std::vector<float> y(ys*n+3),d(ds*n+5);
            char weights=0; sycl::queue q; invocations.clear();
            // Nonzero bases check that the wrapper preserves incoming offsets.
            ggml_sycl_dequantize_mul_mat_vec_q8_0_batched(&weights,y.data()+3,d.data()+5,k,rows,n,ys,ds,&q);
            assert(q.submissions==1);
            verify_mapping(&weights,y.data()+3,d.data()+5,k,rows,n,ys,ds); ++mapping_cases;
        }
#endif
    printf("CPU_CONTRACT_PASSED dispatch=%d launch_mapping=%d (no GPU arithmetic)\n",dispatch_cases,mapping_cases);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    runpy.run_path(str(ROOT / 'check-source.py'))['check'](args.parent, args.candidate)
    old, new = [(root / SYCL / 'dmmv.cpp').read_text() for root in (args.parent, args.candidate)]
    start = new.index('// One independent work-group per (row pair, activation column).')
    end = new.index('#endif // GGML_SYCL_DMMV_HAS_ESIMD', start)
    added = new[start:end]
    restored = (new[:start] + new[end:]).replace('template <int WG, typename Item>', 'template <int WG>', 1)
    restored = restored.replace('const Item & it)', 'const sycl::nd_item<1> & it)', 1)
    assert restored == old, 'singleton arithmetic/selector/launch or other DMMV code changed'
    assert added.count('stream->submit(') == 1
    assert 'wait(' not in added and 'to_fp16' not in added
    parent, candidate = [(root / SYCL / 'ggml-sycl.cpp').read_text() for root in (args.parent, args.candidate)]
    marker, after = '                if (op == ggml_sycl_op_dequantize_mul_mat_vec &&', '\n                // copy dst to host'
    before = parent[parent.index(marker):parent.index(after, parent.index(marker))]
    dispatch = candidate[candidate.index(marker):candidate.index(after, candidate.index(marker))]
    restored = candidate.replace(dispatch, before, 1).replace(
        '// checks are retained; only the marked wide path batches its ESIMD submission.',
        '// checks and the existing per-column DMMV callback are retained.', 1)
    assert restored == parent, 'eligibility, IQ2, tracing or another callsite changed'
    guards = runpy.run_path(str(ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/test-guards.py'))
    names = ['ggml_sycl_verify_singletons_disjoint', 'ggml_sycl_latent_dmmv2_eligible',
             'ggml_sycl_iq2_single2_range', 'ggml_sycl_iq2_single2_eligible']
    actual = ''.join(guards['function'](candidate, name) for name in names)
    previous = ''.join(guards['function'](parent, name) for name in names[1:])
    for name in names[1:]:
        previous = previous.replace(name, name + '_parent')
    fixture = guards['TESTS'].split('int main()', 1)[0]
    header = (args.candidate / SYCL / 'dmmv.hpp').read_text().replace('#include "common.hpp"', '')
    print('SOURCE_IDENTITY_PASSED singleton body/stripe/reductions/1D launch, all guards, IQ2, tracing unchanged', flush=True)
    scratch = ROOT / 'worker-scratch'
    scratch.mkdir(exist_ok=True)
    for esimd in (True, False):
        stubs = guards['STUBS']
        if not esimd:
            stubs = stubs.replace('#define GGML_SYCL_DMMV_HAS_ESIMD', '')
        code = stubs + RECORDING_SYCL + actual + previous + fixture
        code += '#ifdef GGML_SYCL_DMMV_HAS_ESIMD\n#define __INTEL_LLVM_COMPILER 1\n' + header + added + '#endif\n'
        code += DISPATCH_PREFIX + dispatch + '}\n' + TESTS
        with tempfile.TemporaryDirectory(prefix='cpu-contract-', dir=scratch) as tmp:
            src, exe = Path(tmp) / 'contract.cpp', Path(tmp) / 'contract'
            src.write_text(code)
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-attributes',
                            '-Wno-unused-parameter', '-Wno-unused-function', '-fsanitize=undefined',
                            '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True, timeout=30)
            print('recording ESIMD path:', esimd, flush=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    main()

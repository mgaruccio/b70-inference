#!/usr/bin/env python3
"""Compile the actual patched host guards with CPU-only tensor/buffer/queue stubs.

Tests predicates, offsets and preflight atomicity, NOT kernels, USM or GPU ordering.
Temporary compiler inputs/binaries stay under this package and are removed.
"""
import argparse
import runpy
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def function(source, name):
    start = source.index('static bool ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


STUBS = r'''
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <stdexcept>
#include <functional>
#define GGML_MAX_DIMS 4
#define GGML_SYCL_DMMV_HAS_ESIMD
#define GGML_SYCL_F16
#define GGML_SYCL_DMMV_X 32
#define QK8_0 32
#define QK8_1 32
#define QK_K 256
#define MATRIX_ROW_PADDING 512
#define GGML_ABORT(msg) throw std::runtime_error(msg)
enum { GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_IQ2_XXS, GGML_TYPE_F16 };
enum { GGML_OP_NONE, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID, GGML_OP_RESHAPE, GGML_OP_ADD };
enum { GGML_TENSOR_FLAG_INPUT=1, GGML_TENSOR_FLAG_OUTPUT=2, GGML_TENSOR_FLAG_COMPUTE=4 };
enum { GGML_PREC_F32=10, LLM_ARCH_NEMOTRON_H_MOE, LLM_ARCH_DFLASH, LLM_ARCH_NEMOTRON_H,
       LLM_ARCH_GLM4, LLM_ARCH_GLM4_MOE, LLM_ARCH_JAIS2 };
struct block_iq2_xxs { char data[66]; };
struct feature { bool reorder = false; };
struct ggml_tensor_extra_gpu { feature optimized_feature; };
struct ggml_backend_sycl_buffer_context { int device = 0; };
struct buffer { void * context; void * base; size_t size; bool sycl=true, split=false; };
struct ggml_tensor {
    int type=GGML_TYPE_F32, op=GGML_OP_NONE, flags=0;
    int64_t ne[4]={1,1,1,1}; size_t nb[4]={4,4,4,4};
    ggml_tensor * src[3]={}; ggml_tensor * view_src=nullptr; size_t view_offs=0;
    struct buffer * buffer=nullptr; void * data=nullptr; void * extra=nullptr;
    char name[128]={}; int32_t op_params[16]={};
};
struct ggml_backend_sycl_context { int device=0; feature opt_feature{true}; };
bool g_ggml_sycl_enable_esimd=true, g_ggml_sycl_enable_optimize=true, f16_ok=true, reorder_ok=true;
bool ggml_backend_buffer_is_sycl_split(buffer * b) { return b->split; }
bool ggml_backend_buffer_is_sycl(buffer * b) { return b->sycl; }
void * ggml_backend_buffer_get_base(buffer * b) { return b->base; }
size_t ggml_backend_buffer_get_size(buffer * b) { return b->size; }
int32_t ggml_get_op_params_i32(const ggml_tensor * t, int i) { return t->op_params[i]; }
bool ggml_prec_set_acc(ggml_tensor * t, int p) { t->op_params[0]=p; return t->op==GGML_OP_MUL_MAT || t->op==GGML_OP_MUL_MAT_ID; }
#define GGML_ASSERT(x) assert(x)
int64_t ggml_nelements(const ggml_tensor * t) { return t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3]; }
size_t block(int type) { return type==GGML_TYPE_Q8_0 ? 32 : type==GGML_TYPE_IQ2_XXS ? 256 : 1; }
size_t typesize(int type) { return type==GGML_TYPE_Q8_0 ? 34 : type==GGML_TYPE_IQ2_XXS ? 66 : 4; }
size_t ggml_nbytes(const ggml_tensor * t) {
    size_t bytes=t->ne[0]/block(t->type)*t->nb[0];
    for(int i=1;i<4;++i) bytes+=(t->ne[i]-1)*t->nb[i];
    return bytes;
}
bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t stride=typesize(t->type);
    if (t->nb[0]!=stride) return false;
    stride*=t->ne[0]/block(t->type);
    for(int i=1;i<4;++i) { if(t->ne[i]!=1 && t->nb[i]!=stride) return false; stride*=t->ne[i]; }
    return true;
}
bool ggml_sycl_src1_f16_ok(const ggml_tensor *) { return f16_ok; }
static bool should_reorder_tensor(ggml_backend_sycl_context &, const ggml_tensor *) { return reorder_ok; }
struct call { ggml_tensor input, ids, output; };
std::vector<call> calls;
int preflight=0, fail_preflight=-1; bool fail_dispatch=false;
static bool ggml_sycl_mul_mat_id_mmvq_fused(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor * b,
                                          const ggml_tensor * ids, ggml_tensor * dst) {
    calls.push_back({*b,*ids,*dst}); return !fail_dispatch;
}
'''

TESTS = r'''
struct fixture {
    ggml_backend_sycl_context ctx;
    ggml_backend_sycl_buffer_context device[4];
    ggml_tensor_extra_gpu extra[4];
    std::vector<unsigned char> storage[4];
    buffer buffers[4];
    ggml_tensor a,b,ids,d;
    bool iq;
    fixture(bool iq, int n, bool mark, bool up=true) : iq(iq) {
        ggml_tensor * ts[]={&a,&b,&ids,&d};
        for(int i=0;i<4;++i) { storage[i].resize(1<<20); buffers[i]={&device[i],storage[i].data(),storage[i].size()};
            ts[i]->buffer=&buffers[i]; ts[i]->data=storage[i].data(); ts[i]->extra=&extra[i]; }
        shape(a,iq?GGML_TYPE_IQ2_XXS:GGML_TYPE_Q8_0,256,32,iq?16:1);
        shape(b,GGML_TYPE_F32,256,iq?(up?1:9):n,iq?n:1);
        shape(ids,GGML_TYPE_I32,9,n,1); ids.nb[1]=64; ids.nb[2]=ids.nb[3]=64*n;
        shape(d,GGML_TYPE_F32,32,iq?9:n,iq?n:1);
        d.op=iq?GGML_OP_MUL_MAT_ID:GGML_OP_MUL_MAT; d.src[0]=&a; d.src[1]=&b; d.src[2]=&ids;
        d.op_params[0]=mark?GGML_PREC_F32:0;
        d.op_params[15]=mark?0x4e535631:0;
        std::strcpy(a.name,iq?"blk.1.ffn_up_exps.weight":"blk.1.ffn_latent_down.weight");
    }
    static void shape(ggml_tensor & t,int type,int64_t x,int64_t y,int64_t z) {
        t.type=type; t.ne[0]=x; t.ne[1]=y; t.ne[2]=z; t.ne[3]=1;
        t.nb[0]=typesize(type); t.nb[1]=t.nb[0]*x/block(type); t.nb[2]=t.nb[1]*y; t.nb[3]=t.nb[2]*z;
    }
    bool eligible(bool parent=false) {
        if(iq) return parent?ggml_sycl_iq2_single2_eligible_parent(ctx,&a,&b,&ids,&d):ggml_sycl_iq2_single2_eligible(ctx,&a,&b,&ids,&d);
        return parent?ggml_sycl_latent_dmmv2_eligible_parent(ctx,&a,&b,&d):ggml_sycl_latent_dmmv2_eligible(ctx,&a,&b,&d);
    }
};
int main() {
    int cases=0;
    for(bool iq:{false,true}) for(bool up:{false,true}) for(int n=1;n<=9;++n) for(bool mark:{false,true}) {
        fixture f(iq,n,mark,up);
        assert(f.eligible()==(n==2 || (mark && n>=3 && n<=8)));
        if(n<=2 || n==9 || !mark) assert(f.eligible()==f.eligible(true));
        if(iq && f.eligible()) {
            calls.clear(); preflight=0;
            assert(ggml_sycl_mul_mat_id_iq2_single2(f.ctx,&f.a,&f.b,&f.ids,&f.d)); assert(calls.size()==size_t(n));
            for(int col=0;col<n;++col) {
                assert(calls[col].input.data==(char*)f.b.data+col*f.b.nb[2]);
                assert(calls[col].ids.data==(char*)f.ids.data+col*f.ids.nb[1]);
                assert(calls[col].output.data==(char*)f.d.data+col*f.d.nb[2]);
                assert(calls[col].input.ne[2]==1 && calls[col].output.ne[2]==1 && calls[col].ids.ne[1]==1);
                assert(!memcmp(calls[col].input.nb,f.b.nb,sizeof(f.b.nb)));
                assert(!memcmp(calls[col].ids.nb,f.ids.nb,sizeof(f.ids.nb)));
                assert(!memcmp(calls[col].output.nb,f.d.nb,sizeof(f.d.nb)));
                assert(calls[col].output.op_params[0]==f.d.op_params[0]);
                assert(calls[col].output.op_params[15]==f.d.op_params[15]);
            }
            calls.clear(); preflight=0; fail_preflight=n; // last column, before any enqueue
            assert(!ggml_sycl_mul_mat_id_iq2_single2(f.ctx,&f.a,&f.b,&f.ids,&f.d) && calls.empty());
            fail_preflight=-1;
        }
        ++cases;
    }
    // Precision is independent of target identity: include GLM-like attention output.
    for(bool iq:{false,true}) for(int n=1;n<=9;++n) for(bool f32:{false,true})
        for(int32_t annotation:{0,0x4e535631,0x4e535630,-1}) {
            fixture f(iq,n,f32);
            if(!iq) std::strcpy(f.a.name,"blk.7.attn_output.weight");
            f.d.op_params[15]=annotation;
            const bool expected=n==2 || (n>=3 && n<=8 && f32 && annotation==0x4e535631);
            assert(f.eligible()==expected);
            if(n<=2 || !f32 || annotation!=0x4e535631) assert(f.eligible()==f.eligible(true));
            if(iq) {
                calls.clear(); preflight=0;
                assert(ggml_sycl_mul_mat_id_iq2_single2(f.ctx,&f.a,&f.b,&f.ids,&f.d)==expected);
                assert(calls.size()==(expected?size_t(n):0));
            }
            ++cases;
        }
    std::vector<std::function<void(fixture&)>> invalid={
        [](fixture&f){f.d.src[0]=&f.b;}, [](fixture&f){f.b.type=GGML_TYPE_F16;},
        [](fixture&f){f.d.ne[0]++;}, [](fixture&f){f.b.nb[2]+=4; f.b.ne[2]=4;},
        [](fixture&f){f.a.ne[0]=INT64_MAX;}, [](fixture&f){f.device[0].device=1;},
        [](fixture&f){f.buffers[0].sycl=false;}, [](fixture&f){f.d.data=(char*)f.d.data+1;},
        [](fixture&f){f.buffers[3].size=ggml_nbytes(&f.d)-1;},
        [](fixture&f){f.d.buffer=f.b.buffer; f.d.data=f.b.data;},
        [](fixture&f){f.buffers[3].base=(void*)(UINTPTR_MAX-15); f.buffers[3].size=32;},
    };
    for(bool iq:{false,true}) for(int n:{3,8}) for(auto &mutate:invalid) {
        fixture f(iq,n,true); mutate(f); assert(!f.eligible());
        if(iq) { calls.clear(); assert(!ggml_sycl_mul_mat_id_iq2_single2(f.ctx,&f.a,&f.b,&f.ids,&f.d) && calls.empty()); }
        ++cases;
    }
    for(int n:{3,8}) {
        fixture f(true,n,true); f.ids.nb[1]=SIZE_MAX/4*4; assert(!f.eligible()); ++cases;
        fixture g(true,n,true); g.ids.nb[2]++; assert(!g.eligible()); ++cases;
        fixture h(true,n,true); h.extra[0].optimized_feature.reorder=true; assert(!h.eligible()); ++cases;
        fixture q(false,n,true); f16_ok=false; assert(!q.eligible()); f16_ok=true;
        reorder_ok=false; assert(!q.eligible()); reorder_ok=true; ++cases;
        q.a.view_src=&q.b; assert(!q.eligible()); ++cases;
    }
    fixture view(false,4,true);
    std::strcpy(view.a.name,"blk.7.attn_output.weight");
    ggml_tensor base=view.b; view.b.view_src=&base; view.b.op=GGML_OP_RESHAPE;
    assert(view.eligible()); ++cases;
    base.ne[3]=INT64_MAX; assert(!view.eligible()); ++cases;
    fixture f(true,3,true); fail_dispatch=true; calls.clear(); preflight=0;
    bool aborted=false; try { ggml_sycl_mul_mat_id_iq2_single2(f.ctx,&f.a,&f.b,&f.ids,&f.d); } catch(const std::runtime_error&) { aborted=true; }
    assert(aborted && calls.size()==1); fail_dispatch=false; ++cases;
    // Exercise the actual graph producer predicates, including MTP and DFlash exclusions.
    graph g;
    for(bool iq:{false,true}) for(int n=1;n<=9;++n) for(int outputs:{0,1,n})
        for(int arch:{LLM_ARCH_NEMOTRON_H_MOE,LLM_ARCH_DFLASH,LLM_ARCH_NEMOTRON_H,LLM_ARCH_GLM4,LLM_ARCH_GLM4_MOE,LLM_ARCH_JAIS2}) {
            fixture f(iq,n,false); g.n_tokens=n; g.n_outputs=outputs; g.arch=arch;
            if(!iq) std::strcpy(f.a.name,"blk.7.attn_output.weight");
            g.mark(iq,&f.a,&f.b,&f.b,&f.d,&f.ids,nullptr);
            const bool expected=arch==LLM_ARCH_NEMOTRON_H_MOE && n>=3 && n<=8 && outputs==n;
            assert((f.d.op_params[0]==GGML_PREC_F32)==expected);
            assert(f.d.op_params[15]==(expected?0x4e535631:0)); ++cases;
        }
    // Existing unrelated F32 requests are retained but never acquire target scope.
    for(bool iq:{false,true}) for(int n=3;n<=8;++n)
        for(int arch:{LLM_ARCH_GLM4,LLM_ARCH_GLM4_MOE,LLM_ARCH_JAIS2}) {
            fixture f(iq,n,true); f.d.op_params[15]=0;
            if(!iq) std::strcpy(f.a.name,"blk.7.attn_output.weight");
            g.arch=arch; g.n_tokens=g.n_outputs=n;
            g.mark(iq,&f.a,&f.b,&f.b,&f.d,&f.ids,nullptr);
            assert(f.d.op_params[0]==GGML_PREC_F32 && f.d.op_params[15]==0 && !f.eligible()); ++cases;
        }
    g.arch=LLM_ARCH_NEMOTRON_H_MOE; g.n_tokens=g.n_outputs=4;
    for(bool iq:{false,true}) for(int control=0;control<8;++control) {
        fixture f(iq,4,false); if(!iq) std::strcpy(f.a.name,"blk.7.attn_q.weight");
        ggml_tensor * input=&f.b, *slots=nullptr;
        if(control==0) std::strcpy(f.a.name,iq?"blk.52.ffn_up_exps.weight":"blk.52.attn_q.weight");
        if(control==1) f.d.op=GGML_OP_ADD;
        if(control==2) input=&f.a;
        if(control==3) f.d.src[0]=&f.b;
        if(control==4) std::strcpy(f.a.name,"blk.1.unrelated.weight");
        if(control==5) g.n_outputs=1;
        if(control==6) g.n_tokens=5;
        if(control==7) f.d.type=GGML_TYPE_F16;
        g.mark(iq,&f.a,input,&f.b,&f.d,&f.ids,slots); assert(f.d.op_params[0]==0 && f.d.op_params[15]==0);
        g.n_tokens=g.n_outputs=4; ++cases;
    }
    for(int layer:{0,51,52}) {
        fixture f(false,4,false); g.il=layer; g.latent(&f.d,&f.a,&f.b);
        assert((f.d.op_params[0]==10)==(layer<52)); assert(f.d.op_params[15]==(layer<52?0x4e535631:0)); ++cases;
    }
    for(const char *name:{"blk.999999999999999999999999999999.attn_q.weight","blk.-1.attn_q.weight","blk.attn_q.weight"}) {
        fixture f(false,4,false); std::strcpy(f.a.name,name); assert(!nemotron_verify_weight(&f.a,52,".attn_q.weight")); ++cases;
    }
    std::printf("CPU_GUARDS_PASSED cases=%d (stubs; no GPU execution)\n",cases);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    runpy.run_path(str(ROOT / 'check-source.py'))['check'](args.parent, args.candidate)
    rel = 'ggml/src/ggml-sycl/ggml-sycl.cpp'
    parent, candidate = [(p / rel).read_text() for p in (args.parent, args.candidate)]
    names = ['ggml_sycl_verify_singletons_disjoint', 'ggml_sycl_latent_dmmv2_eligible',
             'ggml_sycl_iq2_single2_range', 'ggml_sycl_iq2_single2_eligible']
    code = STUBS + ''.join(function(candidate, name) for name in names)
    oldnames = names[1:]
    old = ''.join(function(parent, name) for name in oldnames)
    for name in oldnames:
        old = old.replace(name, name + '_parent')
    code += old
    layout = function(candidate, 'ggml_sycl_mul_mat_id_mmvq_layout')
    code += layout.replace('ggml_sycl_mul_mat_id_mmvq_layout(', 'layout_actual(')
    code += '''static bool ggml_sycl_mul_mat_id_mmvq_layout(const ggml_tensor *a, const ggml_tensor *b, const ggml_tensor *ids, const ggml_tensor *d) {
        return ++preflight != fail_preflight && layout_actual(a,b,ids,d);
    }\n'''
    code += function(candidate, 'ggml_sycl_mul_mat_id_iq2_single2')
    graph = (args.candidate / 'src/llama-graph.cpp').read_text()
    model = (args.candidate / 'src/models/nemotron-h.cpp').read_text()
    fixtures = (args.candidate / 'tests/test-backend-ops.cpp').read_text()
    assert candidate.count('ggml_get_op_params_i32(dst, 15) != 0x4e535631') == 2
    assert graph.count('res->op_params[15] = 0x4e535631;') == 2
    assert model.count('out->op_params[15] = 0x4e535631;') == 1
    assert fixtures.count('wide->op_params[15] = annotation == 0 ? 0 : (0x4e535631 ^ (annotation == 2));') == 2
    print('SCOPE_MARKER_CONSISTENCY_PASSED word=15 value=0x4e535631 producers=3 guards=2 fixtures=2', flush=True)
    code += function(graph, 'nemotron_verify_weight')
    q8 = graph.split('    // Mark the actual direct Q8 producer,', 1)[1].split('    return res;', 1)[0]
    q8 = '    // Mark the actual direct Q8 producer,' + q8
    iq2 = graph.split('    // Both expert-up and expert-down', 1)[1].split('    return res;', 1)[0]
    iq2 = '    // Both expert-up and expert-down' + iq2
    latent = model.split('        const auto mark_latent_verify', 1)[1].split('        };', 1)[0]
    code += '''struct graph {
        int arch=LLM_ARCH_NEMOTRON_H_MOE, il=1; int64_t n_layer=52,n_tokens=4,n_outputs=4;
        void mark(bool iq, ggml_tensor*w,ggml_tensor*input,ggml_tensor*cur,ggml_tensor*res,ggml_tensor*ids,ggml_tensor*slots) {
        if(iq) {\n''' + iq2 + '\n} else {\n' + q8 + '\n}}\n'
    code += 'void latent(ggml_tensor*out,ggml_tensor*w,ggml_tensor*input) { const auto mark_latent_verify' + latent + '}; mark_latent_verify(out,w,input); }\n};\n'
    code += TESTS
    with tempfile.TemporaryDirectory(prefix='.cpu-guards-', dir=ROOT) as tmp:
        src, exe = Path(tmp) / 'guards.cpp', Path(tmp) / 'guards'
        src.write_text(code)
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                        '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True, timeout=30)
        subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    main()

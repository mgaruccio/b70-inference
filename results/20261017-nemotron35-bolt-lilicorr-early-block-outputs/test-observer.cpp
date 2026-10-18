// Remote CPU spy only. test-trace.py inserts the ACTUAL patched helper below.
// SYCL queue/filesystem boundaries are spied; no GPU, model or linked runtime.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

static std::string root;
static int last_capture_fd = -1;
static std::string mapped(const char * p) {
    const std::string path(p), prefix("/early-block-outputs");
    if (path == prefix || path.rfind(prefix + "/", 0) == 0) return root + path.substr(prefix.size());
    return path;
}
static int spy_open(const char * p, int flags, mode_t mode = 0) {
    const int fd = ::open(mapped(p).c_str(), flags, mode);
    if (std::string(p).find(".f32.bin") != std::string::npos) last_capture_fd = fd;
    return fd;
}
static int spy_lstat(const char * p, struct stat * st) { return ::lstat(mapped(p).c_str(), st); }
struct Abort : std::runtime_error { using std::runtime_error::runtime_error; };
#undef GGML_ABORT
#define GGML_ABORT(format, reason) throw Abort(reason)
#define GGML_LOG_INFO(...) std::printf(__VA_ARGS__)
#define open spy_open
#define lstat spy_lstat

struct SpyQueue;
struct SpyEvent { SpyQueue * q; void wait_and_throw(); };
struct SpyQueue {
    bool ordered = true, fail_submit = false, fail_async = false;
    int copies = 0, events = 0, waits = 0;
    void * host = nullptr;
    const void * source = nullptr;
    size_t bytes = 0;
    bool is_in_order() const { return ordered; }
    SpyEvent memcpy(void * h, const void * s, size_t n) {
        ++copies; host=h; source=s; bytes=n;
        if (fail_submit) throw std::runtime_error("submit");
        return {this};
    }
    void complete() { assert(host && source); std::memcpy(host,source,bytes); host=nullptr; }
    void wait() { ++waits; complete(); } // Writes host AFTER exception; sanitizer checks lifetime.
};
void SpyEvent::wait_and_throw() {
    ++q->events;
    if (q->fail_async) throw std::runtime_error("async");
    q->complete();
}
using queue_ptr = SpyQueue *;
struct ggml_backend_sycl_context { int device; queue_ptr q; queue_ptr stream() const { return q; } };
struct ggml_backend_sycl_buffer_context { int device; void * base; };
static ggml_backend_buffer_type sycl_type{}, cpu_type{}, split_type{};
static int metadata_reads = 0;
bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t b) { return b->buft == &sycl_type; }
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t b) {
    assert(b->buft == &sycl_type); ++metadata_reads;
    return static_cast<ggml_backend_sycl_buffer_context *>(b->context)->base;
}
size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t b) {
    assert(b->buft == &sycl_type); ++metadata_reads; return b->size;
}
bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t n=4; for (int i=0;i<4;++i) { if (t->nb[i]!=n) return false; n*=t->ne[i]; } return true;
}
const char * ggml_type_name(ggml_type type) { return type == GGML_TYPE_F32 ? "f32" : "f16"; }
const char * ggml_op_name(ggml_op op) { return op == GGML_OP_ADD ? "ADD" : "RMS_NORM"; }

// ACTUAL_OBSERVER_INSERT

int main(int argc, char ** argv) {
    assert(argc == 3); root=argv[2]; const std::string mode=argv[1];
    SpyQueue queue, other_queue;
    ggml_backend_sycl_context ctx{0,&queue}, other_ctx{0,&queue};
    const int n = mode == "full1" ? 1 : 2;
    std::vector<float> values(4096*n+32, 7.0f), norm(4096*n,-99.0f);
    ggml_backend_sycl_buffer_context bc{0,values.data()};
    ggml_backend_buffer buffer{}; buffer.buft=&sycl_type; buffer.context=&bc; buffer.size=values.size()*4;
    ggml_tensor source{}, node{};
    source.op=GGML_OP_ADD; source.type=GGML_TYPE_F32;
    source.ne[0]=4096; source.ne[1]=n; source.ne[2]=source.ne[3]=1;
    source.nb[0]=4; source.nb[1]=16384; source.nb[2]=source.nb[3]=16384*n;
    source.data=values.data()+16; source.buffer=&buffer; std::strcpy(source.name,"nemotron_h_block_out-0");
    node.op=GGML_OP_RMS_NORM; node.type=GGML_TYPE_F32; node.src[0]=&source;
    node.data=norm.data(); std::strcpy(node.name,"norm-1");
    const auto saved=source; const auto saved_buffer=buffer;
    auto arm = [&] { std::ofstream out(root+"/armed"); out << n << '\n'; out.close(); };
    if (mode=="span") {
        assert(bolt_block_span(ctx,&source,n)==size_t(16384*n));
        source.extra=(void *)1; // F32 extra must never be read, even if unusable.
        assert(bolt_block_span(ctx,&source,n)==size_t(16384*n));
        int excluded=0;
        auto reject = [&](auto mutate) {
            source=saved;buffer=saved_buffer;bc.device=0;bc.base=values.data();metadata_reads=0;
            mutate();
            bool rejected=false; try { bolt_block_span(ctx,&source,n); } catch(const Abort &) { rejected=true; }
            assert(rejected); ++excluded;
        };
        reject([&]{source.op=GGML_OP_VIEW;}); reject([&]{source.type=GGML_TYPE_F16;});
        reject([&]{source.ne[0]=INT64_MAX;}); reject([&]{source.ne[1]=INT64_MAX;});
        reject([&]{source.ne[1]=-1;}); reject([&]{source.ne[2]=2;}); reject([&]{source.ne[3]=2;});
        reject([&]{source.view_src=&node;}); reject([&]{source.view_offs=4;});
        for(int i=0;i<4;++i) reject([&]{source.nb[i]=SIZE_MAX;});
        reject([&]{source.data=nullptr;}); reject([&]{source.data=(char *)source.data+1;});
        reject([&]{source.buffer=nullptr;}); reject([&]{buffer.buft=nullptr;});
        for(auto * type : {&cpu_type,&split_type}) {
            reject([&]{buffer.buft=type;buffer.context=(void *)1;source.extra=(void *)1;});
            assert(metadata_reads==0); // foreign backend metadata must NEVER be touched
        }
        reject([&]{buffer.context=nullptr;}); reject([&]{bc.device=1;});
        reject([&]{bc.base=nullptr;}); reject([&]{buffer.size=4;});
        reject([&]{buffer.size=SIZE_MAX;});
        reject([&]{source.data=(char *)bc.base+buffer.size;});
        reject([&]{source.data=(void *)((uintptr_t)bc.base-4);});
        reject([&]{source.data=(void *)((uintptr_t)bc.base+buffer.size-4);});
        reject([&]{bc.base=(void *)(UINTPTR_MAX-3);source.data=bc.base;});
        std::printf("span: %d metadata/overflow/ownership/view exclusions passed\n",excluded);
        return 0;
    }
    if (mode=="names") {
        assert(bolt_block_index(nullptr)==-1);
        for (int i=0;i<8;++i) {
            std::snprintf(source.name,sizeof(source.name),"nemotron_h_block_out-%d",i);
            assert(bolt_block_index(&source)==i);
        }
        for (auto name : {"nemotron_h_block_out-8","nemotron_h_block_out-00","nemotron_h_block_out-1.extra",
                          "SYCL0#nemotron_h_block_out-0#0","l_out-1","attn_norm-1","nemotron_h_block_out--1"}) {
            std::strcpy(source.name,name); assert(bolt_block_index(&source)==-1);
        }
        std::puts("names: exact closed set; no copied/view/internal aliases passed"); return 0;
    }
    ::setenv("BOLT_BLOCK_OUTPUT_DIR","/early-block-outputs",1);
    if(mode=="full1" || mode=="full2") {
        bolt_block_observe(ctx,&node); assert(queue.copies==0); // unarmed
        arm();
        for(int i=0;i<8;++i) {
            std::snprintf(source.name,sizeof(source.name),"nemotron_h_block_out-%d",i);
            std::snprintf(node.name,sizeof(node.name),"norm-%d",i+1);
            for(int c=0;c<n;++c) for(int j=0;j<4096;++j) values[16+c*4096+j]=i+c*100+j/4096.0f;
            const auto before=source, consumer=node;
            bolt_block_observe(ctx,&node);
            assert(std::memcmp(&source,&before,sizeof(source))==0 && std::memcmp(&node,&consumer,sizeof(node))==0);
            assert(::fcntl(last_capture_fd,F_GETFD)==-1 && errno==EBADF);
            const std::string path=root+"/block-"+std::to_string(i)+".f32.bin";
            std::ifstream in(path,std::ios::binary); std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});
            assert(bytes.size()==size_t(16384*n) && std::memcmp(bytes.data(),source.data,bytes.size())==0);
            assert(queue.source==source.data && queue.source!=node.data);
        }
        ::unlink((root+"/armed").c_str());
        for(int step=0;step<128;++step) for(int i=0;i<8;++i) {
            std::snprintf(source.name,sizeof(source.name),"nemotron_h_block_out-%d",i);
            bolt_block_observe(ctx,&node);
        }
        assert(queue.copies==8 && queue.events==8 && queue.waits==0);
        assert(std::all_of(norm.begin(),norm.end(),[](float f){return f==-99.0f;}));
        std::puts("source snapshot, unchanged tensors/output, same queue, full bytes and once-only128 passed"); return 0;
    }
    arm();
    if(mode=="unarmed") ::unlink((root+"/armed").c_str());
    if(mode=="noenv") ::unsetenv("BOLT_BLOCK_OUTPUT_DIR");
    if(mode=="prefill") source.ne[1]=46;
    if(mode=="wrongop") node.op=GGML_OP_ADD;
    if(mode=="copyname") std::strcpy(source.name,"SYCL0#nemotron_h_block_out-0#0");
    if(mode=="nullsource") node.src[0]=nullptr;
    if(mode=="unarmed" || mode=="noenv" || mode=="prefill" || mode=="wrongop" || mode=="copyname" || mode=="nullsource") {
        bolt_block_observe(ctx,&node); assert(queue.copies==0); std::puts("excluded before copy"); return 0;
    }
    if(mode=="wrongpath") ::setenv("BOLT_BLOCK_OUTPUT_DIR","/tmp/unsafe",1);
    if(mode=="width") source.ne[1]=1;
    if(mode=="order") std::strcpy(source.name,"nemotron_h_block_out-1");
    if(mode=="queue") queue.ordered=false;
    if(mode=="nullqueue") ctx.q=nullptr;
    if(mode=="stale") { std::ofstream out(root+"/block-0.f32.bin"); out << "stale"; }
    if(mode=="filelink") { assert(::symlink("armed",(root+"/block-0.f32.bin").c_str())==0); }
    if(mode=="markerlink") {
        ::unlink((root+"/armed").c_str()); std::ofstream out(root+"/other"); out << "2\n"; out.close();
        assert(::symlink("other",(root+"/armed").c_str())==0);
    }
    if(mode=="markerhardlink") assert(::link((root+"/armed").c_str(),(root+"/other").c_str())==0);
    if(mode=="markerbad") { std::ofstream out(root+"/armed"); out << "2X"; }
    if(mode=="markerfifo") { ::unlink((root+"/armed").c_str()); assert(::mkfifo((root+"/armed").c_str(),0600)==0); }
    if(mode=="duplicate" || mode=="backend" || mode=="changequeue") {
        bolt_block_observe(ctx,&node);
        if(mode!="duplicate") std::strcpy(source.name,"nemotron_h_block_out-1");
        if(mode=="changequeue") ctx.q=&other_queue;
    }
    if(mode=="async") queue.fail_async=true;
    if(mode=="submit") queue.fail_submit=true;
    bool rejected=false;
    try { bolt_block_observe(mode=="backend" ? other_ctx : ctx,&node); }
    catch(const std::exception & e) { std::printf("rejected %s: %s\n",mode.c_str(),e.what()); rejected=true; }
    assert(rejected);
    if(mode=="async" || mode=="submit") {
        assert(queue.copies==1 && queue.waits==1 && queue.host==nullptr);
        assert(::fcntl(last_capture_fd,F_GETFD)==-1 && errno==EBADF);
        std::puts("exception completion with host alive and descriptor closed passed");
    }
}

// CPU contract harness. Actual patched method/destructor and native copy/dup bodies
// are inserted by the test. Backend operations below are fakes, NOT GPU execution.
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-opt.h"
#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::string> events, logs;
static int contexts = 0, buffers = 0, init_calls = 0, fail_init_at = 0;
static bool fail_alloc = false, fail_read = false, corrupt_read = false;
static bool supported = true, contiguous = true;
static size_t alloc_size = 136;
static void log_message(const char * fmt, ...) {
    char out[1024]; va_list args; va_start(args, fmt);
    vsnprintf(out, sizeof(out), fmt, args); va_end(args); logs.emplace_back(out);
}
#define LLAMA_LOG_INFO(...) log_message(__VA_ARGS__)
#define LLAMA_LOG_WARN(...) log_message(__VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) log_message(__VA_ARGS__)
#define GGML_LOG_DEBUG(...) log_message(__VA_ARGS__)

struct ggml_context { std::vector<ggml_tensor *> tensors; };
struct ggml_backend_reg { const char * name = "SYCL"; };
struct ggml_backend_buffer_type { bool host = false; };
struct ggml_backend_device {
    ggml_backend_reg reg;
    ggml_backend_buffer_type buft;
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_GPU;
};
struct ggml_backend { ggml_backend_device * device; };
struct ggml_backend_buffer {
    bool host = false, owned = false;
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes, pending;
    ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_ANY;
};
struct ggml_backend_sched {};

extern "C" {
void ggml_abort(const char *, int, const char *, ...) { std::abort(); }
ggml_context * ggml_init(ggml_init_params) {
    if (++init_calls == fail_init_at) { return nullptr; }
    ++contexts; return new ggml_context;
}
void ggml_free(ggml_context * ctx) {
    events.push_back("metadata-free");
    for (auto * t : ctx->tensors) { delete t; }
    delete ctx; --contexts;
}
size_t ggml_tensor_overhead() { return sizeof(ggml_tensor) + 32; }
ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n, const int64_t * ne) {
    auto * t = new ggml_tensor{};
    t->type = type;
    for (int i = 0; i < 4; ++i) { t->ne[i] = i < n ? ne[i] : 1; }
    t->nb[0] = type == GGML_TYPE_Q8_0 ? 34 : 4;
    t->nb[1] = t->nb[0] * t->ne[0] / (type == GGML_TYPE_Q8_0 ? 32 : 1);
    t->nb[2] = t->nb[1] * t->ne[1]; t->nb[3] = t->nb[2] * t->ne[2];
    ctx->tensors.push_back(t); return t;
}
ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t n) {
    return ggml_new_tensor(ctx, type, 1, &n);
}
ggml_tensor * ggml_get_rows(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    int64_t ne[] = {a->ne[0], b->ne[0]};
    auto * r = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    r->op = GGML_OP_GET_ROWS; r->src[0] = a; r->src[1] = b; return r;
}
ggml_tensor * ggml_set_name(ggml_tensor * t, const char * name) { snprintf(t->name, sizeof(t->name), "%s", name); return t; }
size_t ggml_nbytes(const ggml_tensor * t) { return t->nb[3] * t->ne[3]; }
bool ggml_is_contiguous(const ggml_tensor *) { return contiguous; }
bool ggml_are_same_layout(const ggml_tensor * a, const ggml_tensor * b) {
    return a->type == b->type && !memcmp(a->ne, b->ne, sizeof(a->ne)) && !memcmp(a->nb, b->nb, sizeof(a->nb));
}
const char * ggml_type_name(ggml_type t) { assert(t == GGML_TYPE_Q8_0); return "q8_0"; }
bool ggml_backend_buffer_is_host(ggml_backend_buffer_t b) { return b->host; }
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t b) { return b->device; }
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t d) { return &d->reg; }
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t d) { return d->type; }
const char * ggml_backend_reg_name(ggml_backend_reg_t r) { return r->name; }
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t b) { return &b->device->buft; }
bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t b) { return b->host; }
size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor *) { return alloc_size; }
bool ggml_backend_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * t) {
    assert(t->op == GGML_OP_GET_ROWS && t->src[0]->type == GGML_TYPE_Q8_0); return supported;
}
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ggml_context * ctx, ggml_backend_buffer_type_t) {
    if (fail_alloc) { return nullptr; }
    assert(ctx->tensors.size() == 1);
    auto * b = new ggml_backend_buffer;
    b->owned = true; b->tensor = ctx->tensors[0]; b->bytes.resize(ggml_nbytes(b->tensor));
    b->tensor->buffer = b; b->tensor->data = b->bytes.data(); b->tensor->extra = b;
    ++buffers; events.push_back("allocate"); return b;
}
void ggml_backend_buffer_free(ggml_backend_buffer_t b) {
    if (!b) { return; }
    assert(b->owned && contexts > 0);
    // Native SYCL buffer destruction drains staging even on a construction failure.
    b->pending.clear(); events.push_back("buffer-free"); delete b; --buffers;
}
void ggml_backend_buffer_set_usage(ggml_backend_buffer_t b, ggml_backend_buffer_usage u) { b->usage = u; }
void ggml_backend_tensor_set(ggml_tensor * t, const void * p, size_t offset, size_t size) {
    assert(offset == 0 && size == t->buffer->bytes.size());
    t->buffer->pending.assign((const uint8_t *)p, (const uint8_t *)p + size);
    events.push_back("copy-queued");
}
void ggml_backend_tensor_get(const ggml_tensor * t, void * p, size_t offset, size_t size) {
    auto * b = t->buffer;
    if (fail_read) { throw std::runtime_error("read failure"); }
    assert(!b->pending.empty());
    std::copy(b->pending.begin(), b->pending.end(), b->bytes.begin()); b->pending.clear();
    memcpy(p, b->bytes.data() + offset, size);
    if (corrupt_read) { *(uint8_t *)p ^= 1; }
    events.push_back("upload-fenced");
}
bool ggml_backend_buffer_copy_tensor(const ggml_tensor *, ggml_tensor *) { assert(false); return false; }
const char * ggml_backend_name(ggml_backend_t) { return "SYCL0"; }
const char * ggml_backend_buffer_name(ggml_backend_buffer_t b) { return b->host ? "SYCL_Host" : "SYCL0"; }
size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t b) { return b->bytes.size(); }
void ggml_backend_free(ggml_backend_t b) { events.push_back("backend-free"); delete b; }
void ggml_backend_sched_free(ggml_backend_sched_t s) { if (s) { assert(buffers > 0); events.push_back("scheduler-free"); delete s; } }
size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t, ggml_backend_t) { return 0; }
const char * ggml_backend_buft_name(ggml_backend_buffer_type_t) { return "SYCL0"; }
void ggml_opt_free(ggml_opt_context_t) {}
}
// EXTRACTED_DUP
// EXTRACTED_COPY

constexpr int LLM_ARCH_DFLASH = 1;
struct llama_context;
struct llama_hparams { bool no_alloc = false, vocab_only = false; int lilicorr_candidate_topk = 3; };
struct llama_model {
    int arch = LLM_ARCH_DFLASH;
    ggml_tensor * tok_embd = nullptr;
    void * lilicorr_pass_hidden_proj = (void *)1;
    llama_hparams hparams;
    struct device { ggml_backend_dev_t dev; };
    std::vector<device> devices;
    uint32_t gpu_layers = 5;
    uint32_t n_gpu_layers() const { return gpu_layers; }
};
struct llama_cparams {
    bool op_offload = true;
    llama_context * ctx_other = nullptr;
    // EXTRACTED_CPARAMS_POINTER
};
struct llama_context {
    const llama_model & model;
    llama_cparams cparams;
    // EXTRACTED_OWNERS
    // Relative order mirrors the real header: owners, scheduler, backends.
    ggml_backend_sched_ptr sched;
    std::vector<ggml_backend_ptr> backends;
    ggml_opt_context_t opt_ctx = nullptr;
    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t> backend_buf_exp_size;
    void synchronize() { events.push_back("synchronize"); }
    explicit llama_context(const llama_model & m) : model(m) {}
    llama_context(const llama_model & m, llama_context * other, ggml_backend_dev_t dev, bool fail_after) : model(m) {
        cparams.ctx_other = other;
        backends.emplace_back(new ggml_backend{dev});
        init_lilicorr_device_embedding();
        if (fail_after) { throw std::runtime_error("later constructor failure"); }
    }
    ~llama_context();
};
const llama_model * llama_get_model(const llama_context * c) { return &c->model; }
// EXTRACTED_INIT
// EXTRACTED_DESTRUCTOR

struct fixture {
    ggml_backend_device device;
    ggml_backend_buffer host;
    ggml_context_ptr metadata{ggml_init({})};
    ggml_tensor * source;
    llama_model target_model, draft_model;
    llama_context target{target_model};
    fixture() {
        host.host = true; host.bytes.resize(136);
        for (size_t i = 0; i < host.bytes.size(); ++i) { host.bytes[i] = uint8_t(i * 13 + 1); }
        int64_t ne[] = {32, 4}; source = ggml_new_tensor(metadata.get(), GGML_TYPE_Q8_0, 2, ne);
        source->buffer = &host; source->data = host.bytes.data(); target_model.tok_embd = source;
        target_model.arch = 0; draft_model.devices.push_back({&device});
    }
    void setup(llama_context & c) {
        c.cparams.ctx_other = &target;
        c.backends.emplace_back(new ggml_backend{&device});
    }
};
static void reset() {
    assert(contexts == 0 && buffers == 0);
    init_calls = fail_init_at = 0; fail_alloc = fail_read = corrupt_read = false;
    supported = contiguous = true; alloc_size = 136; events.clear(); logs.clear();
}

int main() {
    reset();
    {
        fixture f; auto original = *f.source; auto bytes = f.host.bytes;
        {
            llama_context a(f.draft_model), b(f.draft_model); f.setup(a); f.setup(b);
            a.init_lilicorr_device_embedding(); b.init_lilicorr_device_embedding();
            auto * x = a.cparams.lilicorr_tok_embd; auto * y = b.cparams.lilicorr_tok_embd;
            assert(x && y && x != y && x != f.source);
            assert(ggml_are_same_layout(x, f.source) && x->extra != f.source->extra);
            assert(x->buffer->bytes == bytes && y->buffer->bytes == bytes);
            assert(x->buffer->pending.empty() && y->buffer->pending.empty());
            assert(x->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == bytes);
            assert(f.target_model.tok_embd == f.source && f.draft_model.tok_embd == nullptr);
            assert(logs.size() == 2 && logs[0].find("active backend=SYCL0") != std::string::npos);
            a.sched.reset(new ggml_backend_sched); b.sched.reset(new ggml_backend_sched);
        }
        assert(buffers == 0 && contexts == 1);
        auto sync = std::find(events.begin(), events.end(), "synchronize");
        auto sched = std::find(events.begin(), events.end(), "scheduler-free");
        auto free = std::find(events.begin(), events.end(), "buffer-free");
        assert(sync < sched && sched < free);
    }
    for (int guard = 0; guard < 24; ++guard) {
        reset();
        fixture f; llama_context c(f.draft_model); f.setup(c);
        switch (guard) {
            case 0: f.draft_model.arch = 0; break;
            case 1: f.draft_model.lilicorr_pass_hidden_proj = nullptr; break;
            case 2: f.draft_model.hparams.lilicorr_candidate_topk = 0; break;
            case 3: f.draft_model.tok_embd = f.source; break;
            case 4: c.cparams.ctx_other = nullptr; break;
            case 5: c.cparams.op_offload = false; break;
            case 6: f.draft_model.hparams.vocab_only = true; break;
            case 7: f.draft_model.hparams.no_alloc = true; break;
            case 8: f.draft_model.gpu_layers = 0; break;
            case 9: f.draft_model.devices.push_back({&f.device}); break;
            case 10: f.source->buffer = nullptr; break;
            case 11: f.source->data = nullptr; break;
            case 12: f.host.host = false; break;
            case 13: f.source->extra = (void *)1; break;
            case 14: f.source->view_src = f.source; break;
            case 15: f.source->type = GGML_TYPE_F16; break;
            case 16: contiguous = false; break;
            case 17: f.source->ne[2] = 2; break;
            case 18: f.device.reg.name = "CUDA"; break;
            case 19: f.device.type = GGML_BACKEND_DEVICE_TYPE_CPU; break;
            case 20: f.draft_model.devices[0].dev = nullptr; break;
            case 21: f.device.buft.host = true; break;
            case 22: alloc_size = size_t(4) << 30; break;
            case 23: supported = false; break;
        }
        auto original = *f.source; auto bytes = f.host.bytes;
        c.init_lilicorr_device_embedding();
        assert(!c.cparams.lilicorr_tok_embd && buffers == 0 && contexts == 1);
        assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == bytes);
        assert(guard < 3 ? logs.empty() : (logs.size() == 1 && logs[0].find("inactive") != std::string::npos));
    }
    for (int failure = 0; failure < 6; ++failure) {
        reset();
        fixture f; auto original = *f.source; auto bytes = f.host.bytes;
        if (failure == 0) { fail_init_at = init_calls + 1; }
        if (failure == 1) { fail_init_at = init_calls + 2; }
        fail_alloc = failure == 2; fail_read = failure == 3; corrupt_read = failure == 4;
        bool caught = false;
        try {
            // The context destructor is NOT invoked if its constructor throws.
            llama_context c(f.draft_model, &f.target, &f.device, failure == 5);
        } catch (const std::runtime_error &) { caught = true; }
        assert(caught && buffers == 0 && contexts == 1);
        assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == bytes);
        if (failure != 5) { assert(logs.empty()); }
    }
    reset();
    puts("extracted placement: 24 guards, independent owners, fenced standard-byte copy, 6 failure paths passed");
}

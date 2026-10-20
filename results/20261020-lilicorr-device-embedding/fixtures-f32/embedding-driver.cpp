// CPU-only contract/numeric harness: extracted production initializer/destructor,
// native CPU GET_ROWS and actual native Q8 type traits/conversion (separate C TU).
// Backend/device operations are fakes; this is NOT GPU execution.
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-opt.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
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
static bool fail_alloc = false, fail_read = false, corrupt_read = false, fail_set = false;
static bool supported = true, contiguous = true;
static size_t alloc_size = 0, uploads = 0, fences = 0, queued_bytes = 0;
static size_t fail_read_at = 0;
static ggml_tensor ** watched_override = nullptr;
static void log_message(const char * fmt, ...) {
    char out[1024]; va_list args; va_start(args, fmt);
    vsnprintf(out, sizeof(out), fmt, args); va_end(args); logs.emplace_back(out);
}
#define LLAMA_LOG_INFO(...) log_message(__VA_ARGS__)
#define LLAMA_LOG_WARN(...) log_message(__VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) log_message(__VA_ARGS__)

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
    const uint8_t * pending_source = nullptr;
    size_t pending_offset = 0, next_offset = 0;
    ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_ANY;
};
struct ggml_backend_sched {};

static void drain(ggml_backend_buffer * b) {
    if (b->pending.empty()) { return; }
    // Retain the caller's actual scratch pointer until the default-queue fence.
    // ASan catches early free; snapshot comparison catches premature reuse.
    assert(!memcmp(b->pending_source, b->pending.data(), b->pending.size()));
    memcpy(b->bytes.data() + b->pending_offset, b->pending_source, b->pending.size());
    // Poison consumed scratch so the next chunk must fully overwrite its range.
    memset(const_cast<uint8_t *>(b->pending_source), 0xa5, b->pending.size());
    b->pending.clear(); b->pending_source = nullptr;
}

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
bool ggml_backend_buffer_is_host(ggml_backend_buffer_t b) { return b->host; }
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t b) { return b->device; }
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t d) { return &d->reg; }
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t d) { return d->type; }
const char * ggml_backend_reg_name(ggml_backend_reg_t r) { return r->name; }
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t b) { return &b->device->buft; }
bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t b) { return b->host; }
size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) { assert(t->type == GGML_TYPE_F32); return alloc_size ? alloc_size : ggml_nbytes(t); }
bool ggml_backend_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * t) {
    assert(t->op == GGML_OP_GET_ROWS && t->src[0]->type == GGML_TYPE_F32); return supported;
}
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ggml_context * ctx, ggml_backend_buffer_type_t) {
    if (fail_alloc) { return nullptr; }
    assert(ctx->tensors.size() == 1);
    auto * b = new ggml_backend_buffer;
    b->owned = true; b->tensor = ctx->tensors[0]; b->bytes.resize(ggml_nbytes(b->tensor), 0xcd);
    assert(b->tensor->type == GGML_TYPE_F32 && !b->tensor->extra);
    b->tensor->buffer = b; b->tensor->data = b->bytes.data();
    ++buffers; events.push_back("allocate"); return b;
}
void ggml_backend_buffer_free(ggml_backend_buffer_t b) {
    if (!b) { return; }
    assert(b->owned && contexts > 0);
    // Native SYCL buffer destruction drains staging even on a construction failure.
    drain(b); events.push_back("buffer-free"); delete b; --buffers;
}
void ggml_backend_buffer_set_usage(ggml_backend_buffer_t b, ggml_backend_buffer_usage u) { b->usage = u; }
void ggml_backend_tensor_set(ggml_tensor * t, const void * p, size_t offset, size_t size) {
    auto * b = t->buffer;
    assert(watched_override && !*watched_override);
    assert(b->pending.empty() && !b->pending_source && offset == b->next_offset);
    assert(size && size <= (size_t(16) << 20) && offset + size <= b->bytes.size());
    assert(offset % t->nb[1] == 0 && size % t->nb[1] == 0);
    assert(std::all_of(b->bytes.begin() + offset, b->bytes.begin() + offset + size,
                      [](uint8_t v) { return v == 0xcd; }));
    b->pending_source = static_cast<const uint8_t *>(p);
    b->pending.assign(b->pending_source, b->pending_source + size);
    b->pending_offset = offset; b->next_offset = offset + size;
    ++uploads; queued_bytes += size; events.push_back("copy-queued");
    if (fail_set) { throw std::runtime_error("queued set failure"); }
}
void ggml_backend_tensor_get(const ggml_tensor * t, void * p, size_t offset, size_t size) {
    auto * b = t->buffer;
    assert(watched_override && !*watched_override);
    assert(!b->pending.empty() && size == sizeof(float));
    assert(offset + size == b->pending_offset + b->pending.size());
    if (fail_read || uploads == fail_read_at) { throw std::runtime_error("read failure"); }
    drain(b);
    memcpy(p, b->bytes.data() + offset, size);
    if (corrupt_read) { *(uint8_t *)p ^= 1; }
    ++fences; events.push_back("upload-fenced");
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
// EXTRACTED_NEW2D
struct ggml_compute_params { int ith, nth; }; // Only these fields are read by this native function.
// EXTRACTED_CPU_GET_ROWS

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
        watched_override = &cparams.lilicorr_tok_embd;
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
    explicit fixture(int64_t nrows = 4, int64_t ncols = 32) {
        int64_t ne[] = {ncols, nrows}; source = ggml_new_tensor(metadata.get(), GGML_TYPE_Q8_0, 2, ne);
        host.host = true; host.bytes.resize(ggml_nbytes(source));
        auto * blocks = reinterpret_cast<block_q8_0 *>(host.bytes.data());
        const int8_t values[] = {-128, -127, -3, -1, 0, 1, 3, 127};
        for (int64_t i = 0; i < nrows * ncols / QK8_0; ++i) {
            blocks[i].d = ggml_fp16_t(i & 0xffff); // All half bit patterns, including signed zero.
            for (int j = 0; j < QK8_0; ++j) { blocks[i].qs[j] = values[j % 8]; }
        }
        source->buffer = &host; source->data = host.bytes.data(); target_model.tok_embd = source;
        target_model.arch = 0; draft_model.devices.push_back({&device});
    }
    void setup(llama_context & c) {
        c.cparams.ctx_other = &target;
        watched_override = &c.cparams.lilicorr_tok_embd;
        c.backends.emplace_back(new ggml_backend{&device});
    }
};
static void reset() {
    assert(contexts == 0 && buffers == 0);
    init_calls = fail_init_at = 0; fail_alloc = fail_read = corrupt_read = fail_set = false;
    supported = contiguous = true; alloc_size = uploads = fences = queued_bytes = fail_read_at = 0;
    watched_override = nullptr; events.clear(); logs.clear();
}

static void test_numeric(int64_t ncols) {
    const int64_t chunk_rows = (size_t(16) << 20) / (ncols * sizeof(float));
    for (int64_t nrows : {chunk_rows - 1, chunk_rows, chunk_rows + 3, 2 * chunk_rows + 5}) {
        reset();
        fixture f(nrows, ncols); auto original = *f.source; auto source_bytes = f.host.bytes;
        // Actual native trait/function, not a hand-written quantization formula.
        std::vector<float> reference(size_t(nrows) * ncols);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(f.source->data, reference.data(), reference.size());
        {
            llama_context c(f.draft_model); f.setup(c); c.init_lilicorr_device_embedding();
            auto * x = c.cparams.lilicorr_tok_embd;
            assert(x && x->type == GGML_TYPE_F32 && !x->extra);
            assert(ggml_nbytes(x) == reference.size() * sizeof(float));
            assert(!memcmp(x->data, reference.data(), ggml_nbytes(x)));
            assert(uploads == size_t((nrows + chunk_rows - 1) / chunk_rows) && uploads == fences);
            assert(queued_bytes == ggml_nbytes(x) && x->buffer->pending.empty());
            auto bits = [x](size_t block, size_t col) {
                uint32_t value;
                memcpy(&value, static_cast<const uint8_t *>(x->data) + (block * QK8_0 + col) * 4, 4);
                return value;
            };
            assert(bits(0, 0) == 0x80000000 && bits(0, 4) == 0); // +0 scale * negative / zero
            assert(bits(0x8000, 0) == 0 && bits(0x8000, 4) == 0x80000000); // -0 scale
            assert(bits(1, 7) == 0x36fe0000); // smallest half subnormal * 127, exact F32
            assert(bits(0x0400, 7) == 0x3bfe0000); // smallest normal half * 127
            assert(bits(0x3555, 6) == 0x3f7ff000); // 0.999755859375; half multiplication rounds to 1
            assert(bits(0x7bff, 7) == 0x4afde040); // maximum finite half * 127 stays finite
            // Execute the ORIGINAL CPU GET_ROWS function with real trait dispatch,
            // including row ids on both sides of chunk boundaries and signed zeros.
            ggml_context_ptr rows_ctx(ggml_init({}));
            std::vector<int32_t> ids{0, 1, int32_t(0x8000 / (ncols / QK8_0)), int32_t(0x3555 / (ncols / QK8_0)), int32_t(0x7bff / (ncols / QK8_0)),
                                     int32_t(chunk_rows - 1), int32_t(chunk_rows),
                                     int32_t(chunk_rows + 1), int32_t(nrows - 1)};
            for (auto & id : ids) { id %= nrows; }
            auto * indices = ggml_new_tensor_1d(rows_ctx.get(), GGML_TYPE_I32, ids.size());
            indices->data = ids.data();
            auto * selected = ggml_get_rows(rows_ctx.get(), f.source, indices);
            std::vector<float> cpu_rows(ids.size() * ncols); selected->data = cpu_rows.data();
            for (int ith = 0; ith < 3; ++ith) {
                const ggml_compute_params params{ith, 3};
                ggml_compute_forward_get_rows_q(&params, selected);
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                assert(!memcmp(cpu_rows.data() + i * ncols,
                               static_cast<const uint8_t *>(x->data) + size_t(ids[i]) * x->nb[1], x->nb[1]));
            }
        }
        assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == source_bytes);
        assert(contexts == 1 && buffers == 0);
    }
    reset();
    printf("native CPU Q8 traits/GET_ROWS: ncols=%zu, bit-exact F32, all 65536 half patterns, int8 extremes, signed zero, 16MiB boundaries/tails, fenced poisoned scratch passed\n", size_t(ncols));
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "numeric")) { test_numeric(32); test_numeric(4096); return 0; }
    assert(argc == 1);
    reset();
    {
        fixture f; auto original = *f.source; auto bytes = f.host.bytes;
        std::vector<float> reference(128);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(f.source->data, reference.data(), reference.size());
        {
            llama_context a(f.draft_model), b(f.draft_model); f.setup(a); f.setup(b);
            watched_override = &a.cparams.lilicorr_tok_embd; a.init_lilicorr_device_embedding();
            watched_override = &b.cparams.lilicorr_tok_embd; b.init_lilicorr_device_embedding();
            auto * x = a.cparams.lilicorr_tok_embd; auto * y = b.cparams.lilicorr_tok_embd;
            assert(x && y && x != y && x != f.source);
            assert(x->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && !x->extra && !y->extra);
            assert(!memcmp(x->ne, f.source->ne, sizeof(x->ne)));
            assert(ggml_nbytes(x) == reference.size() * sizeof(float));
            assert(!memcmp(x->data, reference.data(), ggml_nbytes(x)));
            assert(!memcmp(y->data, reference.data(), ggml_nbytes(y)));
            assert(uploads == 2 && fences == 2);
            assert(x->buffer->pending.empty() && y->buffer->pending.empty());
            assert(x->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == bytes);
            assert(f.target_model.tok_embd == f.source && f.draft_model.tok_embd == nullptr);
            assert(logs.size() == 2 && logs[0].find("active backend=SYCL0") != std::string::npos);
            assert(logs[0].find("source_type=q8_0 source_bytes=136 type=f32 tensor_bytes=512 buffer_bytes=512") != std::string::npos);
            a.sched.reset(new ggml_backend_sched); b.sched.reset(new ggml_backend_sched);
        }
        assert(buffers == 0 && contexts == 1);
        auto sync = std::find(events.begin(), events.end(), "synchronize");
        auto sched = std::find(events.begin(), events.end(), "scheduler-free");
        auto free = std::find(events.begin(), events.end(), "buffer-free");
        assert(sync < sched && sched < free);
    }
    for (int guard = 0; guard < 30; ++guard) {
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
            case 22: alloc_size = (size_t(3) << 30) + 1; break;
            case 23: supported = false; break;
            case 24: f.source->ne[0] = 0; break;
            case 25: f.source->ne[1] = 0; break;
            case 26: f.source->ne[0] = 33; break;
            case 27: f.source->ne[0] = (size_t(16) << 20) / sizeof(float) + 32; break;
            case 28: f.source->ne[1] = (size_t(3) << 30) / (32 * sizeof(float)) + 1; break;
            case 29: f.source->ne[3] = 2; break;
        }
        auto original = *f.source; auto bytes = f.host.bytes;
        c.init_lilicorr_device_embedding();
        assert(!c.cparams.lilicorr_tok_embd && buffers == 0 && contexts == 1);
        assert(!memcmp(&original, f.source, sizeof(original)) && f.host.bytes == bytes);
        assert(guard < 3 ? logs.empty() : (logs.size() == 1 && logs[0].find("inactive") != std::string::npos));
    }
    for (int failure = 0; failure < 8; ++failure) {
        reset();
        fixture f(failure == 7 ? 131075 : 4); auto original = *f.source; auto bytes = f.host.bytes;
        if (failure == 0) { fail_init_at = init_calls + 1; }
        if (failure == 1) { fail_init_at = init_calls + 2; }
        fail_alloc = failure == 2; fail_read = failure == 3; corrupt_read = failure == 4;
        fail_set = failure == 6; fail_read_at = failure == 7 ? 2 : 0;
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
    puts("extracted placement: 30 guards, independent F32 owners, chunk/scratch fences, 8 failure paths passed");
}

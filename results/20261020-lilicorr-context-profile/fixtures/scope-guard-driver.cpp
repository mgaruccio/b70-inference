// CPU-only driver: Python inserts the ACTUAL patched helpers and graph_compute.
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

static thread_local bool fail_allocation = false;
void * operator new(std::size_t n) {
    if (fail_allocation) { fail_allocation = false; throw std::bad_alloc(); }
    if (void * p = std::malloc(n)) { return p; }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

// EXTRACTED_BACKEND_SCOPE

struct backend { int kind; };
using ggml_backend_t = backend *;
using ggml_backend_ptr = std::unique_ptr<backend>;
using ggml_threadpool_t = void *;
static thread_local bool other_flag = true;
static int calls[128], n_calls = 0, lookups = 0, throw_lookup = 0;
static bool set_first(bool value) noexcept {
    calls[n_calls++] = 1 + 100 * value;
    return set_lilicorr_profile_scope(value);
}
static bool set_other(bool value) noexcept {
    calls[n_calls++] = 2 + 100 * value;
    const bool previous = other_flag; other_flag = value; return previous;
}
static backend * ggml_backend_get_device(backend * b) { return b; }
static backend * ggml_backend_dev_backend_reg(backend * b) { return b; }
static void * ggml_backend_reg_get_proc_address(backend * reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) { return nullptr; }
    assert(std::strcmp(name, "set_lilicorr_profile_scope") == 0);
    if (++lookups == throw_lookup) { throw std::runtime_error("lookup failure"); }
    if (reg->kind == 1) { return reinterpret_cast<void *>(set_first); }
    if (reg->kind == 2) { return reinterpret_cast<void *>(set_other); }
    return nullptr;
}
static void ggml_backend_cpu_set_threadpool(ggml_backend_t, ggml_threadpool_t) {}

// EXTRACTED_CONTEXT_SCOPE

using ggml_status = int;
static constexpr int GGML_STATUS_SUCCESS = 0;
struct ggml_cgraph {};
struct llama_context;
struct scheduler { bool expected = false; int mode = 0; llama_context * nested = nullptr; };
struct llama_context {
    struct { struct { int lilicorr_candidate_topk = 0; } hparams; } model;
    struct { int n_threads = 1, n_threads_batch = 1; } cparams;
    ggml_threadpool_t threadpool = nullptr, threadpool_batch = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<std::pair<ggml_backend_t, void (*)(ggml_backend_t, int)>> set_n_threads_fns;
    struct { void reset() {} } copy_experts;
    std::vector<ggml_backend_ptr> backends;
    std::unique_ptr<scheduler> sched = std::make_unique<scheduler>();
    ggml_status graph_compute(ggml_cgraph *, bool);
    explicit llama_context(bool drafter) {
        model.hparams.lilicorr_candidate_topk = drafter ? 8 : 0;
        for (int k : {1, 0, 2, 1}) { backends.push_back(std::make_unique<backend>(backend{k})); }
    }
};
static int segments = 0, errors = 0;
static const auto submitting_thread = std::this_thread::get_id();
#define LLAMA_LOG_ERROR(...) do { assert(!lilicorr_sycl_profile_context); ++errors; } while (0)
static ggml_status ggml_backend_sched_graph_compute_async(scheduler * sched, ggml_cgraph * gf) {
    // The qualified scheduler uses this same direct, same-host-thread call chain.
    for (int i = 0; i < 3; ++i) {
        assert(std::this_thread::get_id() == submitting_thread);
        assert(lilicorr_sycl_profile_context == sched->expected);
        ++segments;
    }
    if (sched->nested) {
        sched->nested->graph_compute(gf, false);
        assert(lilicorr_sycl_profile_context == sched->expected);
    }
    if (sched->mode == 2) { throw std::runtime_error("submit failure"); }
    return sched->mode == 1 ? -1 : GGML_STATUS_SUCCESS;
}

// EXTRACTED_GRAPH_COMPUTE

int main() {
    llama_context draft(true), target(false);
    ggml_cgraph graph;
    unsetenv("LLAMA_LILICORR_PROFILE");
    draft.graph_compute(&graph, false);
    assert(lookups == 0 && n_calls == 0 && !lilicorr_sycl_profile_context);
    for (const char * off : {"", "0", "01", "true", "1 "}) {
        setenv("LLAMA_LILICORR_PROFILE", off, 1);
        draft.graph_compute(&graph, false);
        assert(lookups == 0 && n_calls == 0);
    }
    setenv("LLAMA_LILICORR_PROFILE", "1", 1);
    target.graph_compute(&graph, false);
    assert(lookups == 0 && !lilicorr_sycl_profile_context); // native stand-alone also defaults false
    draft.sched->expected = true;
    draft.graph_compute(&graph, true);
    assert(!lilicorr_sycl_profile_context && other_flag && lilicorr_context_profile_depth == 0);
    assert(n_calls == 6 && calls[3] == 101 && calls[4] == 102 && calls[5] == 1); // reverse, duplicate reg
    n_calls = 0;
    {
        lilicorr_context_profile_scope outer(draft.backends, true);
        std::thread independent([] { assert(!lilicorr_sycl_profile_context && lilicorr_context_profile_depth == 0); });
        independent.join();
        target.graph_compute(&graph, false); // nested target must suspend, then restore true
        draft.graph_compute(&graph, false);  // nested drafter must restore true, not false
        assert(lilicorr_sycl_profile_context);
        outer.restore();
        assert(!lilicorr_sycl_profile_context);
        const int restored = n_calls;
        outer.restore(); assert(n_calls == restored);
    }
    assert(!lilicorr_sycl_profile_context && lilicorr_context_profile_depth == 0);
    n_calls = 0; draft.sched->nested = &target;
    draft.graph_compute(&graph, false); // same-thread nested context through actual graph_compute
    draft.sched->nested = nullptr;
    for (int mode : {1, 2}) {
        n_calls = 0; draft.sched->mode = mode;
        bool threw = false;
        try { assert(draft.graph_compute(&graph, false) == -1); }
        catch (const std::runtime_error &) { threw = true; }
        assert(threw == (mode == 2));
        assert(!lilicorr_sycl_profile_context && other_flag && lilicorr_context_profile_depth == 0);
    }
    draft.sched->mode = 0; n_calls = 0;
    fail_allocation = true;
    try { draft.graph_compute(&graph, false); assert(false); } catch (const std::bad_alloc &) {}
    assert(n_calls == 0 && !lilicorr_sycl_profile_context && lilicorr_context_profile_depth == 0);
    lookups = 0; throw_lookup = 3;
    try { draft.graph_compute(&graph, false); assert(false); } catch (const std::runtime_error &) {}
    assert(n_calls == 0 && !lilicorr_sycl_profile_context && lilicorr_context_profile_depth == 0);
    throw_lookup = 0;
    {
        lilicorr_context_profile_scope outer(draft.backends, true);
        n_calls = 0; fail_allocation = true;
        try { target.graph_compute(&graph, false); assert(false); } catch (const std::bad_alloc &) {}
        assert(n_calls == 0 && lilicorr_sycl_profile_context && lilicorr_context_profile_depth == 1);
    }
    assert(!lilicorr_sycl_profile_context && other_flag && errors == 1 && segments > 20);
}

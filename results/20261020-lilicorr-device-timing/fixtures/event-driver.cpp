// CPU mocks exercise the extracted production diagnostic and its two real launch sites.
// No SYCL SDK, device execution, numerical, or runtime API compatibility claim.
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

static std::atomic<int> clocks{0}, queries{0}, premature{0}, waits{0};
static thread_local bool copy_error = false;
static std::mutex log_mutex;
static std::string logs;
static int64_t ggml_time_us() { return ++clocks; }
static void log_message(const char * format, ...) {
    char line[2048]; va_list args; va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args); va_end(args);
    std::lock_guard<std::mutex> lock(log_mutex); logs += line;
}
#define GGML_LOG_INFO(...) log_message(__VA_ARGS__)
#define GGML_SYCL_DEBUG(...) ((void) 0)
#define GGML_UNUSED(x) ((void) (x))
#define GGML_ASSERT(x) assert(x)
#define SYCL_CHECK(x) (x)
#define CHECK_TRY_ERROR(x) (x)
constexpr int GGML_MAX_NAME = 64, WARP_SIZE = 16, QK8_0 = 32, QK8_1 = 32, GGML_SYCL_MMV_Y = 1;
constexpr int GGML_OP_MUL_MAT = 1, GGML_TYPE_Q8_0 = 2, GGML_GLU_OP_SWIGLU = 3;
struct ggml_tensor {
    int op = GGML_OP_MUL_MAT, type = GGML_TYPE_Q8_0;
    int64_t ne[4] = {4096, 8, 1, 1};
    char name[GGML_MAX_NAME] = "output";
    ggml_tensor * src[2] = {};
};
struct ggml_cgraph { int n_nodes = 826; };

namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
namespace property { namespace queue { struct enable_profiling {}; struct in_order {}; } }
enum class aspect { queue_profiling, fp16 };
namespace info {
namespace event { struct command_execution_status {}; }
namespace event_profiling { struct command_start {}; struct command_end {}; }
enum class event_command_status { submitted, complete };
}
struct state { bool complete = false, query_error = false; uint64_t start = 1000, end = 1075; };
struct event {
    std::shared_ptr<state> value;
    explicit event(std::shared_ptr<state> v) : value(std::move(v)) {}
    event(const event & e) : value(e.value) { if (copy_error) { throw exception("event-copy"); } }
    event(event &&) noexcept = default;
    event & operator=(const event & e) {
        if (copy_error) { throw exception("event-copy"); } value = e.value; return *this;
    }
    event & operator=(event &&) noexcept = default;
    template<class T> info::event_command_status get_info() const {
        static_assert(std::is_same_v<T, info::event::command_execution_status>);
        return value->complete ? info::event_command_status::complete : info::event_command_status::submitted;
    }
    template<class T> uint64_t get_profiling_info() const {
        ++queries;
        if (!value->complete) { ++premature; throw exception("premature-query"); }
        if (value->query_error) { throw exception("unsupported-event-profiling"); }
        return std::is_same_v<T, info::event_profiling::command_start> ? value->start : value->end;
    }
    // Deliberately no event::wait: an added wait in production fails this CPU compilation.
};
template<int N> struct range {
    std::array<size_t, N> dims;
    template<class... A> range(A... a) : dims{static_cast<size_t>(a)...} { static_assert(sizeof...(A) == N); }
    range operator*(const range & r) const { range out = *this; for (int i=0; i<N; ++i) { out.dims[i] *= r.dims[i]; } return out; }
};
template<int N> struct nd_range { range<N> global, local; nd_range(range<N> g, range<N> l) : global(g), local(l) {} };
template<int N> struct nd_item {};
struct handler {
    template<int N, class F> void parallel_for(nd_range<N> r, F) {
        static_assert(N == 3);
        assert((r.local.dims == std::array<size_t,3>{1, 1, 256})); // WARP_SIZE=16, unchanged
        assert(r.global.dims[2] % 256 == 0);
    }
};
struct queue {
    bool profiling = false, submit_error = false, wait_error = false, property_error = false;
    std::vector<std::shared_ptr<state>> commands;
    int get_device() const { return 0; }
    template<class P> bool has_property() const {
        if (property_error) { throw exception("property-error"); }
        return std::is_same_v<P, property::queue::enable_profiling> && profiling;
    }
    event enqueue() {
        if (submit_error) { throw exception("submit-error"); }
        commands.push_back(std::make_shared<state>()); return event(commands.back());
    }
    template<int N, class F> event parallel_for(nd_range<N> r, F) {
        static_assert(N == 1); assert(r.local.dims[0] == 16); return enqueue();
    }
    template<class F> event submit(F f) { handler h; f(h); return enqueue(); }
    void wait() {
        ++waits; if (wait_error) { throw exception("original-wait-error"); }
        for (auto & c : commands) { c->complete = true; }
    }
};
}
namespace dpct {
using queue_ptr = sycl::queue *;
static void has_capability_or_fail(int, std::initializer_list<sycl::aspect>) {}
}
using queue_ptr = sycl::queue *;
template<int N> struct quantize_and_reorder_q8_1_soa { void operator()(const float *, void *, int, int, sycl::nd_item<1>) const {} };
template<int N> struct quantize_q8_1 { void operator()(const float *, void *, int, int, sycl::nd_item<1>) const {} };
template<int N> struct reorder_vec_dot_q_sycl {};
template<class T, int N> void mul_mat_vec_q_reorder_ncols(const void *, const void *, const void *, float *, int, int, int, int, int, sycl::nd_item<3>) {}
static int ceil_div(int x, int y) { return (x + y - 1) / y; }

// EXTRACTED_DIAGNOSTIC_HEADER

struct ggml_backend_sycl_context {
    int device = 0;
    sycl::queue q;
    std::unique_ptr<lilicorr_device_pending> lilicorr_device_events;
    queue_ptr stream(int = 0, int = 0) { return &q; }
};
struct backend { ggml_backend_sycl_context * context; };
using ggml_backend_t = backend *;
// EXTRACTED_SYNCHRONIZE
// EXTRACTED_SCOPE_GATE
// EXTRACTED_QUANTIZER_LAUNCH
// EXTRACTED_MMVQ_LAUNCH

// Exact changed queue-init body, with CPU queue creation/property recording only.
struct queue_owner {
    sycl::queue _q_in_order, _q_out_of_order, _saved_queue;
    bool supported = true;
    int created = 0;
    bool has(sycl::aspect) const { return supported; }
    sycl::queue default_queue() { return _q_in_order; }
    template<class... P> sycl::queue create_queue_impl(bool exceptions, P...) {
        assert(exceptions); ++created; sycl::queue q;
        q.profiling = (false || ... || std::is_same_v<P, sycl::property::queue::enable_profiling>);
#ifdef DPCT_PROFILING_ENABLED
        q.profiling = true;
#endif
        return q;
    }
    // EXTRACTED_QUEUE_INIT
};

static void sync(ggml_backend_sycl_context & ctx) { backend b{&ctx}; ggml_backend_sycl_synchronize(&b); }
static void launches(ggml_backend_sycl_context & ctx) {
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(nullptr, nullptr, 4096, 8, 4096, &ctx.q);
    reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(nullptr, nullptr, nullptr, 4096, 8192, 4096, 8192, &ctx.q);
}
static void graph(ggml_backend_sycl_context * sycl_ctx, uint64_t profile_graph_id, bool graphs = false) {
    const bool profile_enabled = lilicorr_sycl_profile_context && lilicorr_sycl_profile_requested();
    int g_ggml_sycl_enable_graph = graphs;
    ggml_cgraph storage, * cgraph = &storage;
    // EXTRACTED_GRAPH_SCOPE
    ggml_tensor weight, act, dst; dst.src[0] = &weight; dst.src[1] = &act;
    std::strcpy(weight.name, "blk.0.ffn_up.weight");
    device_scope.set_node(42, &dst);
    launches(*sycl_ctx);
    // Source tensors die/are overwritten before any event query.
    std::strcpy(weight.name, "destroyed"); std::strcpy(dst.name, "destroyed");
}
static void contains(const char * text) { assert(logs.find(text) != std::string::npos); }
static void clear_log() { logs.clear(); }

int main(int argc, char **) {
    setenv("LLAMA_LILICORR_PROFILE", "1", 1); unsetenv("GGML_SYCL_ENABLE_GRAPH");
    set_lilicorr_profile_scope(true);
    if (argc > 1) {
        ggml_backend_sycl_context ctx; ctx.q.profiling = true; graph(&ctx, 999);
        std::atexit([] { assert(queries == 0 && waits == 1); std::cout << "PASS original wait failure: no profiling query\n"; });
        ctx.q.wait_error = true; sync(ctx); return 99; // extracted original catch must exit(1), never query
    }
    for (const char * env : {"", "0", "1", "true", "10"}) {
        setenv("LLAMA_LILICORR_PROFILE", env, 1);
        queue_owner owner; owner.init_queues(); assert(owner.created == 2);
#ifndef DPCT_PROFILING_ENABLED
        assert(owner._q_in_order.profiling == (std::strcmp(env, "1") == 0));
        assert(!owner._q_out_of_order.profiling);
#endif
    }
    setenv("LLAMA_LILICORR_PROFILE", "1", 1);
    queue_owner unsupported; unsupported.supported = false; unsupported.init_queues();
    setenv("GGML_SYCL_ENABLE_GRAPH", "1", 1); queue_owner graphs; graphs.init_queues();
#ifndef DPCT_PROFILING_ENABLED
    assert(!unsupported._q_in_order.profiling && !graphs._q_in_order.profiling);
#endif
    unsetenv("GGML_SYCL_ENABLE_GRAPH");
    std::cout << "PASS exact queue count/off properties/opt-in/unsupported/graphs\n";

    for (bool target : {false, true}) {
        ggml_backend_sycl_context ctx; int before = clocks, q = queries;
        setenv("LLAMA_LILICORR_PROFILE", target ? "1" : "0", 1);
        set_lilicorr_profile_scope(!target); clear_log(); graph(&ctx, 1);
        assert(ctx.q.commands.size() == 2 && !ctx.lilicorr_device_events);
        assert(clocks == before && queries == q && logs.empty());
        for (auto & e : ctx.q.commands) { assert(e.use_count() == 1); }
    }
    setenv("LLAMA_LILICORR_PROFILE", "1", 1); set_lilicorr_profile_scope(true);
    std::cout << "PASS opt-out/target exact submissions with no clocks/queries/retained events/logs\n";

    ggml_backend_sycl_context a, b; a.q.profiling = b.q.profiling = true;
    int q = queries, w = waits; graph(&a, 101); graph(&b, 202);
    assert(queries == q && waits == w && a.lilicorr_device_events->records.size() == 2);
    clear_log(); sync(b); contains("graph=202"); assert(logs.find("graph=101") == std::string::npos);
    assert(a.lilicorr_device_events && !b.lilicorr_device_events && queries == q+4 && waits == w+1);
    clear_log(); std::thread drain([&] { sync(a); }); drain.join();
    contains("graph=101 node=42"); contains("weight=blk.0.ffn_up.weight"); contains("device_elapsed_ns=75");
    contains("attribution=shared_queue_not_pure_drafter"); assert(logs.find("destroyed") == std::string::npos);
    assert(!a.lilicorr_device_events && waits == w+2 && premature == 0);
    for (auto & e : a.q.commands) { assert(e.use_count() == 1); }
    std::cout << "PASS actual launch returns/deferred existing sync/context ownership/thread handoff/copied metadata\n";

    {
        ggml_tensor x, y, dst; dst.src[0]=&x; dst.src[1]=&y;
        lilicorr_device_scope outer(true, false, a.lilicorr_device_events, &a.q, 303, 0, 826);
        outer.set_node(1, &dst);
        { lilicorr_device_scope target(false, false, b.lilicorr_device_events, &b.q, 0, 0, 826); launches(b); }
        assert(!b.lilicorr_device_events && lilicorr_device_active == &outer);
        std::thread other([&] { assert(!lilicorr_device_active); launches(b); set_lilicorr_profile_scope(true); graph(&b, 404); });
        other.join(); assert(lilicorr_device_active == &outer && b.lilicorr_device_events->records[0].graph == 404);
        launches(a); assert(a.lilicorr_device_events->records[0].graph == 303);
        y.ne[1]=7; outer.set_node(2, &dst); launches(a);
        assert(a.lilicorr_device_events->records.size() == 2);
        y.ne[1]=8; outer.set_node(3, &dst);
        quantize_row_q8_1_sycl<quantize_q8_1>(nullptr, nullptr, 4096, 8, 4096, &a.q);
        reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(nullptr, nullptr, nullptr, 4096, 8192, 4096, 8192, &a.q);
        assert(a.lilicorr_device_events->records.size() == 2);
    }
    sync(a); sync(b); assert(!lilicorr_device_active);
    std::cout << "PASS nested target suspension/thread TLS/only selected reordered ncols8 kernels\n";

    clear_log(); graph(&a, 505, true); assert(!a.lilicorr_device_events); contains("graphs_enabled_unsupported");
    ggml_backend_sycl_context no_profile; q = queries; graph(&no_profile, 506); clear_log(); sync(no_profile);
    contains("status=unsupported_profiling"); contains("device_elapsed_ns=unavailable"); assert(queries == q);
    graph(&a, 507); a.q.commands.back()->query_error = true; clear_log(); sync(a); contains("status=profiling_query_error");
    graph(&a, 508); a.q.commands.back()->end = 1; clear_log(); sync(a); contains("status=invalid_timestamps");
    graph(&a, 509); clear_log(); q = queries;
    lilicorr_device_after_wait(a.lilicorr_device_events, 1, 2); // adversarial misuse: defensive completion check must refuse queries
    contains("not_complete_after_wait"); assert(queries == q && premature == 0);
    std::cout << "PASS unsupported/graphs/query failure/invalid timestamps/incomplete events never become zero timings\n";

    {
        ggml_tensor x, y, dst; dst.src[0]=&x; dst.src[1]=&y;
        lilicorr_device_scope scope(true, false, a.lilicorr_device_events, &a.q, 606, 0, 826); scope.set_node(0, &dst);
        launches(b); assert(a.lilicorr_device_events->records.size() == 2);
        q = queries; clear_log(); sync(a); contains("status=foreign_queue"); assert(queries == q);
        a.q.submit_error = true;
        try { launches(a); assert(false); } catch (const sycl::exception &) {}
        a.q.submit_error = false; assert(scope.failed == 1 && !a.lilicorr_device_events);
        copy_error = true; launches(a); copy_error = false; assert(scope.dropped == 2);
        a.q.property_error = true; launches(a); a.q.property_error = false; assert(scope.dropped == 4);
        for (size_t i=0; i < lilicorr_device_pending::limit/2 + 1; ++i) { launches(a); }
        assert(a.lilicorr_device_events->records.size() == lilicorr_device_pending::limit && scope.dropped == 6);
    }
    q = queries; w = waits; clear_log(); a.lilicorr_device_events.reset();
    contains("status=no_existing_synchronize records=2048"); assert(queries == q && waits == w);
    std::cout << "PASS foreign queue/submit exception/property and copy errors/bounded overflow/discard without wait\n";
    assert(premature == 0); std::cout << "ALL CPU MOCK CHECKS PASSED (not device proof)\n";
}

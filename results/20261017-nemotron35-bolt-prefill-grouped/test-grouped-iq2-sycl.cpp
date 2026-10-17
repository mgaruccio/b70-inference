#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/fused-gemm.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr size_t kGuardBytes = 128;
constexpr size_t kTailBytes = 128;
constexpr uint8_t kCanary = 0xA5;
constexpr uint32_t kUntouchedBits = 0x7fc12345u;
static_assert(sizeof(block_iq2_xxs) == 66, "IQ2_XXS must remain a 66-byte raw block");

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

uint32_t bits_of(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct guarded_buffer {
    sycl::queue * queue = nullptr;
    uint8_t * base = nullptr;
    size_t payload_bytes = 0;

    guarded_buffer(sycl::queue * q, size_t payload) :
        queue(q),
        base(sycl::malloc_shared<uint8_t>(kGuardBytes + payload + kTailBytes, *q)),
        payload_bytes(payload) {
        require(base != nullptr, "sycl::malloc_shared returned null");
        reset();
    }

    guarded_buffer(const guarded_buffer &) = delete;
    guarded_buffer & operator=(const guarded_buffer &) = delete;

    ~guarded_buffer() {
        if (base != nullptr) {
            try {
                queue->wait_and_throw();
            } catch (const std::exception & exception) {
                std::fprintf(stderr, "buffer cleanup wait failed: %s\n", exception.what());
            }
            sycl::free(base, *queue);
        }
    }

    size_t total_bytes() const {
        return kGuardBytes + payload_bytes + kTailBytes;
    }

    uint8_t * payload() {
        return base + kGuardBytes;
    }

    const uint8_t * payload() const {
        return base + kGuardBytes;
    }

    template <typename T>
    T * payload_as() {
        return reinterpret_cast<T *>(payload());
    }

    template <typename T>
    const T * payload_as() const {
        return reinterpret_cast<const T *>(payload());
    }

    void reset() {
        std::memset(base, kCanary, total_bytes());
    }

    void reset_f32(size_t elements) {
        require(payload_bytes == elements * sizeof(float), "bad F32 output size");
        reset();
        std::fill_n(reinterpret_cast<uint32_t *>(payload()), elements, kUntouchedBits);
    }

    void copy_payload(const void * source) {
        std::memcpy(payload(), source, payload_bytes);
    }

    bool canaries_intact() const {
        uint8_t expected[kGuardBytes];
        std::memset(expected, kCanary, sizeof(expected));
        return std::memcmp(base, expected, kGuardBytes) == 0 &&
               std::memcmp(base + kGuardBytes + payload_bytes, expected, kTailBytes) == 0;
    }
};

void check_buffer_guards(const guarded_buffer & buffer, const std::string & label) {
    require(buffer.canaries_intact(), label + ": canary or tail changed");
}

void check_unchanged(const guarded_buffer & buffer, const std::vector<uint8_t> & snapshot,
                     const std::string & label) {
    require(snapshot.size() == buffer.total_bytes(), label + ": bad input snapshot size");
    if (std::memcmp(buffer.base, snapshot.data(), snapshot.size()) != 0) {
        size_t first = 0;
        while (first < snapshot.size() && buffer.base[first] == snapshot[first]) {
            ++first;
        }
        std::fprintf(stderr, "%s: input changed at byte %zu: got=0x%02x want=0x%02x\n", label, first,
                     first < snapshot.size() ? buffer.base[first] : 0,
                     first < snapshot.size() ? snapshot[first] : 0);
        fail(label + ": input bytes changed");
    }
}

void check_all_written(const guarded_buffer & output, size_t elements, const std::string & label) {
    const uint32_t * bits = output.payload_as<uint32_t>();
    for (size_t i = 0; i < elements; ++i) {
        if (bits[i] == kUntouchedBits) {
            fail(label + ": output element " + std::to_string(i) + " was not written");
        }
    }
}

struct nonfinite_counts {
    size_t finite = 0;
    size_t infinite = 0;
    size_t nan = 0;
};

nonfinite_counts count_values(const float * values, size_t count) {
    nonfinite_counts result;
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            ++result.nan;
        } else if (std::isinf(values[i])) {
            ++result.infinite;
        } else {
            ++result.finite;
        }
    }
    return result;
}

void report_values(const char * phase, const std::string & label, const float * values, size_t count) {
    const nonfinite_counts counts = count_values(values, count);
    std::printf("IQ2 grouped %s case=%s values=%zu finite=%zu inf=%zu nan=%zu\n", phase, label.c_str(), count,
                counts.finite, counts.infinite, counts.nan);
}

void compare_exact_bytes(const std::vector<uint8_t> & original, const uint8_t * replay, size_t payload_bytes,
                         const std::string & label) {
    require(original.size() == payload_bytes, label + ": bad ORIGINAL payload size");
    if (std::memcmp(original.data(), replay, payload_bytes) == 0) {
        return;
    }
    size_t first_byte = 0;
    while (first_byte < payload_bytes && original[first_byte] == replay[first_byte]) {
        ++first_byte;
    }
    const size_t element = first_byte / sizeof(float);
    uint32_t original_bits = 0;
    uint32_t replay_bits = 0;
    std::memcpy(&original_bits, original.data() + element * sizeof(float), sizeof(original_bits));
    std::memcpy(&replay_bits, replay + element * sizeof(float), sizeof(replay_bits));
    float original_value = 0.0f;
    float replay_value = 0.0f;
    std::memcpy(&original_value, &original_bits, sizeof(original_value));
    std::memcpy(&replay_value, &replay_bits, sizeof(replay_value));
    std::fprintf(stderr,
                 "%s: first ORIGINAL/REPLAY mismatch at element %zu: ORIGINAL=0x%08" PRIx32
                 " (%g) REPLAY=0x%08" PRIx32 " (%g)\n",
                 label.c_str(), element, original_bits, static_cast<double>(original_value), replay_bits,
                 static_cast<double>(replay_value));
    fail(label + ": full-array bitwise comparison failed");
}

void compare_expected(const float * actual, const std::vector<float> & expected, const std::string & label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const uint32_t actual_bits = bits_of(actual[i]);
        const uint32_t expected_bits = bits_of(expected[i]);
        if (actual_bits != expected_bits) {
            std::fprintf(stderr,
                         "%s: first oracle mismatch at element %zu: got=0x%08" PRIx32 " (%g) want=0x%08" PRIx32
                         " (%g)\n",
                         label.c_str(), i, actual_bits, static_cast<double>(actual[i]), expected_bits,
                         static_cast<double>(expected[i]));
            fail(label + ": F16 semantic oracle failed");
        }
    }
}

struct guarded_pool final : ggml_sycl_pool {
    struct allocation {
        uint8_t * base;
        size_t payload;
        size_t total;
    };

    sycl::queue * queue;
    std::unordered_map<void *, allocation> live;

    explicit guarded_pool(sycl::queue * q) : queue(q) {
    }

    void * alloc(size_t size, size_t * actual_size) override {
        const size_t payload = std::max<size_t>(size, 1);
        const size_t total = kGuardBytes + payload + kTailBytes;
        uint8_t * base = sycl::malloc_shared<uint8_t>(total, *queue);
        require(base != nullptr, "scratch sycl::malloc_shared returned null");
        std::memset(base, kCanary, total);
        void * ptr = base + kGuardBytes;
        live.emplace(ptr, allocation { base, payload, total });
        *actual_size = total;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        queue->wait_and_throw();
        const auto it = live.find(ptr);
        require(it != live.end(), "scratch pool freed an unknown pointer");
        const allocation entry = it->second;
        bool intact = size == entry.total;
        uint8_t expected[kGuardBytes];
        std::memset(expected, kCanary, sizeof(expected));
        intact = intact && std::memcmp(entry.base, expected, kGuardBytes) == 0;
        intact = intact &&
                 std::memcmp(entry.base + kGuardBytes + entry.payload, expected, kTailBytes) == 0;
        sycl::free(entry.base, *queue);
        live.erase(it);
        require(intact, "scratch pool guard changed");
    }

    ~guarded_pool() override {
        if (live.empty()) {
            return;
        }
        try {
            queue->wait_and_throw();
        } catch (const std::exception & exception) {
            std::fprintf(stderr, "scratch cleanup wait failed: %s\n", exception.what());
        }
        for (const auto & item : live) {
            sycl::free(item.second.base, *queue);
        }
        live.clear();
    }

    void assert_empty(const std::string & label) const {
        require(live.empty(), label + ": scratch allocation leaked");
    }
};

struct route {
    std::vector<int64_t> widths;
    std::vector<int64_t> offsets;
};

route make_route(const std::vector<int64_t> & widths) {
    route result;
    result.widths = widths;
    result.offsets.push_back(0);
    for (const int64_t width : widths) {
        require(width >= 0, "negative routed width");
        result.offsets.push_back(result.offsets.back() + width);
    }
    return result;
}

int64_t route_rows(const route & routing) {
    return routing.offsets.back();
}

size_t expected_tile_count(const route & routing) {
    size_t result = 0;
    for (const int64_t width : routing.widths) {
        result += static_cast<size_t>((width + 31) / 32);
    }
    return result;
}

size_t route_owner(const route & routing, int64_t row) {
    for (size_t expert = 0; expert < routing.widths.size(); ++expert) {
        if (row < routing.offsets[expert + 1]) {
            return expert;
        }
    }
    fail("row is outside routing offsets");
}

uint16_t scale_bits(size_t block, bool oracle) {
    if (oracle) {
        return 0x3c00u; // +1.0, so the oracle has exactly representable dyadic terms.
    }
    switch (block % 8) {
        case 0: return 0x0000u; // zero
        case 1: return 0x3c00u; // +1
        case 2: return 0xbc00u; // -1
        case 3: return 0x3400u; // +0.25
        case 4: return 0x3800u; // +0.5
        case 5: return 0x4000u; // +2
        case 6: return 0x3a00u; // +0.75
        default: return 0x4200u; // +3
    }
}

void fill_iq2_block(block_iq2_xxs & block, size_t global_block, bool oracle) {
    std::memset(&block, 0, sizeof(block));
    block.d = scale_bits(global_block, oracle);
    for (size_t ib8 = 0; ib8 < QK_K / 32; ++ib8) {
        uint16_t * q2 = block.qs + 4 * ib8;
        uint32_t aux32 = 0;
        uint8_t grid_indices[4] = {};
        for (size_t il = 0; il < 4; ++il) {
            const size_t logical = global_block * 32 + ib8 * 4 + il;
            grid_indices[il] = static_cast<uint8_t>(logical & 0xffu);
            const uint32_t sign_index = static_cast<uint32_t>(logical & 127u);
            aux32 |= sign_index << (7 * il);
        }
        const uint32_t scale_nibble = static_cast<uint32_t>((global_block + ib8) & 15u);
        aux32 |= scale_nibble << 28;
        q2[0] = static_cast<uint16_t>(grid_indices[0] | (static_cast<uint16_t>(grid_indices[1]) << 8));
        q2[1] = static_cast<uint16_t>(grid_indices[2] | (static_cast<uint16_t>(grid_indices[3]) << 8));
        q2[2] = static_cast<uint16_t>(aux32 & 0xffffu);
        q2[3] = static_cast<uint16_t>(aux32 >> 16);
    }
}

enum class data_mode {
    deterministic,
    oracle,
    overflow,
};

std::vector<block_iq2_xxs> make_weights(size_t n_as, int64_t M, int64_t K, data_mode mode) {
    require(K > 0, "weight fixture requires positive K");
    const size_t blocks_per_row = static_cast<size_t>((K + QK_K - 1) / QK_K);
    const size_t blocks_per_expert = static_cast<size_t>(M) * blocks_per_row;
    std::vector<block_iq2_xxs> weights(n_as * blocks_per_expert);
    for (size_t expert = 0; expert < n_as; ++expert) {
        for (int64_t row = 0; row < M; ++row) {
            for (size_t block = 0; block < blocks_per_row; ++block) {
                const size_t index = (expert * static_cast<size_t>(M) + static_cast<size_t>(row)) * blocks_per_row + block;
                fill_iq2_block(weights[index], index, mode == data_mode::oracle);
            }
        }
    }
    return weights;
}

float small_activation(size_t index) {
    static constexpr float values[] = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
                                        0.125f, -0.25f, 3.0f, -3.5f, 0.75f, -0.875f };
    return values[index % (sizeof(values) / sizeof(values[0]))];
}

std::vector<float> make_activations(int64_t total_rows, int64_t K, data_mode mode) {
    std::vector<float> values(static_cast<size_t>(total_rows) * static_cast<size_t>(K));
    for (int64_t row = 0; row < total_rows; ++row) {
        for (int64_t k = 0; k < K; ++k) {
            const size_t index = static_cast<size_t>(row) * static_cast<size_t>(K) + static_cast<size_t>(k);
            values[index] = mode == data_mode::overflow && (index % 29 == 0) ? 100000.0f : small_activation(index);
        }
    }
    return values;
}

void check_raw_coverage(const std::vector<block_iq2_xxs> & weights) {
    bool grids[256] = {};
    bool signs[128] = {};
    bool scales[16] = {};
    bool saw_zero = false;
    bool saw_negative = false;
    bool saw_finite_small = false;
    for (const block_iq2_xxs & block : weights) {
        saw_zero = saw_zero || block.d == 0x0000u;
        saw_negative = saw_negative || block.d == 0xbc00u;
        saw_finite_small = saw_finite_small || block.d == 0x3400u || block.d == 0x3800u;
        for (size_t ib8 = 0; ib8 < QK_K / 32; ++ib8) {
            const uint16_t * q2 = block.qs + 4 * ib8;
            const uint8_t * raw = reinterpret_cast<const uint8_t *>(q2);
            for (size_t il = 0; il < 4; ++il) {
                grids[raw[il]] = true;
            }
            const uint32_t aux32 = static_cast<uint32_t>(q2[2]) | (static_cast<uint32_t>(q2[3]) << 16);
            for (size_t il = 0; il < 4; ++il) {
                signs[(aux32 >> (7 * il)) & 127u] = true;
            }
            scales[(aux32 >> 28) & 15u] = true;
        }
    }
    for (bool seen : grids) {
        require(seen, "IQ2 grid coverage is incomplete");
    }
    for (bool seen : signs) {
        require(seen, "IQ2 sign-index coverage is incomplete");
    }
    for (bool seen : scales) {
        require(seen, "IQ2 scale-nibble coverage is incomplete");
    }
    require(saw_zero && saw_negative && saw_finite_small, "IQ2 scale fixture missed zero/negative/small values");
}

std::vector<float> f16_cpu_reference(const std::vector<block_iq2_xxs> & weights, const route & routing,
                                     int64_t M, int64_t K, const std::vector<float> & activations) {
    const size_t blocks_per_row = static_cast<size_t>(K / QK_K);
    const size_t blocks_per_expert = static_cast<size_t>(M) * blocks_per_row;
    const int64_t total_rows = route_rows(routing);
    std::vector<float> expected(static_cast<size_t>(total_rows) * static_cast<size_t>(M));
    std::vector<float> dequantized(static_cast<size_t>(K));
    for (int64_t routed_row = 0; routed_row < total_rows; ++routed_row) {
        const size_t expert = route_owner(routing, routed_row);
        for (int64_t m = 0; m < M; ++m) {
            const block_iq2_xxs * row_blocks =
                weights.data() + expert * blocks_per_expert + static_cast<size_t>(m) * blocks_per_row;
            dequantize_row_iq2_xxs(row_blocks, dequantized.data(), K);
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const sycl::half a = static_cast<sycl::half>(dequantized[static_cast<size_t>(k)]);
                const sycl::half b = static_cast<sycl::half>(
                    activations[static_cast<size_t>(routed_row) * static_cast<size_t>(K) + static_cast<size_t>(k)]);
                sum += static_cast<float>(a) * static_cast<float>(b);
            }
            expected[static_cast<size_t>(routed_row) * static_cast<size_t>(M) + static_cast<size_t>(m)] = sum;
        }
    }
    return expected;
}

bool run_grouped(const char * phase, const guarded_buffer & weights, size_t expert_stride,
                 const guarded_buffer & activations, guarded_buffer & output, const route & routing,
                 int64_t M, int64_t K, bool reordered, guarded_pool & pool, sycl::queue * queue,
                 size_t * tile_count) {
    std::vector<ggml_sycl_gg_tile> tiles;
    const bool launched = ggml_sycl_grouped_dequant_gemm(
        GGML_TYPE_IQ2_XXS, reordered, weights.payload(), expert_stride, activations.payload_as<float>(),
        GGML_PREC_UNDEFINED, output.payload_as<float>(), routing.offsets.data(), static_cast<int64_t>(routing.widths.size()),
        M, K, route_rows(routing), tiles, pool, queue);
    queue->wait_and_throw();
    *tile_count = tiles.size();
    std::printf("IQ2 grouped %s returned=%s tiles=%zu rows=%" PRId64 " M=%" PRId64 " K=%" PRId64 "\n", phase,
                launched ? "true" : "false", tiles.size(), route_rows(routing), M, K);
    return launched;
}

bool run_original(const guarded_buffer & weights, size_t expert_stride, const guarded_buffer & activations,
                  guarded_buffer & output, const route & routing, int64_t M, int64_t K, guarded_pool & pool,
                  sycl::queue * queue, size_t * tile_count) {
    return run_grouped("ORIGINAL", weights, expert_stride, activations, output, routing, M, K, false, pool, queue,
                       tile_count);
}

bool run_replay(const guarded_buffer & weights, size_t expert_stride, const guarded_buffer & activations,
                guarded_buffer & output, const route & routing, int64_t M, int64_t K, guarded_pool & pool,
                sycl::queue * queue, size_t * tile_count) {
    return run_grouped("REPLAY", weights, expert_stride, activations, output, routing, M, K, false, pool, queue,
                       tile_count);
}

struct valid_case {
    const char * name;
    int64_t M;
    int64_t K;
    std::vector<int64_t> widths;
    data_mode mode;
    bool raw_coverage = false;
    bool oracle = false;
};

void run_valid_case(const valid_case & specification, sycl::queue * queue) {
    const route routing = make_route(specification.widths);
    const int64_t total_rows = route_rows(routing);
    require(total_rows > 0, std::string(specification.name) + ": valid case has no rows");
    const std::vector<block_iq2_xxs> host_weights =
        make_weights(specification.widths.size(), specification.M, specification.K, specification.mode);
    if (specification.raw_coverage) {
        check_raw_coverage(host_weights);
        std::printf("IQ2 raw coverage grids=256 signs=128 scale_nibbles=16 zero=1 negative=1 finite_small=1\n");
    }
    const std::vector<float> host_activations = make_activations(total_rows, specification.K, specification.mode);
    if (specification.mode == data_mode::overflow) {
        require(std::find(host_activations.begin(), host_activations.end(), 100000.0f) != host_activations.end(),
                std::string(specification.name) + ": missing B(100000) activation");
    }

    const size_t blocks_per_row = static_cast<size_t>(specification.K / QK_K);
    const size_t blocks_per_expert = static_cast<size_t>(specification.M) * blocks_per_row;
    const size_t expert_stride = blocks_per_expert * sizeof(block_iq2_xxs);
    guarded_buffer weights(queue, host_weights.size() * sizeof(block_iq2_xxs));
    weights.copy_payload(host_weights.data());
    guarded_buffer activations(queue, host_activations.size() * sizeof(float));
    activations.copy_payload(host_activations.data());
    const std::vector<uint8_t> weights_snapshot(weights.base, weights.base + weights.total_bytes());
    const std::vector<uint8_t> activations_snapshot(activations.base, activations.base + activations.total_bytes());
    const size_t output_elements = static_cast<size_t>(total_rows) * static_cast<size_t>(specification.M);
    guarded_buffer output(queue, output_elements * sizeof(float));
    guarded_pool pool(queue);

    output.reset_f32(output_elements);
    size_t original_tiles = 0;
    const bool original_ok = run_original(weights, expert_stride, activations, output, routing, specification.M,
                                           specification.K, pool, queue, &original_tiles);
    require(original_ok, std::string(specification.name) + ": ORIGINAL returned false");
    require(original_tiles == expected_tile_count(routing), std::string(specification.name) + ": wrong ORIGINAL tile count");
    check_all_written(output, output_elements, std::string(specification.name) + " ORIGINAL");
    check_buffer_guards(output, std::string(specification.name) + " ORIGINAL output");
    check_unchanged(weights, weights_snapshot, std::string(specification.name) + " ORIGINAL weights");
    check_unchanged(activations, activations_snapshot, std::string(specification.name) + " ORIGINAL activations");
    report_values("ORIGINAL", specification.name, output.payload_as<float>(), output_elements);
    const std::vector<uint8_t> original_bytes(output.payload(), output.payload() + output.payload_bytes);

    if (specification.oracle) {
        const std::vector<float> expected = f16_cpu_reference(host_weights, routing, specification.M, specification.K,
                                                              host_activations);
        require(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0.0f; }),
                std::string(specification.name) + ": oracle fixture is a no-op");
        compare_expected(output.payload_as<float>(), expected, std::string(specification.name) + " ORIGINAL");
    }

    output.reset_f32(output_elements);
    size_t replay_tiles = 0;
    const bool replay_ok = run_replay(weights, expert_stride, activations, output, routing, specification.M,
                                      specification.K, pool, queue, &replay_tiles);
    require(replay_ok, std::string(specification.name) + ": REPLAY returned false");
    require(replay_tiles == original_tiles, std::string(specification.name) + ": ORIGINAL/REPLAY tile count differs");
    check_all_written(output, output_elements, std::string(specification.name) + " REPLAY");
    check_buffer_guards(output, std::string(specification.name) + " REPLAY output");
    check_unchanged(weights, weights_snapshot, std::string(specification.name) + " REPLAY weights");
    check_unchanged(activations, activations_snapshot, std::string(specification.name) + " REPLAY activations");
    report_values("REPLAY", specification.name, output.payload_as<float>(), output_elements);
    compare_exact_bytes(original_bytes, output.payload(), output.payload_bytes, std::string(specification.name));
    pool.assert_empty(specification.name);
    std::printf("IQ2 grouped case=%s exact=1 rows=%" PRId64 " M=%" PRId64 " K=%" PRId64 "\n", specification.name,
                total_rows, specification.M, specification.K);
}

enum class negative_gate {
    all_empty,
    bad_k,
    reordered,
    type_mask,
    f32_precision,
    wide_average,
};

void run_negative_case(const char * name, negative_gate gate, int64_t M, int64_t K,
                       const std::vector<int64_t> & widths, sycl::queue * queue) {
    const route routing = make_route(widths);
    int64_t total_rows = route_rows(routing);
    const int64_t call_rows = gate == negative_gate::all_empty ? 1 : total_rows;
    if (gate == negative_gate::all_empty) {
        require(total_rows == 0, std::string(name) + ": all-empty route is not empty");
    }
    const std::vector<block_iq2_xxs> host_weights = make_weights(widths.size(), M, K, data_mode::deterministic);
    const std::vector<float> host_activations = make_activations(call_rows, K, data_mode::deterministic);
    const size_t blocks_per_row = static_cast<size_t>(K / QK_K);
    const size_t expert_stride = static_cast<size_t>(M) * blocks_per_row * sizeof(block_iq2_xxs);
    guarded_buffer weights(queue, host_weights.size() * sizeof(block_iq2_xxs));
    weights.copy_payload(host_weights.data());
    guarded_buffer activations(queue, host_activations.size() * sizeof(float));
    activations.copy_payload(host_activations.data());
    const std::vector<uint8_t> weights_snapshot(weights.base, weights.base + weights.total_bytes());
    const std::vector<uint8_t> activations_snapshot(activations.base, activations.base + activations.total_bytes());
    const size_t output_elements = static_cast<size_t>(call_rows) * static_cast<size_t>(M);
    guarded_buffer output(queue, output_elements * sizeof(float));
    guarded_pool pool(queue);
    output.reset_f32(output_elements);
    std::vector<ggml_sycl_gg_tile> tiles;

    const int saved_types = g_ggml_sycl_xmx_gather_types;
    const int saved_precision = g_ggml_sycl_dynamic_precision;
    if (gate == negative_gate::type_mask) {
        g_ggml_sycl_xmx_gather_types = 0;
    }
    if (gate == negative_gate::f32_precision) {
        g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
    }
    const bool returned = ggml_sycl_grouped_dequant_gemm(
        GGML_TYPE_IQ2_XXS, gate == negative_gate::reordered, weights.payload(), expert_stride,
        activations.payload_as<float>(), GGML_PREC_UNDEFINED, output.payload_as<float>(), routing.offsets.data(),
        static_cast<int64_t>(routing.widths.size()), M, K, call_rows, tiles, pool, queue);
    queue->wait_and_throw();
    g_ggml_sycl_xmx_gather_types = saved_types;
    g_ggml_sycl_dynamic_precision = saved_precision;

    require(!returned, std::string(name) + ": negative gate returned true");
    require(tiles.empty(), std::string(name) + ": negative gate populated tile schedule");
    const uint32_t * output_bits = output.payload_as<uint32_t>();
    for (size_t i = 0; i < output_elements; ++i) {
        if (output_bits[i] != kUntouchedBits) {
            std::fprintf(stderr, "%s: negative gate wrote element %zu: 0x%08" PRIx32 "\n", name, i,
                         output_bits[i]);
            fail(std::string(name) + ": output was not untouched");
        }
    }
    check_buffer_guards(output, std::string(name) + " output");
    check_unchanged(weights, weights_snapshot, std::string(name) + " weights");
    check_unchanged(activations, activations_snapshot, std::string(name) + " activations");
    pool.assert_empty(name);
    std::printf("IQ2 grouped negative=%s returned=false output_untouched=1\n", name);
}

bool has_size(size_t max_size, size_t exact_size, size_t required) {
    return max_size >= required || exact_size == required;
}

void check_legal_combo(sycl::queue * queue) {
    const sycl::device device = queue->get_device();
    const auto combinations =
        device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    const auto subgroup_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    const bool has_sg16 = std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(16)) != subgroup_sizes.end();
    bool legal = false;
    for (const auto & combination : combinations) {
        if (combination.atype == mx::matrix_type::fp16 && combination.btype == mx::matrix_type::fp16 &&
            combination.ctype == mx::matrix_type::fp32 && combination.dtype == mx::matrix_type::fp32 &&
            has_size(combination.max_msize, combination.msize, 8) &&
            has_size(combination.max_nsize, combination.nsize, 16) &&
            has_size(combination.max_ksize, combination.ksize, 16)) {
            legal = true;
            break;
        }
    }
    std::printf("IQ2 grouped legal_combo=f16 TM8 TN16 TK16 SG16 available=%d subgroup16=%d shape_mask=0x%x\n",
                legal ? 1 : 0, has_sg16 ? 1 : 0, g_ggml_sycl_xmx_gather_shapes);
    require(g_ggml_sycl_xmx_gather_shapes == GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT,
            "shape mask must remain default 0xff");
    require((g_ggml_sycl_xmx_gather_shapes & 1) != 0 && has_sg16 && legal,
            "production f16 TM8 TN16 TK16 SG16 selection is unavailable");
}

struct global_state_restore {
    int fusion;
    int gather_types;
    int gather_shapes;
    int dynamic_precision;
    int required_precision;

    ~global_state_restore() {
        g_ggml_sycl_enable_fusion = fusion;
        g_ggml_sycl_xmx_gather_types = gather_types;
        g_ggml_sycl_xmx_gather_shapes = gather_shapes;
        g_ggml_sycl_dynamic_precision = dynamic_precision;
        g_ggml_sycl_dynamic_required_precision = required_precision;
    }
};

} // namespace

int main() {
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_sycl_init(0);
        require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
        sycl::queue * queue = &dpct::get_device(0).default_queue();
        std::printf("IQ2 grouped device=%s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());

        const global_state_restore restore {
            g_ggml_sycl_enable_fusion,
            g_ggml_sycl_xmx_gather_types,
            g_ggml_sycl_xmx_gather_shapes,
            g_ggml_sycl_dynamic_precision,
            g_ggml_sycl_dynamic_required_precision,
        };
        require(g_ggml_sycl_xmx_gather_shapes == GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT,
                "test requires the default global shape mask 0xff");
        g_ggml_sycl_enable_fusion = 1;
        g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_IQ2_XXS;
        g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_F16;
        require((g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS) != 0,
                "IQ2_XXS gather type is disabled");
        check_legal_combo(queue);

        const std::vector<valid_case> valid_cases = {
            { "oracle-m1-k256", 1, 256, { 1 }, data_mode::oracle, false, true },
            { "raw-routed-widths", 1, 2048, { 1, 0, 15, 0, 16, 17, 0, 31, 32, 0, 33, 63, 64 },
              data_mode::deterministic, true, false },
            { "skew-65-1", 15, 256, { 65, 1 }, data_mode::deterministic, false, false },
            { "skew-65-63", 16, 512, { 65, 0, 63 }, data_mode::deterministic, false, false },
            { "tail-m17", 17, 1024, { 1, 17, 31 }, data_mode::deterministic, false, false },
            { "up-k1024-m1536", 1536, 1024, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "up-k1024-m2048", 2048, 1024, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "up-k1024-m2304", 2304, 1024, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "up-k1024-m2560", 2560, 1024, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "down-m1024-k1536", 1024, 1536, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "down-m1024-k2048", 1024, 2048, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
            { "down-m1024-k2304", 1024, 2304, { 1, 15, 16, 17 }, data_mode::overflow, false, false },
            { "down-m1024-k2560", 1024, 2560, { 1, 15, 16, 17 }, data_mode::deterministic, false, false },
        };
        for (const valid_case & specification : valid_cases) {
            run_valid_case(specification, queue);
        }

        run_negative_case("all-empty", negative_gate::all_empty, 1, 256, { 0, 0, 0 }, queue);
        run_negative_case("k-nonmultiple256", negative_gate::bad_k, 1, 257, { 1 }, queue);
        run_negative_case("iq2-reordered", negative_gate::reordered, 1, 256, { 1, 15 }, queue);
        run_negative_case("type-mask-disabled", negative_gate::type_mask, 1, 256, { 1, 15 }, queue);
        run_negative_case("dynamic-precision-f32", negative_gate::f32_precision, 1, 256, { 1, 15 }, queue);
        run_negative_case("wide-average-65", negative_gate::wide_average, 1, 256, { 65 }, queue);

        queue->wait_and_throw();
        ggml_backend_free(backend);
        backend = nullptr;
        std::printf("IQ2 grouped native ORIGINAL/REPLAY PASS valid_cases=%zu negative_cases=6\n", valid_cases.size());
        return 0;
    } catch (const sycl::exception & exception) {
        std::fprintf(stderr, "IQ2 grouped native SYCL failure: %s\n", exception.what());
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "IQ2 grouped native failure: %s\n", exception.what());
    }
    if (backend != nullptr) {
        ggml_backend_free(backend);
    }
    return 1;
}

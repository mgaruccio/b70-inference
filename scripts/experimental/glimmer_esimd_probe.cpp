// Standalone correctness screen and synchronized microbenchmark for the pinned
// Intel llm-scaler INT4 DPAS header.  The header is intentionally not vendored;
// pass its directory with -I when building this probe.
//
// Upstream header (Apache-2.0, Intel llm-scaler repository):
// https://raw.githubusercontent.com/intel/llm-scaler/ede4320a24a67f664fb53081d2623f9efe9a75b7/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/int4_GEMM.h
//
// Example build in intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04:
//   icpx -O2 -std=c++17 -fsycl -fsycl-targets=spir64_gen \
//     -I"$LLM_SCALER_SRC/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels" \
//     scripts/experimental/glimmer_esimd_probe.cpp -o glimmer_esimd_probe
// Quick compile/run screen (the default also runs the two exact large shapes):
//   ./glimmer_esimd_probe --small-only --warmup 5 --iters 20

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The include directory above must contain the unmodified upstream header.
#include "int4_GEMM.h"

namespace {

constexpr std::uint32_t kGroupSize = 128;
constexpr float kAbsTolerance = 0.05F;
constexpr float kRelativeTolerance = 0.02F;

struct Options {
    std::size_t warmup = 5;
    std::size_t iterations = 20;
    std::size_t weight_banks = 1;
    bool small_only = false;
};

enum class WeightPattern {
    kRegular,
    kSignedExtremes,
    kAllZeroPointEight,
};

struct CaseSpec {
    const char* name;
    std::uint32_t m;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t seed;
    WeightPattern pattern;
    bool large;
};

constexpr std::array<CaseSpec, 6> kCases{{
    {"tiny_m2_n16_k128", 2, 16, 128, 3, WeightPattern::kRegular, false},
    {"wide_m2_n32_k256", 2, 32, 256, 11, WeightPattern::kRegular, false},
    {"signed_m17_n16_k128", 17, 16, 128, 19, WeightPattern::kSignedExtremes, false},
    {"allzp8_m2_n16_k128", 2, 16, 128, 23, WeightPattern::kAllZeroPointEight, false},
    {"gate_m32_n39936_k6656", 32, 39936, 6656, 31, WeightPattern::kRegular, true},
    {"down_m32_n6656_k19968", 32, 6656, 19968, 37, WeightPattern::kRegular, true},
}};

std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                std::ostringstream out;
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(c);
                escaped += out.str();
            } else {
                escaped.push_back(static_cast<char>(c));
            }
        }
    }
    return escaped;
}

void emit_error(std::string_view stage, std::string_view message) {
    std::cerr << "{\"type\":\"error\",\"stage\":\""
              << json_escape(stage) << "\",\"message\":\""
              << json_escape(message) << "\"}\n";
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* what) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error(std::string("size overflow for ") + what);
    }
    return lhs * rhs;
}

std::size_t parse_size(std::string_view text, const char* option) {
    if (text.empty()) {
        throw std::invalid_argument(std::string("missing value for ") + option);
    }
    std::size_t consumed = 0;
    const std::string value(text);
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid positive value for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--small-only") {
            options.small_only = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: glimmer_esimd_probe [--small-only] [--warmup N] "
                   "[--iters N] [--weight-banks 1|3]\n";
            std::exit(0);
        }

        auto read_value = [&](std::string_view name) -> std::string_view {
            if (arg == name) {
                if (++i >= argc) {
                    throw std::invalid_argument(std::string("missing value for ") +
                                                std::string(name));
                }
                return std::string_view(argv[i]);
            }
            const std::string prefix = std::string(name) + "=";
            if (arg.substr(0, prefix.size()) == prefix) {
                return arg.substr(prefix.size());
            }
            return {};
        };

        if (const auto value = read_value("--warmup"); !value.empty()) {
            options.warmup = parse_size(value, "--warmup");
        } else if (const auto value = read_value("--iters"); !value.empty()) {
            options.iterations = parse_size(value, "--iters");
        } else if (const auto value = read_value("--weight-banks"); !value.empty()) {
            options.weight_banks = parse_size(value, "--weight-banks");
            if (options.weight_banks != 1 && options.weight_banks != 3) {
                throw std::invalid_argument("--weight-banks must be 1 or 3");
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    return options;
}

std::uint8_t nibble_for(const CaseSpec& spec, std::uint32_t n, std::uint32_t k) {
    if (spec.pattern == WeightPattern::kAllZeroPointEight) {
        return 8;
    }
    if (spec.pattern == WeightPattern::kSignedExtremes) {
        // Include both signed endpoints (-8 and +7), zero-point 8, and every
        // intermediate nibble in a deterministic, non-repeating local pattern.
        constexpr std::array<std::uint8_t, 16> signed_pattern{{
            0, 15, 8, 7, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9,
        }};
        return signed_pattern[(n * 17U + k * 7U + spec.seed) & 15U];
    }
    // The n/k coefficients make every small test exercise a distinct packed
    // byte pattern rather than a constant or a single repeated nibble.
    return static_cast<std::uint8_t>(
        (n * 13U + k * 5U + (k / kGroupSize) * 3U + spec.seed) & 15U);
}

float scale_for(const CaseSpec& spec, std::uint32_t n, std::uint32_t group) {
    // Deliberately non-power-of-two scales.  The group and output-column terms
    // make the transposed [N, K/128] scale layout observable in the check.
    const std::uint32_t code = (n * 13U + group * 17U + spec.seed * 5U) % 37U;
    return 0.0095F + 0.00041F * static_cast<float>(code);
}

void fill_host_inputs(const CaseSpec& spec, std::vector<sycl::half>& input,
                      std::vector<std::uint8_t>& weight,
                      std::vector<sycl::half>& scale) {
    if (spec.k == 0 || spec.k % kGroupSize != 0 || spec.n % 16 != 0) {
        throw std::invalid_argument("probe shape violates K/128 or N/16 header requirements");
    }

    const std::size_t input_elements = checked_product(spec.m, spec.k, "input");
    const std::size_t weight_elements = checked_product(spec.n, spec.k / 2, "weight");
    const std::size_t groups = spec.k / kGroupSize;
    const std::size_t scale_elements = checked_product(spec.n, groups, "scale");
    input.resize(input_elements);
    weight.assign(weight_elements, 0);
    scale.resize(scale_elements);

    for (std::uint32_t row = 0; row < spec.m; ++row) {
        for (std::uint32_t k = 0; k < spec.k; ++k) {
            const std::uint32_t code =
                (row * 113U + k * 17U + spec.seed * 19U) % 127U;
            const float value = (static_cast<int>(code) - 63) / 256.0F;
            input[static_cast<std::size_t>(row) * spec.k + k] = sycl::half(value);
        }
    }

    for (std::uint32_t n = 0; n < spec.n; ++n) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            scale[static_cast<std::size_t>(n) * groups + group] =
                sycl::half(scale_for(spec, n, group));
        }
        const std::size_t weight_row = static_cast<std::size_t>(n) * (spec.k / 2);
        for (std::uint32_t k = 0; k < spec.k; k += 2) {
            const std::uint8_t lo = nibble_for(spec, n, k);
            const std::uint8_t hi = nibble_for(spec, n, k + 1);
            weight[weight_row + k / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
    }
}

template <typename T>
T* device_alloc(std::size_t count, sycl::queue& queue, const char* what) {
    T* ptr = sycl::malloc_device<T>(count, queue);
    if (ptr == nullptr) {
        throw std::runtime_error(std::string("USM allocation failed for ") + what);
    }
    return ptr;
}
// --weight-banks=3 duplicates the packed weights once during setup and rotates
// resident USM pointers during timing; it is not a PCIe streaming claim.

struct DeviceBuffers {
    sycl::queue* queue;
    sycl::half* input = nullptr;
    sycl::half* scale = nullptr;
    sycl::half* output = nullptr;
    std::vector<std::uint8_t*> weight_banks;

    explicit DeviceBuffers(sycl::queue& q) : queue(&q) {}
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        for (std::uint8_t* ptr : weight_banks) {
            if (ptr != nullptr) {
                try {
                    sycl::free(ptr, *queue);
                } catch (...) {
                }
            }
        }
        if (output != nullptr) {
            try {
                sycl::free(output, *queue);
            } catch (...) {
            }
        }
        if (scale != nullptr) {
            try {
                sycl::free(scale, *queue);
            } catch (...) {
            }
        }
        if (input != nullptr) {
            try {
                sycl::free(input, *queue);
            } catch (...) {
            }
        }
    }
};

void allocate_device_buffers(const CaseSpec& spec, std::size_t bank_count,
                             sycl::queue& queue, DeviceBuffers& buffers) {
    const std::size_t input_elements = checked_product(spec.m, spec.k, "input");
    const std::size_t weight_elements = checked_product(spec.n, spec.k / 2, "weight");
    const std::size_t scale_elements =
        checked_product(spec.n, spec.k / kGroupSize, "scale");
    const std::size_t output_elements = checked_product(spec.m, spec.n, "output");

    buffers.input = device_alloc<sycl::half>(input_elements, queue, "input");
    buffers.scale = device_alloc<sycl::half>(scale_elements, queue, "scale");
    buffers.output = device_alloc<sycl::half>(output_elements, queue, "output");
    buffers.weight_banks.resize(bank_count, nullptr);
    for (std::size_t bank = 0; bank < bank_count; ++bank) {
        buffers.weight_banks[bank] =
            device_alloc<std::uint8_t>(weight_elements, queue, "weight bank");
    }
}

void copy_to_device(const CaseSpec& spec, const std::vector<sycl::half>& input,
                    const std::vector<std::uint8_t>& weight,
                    const std::vector<sycl::half>& scale, DeviceBuffers& buffers) {
    sycl::queue& queue = *buffers.queue;
    const std::size_t input_bytes = input.size() * sizeof(sycl::half);
    const std::size_t weight_bytes = weight.size() * sizeof(std::uint8_t);
    const std::size_t scale_bytes = scale.size() * sizeof(sycl::half);
    queue.memcpy(buffers.input, input.data(), input_bytes);
    queue.memcpy(buffers.scale, scale.data(), scale_bytes);
    for (std::uint8_t* bank : buffers.weight_banks) {
        queue.memcpy(bank, weight.data(), weight_bytes);
    }
    queue.wait_and_throw();
    (void)spec;
}

void invoke_upstream_dispatcher(const CaseSpec& spec, DeviceBuffers& buffers,
                                std::size_t bank) {
    // This is the actual upstream dispatcher, called without a wrapper or
    // altered kernel.  Its void return is why timings synchronize the queue.
    GEMM_int4_pgrp_host(
        buffers.input, buffers.weight_banks.at(bank), buffers.scale, buffers.output,
        spec.m, spec.n, spec.k, *buffers.queue);
}

std::vector<std::uint32_t> sample_indices(std::uint32_t count, std::size_t limit) {
    const std::size_t wanted = std::min<std::size_t>(count, limit);
    std::vector<std::uint32_t> indices;
    indices.reserve(wanted);
    if (wanted == 0) {
        return indices;
    }
    if (wanted == 1) {
        indices.push_back(0);
        return indices;
    }
    for (std::size_t i = 0; i < wanted; ++i) {
        const auto value = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(i) * (count - 1)) / (wanted - 1));
        if (indices.empty() || indices.back() != value) {
            indices.push_back(value);
        }
    }
    return indices;
}

struct CheckMetrics {
    bool finite = true;
    bool pass = false;
    std::size_t nonfinite_count = 0;
    std::size_t sample_count = 0;
    double max_abs = 0.0;
    double max_relative = 0.0;
    double rms = 0.0;
    double zero_max_abs = 0.0;
};

CheckMetrics check_output(const CaseSpec& spec, const std::vector<sycl::half>& input,
                          const std::vector<std::uint8_t>& weight,
                          const std::vector<sycl::half>& scale,
                          const std::vector<sycl::half>& output) {
    CheckMetrics metrics;
    for (const sycl::half value : output) {
        const float converted = static_cast<float>(value);
        if (!std::isfinite(converted)) {
            metrics.finite = false;
            ++metrics.nonfinite_count;
            continue;
        }
        if (spec.pattern == WeightPattern::kAllZeroPointEight) {
            metrics.zero_max_abs =
                std::max(metrics.zero_max_abs, static_cast<double>(std::fabs(converted)));
        }
    }
    if (!metrics.finite) {
        return metrics;
    }

    const std::vector<std::uint32_t> rows = sample_indices(spec.m, 64);
    const std::vector<std::uint32_t> columns = sample_indices(spec.n, 64);
    double sum_squared_error = 0.0;
    bool within_tolerance = true;
    const std::size_t half_k = spec.k / 2;
    const std::size_t groups = spec.k / kGroupSize;

    for (std::uint32_t row : rows) {
        for (std::uint32_t column : columns) {
            float expected = 0.0F;
            for (std::uint32_t k = 0; k < spec.k; ++k) {
                const std::uint8_t packed =
                    weight[static_cast<std::size_t>(column) * half_k + k / 2];
                const std::uint8_t quantized = (k & 1U) == 0 ? (packed & 0x0FU)
                                                              : (packed >> 4);
                const float a = static_cast<float>(input[static_cast<std::size_t>(row) * spec.k + k]);
                const float s = static_cast<float>(
                    scale[static_cast<std::size_t>(column) * groups + k / kGroupSize]);
                const float b = (static_cast<float>(quantized) - 8.0F) * s;
                expected = std::fma(a, b, expected);
            }

            const float actual = static_cast<float>(
                output[static_cast<std::size_t>(row) * spec.n + column]);
            const double absolute = std::fabs(static_cast<double>(actual) - expected);
            const double relative =
                absolute / std::max(std::fabs(static_cast<double>(expected)), 1.0e-6);
            metrics.max_abs = std::max(metrics.max_abs, absolute);
            metrics.max_relative = std::max(metrics.max_relative, relative);
            sum_squared_error += absolute * absolute;
            ++metrics.sample_count;
            if (absolute > kAbsTolerance + kRelativeTolerance *
                                  std::fabs(static_cast<double>(expected))) {
                within_tolerance = false;
            }
        }
    }

    metrics.rms = metrics.sample_count == 0
                      ? 0.0
                      : std::sqrt(sum_squared_error / metrics.sample_count);
    const bool zero_ok = spec.pattern != WeightPattern::kAllZeroPointEight ||
                         metrics.zero_max_abs == 0.0;
    metrics.pass = metrics.finite && within_tolerance && zero_ok;
    return metrics;
}

void emit_check(const CaseSpec& spec, const CheckMetrics& metrics) {
    std::cout << std::setprecision(9)
              << "{\"type\":\"correctness\",\"shape\":\""
              << json_escape(spec.name) << "\",\"m\":" << spec.m
              << ",\"n\":" << spec.n << ",\"k\":" << spec.k
              << ",\"reference\":\"fp32_dequant_matmul\",\"finite\":"
              << (metrics.finite ? "true" : "false")
              << ",\"nonfinite_count\":" << metrics.nonfinite_count
              << ",\"samples\":" << metrics.sample_count;
    if (metrics.finite) {
        std::cout << ",\"max_abs\":" << metrics.max_abs
                  << ",\"max_relative\":" << metrics.max_relative
                  << ",\"rms\":" << metrics.rms;
        if (spec.pattern == WeightPattern::kAllZeroPointEight) {
            std::cout << ",\"zero_max_abs\":" << metrics.zero_max_abs;
        }
    } else {
        std::cout << ",\"max_abs\":null,\"max_relative\":null,\"rms\":null";
    }
    std::cout << ",\"abs_tol\":" << kAbsTolerance
              << ",\"rel_tol\":" << kRelativeTolerance
              << ",\"pass\":" << (metrics.pass ? "true" : "false") << "}\n";
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if ((values.size() & 1U) != 0) {
        return *middle;
    }
    const double upper = *middle;
    const auto lower_middle = values.begin() + values.size() / 2 - 1;
    std::nth_element(values.begin(), lower_middle, values.end());
    return (*lower_middle + upper) / 2.0;
}

void emit_benchmark(const CaseSpec& spec, const Options& options,
                    const std::vector<double>& timings_us, std::size_t setup_bytes,
                    std::size_t device_bytes) {
    std::vector<double> deviations;
    const double middle = median(timings_us);
    deviations.reserve(timings_us.size());
    for (double timing : timings_us) {
        deviations.push_back(std::fabs(timing - middle));
    }
    const auto [minimum, maximum] =
        std::minmax_element(timings_us.begin(), timings_us.end());
    std::cout << std::setprecision(9)
              << "{\"type\":\"benchmark\",\"shape\":\""
              << json_escape(spec.name) << "\",\"m\":" << spec.m
              << ",\"n\":" << spec.n << ",\"k\":" << spec.k
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"weight_banks\":" << options.weight_banks
              << ",\"bank_mode\":\"resident_rotation\""
              << ",\"timing\":\"host_dispatch_plus_synchronized_wait\""
              << ",\"median_us\":" << middle
              << ",\"mad_us\":" << median(std::move(deviations))
              << ",\"min_us\":" << *minimum << ",\"max_us\":" << *maximum
              << ",\"setup_bytes_excluded\":" << setup_bytes
              << ",\"device_buffer_bytes\":" << device_bytes
              << ",\"preprocessing_excluded\":true}\n";
}

bool run_case(const CaseSpec& spec, const Options& options, sycl::queue& queue) {
    std::vector<sycl::half> input;
    std::vector<std::uint8_t> weight;
    std::vector<sycl::half> scale;
    fill_host_inputs(spec, input, weight, scale);

    DeviceBuffers buffers(queue);
    allocate_device_buffers(spec, options.weight_banks, queue, buffers);
    copy_to_device(spec, input, weight, scale, buffers);

    invoke_upstream_dispatcher(spec, buffers, 0);
    queue.wait_and_throw();
    std::vector<sycl::half> output(
        checked_product(spec.m, spec.n, "output copy"), sycl::half(0.0F));
    queue.memcpy(output.data(), buffers.output,
                 output.size() * sizeof(sycl::half));
    queue.wait_and_throw();

    const CheckMetrics metrics = check_output(spec, input, weight, scale, output);
    emit_check(spec, metrics);

    std::vector<double> timings_us;
    timings_us.reserve(options.iterations);
    for (std::size_t i = 0; i < options.warmup; ++i) {
        invoke_upstream_dispatcher(spec, buffers, i % options.weight_banks);
        queue.wait_and_throw();
    }
    for (std::size_t i = 0; i < options.iterations; ++i) {
        queue.wait_and_throw();
        const auto begin = std::chrono::steady_clock::now();
        invoke_upstream_dispatcher(spec, buffers, i % options.weight_banks);
        queue.wait_and_throw();
        const auto end = std::chrono::steady_clock::now();
        timings_us.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
    }

    const std::size_t input_bytes = input.size() * sizeof(sycl::half);
    const std::size_t weight_bytes = weight.size() * sizeof(std::uint8_t);
    const std::size_t scale_bytes = scale.size() * sizeof(sycl::half);
    const std::size_t output_bytes = output.size() * sizeof(sycl::half);
    const std::size_t setup_bytes =
        input_bytes + scale_bytes + weight_bytes * options.weight_banks;
    const std::size_t device_bytes =
        input_bytes + scale_bytes + output_bytes + weight_bytes * options.weight_banks;
    emit_benchmark(spec, options, timings_us, setup_bytes, device_bytes);
    return metrics.pass;
}

std::string compiler_info() {
#ifdef __INTEL_LLVM_COMPILER
    return "intel-llvm-" + std::to_string(__INTEL_LLVM_COMPILER);
#elif defined(__clang_version__)
    return std::string("clang-") + __clang_version__;
#else
    return "unknown";
#endif
}

void emit_device_info(const sycl::queue& queue, const Options& options) {
    const sycl::device device = queue.get_device();
    const auto name = device.get_info<sycl::info::device::name>();
    const auto vendor = device.get_info<sycl::info::device::vendor>();
    const auto driver = device.get_info<sycl::info::device::driver_version>();
    std::cout << "{\"type\":\"device\",\"name\":\""
              << json_escape(name) << "\",\"vendor\":\""
              << json_escape(vendor) << "\",\"driver\":\""
              << json_escape(driver) << "\",\"compiler\":\""
              << json_escape(compiler_info()) << "\",\"weight_banks\":"
              << options.weight_banks << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        sycl::queue queue{
            sycl::gpu_selector_v,
            sycl::property_list{sycl::property::queue::enable_profiling{}}};
        emit_device_info(queue, options);

        bool all_passed = true;
        for (const CaseSpec& spec : kCases) {
            if (options.small_only && spec.large) {
                continue;
            }
            try {
                all_passed = run_case(spec, options, queue) && all_passed;
            } catch (const sycl::exception& error) {
                emit_error(spec.name, error.what());
                all_passed = false;
            } catch (const std::exception& error) {
                emit_error(spec.name, error.what());
                all_passed = false;
            }
        }
        return all_passed ? 0 : 2;
    } catch (const sycl::exception& error) {
        emit_error("initialization", error.what());
        return 3;
    } catch (const std::exception& error) {
        emit_error("arguments_or_initialization", error.what());
        return 3;
    }
}

// CPU-only: compile the actual per-chunk producer with mocked existing getter/process APIs.
#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static int64_t clock_us = 0;
static int clock_calls = 0, getter_calls = 0, process_calls = 0, projection_calls = 0, process_rc = 0;
static bool projection_available = true;
static std::string logs, order;
static int64_t ggml_time_us() { ++clock_calls; return ++clock_us; }
static void log_message(const char * format, ...) {
    char line[4096]; va_list args; va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args); va_end(args); logs += line; clock_us += 7;
}
#define LOG_INF(...) log_message(__VA_ARGS__)
#define LOG_ERR(...) log_message(__VA_ARGS__)
#define GGML_ABORT(...) throw std::runtime_error("original abort")
using llama_seq_id = int;
using llama_pos = int;
struct context { int id; };
struct view { const float * data; size_t stride, size; };
struct token { std::array<int,4> pos{}; const float * embd = nullptr; };
struct fake_batch {
    std::vector<token> tokens;
    void clear() { tokens.clear(); }
    void add_embd(view v, const int * pos, int seq, bool logits) {
        assert(v.stride == 1 && v.size == 24576 && seq == 1 && !logits);
        tokens.push_back({{pos[0],pos[1],pos[2],pos[3]},v.data});
    }
    const fake_batch & get() const { return *this; }
};
constexpr int LLAMA_PROCESS_TYPE_DECODE = 1;
static std::array<std::vector<float>,6> layers;
static std::vector<float> projection;
static const float * llama_get_embeddings_layer_inp(context * ctx, uint32_t layer) {
    assert(ctx->id == 0 && layer == (uint32_t) getter_calls);
    ++getter_calls; order += 'g'; clock_us += 11; return layers[layer].data();
}
static int llama_process(context * ctx, int type, const fake_batch & batch) {
    assert(ctx->id == 1 && type == LLAMA_PROCESS_TYPE_DECODE && getter_calls == 6);
    ++process_calls; order += 'p'; clock_us += 23;
    for (size_t row=0; row<batch.tokens.size(); ++row) {
        for (size_t layer=0; layer<6; ++layer) {
            for (size_t col=0; col<4096; ++col) {
                assert(batch.tokens[row].embd[layer*4096+col] == layers[layer][(row+3)*4096+col]);
            }
        }
    }
    return process_rc;
}
static const float * llama_get_embeddings_nextn(context * ctx) {
    assert(ctx->id == 1 && process_calls == 1); ++projection_calls; order += 'r'; clock_us += 37;
    return projection_available ? projection.data() : nullptr;
}
// EXTRACTED_PROFILE_GATE
// EXTRACTED_COMPONENT_HELPER

static bool run_chunk(int n_chunk, bool is_lilicorr) {
    const bool profile = lilicorr_profile_enabled(is_lilicorr);
    context target{0}, draft{1}; auto * ctx_tgt = &target; auto * ctx_dft = &draft;
    const int seq_id = 1, offset = 2, n_rows = n_chunk + offset, n_tokens = n_rows + 3;
    const int n_embd_tgt = 4096, n_embd_enc = 24576, n_embd_dec = 4096, lilicorr_rows = 32, block_size = 8;
    const uint32_t target_layer_ids_n = 6;
    const int target_layer_ids[6] = {0,1,2,3,4,5};
    const int i_batch_beg[2] = {0,1};
    const int64_t profile_begin = profile ? ggml_time_us() : 0;
    std::vector<float> features_buf, lilicorr_embd(2*lilicorr_rows*n_embd_dec, -1);
    std::vector<int> lilicorr_pos(2*lilicorr_rows, -1);
    fake_batch batch_in, batch_inject;
    for (int i=0; i<n_tokens; ++i) { batch_in.tokens.push_back({{100+i,100+i,100+i,0},nullptr}); }
    for (size_t layer=0; layer<6; ++layer) {
        layers[layer].resize(n_tokens*n_embd_tgt);
        for (int row=0; row<n_tokens; ++row) {
            for (int col=0; col<n_embd_tgt; ++col) { layers[layer][row*n_embd_tgt+col] = layer*100000+row*100+col%97; }
        }
    }
    projection.resize(n_chunk*n_embd_dec);
    for (size_t i=0; i<projection.size(); ++i) { projection[i] = float(i+1); }
    // EXTRACTED_PROCESS_CHUNK
    if (is_lilicorr && projection_available) {
        const int first = std::max(0, n_chunk-block_size-1);
        for (int row=first; row<n_chunk; ++row) {
            const int pos = 103+row;
            const size_t slot = (size_t) seq_id*lilicorr_rows + pos%lilicorr_rows;
            assert(lilicorr_pos[slot] == pos);
            for (int col=0; col<n_embd_dec; ++col) { assert(lilicorr_embd[slot*n_embd_dec+col] == projection[row*n_embd_dec+col]); }
        }
    }
    return true;
}
static void contains(const std::string & text) { assert(logs.find(text) != std::string::npos); }
int main() {
    for (int width : {4,5,46}) {
        for (int mode=0; mode<5; ++mode) {
            logs.clear(); order.clear(); clock_calls=getter_calls=process_calls=projection_calls=0; clock_us=0;
            process_rc = mode == 3 ? 7 : 0; projection_available = mode != 4;
            setenv("LLAMA_LILICORR_PROFILE", mode == 0 ? "0" : "1", 1);
            const bool result = run_chunk(width, mode != 1);
            assert(result == (process_rc == 0));
            assert(getter_calls == 6 && process_calls == 1);
            const bool read_projection = mode != 1 && mode != 3;
            assert(projection_calls == int(read_projection));
            assert(order == (read_projection ? "ggggggpr" : "ggggggp"));
            if (mode <= 1) {
                assert(clock_calls == 0 && logs.empty());
            } else {
                contains("LLAMA_LILICORR_PROCESS_COMPONENTS v=1 stage=chunk");
                contains("seq=1 batch_rows=" + std::to_string(width+5) + " seq_rows=" + std::to_string(width+2) + " offset=2 rows=" + std::to_string(width));
                contains("getter_calls=6 target_getter_host_us=72");
                contains("feature_copy_calls=" + std::to_string(6*width));
                contains("feature_copy_bytes=" + std::to_string((size_t) width*24576*sizeof(float)));
                contains("feature_copy_loop_host_us=6"); contains("process_host_us=24");
                contains("getter_attribution=target_wait_and_reorder_possible");
                contains("projection_attribution=injection_or_shared_queue_drain"); contains("device_time=unmeasured");
                contains("log_host_us=8");
                if (mode == 3) {
                    contains("status=process_error"); contains("projection_seen=0"); contains("projection_getter_host_us=unavailable"); contains("ring_copy_loop_host_us=unavailable");
                } else {
                    contains("projection_getter_host_us=38");
                    if (mode == 4) { contains("projected_present=0"); contains("ring_copy_calls=0"); contains("ring_copy_loop_host_us=unavailable"); }
                    else { contains("projected_present=1"); contains("ring_copy_calls=" + std::to_string(std::min(width,9))); }
                }
            }
        }
    }
    std::cout << "PASS extracted process chunk: widths4/5/46, six real CPU gather loops, ring bytes, existing API order/counts, opt-out/non-LiliCorr guards, rc/missing-projection negatives, getter attribution and separate logging cost\n";
}

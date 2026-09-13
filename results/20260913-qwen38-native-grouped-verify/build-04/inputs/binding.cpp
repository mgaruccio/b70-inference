#include <torch/all.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <cmath>

#include "paged_decode.hpp"

namespace {
constexpr int kPage = 1664;
constexpr int kSplits = 32;

void check_metadata(const at::Tensor& t, const at::Tensor& q,
                    at::IntArrayRef shape) {
  TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kInt &&
                  t.sizes() == shape && t.is_contiguous(),
              "b70_grouped_verify: invalid device int32 metadata");
}

void check_scale(const at::Tensor& t, const at::Tensor& q) {
  TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kFloat &&
                  t.numel() > 0,
              "b70_grouped_verify: descale must be device float32");
  for (int64_t i = 0; i < t.dim(); ++i) {
    TORCH_CHECK(t.size(i) <= 1 || t.stride(i) == 0,
                "b70_grouped_verify: only scalar broadcast descales supported");
  }
}

at::Tensor b70_grouped_verify_forward(const at::Tensor& q, const at::Tensor& k,
                   const at::Tensor& v, const at::Tensor& used,
                   const at::Tensor& table, const at::Tensor& cu,
                   const at::Tensor& ks, const at::Tensor& vs,
                   int64_t max_k, double scale) {
  TORCH_CHECK(q.is_xpu() && q.scalar_type() == at::kHalf &&
                  q.sizes() == at::IntArrayRef({1, 120, 256}) && q.is_contiguous(),
              "b70_grouped_verify: expected packed FP16 Q[1,120,256]");
  c10::DeviceGuard guard(q.device());
  for (const auto* cache : {&k, &v}) {
    TORCH_CHECK(cache->device() == q.device() &&
                    cache->scalar_type() == at::ScalarType::Float8_e4m3fn &&
                    cache->sizes() == at::IntArrayRef({176, kPage, 4, 256}) &&
                    cache->strides() == at::IntArrayRef({kPage * 4 * 256, 256,
                                                        kPage * 256, 1}),
                "b70_grouped_verify: expected FP8 e4m3fn HND KV[176,1664,4,256]");
  }
  check_metadata(used, q, {1});
  check_metadata(table, q, {1, 128});
  check_metadata(cu, q, {2});
  check_scale(ks, q);
  check_scale(vs, q);
  TORCH_CHECK(max_k >= 1 && max_k <= 212992 && std::isfinite(scale) && scale > 0,
              "b70_grouped_verify: invalid fixed bound or softmax scale");
  // The Python seam generates cu=[0,1] and clamp_min(used,1) ON DEVICE.
  // As in native, used<=max_k and valid page IDs are caller preconditions;
  // no host reads/synchronization of graph-mutable tensor contents occur here.
  auto output = at::empty({1, 120, 256}, q.options());
  auto partial = at::empty({1, 120 * kSplits, 256}, q.options());
  auto sums = at::empty({1, 120, kSplits}, q.options().dtype(at::kFloat));
  auto maxima = at::empty_like(sums);

  paged_decode_args_t args{};
  args.query = q.data_ptr();
  args.key = k.data_ptr();
  args.value = v.data_ptr();
  args.out = output.data_ptr();
  args.tem_out = partial.data_ptr();
  args.exp_sums = sums.data_ptr();
  args.max_logits = maxima.data_ptr();
  args.block_table = table.data_ptr();
  args.cu_seqlens_q = cu.data_ptr();
  // Despite the field name, native decode reads lengths[batch], not a prefix sum.
  args.cu_seqlens_k = used.data_ptr();
  args.max_queries = 1;
  args.max_keys = static_cast<int>(max_k);
  args.total_seqlen_q = 1;
  args.page_stride_elements =
      static_cast<int>(get_paged_kv_cache_page_stride_elements(k));
  args.total_seqlen_k =
      static_cast<int>(get_paged_kv_cache_effective_total_seqlen(k));
  args.k_scale = ks.data_ptr();
  args.v_scale = vs.data_ptr();
  args.sm_scale = static_cast<float>(scale);
  args.batch_size = 1;
  args.num_heads_q = 120;
  args.num_heads_k = 4;
  args.head_size = args.v_head_size = 256;
  args.max_blocks_per_seq = 128;
  args.block_size = kPage;
  args.is_varlen = args.is_paged = true;
  args.num_kv_splits = kSplits;
  args.k_stride_page = k.stride(0);
  args.k_stride_seq = k.stride(1);
  args.k_stride_heads = k.stride(2);
  args.v_stride_page = v.stride(0);
  args.v_stride_seq = v.stride(1);
  args.v_stride_heads = v.stride(2);
  args.q_stride_seq = q.stride(0);
  args.q_stride_heads = q.stride(1);

  using Policy = decode_policy_q16_h256_p64;
  using RowStride = cute::Stride<int, cute::_1, int, int>;
  using VStride = cute::Stride<cute::_1, int, int, int>;
  using Config = PagedDecodeConfig<
      Policy::ShapeQK, Policy::ShapePV, Policy::ShapeOut,
      Policy::SubgroupLayoutQK, void, 1,
      false, false, false,  // native causal/local/sink remain off
      cutlass::half_t, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
      cutlass::half_t, void,
      RowStride, RowStride, VStride, RowStride, RowStride,
      void, void, void, void, true>;  // only this instantiation: PackedVerify
  auto& queue = c10::xpu::getCurrentXPUStream(q.get_device()).queue();
  Config::kernel_dispatch(queue, args);
  return output;
}
}  // namespace

// No _vllm_fa2_C registration, API replacement, or exported native entrypoint.
TORCH_LIBRARY(b70_grouped_verify, ops) {
  ops.def("forward(Tensor q, Tensor k, Tensor v, Tensor used, Tensor table, "
          "Tensor cu, Tensor ks, Tensor vs, int max_k, float scale) -> Tensor");
  ops.impl("forward", torch::kXPU, &b70_grouped_verify_forward);
}

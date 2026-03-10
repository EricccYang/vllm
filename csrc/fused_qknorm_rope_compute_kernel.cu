/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * fused_qknorm_rope_compute_kernel.cu
 *
 * Variant of fused_qknorm_rope_improve_kernel.cu that computes RoPE cos/sin
 * on-the-fly via __sincosf + powf instead of reading from a pre-computed cache.
 *
 * Hypothesis for H100: SFU throughput for __sincosf + powf is sufficient that
 * it is cheaper than the extra HBM round-trip for the cos/sin cache,
 * particularly at small batch sizes where the cache is cold.
 *
 * Key differences vs fused_qknorm_rope_improve_kernel.cu:
 *
 *   1. No cos_sin_cache input.  cos/sin are computed as:
 *        freq  = powf(rope_base, -2 * half_dim / rotary_dim)
 *        theta = position_id * freq           (with optional YaRN scaling)
 *        __sincosf(theta, &sin_val, &cos_val)
 *      Supports YaRN-style linear ramp scaling via rope_factor/low/high and
 *      an optional attention_factor multiplier.
 *
 *   2. NTokenHeads variant: cos/sin are computed once per warp into register
 *      arrays before the per-head loop.  Because every head processed by the
 *      warp belongs to the same token (same position), the values are
 *      identical across heads and can be reused at zero extra cost.  The
 *      computation is overlapped with the async QKV load and weight preload,
 *      so SFU cycles are effectively hidden.  Shared memory holds QKV data
 *      only (no cos/sin region), reducing smem consumption and improving
 *      occupancy.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#define VLLM_NVTX_RANGE_PUSH(name) ((void)0)
#define VLLM_NVTX_RANGE_POP() ((void)0)

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define CHECK_TYPE(x, st)                                               \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);   \
  CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL
  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
  #endif
#else
  #define FINAL_MASK 0xffffffff
#endif

// ---------------------------------------------------------------------------
// Helpers shared within this TU
// ---------------------------------------------------------------------------
namespace tensorrt_llm::common_compute {

template <typename T, int num>
struct packed_as;
template <>
struct packed_as<uint, 1> {
  using type = uint;
};
template <>
struct packed_as<uint, 2> {
  using type = uint2;
};
template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace tensorrt_llm::common_compute

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
namespace tensorrt_llm::kernels {

// cp.async helpers (identical to those in improve_kernel.cu)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
__device__ __forceinline__ void compute_cp_async4(void* smem_ptr,
                                                  const void* glob_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(glob_ptr));
}
__device__ __forceinline__ void compute_cp_async_ca(void* smem_ptr,
                                                    const void* glob_ptr,
                                                    int size_bytes) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if (size_bytes == 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else if (size_bytes == 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  }
}
__device__ __forceinline__ void compute_cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int n>
__device__ __forceinline__ void compute_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(n));
}
#else
__device__ __forceinline__ void compute_cp_async4(void* smem_ptr,
                                                  const void* glob_ptr) {
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
}
__device__ __forceinline__ void compute_cp_async_ca(void* smem_ptr,
                                                    const void* glob_ptr,
                                                    int size_bytes) {
  if (size_bytes == 4)
    *reinterpret_cast<uint32_t*>(smem_ptr) =
        *reinterpret_cast<const uint32_t*>(glob_ptr);
  else if (size_bytes == 8)
    *reinterpret_cast<uint64_t*>(smem_ptr) =
        *reinterpret_cast<const uint64_t*>(glob_ptr);
  else
    *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
}
__device__ __forceinline__ void compute_cp_async_fence() {}
template <int n>
__device__ __forceinline__ void compute_cp_async_wait() {}
#endif

// Compute a single RoPE frequency with optional YaRN linear-ramp scaling.
// half_dim: the half-index (0..rotary_dim/2-1) used in the base formula.
__device__ __forceinline__ float compute_rope_freq(int const half_dim,
                                                   int const rotary_dim,
                                                   float const rope_base,
                                                   float const rope_factor,
                                                   float rope_low,
                                                   float rope_high) {
  float freq =
      powf(rope_base,
           -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim));
  if (rope_factor != 1.0f) {
    float const inv_freq_extrapolation = freq;
    float const inv_freq_interpolation = freq / rope_factor;
    if (fabsf(rope_low - rope_high) <= 1e-6f) rope_high += 0.001f;
    float const linear_func =
        (static_cast<float>(half_dim) - rope_low) / (rope_high - rope_low);
    float const ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float const ext_factor = 1.0f - ramp_func;
    freq = inv_freq_interpolation * (1.0f - ext_factor) +
           inv_freq_extrapolation * ext_factor;
  }
  return freq;
}

// ---------------------------------------------------------------------------
// Kernel 1: one warp processes one (token, head) pair.
//   Like fusedQKNormRopeImproveKernel but computes cos/sin on-the-fly.
// ---------------------------------------------------------------------------
template <typename scalar_t_in, int head_dim, bool interleave>
__global__ void fusedQKNormRopeComputeKernel(
    void* qkv_void,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    void const* q_weight_void,
    void const* k_weight_void,
    int64_t const* position_ids,
    int const num_tokens,
    int const rotary_dim,
    float const rope_base,
    float const rope_factor,
    float const rope_low,
    float const rope_high,
    float const attention_factor) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif

  using Converter = vllm::_typeConvert<scalar_t_in>;
  static_assert(Converter::exists,
                "Input QKV dtype not supported for this arch/toolkit.");
  using T_in  = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;

  T_in* qkv           = reinterpret_cast<T_in*>(qkv_void);
  T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
  T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);

  int const warpsPerBlock = blockDim.x / 32;
  int const warpId  = threadIdx.x / 32;
  int const laneId  = threadIdx.x % 32;

  int const globalWarpIdx  = blockIdx.x * warpsPerBlock + warpId;
  int const total_qk_heads = num_heads_q + num_heads_k;
  int const tokenIdx       = globalWarpIdx / total_qk_heads;
  int const localHeadIdx   = globalWarpIdx % total_qk_heads;

  if (tokenIdx >= num_tokens) return;

  bool const isQ    = localHeadIdx < num_heads_q;
  int  const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
  int  const num_heads = num_heads_q + num_heads_k + num_heads_v;

  static_assert(head_dim % (32 * 2) == 0,
                "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename tensorrt_llm::common_compute::packed_as<uint, vecSize>::type;

  int const offsetWarp =
      isQ ? tokenIdx * num_heads * head_dim + headIdx * head_dim
          : tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                headIdx * head_dim;
  int const offsetThread = offsetWarp + laneId * numElemsPerThread;

  // === Part 1: Load + RMSNorm ===
  float sumOfSquares = 0.0f;
  {
    vec_T const vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
    for (int i = 0; i < num_packed_elems; ++i) {
      T2_in packed_val = *(reinterpret_cast<T2_in const*>(&vec) + i);
      float2 vals      = Converter::convert(packed_val);
      sumOfSquares += vals.x * vals.x + vals.y * vals.y;
      elements[2 * i]     = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  sumOfSquares = tensorrt_llm::common_compute::warpReduceSum(sumOfSquares);
  float const rms_rcp =
      rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
  for (int i = 0; i < numElemsPerThread; ++i) {
    int const dim    = laneId * numElemsPerThread + i;
    float const weight = isQ ? Converter::convert(q_weight[dim])
                             : Converter::convert(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }

  // === Part 2: RoPE — cos/sin computed on-the-fly ===
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  float const pos_f      = static_cast<float>(position_ids[tokenIdx]);

  if (laneId < rotary_lanes) {
    if constexpr (interleave) {
#pragma unroll
      for (int i = 0; i < numElemsPerThread / 2; ++i) {
        int const idx0    = 2 * i;
        int const idx1    = 2 * i + 1;
        int const dim_idx = laneId * numElemsPerThread + idx0;
        int const half_dim = dim_idx / 2;
        float const freq  = compute_rope_freq(half_dim, rotary_dim, rope_base,
                                              rope_factor, rope_low, rope_high);
        float cos_val, sin_val;
        __sincosf(pos_f * freq, &sin_val, &cos_val);
        float const v0 = elements[idx0], v1 = elements[idx1];
        elements[idx0] = (v0 * cos_val - v1 * sin_val) * attention_factor;
        elements[idx1] = (v0 * sin_val + v1 * cos_val) * attention_factor;
      }
    } else {
      __syncwarp();
      int const pairOffset = (rotary_dim / 2) / numElemsPerThread;
#pragma unroll
      for (int i = 0; i < numElemsPerThread; ++i) {
        float elem2 = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);
        if (laneId < pairOffset) elem2 = -elem2;
        int dim_idx = laneId * numElemsPerThread + i;
        dim_idx     = (dim_idx * 2) % rotary_dim;
        int const half_dim = dim_idx / 2;
        float const freq   = compute_rope_freq(half_dim, rotary_dim, rope_base,
                                               rope_factor, rope_low, rope_high);
        float cos_val, sin_val;
        __sincosf(pos_f * freq, &sin_val, &cos_val);
        elements[i] = (elements[i] * cos_val + elem2 * sin_val) * attention_factor;
      }
      __syncwarp();
    }
  }

  // Store
  {
    vec_T vec;
    constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
    for (int i = 0; i < num_packed_elems; ++i) {
      T2_in packed_val = Converter::convert(
          make_float2(elements[2 * i], elements[2 * i + 1]));
      *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
    }
    *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
  }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ---------------------------------------------------------------------------
// Kernel 2: one warp processes HEADS_PER_WARP (token, head) pairs.
//   cos/sin computed once into registers before the head loop, overlapped
//   with async QKV load and weight preload.  No smem used for cos/sin.
// ---------------------------------------------------------------------------
template <typename scalar_t_in, int head_dim, bool interleave, int HEADS_PER_WARP>
__global__ void fusedQKNormRopeComputeKernelNTokenHeads(
    void* qkv_void,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    void const* q_weight_void,
    void const* k_weight_void,
    int64_t const* position_ids,
    int const num_tokens,
    int const rotary_dim,
    float const rope_base,
    float const rope_factor,
    float const rope_low,
    float const rope_high,
    float const attention_factor) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif

  using Converter = vllm::_typeConvert<scalar_t_in>;
  static_assert(Converter::exists,
                "Input QKV dtype not supported for this arch/toolkit.");
  using T_in  = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;

  extern __shared__ char smem_storage[];

  T_in* qkv            = reinterpret_cast<T_in*>(qkv_void);
  T_in const* q_weight  = reinterpret_cast<T_in const*>(q_weight_void);
  T_in const* k_weight  = reinterpret_cast<T_in const*>(k_weight_void);

  int const warpsPerBlock = blockDim.x / 32;
  int const warpId  = threadIdx.x / 32;
  int const laneId  = threadIdx.x % 32;

  int const total_qk_heads      = num_heads_q + num_heads_k;
  int const num_heads            = num_heads_q + num_heads_k + num_heads_v;
  int const head_chunks_per_token =
      (total_qk_heads + HEADS_PER_WARP - 1) / HEADS_PER_WARP;

  int const warp_global    = blockIdx.x * warpsPerBlock + warpId;
  int const tokenIdx       = warp_global / head_chunks_per_token;
  int const headChunk      = warp_global % head_chunks_per_token;
  int const first_head     = headChunk * HEADS_PER_WARP;
  int const num_heads_this_warp =
      (first_head + HEADS_PER_WARP <= total_qk_heads)
          ? HEADS_PER_WARP
          : (total_qk_heads - first_head);

  if (tokenIdx >= num_tokens) return;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  constexpr int elemSizeBytes     = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T =
      typename tensorrt_llm::common_compute::packed_as<uint, vecSize>::type;

  // Smem layout: QKV tiles only (no cos/sin region).
  // [warp0: HEADS_PER_WARP * 32 * elemSizeBytes] [warp1: ...] ...
  int const qkv_tile_bytes   = 32 * elemSizeBytes;
  char* const this_warp_smem =
      smem_storage + warpId * (HEADS_PER_WARP * qkv_tile_bytes);

  // === Step 1: Async copy all heads' QKV data into smem (group 0) ===
  for (int k = 0; k < num_heads_this_warp; ++k) {
    int const localHeadIdx = first_head + k;
    bool const isQ  = localHeadIdx < num_heads_q;
    int const hIdx  = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
    int const offWarp =
        isQ ? tokenIdx * num_heads * head_dim + hIdx * head_dim
            : tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                  hIdx * head_dim;
    int const offThread = offWarp + laneId * numElemsPerThread;
    char* smem_dst =
        this_warp_smem + k * qkv_tile_bytes + laneId * elemSizeBytes;
    compute_cp_async_ca(smem_dst,
                        reinterpret_cast<const char*>(&qkv[offThread]),
                        elemSizeBytes);
  }
  compute_cp_async_fence();  // commit group 0 (QKV)

  // === Step 2 (overlap with HBM fetch): preload weights + compute cos/sin ===
  //
  // Both happen while the async copy for QKV is still in flight.

  // 2a. Load norm weights into registers (same for all HEADS_PER_WARP heads).
  float q_w[numElemsPerThread];
  float k_w[numElemsPerThread];
#pragma unroll
  for (int i = 0; i < numElemsPerThread; ++i) {
    int const dim = laneId * numElemsPerThread + i;
    q_w[i] = Converter::convert(q_weight[dim]);
    k_w[i] = Converter::convert(k_weight[dim]);
  }

  // 2b. Compute cos/sin for this token once; reuse across all heads.
  //     All heads belong to the same token → same position → same cos/sin.
  float cos_reg[numElemsPerThread];
  float sin_reg[numElemsPerThread];

  int const rotary_lanes = rotary_dim / numElemsPerThread;
  float const pos_f      = static_cast<float>(position_ids[tokenIdx]);

  if (laneId < rotary_lanes) {
    if constexpr (interleave) {
      // Compute one (cos, sin) pair per element pair; store at pair index i.
#pragma unroll
      for (int i = 0; i < numElemsPerThread / 2; ++i) {
        int const dim_idx  = laneId * numElemsPerThread + 2 * i;
        int const half_dim = dim_idx / 2;
        float const freq   = compute_rope_freq(half_dim, rotary_dim, rope_base,
                                               rope_factor, rope_low, rope_high);
        __sincosf(pos_f * freq, &sin_reg[i], &cos_reg[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < numElemsPerThread; ++i) {
        int dim_idx        = laneId * numElemsPerThread + i;
        dim_idx            = (dim_idx * 2) % rotary_dim;
        int const half_dim = dim_idx / 2;
        float const freq   = compute_rope_freq(half_dim, rotary_dim, rope_base,
                                               rope_factor, rope_low, rope_high);
        __sincosf(pos_f * freq, &sin_reg[i], &cos_reg[i]);
      }
    }
  }

  // === Step 3: Wait for QKV async copy to complete ===
  compute_cp_async_wait<0>();

  // === Step 4: Process each head — norm + RoPE + store ===
  float elements[numElemsPerThread];

  for (int k = 0; k < num_heads_this_warp; ++k) {
    int const localHeadIdx = first_head + k;
    bool const isQ  = localHeadIdx < num_heads_q;
    int const hIdx  = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const offsetWarp =
        isQ ? tokenIdx * num_heads * head_dim + hIdx * head_dim
            : tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                  hIdx * head_dim;
    int const offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Part 1: RMSNorm from smem
    float sumOfSquares = 0.0f;
    {
      char const* smem_src =
          this_warp_smem + k * qkv_tile_bytes + laneId * elemSizeBytes;
      vec_T const vec = *reinterpret_cast<vec_T const*>(smem_src);
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; ++i) {
        T2_in packed_val = *(reinterpret_cast<T2_in const*>(&vec) + i);
        float2 vals      = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x + vals.y * vals.y;
        elements[2 * i]     = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    sumOfSquares = tensorrt_llm::common_compute::warpReduceSum(sumOfSquares);
    float const rms_rcp =
        rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
    for (int i = 0; i < numElemsPerThread; ++i)
      elements[i] *= rms_rcp * (isQ ? q_w[i] : k_w[i]);

    // Part 2: RoPE using pre-computed cos/sin from registers
    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          float const v0 = elements[idx0], v1 = elements[idx1];
          // cos_reg[i]/sin_reg[i] were stored at pair index i above
          elements[idx0] =
              (v0 * cos_reg[i] - v1 * sin_reg[i]) * attention_factor;
          elements[idx1] =
              (v0 * sin_reg[i] + v1 * cos_reg[i]) * attention_factor;
        }
      } else {
        __syncwarp();
        int const pairOffset = (rotary_dim / 2) / numElemsPerThread;
#pragma unroll
        for (int i = 0; i < numElemsPerThread; ++i) {
          float elem2 = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);
          if (laneId < pairOffset) elem2 = -elem2;
          elements[i] =
              (elements[i] * cos_reg[i] + elem2 * sin_reg[i]) * attention_factor;
        }
        __syncwarp();
      }
    }

    // Store
    {
      vec_T vec;
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; ++i) {
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
    }
  }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ---------------------------------------------------------------------------
// DISPATCH_INTERLEAVE macro
// ---------------------------------------------------------------------------
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...)  \
  if (interleave) {                                       \
    const bool INTERLEAVE = true;                         \
    __VA_ARGS__                                           \
  } else {                                                \
    const bool INTERLEAVE = false;                        \
    __VA_ARGS__                                           \
  }

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
template <typename scalar_t_in>
void launchFusedQKNormRopeCompute(
    void* qkv, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim,
    int const rotary_dim, float const eps, void const* q_weight,
    void const* k_weight, bool const interleave,
    int64_t const* position_ids, int const block_size,
    float const rope_base, float const rope_factor,
    float const rope_low, float const rope_high,
    float const attention_factor, cudaStream_t stream) {
  TORCH_CHECK(block_size == 128 || block_size == 256 || block_size == 512,
              "block_size must be 128, 256, or 512, got ", block_size);
  int const warpsPerBlock = block_size / 32;
  int const totalQKHeads  = num_heads_q + num_heads_k;
  int const totalWarps    = num_tokens * totalQKHeads;
  int const gridSize =
      tensorrt_llm::common_compute::divUp(totalWarps, warpsPerBlock);

  VLLM_NVTX_RANGE_PUSH("FusedQKNormRopeCompute");
  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeComputeKernel<scalar_t_in, 64, INTERLEAVE>
            <<<dim3(gridSize), dim3(block_size), 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps,
                q_weight, k_weight, position_ids, num_tokens, rotary_dim,
                rope_base, rope_factor, rope_low, rope_high, attention_factor);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeComputeKernel<scalar_t_in, 128, INTERLEAVE>
            <<<dim3(gridSize), dim3(block_size), 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps,
                q_weight, k_weight, position_ids, num_tokens, rotary_dim,
                rope_base, rope_factor, rope_low, rope_high, attention_factor);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeComputeKernel<scalar_t_in, 256, INTERLEAVE>
            <<<dim3(gridSize), dim3(block_size), 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps,
                q_weight, k_weight, position_ids, num_tokens, rotary_dim,
                rope_base, rope_factor, rope_low, rope_high, attention_factor);
      });
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head_dim for fusedQKNormRopeCompute: ", head_dim);
  }
  VLLM_NVTX_RANGE_POP();
}

template <typename scalar_t_in>
void launchFusedQKNormRopeComputeNTokenHeads(
    void* qkv, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim,
    int const rotary_dim, float const eps, void const* q_weight,
    void const* k_weight, bool const interleave,
    int64_t const* position_ids, int const block_size,
    int const token_heads_per_warp,
    float const rope_base, float const rope_factor,
    float const rope_low, float const rope_high,
    float const attention_factor, cudaStream_t stream) {
  TORCH_CHECK(block_size == 128 || block_size == 256 || block_size == 512,
              "block_size must be 128, 256, or 512, got ", block_size);
  TORCH_CHECK(token_heads_per_warp == 2 || token_heads_per_warp == 4 ||
                  token_heads_per_warp == 8,
              "token_heads_per_warp must be 2, 4, or 8, got ",
              token_heads_per_warp);

  int const warpsPerBlock       = block_size / 32;
  int const totalQKHeads         = num_heads_q + num_heads_k;
  int const head_chunks_per_token =
      (totalQKHeads + token_heads_per_warp - 1) / token_heads_per_warp;
  int const total_warps =
      num_tokens * head_chunks_per_token;
  int const gridSize =
      tensorrt_llm::common_compute::divUp(total_warps, warpsPerBlock);

  // Smem: QKV tiles only (no cos/sin).
  // Each tile = 32 threads × elemSizeBytes = 32 × (head_dim/32 × 2) = 2×head_dim bytes.
  size_t const smem_bytes =
      static_cast<size_t>(warpsPerBlock) *
      static_cast<size_t>(token_heads_per_warp) *
      2u * static_cast<size_t>(head_dim);

  VLLM_NVTX_RANGE_PUSH("FusedQKNormRopeComputeNTokenHeads");
#define LAUNCH_COMPUTE_N(N)                                                      \
  do {                                                                           \
    switch (head_dim) {                                                          \
      case 64:                                                                   \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                            \
          fusedQKNormRopeComputeKernelNTokenHeads<scalar_t_in, 64, INTERLEAVE,  \
                                                  (N)>                           \
              <<<dim3(gridSize), dim3(block_size), smem_bytes, stream>>>(        \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,    \
                  k_weight, position_ids, num_tokens, rotary_dim, rope_base,    \
                  rope_factor, rope_low, rope_high, attention_factor);           \
        });                                                                      \
        break;                                                                   \
      case 128:                                                                  \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                            \
          fusedQKNormRopeComputeKernelNTokenHeads<scalar_t_in, 128, INTERLEAVE, \
                                                  (N)>                           \
              <<<dim3(gridSize), dim3(block_size), smem_bytes, stream>>>(        \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,    \
                  k_weight, position_ids, num_tokens, rotary_dim, rope_base,    \
                  rope_factor, rope_low, rope_high, attention_factor);           \
        });                                                                      \
        break;                                                                   \
      case 256:                                                                  \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                            \
          fusedQKNormRopeComputeKernelNTokenHeads<scalar_t_in, 256, INTERLEAVE, \
                                                  (N)>                           \
              <<<dim3(gridSize), dim3(block_size), smem_bytes, stream>>>(        \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,    \
                  k_weight, position_ids, num_tokens, rotary_dim, rope_base,    \
                  rope_factor, rope_low, rope_high, attention_factor);           \
        });                                                                      \
        break;                                                                   \
      default:                                                                   \
        TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);                  \
    }                                                                            \
  } while (0)

  if (token_heads_per_warp == 2)
    LAUNCH_COMPUTE_N(2);
  else if (token_heads_per_warp == 4)
    LAUNCH_COMPUTE_N(4);
  else
    LAUNCH_COMPUTE_N(8);

#undef LAUNCH_COMPUTE_N
  VLLM_NVTX_RANGE_POP();
}

}  // namespace tensorrt_llm::kernels

// ---------------------------------------------------------------------------
// Public C++ / Torch API
// ---------------------------------------------------------------------------

void fused_qk_norm_rope_compute(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    bool is_neox,
    torch::Tensor& position_ids,
    int64_t block_size,
    double rope_base,
    double rope_factor,
    double rope_low,
    double rope_high,
    double attention_factor) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_TYPE(position_ids, torch::kInt64);

  TORCH_CHECK(qkv.dim() == 2, "QKV must be 2D: [num_tokens, ...]");
  TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D");
  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim,
              "q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim,
              "k_weight size must match head_dim");
  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type(),
              "qkv, q_weight and k_weight must share dtype");

  int64_t const num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "position_ids length must match num_tokens");
  TORCH_CHECK(qkv.size(1) == (num_heads_q + num_heads_k + num_heads_v) * head_dim,
              "QKV size(1) mismatch");

  auto const stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qk_norm_rope_compute_kernel", [&] {
        tensorrt_llm::kernels::launchFusedQKNormRopeCompute<scalar_t>(
            qkv.data_ptr(), static_cast<int>(num_tokens),
            static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
            static_cast<int>(num_heads_v), static_cast<int>(head_dim),
            static_cast<int>(head_dim),  // rotary_dim = head_dim (full RoPE)
            static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
            !is_neox,
            reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
            static_cast<int>(block_size), static_cast<float>(rope_base),
            static_cast<float>(rope_factor), static_cast<float>(rope_low),
            static_cast<float>(rope_high), static_cast<float>(attention_factor),
            stream);
      });
}

void fused_qk_norm_rope_compute_n_token_heads(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    bool is_neox,
    torch::Tensor& position_ids,
    int64_t block_size,
    int64_t token_heads_per_warp,
    double rope_base,
    double rope_factor,
    double rope_low,
    double rope_high,
    double attention_factor) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_TYPE(position_ids, torch::kInt64);

  TORCH_CHECK(qkv.dim() == 2, "QKV must be 2D: [num_tokens, ...]");
  TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D");
  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim,
              "q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim,
              "k_weight size must match head_dim");
  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type(),
              "qkv, q_weight and k_weight must share dtype");
  TORCH_CHECK(token_heads_per_warp == 2 || token_heads_per_warp == 4 ||
                  token_heads_per_warp == 8,
              "token_heads_per_warp must be 2, 4, or 8, got ",
              token_heads_per_warp);

  int64_t const num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "position_ids length must match num_tokens");
  TORCH_CHECK(qkv.size(1) == (num_heads_q + num_heads_k + num_heads_v) * head_dim,
              "QKV size(1) mismatch");

  auto const stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qk_norm_rope_compute_n_token_heads_kernel", [&] {
        tensorrt_llm::kernels::launchFusedQKNormRopeComputeNTokenHeads<scalar_t>(
            qkv.data_ptr(), static_cast<int>(num_tokens),
            static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
            static_cast<int>(num_heads_v), static_cast<int>(head_dim),
            static_cast<int>(head_dim),  // rotary_dim = head_dim (full RoPE)
            static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
            !is_neox,
            reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
            static_cast<int>(block_size), static_cast<int>(token_heads_per_warp),
            static_cast<float>(rope_base), static_cast<float>(rope_factor),
            static_cast<float>(rope_low), static_cast<float>(rope_high),
            static_cast<float>(attention_factor), stream);
      });
}

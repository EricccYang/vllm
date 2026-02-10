# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int8 prefill KV: plan metadata (build) and dequant/fill-from-activation (forward)."""

import torch

import numpy as np

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

if HAS_TRITON:

    @triton.jit
    def _dequant_int8_pages_into_buffer_kernel(
        kv_cache_ptr,
        physical_page_indices_ptr,
        buffer_ptr,
        k_scale: tl.float32,
        v_scale: tl.float32,
        page_stride: tl.int64,
        k_elems: tl.int64,
        num_elems_per_page: tl.int64,
        BLOCK_ELEMS: tl.constexpr,
        OUT_BF16: tl.constexpr,
    ):
        """Fused gather + int8->float * scale + cast + scatter for one page."""
        page_idx = tl.program_id(axis=0)
        physical_idx = tl.load(physical_page_indices_ptr + page_idx).to(tl.int64)
        base_kv = physical_idx * page_stride
        base_buf = physical_idx * page_stride

        for start in range(0, num_elems_per_page, BLOCK_ELEMS):
            offs = start + tl.arange(0, BLOCK_ELEMS)
            mask = offs < num_elems_per_page
            val = tl.load(kv_cache_ptr + base_kv + offs, mask=mask)
            val_float = val.to(tl.float32)
            scale = tl.where(offs < k_elems, k_scale, v_scale)
            val_scaled = val_float * scale
            if OUT_BF16:
                out_val = val_scaled.to(tl.bfloat16)
            else:
                out_val = val_scaled.to(tl.float16)
            tl.store(buffer_ptr + base_buf + offs, out_val, mask=mask)


def dequant_int8_pages_to_contiguous(
    kv_cache: torch.Tensor,
    physical_page_indices: torch.Tensor,
    k_scale: float | torch.Tensor,
    v_scale: float | torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize int8 KV cache at given physical block ids into a contiguous buffer.

    Used when FI backend is used with int8 KV cache: we only dequant the pages
    needed for the current batch so FlashInfer receives a bf16/fp16 cache with
    contiguous block indices 0..num_pages-1.

    Args:
        kv_cache: int8, shape (num_blocks, 2, block_size, num_kv_heads, head_size)
            or HND layout.
        physical_page_indices: shape (num_pages,) int32, physical block ids.
        k_scale, v_scale: scalar scale for K and V (float or scalar tensor).
        out_dtype: bfloat16 or float16.
        out: optional preallocated buffer (max_num_pages, 2, ...). When provided
            and large enough, result is written here to avoid per-step allocation.

    Returns:
        Contiguous tensor shape (num_pages, 2, ...) in out_dtype.
    """
    # info_once dedup is by (msg, *args); use static msg so we log once per process
    logger.info_once("int8_prefill_kv: dequant_int8_pages_to_contiguous")
    # Do not create new tensors here: torch.tensor(..., device=...) is not
    # allowed during CUDA graph capture (cudaErrorStreamCaptureUnsupported).
    # Output shape (num_pages, ...) is variable-length; must not be captured.
    # mul_ accepts both Python float and scalar tensor.
    num_pages = physical_page_indices.shape[0]
    buf = kv_cache.index_select(0, physical_page_indices).float()
    buf[:, 0].mul_(k_scale)
    buf[:, 1].mul_(v_scale)
    buf = buf.to(out_dtype)
    if (
        out is not None
        and out.shape[0] >= num_pages
        and out.dtype == out_dtype
        and out.device == buf.device
    ):
        out[:num_pages].copy_(buf)
        return out[:num_pages]
    return buf


def _dequant_int8_pages_into_buffer_torch(
    kv_cache: torch.Tensor,
    physical_page_indices: torch.Tensor,
    k_scale: float | torch.Tensor,
    v_scale: float | torch.Tensor,
    out_dtype: torch.dtype,
    buffer: torch.Tensor,
) -> None:
    """Fallback: multiple torch ops (gather, float, mul_, scatter)."""
    tmp = kv_cache.index_select(0, physical_page_indices).float()
    tmp[:, 0].mul_(k_scale)
    tmp[:, 1].mul_(v_scale)
    buffer[physical_page_indices] = tmp.to(out_dtype)


def dequant_int8_pages_into_buffer(
    kv_cache: torch.Tensor,
    physical_page_indices: torch.Tensor,
    k_scale: float | torch.Tensor,
    v_scale: float | torch.Tensor,
    out_dtype: torch.dtype,
    buffer: torch.Tensor,
) -> None:
    """Dequantize int8 KV pages into a full-size buffer at original block indices.

    Writes only at buffer[physical_page_indices]; no contiguous reindexing.
    Caller passes the same buffer + original paged_kv_indices to the attention
    backend. Buffer must have shape (num_blocks, 2, block_size, num_kv_heads,
    head_size) matching kv_cache.
    Uses a fused Triton kernel when available; otherwise falls back to torch.
    """
    logger.info_once("int8_prefill_kv: dequant_int8_pages_into_buffer")
    num_pages = physical_page_indices.shape[0]
    if num_pages == 0:
        return

    if not HAS_TRITON:
        _dequant_int8_pages_into_buffer_torch(
            kv_cache,
            physical_page_indices,
            k_scale,
            v_scale,
            out_dtype,
            buffer,
        )
        return

    # Triton kernel assumes contiguous last 4 dims per block
    if not kv_cache.is_contiguous() or not buffer.is_contiguous():
        _dequant_int8_pages_into_buffer_torch(
            kv_cache,
            physical_page_indices,
            k_scale,
            v_scale,
            out_dtype,
            buffer,
        )
        return

    # Fused Triton path: one kernel (gather + dequant + scale + scatter)
    k_scale_val = float(k_scale.item() if isinstance(k_scale, torch.Tensor) else k_scale)
    v_scale_val = float(v_scale.item() if isinstance(v_scale, torch.Tensor) else v_scale)
    _, two, block_size, num_kv_heads, head_size = kv_cache.shape
    assert two == 2
    k_elems = block_size * num_kv_heads * head_size
    num_elems_per_page = 2 * k_elems
    page_stride = num_elems_per_page

    OUT_BF16 = out_dtype == torch.bfloat16
    BLOCK_ELEMS = 1024
    grid = (num_pages,)
    _dequant_int8_pages_into_buffer_kernel[grid](
        kv_cache_ptr=kv_cache,
        physical_page_indices_ptr=physical_page_indices,
        buffer_ptr=buffer,
        k_scale=k_scale_val,
        v_scale=v_scale_val,
        page_stride=page_stride,
        k_elems=k_elems,
        num_elems_per_page=num_elems_per_page,
        BLOCK_ELEMS=BLOCK_ELEMS,
        OUT_BF16=OUT_BF16,
    )


def plan_int8_prefill_for_fi(
    paged_kv_indptr_prefill_cpu: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr_np: np.ndarray,
    prefill_start: int,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute plan inputs for FlashInfer int8 prefill (contiguous buffer layout).

    FI does not support int8 KV natively; we plan with contiguous indices
    0..num_prefill_pages-1 and bf16. At forward we dequant only these pages
    into a contiguous buffer.

    Args:
        paged_kv_indptr_prefill_cpu: (num_prefills+1,) cumulative page counts.
        paged_kv_indices: full paged_kv_indices tensor.
        paged_kv_indptr_np: full indptr array (length >= num_reqs+1) for indexing.
        prefill_start: first prefill request index in the batch.
        num_reqs: total number of requests in the batch.

    Returns:
        paged_kv_indptr_prefill_0based: (num_prefills+1,) 0-based for contiguous buffer.
        paged_kv_indices_prefill: (num_prefill_pages,) [0, 1, ..., num_prefill_pages-1].
        physical_page_indices_prefill: (num_prefill_pages,) physical block ids.
    """
    logger.info_once("int8_prefill_kv: plan_int8_prefill_for_fi")
    prefill_page_start = int(paged_kv_indptr_np[prefill_start])
    num_prefill_pages = int(
        paged_kv_indptr_np[num_reqs] - prefill_page_start
    )
    paged_kv_indptr_prefill_0based = (
        paged_kv_indptr_prefill_cpu - paged_kv_indptr_prefill_cpu[0]
    )
    paged_kv_indices_prefill = torch.arange(
        num_prefill_pages,
        device=paged_kv_indices.device,
        dtype=torch.int32,
    )
    physical_page_indices_prefill = (
        paged_kv_indices[
            prefill_page_start : prefill_page_start + num_prefill_pages
        ].clone()
    )
    return (
        paged_kv_indptr_prefill_0based,
        paged_kv_indices_prefill,
        physical_page_indices_prefill,
    )


def fill_prefill_kv_from_activation(
    prefill_kv: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    physical_page_indices: torch.Tensor,
) -> None:
    """Overwrite prefill_kv with activation (key, value) for slots written this round.

    prefill_kv is (num_pages, 2, block_size, num_kv_heads, head_size). We overwrite
    positions that correspond to tokens in [num_decode_tokens, num_decode_tokens +
    num_prefill_tokens) using the passed-in key, value, so we avoid quantize-dequant
    round-trip for this round's KV. Slots from previous rounds (chunked prefill)
    are left as-is (already filled by dequant).
    """
    logger.info_once("int8_prefill_kv: fill_prefill_kv_from_activation")
    page_size = prefill_kv.shape[2]
    t_vals = torch.arange(
        num_decode_tokens,
        num_decode_tokens + num_prefill_tokens,
        device=key.device,
        dtype=torch.long,
    )
    slots = slot_mapping[t_vals]
    block_ids = slots // page_size
    pos_in_page = slots % page_size
    # L such that physical_page_indices[L] == block_id for each token
    match = physical_page_indices.unsqueeze(0) == block_ids.unsqueeze(1)
    valid = match.any(dim=1)
    if not valid.any():
        return
    L_vals = match.long().argmax(dim=1)
    prefill_kv[L_vals[valid], 0, pos_in_page[valid], :, :] = key[t_vals[valid]]
    prefill_kv[L_vals[valid], 1, pos_in_page[valid], :, :] = value[t_vals[valid]]


def fill_prefill_kv_from_activation_into_buffer(
    buffer: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    num_prefill_tokens: int,
) -> None:
    """Overwrite buffer with activation (key, value) for prefill slots this round.

    buffer has shape (num_blocks, 2, block_size, num_kv_heads, head_size).
    Writes to buffer[block_id, 0, pos, :, :] and buffer[block_id, 1, pos, :, :]
    for tokens in [num_decode_tokens, num_decode_tokens + num_prefill_tokens)
    using slot_mapping (slot = block_id * page_size + pos).
    """
    logger.info_once("int8_prefill_kv: fill_prefill_kv_from_activation_into_buffer")
    page_size = buffer.shape[2]
    t_vals = torch.arange(
        num_decode_tokens,
        num_decode_tokens + num_prefill_tokens,
        device=key.device,
        dtype=torch.long,
    )
    slots = slot_mapping[t_vals]
    block_ids = slots // page_size
    pos_in_page = slots % page_size
    buffer[block_ids, 0, pos_in_page, :, :] = key[t_vals]
    buffer[block_ids, 1, pos_in_page, :, :] = value[t_vals]


def build_int8_prefill_kv_buffer(
    kv_cache: torch.Tensor,
    physical_page_indices: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: float | torch.Tensor,
    v_scale: float | torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Build prefill KV buffer: dequant all pages, then overwrite this round from activation.

    We dequant all prefill pages because a page can be "mixed" (some slots from
    previous chunks, some from this round). If we only dequant "previous-round"
    pages and skip pages that have any this-round token, the previous-round slots
    on mixed pages would stay uninitialized. So we dequant everything, then
    overwrite this round's slots with activation to avoid quantize-dequant for
    the current forward's KV.

    A future optimization could skip dequant for pages that are known to be
    100% this round (e.g. when metadata indicates no chunked prefill).

    Args:
        kv_cache: int8 KV cache tensor.
        physical_page_indices: (num_pages,) physical block ids for prefill pages.
        slot_mapping: (num_actual_tokens,) slot for each token.
        num_decode_tokens: number of decode tokens in the batch.
        num_prefill_tokens: number of prefill tokens.
        key, value: current forward's key/value (activation).
        k_scale, v_scale: scales for int8 dequant (float or scalar tensor).
        out_dtype: dtype of the output buffer (e.g. bfloat16).

    Returns:
        prefill_kv: (num_pages, 2, block_size, num_kv_heads, head_size) in out_dtype.

    Note:
        physical_page_indices is variable-length (num_prefill_pages per batch).
        This path is only used when num_prefill_tokens > 0; FlashInfer uses
        decode cudagraph only when pure_decode, so this must not run under
        FULL cudagraph replay (see assert in flashinfer.py forward).
    """
    logger.info_once("int8_prefill_kv: build_int8_prefill_kv_buffer")
    num_pages = physical_page_indices.shape[0]
    if num_pages == 0:
        return torch.empty(
            (0,) + tuple(kv_cache.shape[1:]),
            dtype=out_dtype,
            device=kv_cache.device,
        )

    prefill_kv = dequant_int8_pages_to_contiguous(
        kv_cache,
        physical_page_indices,
        k_scale,
        v_scale,
        out_dtype,
    )

    fill_prefill_kv_from_activation(
        prefill_kv,
        key,
        value,
        slot_mapping,
        num_decode_tokens,
        num_prefill_tokens,
        physical_page_indices,
    )
    return prefill_kv

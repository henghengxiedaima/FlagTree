# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone benchmark: fp8_fp4_mqa_logits — vLLM vs FlagGems vs TLE.

Three implementations compared for correctness and performance:
  1. vLLM (DeepGEMM)
  2. FlagGems
  3. TLE (standalone kernel — zero optimization, baseline for future TLE primitive work)

Usage:
    python bench_fp8_fp4_mqa_logits.py
"""

import time
import torch
import triton
import triton.language as tl

# =============================================================================
# Imports with graceful fallback
# =============================================================================

try:
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits as vllm_fp8_fp4_mqa_logits
    from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8 as _vllm_cast_to_fp8
    VLLM_AVAILABLE = True
    print("[INFO] vLLM (DeepGEMM) — available")
except ImportError as e:
    VLLM_AVAILABLE = False
    _vllm_cast_to_fp8 = None
    print(f"[WARN] vLLM not available: {e}")

# FlagGems is inlined below — no external import needed.


# =============================================================================
# Fallback FP8 quantization
# =============================================================================

def _simple_per_dim_cast_to_fp8(x, dims, is_scale_transposed=False):
    """Simple per-dimension FP8 cast: amax-scale quantize along given dims."""
    amax = x.abs()
    for d in sorted(dims, reverse=True):
        amax = amax.amax(dim=d, keepdim=True)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = amax / fp8_max
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    for d in sorted(dims, reverse=True):
        scale = scale.squeeze(d)
    return x_fp8, scale


def cast_to_fp8(x, dims, is_scale_transposed=False):
    """FP8 cast: use vLLM impl if available, otherwise fallback."""
    if _vllm_cast_to_fp8 is not None:
        return _vllm_cast_to_fp8(x, dims, is_scale_transposed)
    return _simple_per_dim_cast_to_fp8(x, dims, is_scale_transposed)


# =============================================================================
# Shared MQA Logits Kernel (used by both FlagGems-ref and TLE wrappers)
# =============================================================================

@triton.jit
def _mqa_logits_kernel(
    Q_ptr, K_ptr, K_scale_ptr, W_ptr, O_ptr,
    M, N,
    H: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_om, stride_on,
    stride_wm, stride_wh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_BLOCK: tl.constexpr,
):
    """Fused FP8 MQA logits — head-batched tiled dot products.

    logits[m,n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * weights[m,h])
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # K loaded once, reused across HEAD_BLOCK head batches
    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None] & (d_offs[None, :] < D), other=0.0,
    )
    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        q = tl.load(
            Q_ptr + m_offs[:, None, None] * stride_qm
                  + hb_offs[None, :, None] * stride_qh
                  + d_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
                 & (hb_offs[None, :, None] < H)
                 & (d_offs[None, None, :] < D),
            other=0.0,
        )

        q_2d = tl.reshape(q, [BLOCK_M * HEAD_BLOCK, D])
        dot = tl.dot(q_2d, tl.trans(k))
        dot = tl.maximum(dot * k_scale[None, :], 0.0)  # k_scale + ReLU

        w = tl.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H), other=0.0,
        )
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        acc += tl.sum(dot_3d, axis=1)

    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


@triton.jit
def _clean_logits_kernel(
    O_ptr, KS_ptr, KE_ptr, M, N,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fill invalid positions with -inf based on per-row [ks, ke) ranges."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    ks = tl.load(KS_ptr + m_offs, mask=m_mask, other=0)
    ke = tl.load(KE_ptr + m_offs, mask=m_mask, other=0)

    invalid_mask = (n_offs[None, :] < ks[:, None]) | (n_offs[None, :] >= ke[:, None])
    write_mask = m_mask[:, None] & n_mask[None, :] & invalid_mask

    neg_inf = float("-inf")
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, neg_inf + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
             mask=write_mask)


# =============================================================================
# TLE Step 1: fused clean-logits kernel
# =============================================================================

@triton.jit
def _tle_mqa_logits_kernel(
    Q_ptr, K_ptr, K_scale_ptr, W_ptr, O_ptr,
    KS_ptr, KE_ptr,
    M, N,
    H: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_om, stride_on,
    stride_wm, stride_wh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
):
    """TLE fused MQA logits — clean-logits inlined into the main kernel.

    logits[m,n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * weights[m,h])
    Invalid positions (n < ks[m] or n >= ke[m]) are filled with -inf in-place.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    m_mask = m_offs < M
    n_mask = n_offs < N

    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None] & (d_offs[None, :] < D), other=0.0,
    )
    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        q = tl.load(
            Q_ptr + m_offs[:, None, None] * stride_qm
                  + hb_offs[None, :, None] * stride_qh
                  + d_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
                 & (hb_offs[None, :, None] < H)
                 & (d_offs[None, None, :] < D),
            other=0.0,
        )

        q_2d = tl.reshape(q, [BLOCK_M * HEAD_BLOCK, D])
        dot = tl.dot(q_2d, tl.trans(k))
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        w = tl.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H), other=0.0,
        )
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        acc += tl.sum(dot_3d, axis=1)

    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    base_mask = m_mask[:, None] & n_mask[None, :]

    if CLEAN_LOGITS:
        ks = tl.load(KS_ptr + m_offs, mask=m_mask, other=0)
        ke = tl.load(KE_ptr + m_offs, mask=m_mask, other=0)
        invalid = (n_offs[None, :] < ks[:, None]) | (n_offs[None, :] >= ke[:, None])
        # Store -inf on invalid positions first, then valid acc on the rest
        neg_inf = float("-inf")
        tl.store(out_ptrs, neg_inf + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
                 mask=base_mask & invalid)
        tl.store(out_ptrs, acc, mask=base_mask & ~invalid)
    else:
        tl.store(out_ptrs, acc, mask=base_mask)


# =============================================================================
# Wrappers — FlagGems-ref (baseline) and TLE (optimized)
# =============================================================================

_DEFAULT_CONFIG = (64, 128, 2, 4, 2)
_CLEAN_M = 8
_CLEAN_N = 128


def fg_fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                           clean_logits=True):
    """FlagGems reference — original two-kernel implementation."""
    q_values, _ = q
    k_values, k_scales = kv
    M, H, D = q_values.shape
    N = k_values.shape[0]
    BLOCK_M, BLOCK_N, HEAD_BLOCK, num_warps, num_stages = _DEFAULT_CONFIG

    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mqa_logits_kernel[grid](
        q_values, k_values, k_scales, weights, logits,
        M, N, H, D,
        q_values.stride(0), q_values.stride(1), q_values.stride(2),
        k_values.stride(0), k_values.stride(1),
        logits.stride(0), logits.stride(1),
        weights.stride(0), weights.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_BLOCK=HEAD_BLOCK,
        num_warps=num_warps, num_stages=num_stages,
    )

    if clean_logits:
        clean_grid = (triton.cdiv(M, _CLEAN_M), triton.cdiv(N, _CLEAN_N))
        _clean_logits_kernel[clean_grid](
            logits, cu_seqlen_ks, cu_seqlen_ke, M, N,
            logits.stride(0), logits.stride(1),
            BLOCK_M=_CLEAN_M, BLOCK_N=_CLEAN_N,
        )

    return logits


def tle_fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                            clean_logits=True):
    """TLE Step 1 — fused clean-logits (single kernel, no second pass)."""
    q_values, _ = q
    k_values, k_scales = kv
    M, H, D = q_values.shape
    N = k_values.shape[0]
    BLOCK_M, BLOCK_N, HEAD_BLOCK, num_warps, num_stages = _DEFAULT_CONFIG

    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tle_mqa_logits_kernel[grid](
        q_values, k_values, k_scales, weights, logits,
        cu_seqlen_ks, cu_seqlen_ke,
        M, N, H, D,
        q_values.stride(0), q_values.stride(1), q_values.stride(2),
        k_values.stride(0), k_values.stride(1),
        logits.stride(0), logits.stride(1),
        weights.stride(0), weights.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_BLOCK=HEAD_BLOCK,
        CLEAN_LOGITS=clean_logits,
        num_warps=num_warps, num_stages=num_stages,
    )

    return logits


# =============================================================================
# Benchmark
# =============================================================================

H = 64
D = 128

BENCH_SHAPES = [
    (1, 1024), (1, 2048), (4, 2048), (4, 4096),        # decode
    (64, 4096), (256, 4096), (1024, 4096), (2048, 4096), (1024, 8192),  # prefill
]

WARMUP_ITERS = 5
BENCH_ITERS = 20


def build_inputs(M, N, device="cuda"):
    torch.manual_seed(42)
    q_bf16 = torch.randn(M, H, D, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scale = cast_to_fp8(k_bf16, (0,), False)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)
    return q_fp8, k_fp8, k_scale, weights, ks, ke


def _time_ms(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def benchmark_shape(M, N):
    device = "cuda"
    q_fp8, k_fp8, k_scale, weights, ks, ke = build_inputs(M, N, device)
    results = {}
    ref = None

    # --- vLLM ---
    if VLLM_AVAILABLE:
        try:
            out = vllm_fp8_fp4_mqa_logits(
                q=(q_fp8, None), kv=(k_fp8, k_scale),
                weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
            )
            ref = out
            for _ in range(WARMUP_ITERS):
                vllm_fp8_fp4_mqa_logits(
                    q=(q_fp8, None), kv=(k_fp8, k_scale),
                    weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
                )
            times = [_time_ms(vllm_fp8_fp4_mqa_logits,
                              q=(q_fp8, None), kv=(k_fp8, k_scale),
                              weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True)
                     for _ in range(BENCH_ITERS)]
            results["vLLM"] = (out, min(times))
        except Exception as e:
            print(f"  [SKIP] vLLM: {e}")

    # --- FlagGems reference (inline kernel) ---
    out = fg_fp8_fp4_mqa_logits(
        q=(q_fp8, None), kv=(k_fp8, k_scale),
        weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
    )
    if ref is None:
        ref = out
    for _ in range(WARMUP_ITERS):
        fg_fp8_fp4_mqa_logits(
            q=(q_fp8, None), kv=(k_fp8, k_scale),
            weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
        )
    times = [_time_ms(fg_fp8_fp4_mqa_logits,
                      q=(q_fp8, None), kv=(k_fp8, k_scale),
                      weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True)
             for _ in range(BENCH_ITERS)]
    results["FlagGems"] = (out, min(times))

    # --- TLE ---
    out = tle_fp8_fp4_mqa_logits(
        q=(q_fp8, None), kv=(k_fp8, k_scale),
        weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
    )
    if ref is None:
        ref = out
    for _ in range(WARMUP_ITERS):
        tle_fp8_fp4_mqa_logits(
            q=(q_fp8, None), kv=(k_fp8, k_scale),
            weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True,
        )
    times = [_time_ms(tle_fp8_fp4_mqa_logits,
                      q=(q_fp8, None), kv=(k_fp8, k_scale),
                      weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, clean_logits=True)
             for _ in range(BENCH_ITERS)]
    results["TLE"] = (out, min(times))

    # --- Correctness vs ref ---
    if ref is not None:
        for name in list(results.keys()):
            out, t = results[name]
            diff = (ref.float() - out.float()).abs().max().item()
            status = "PASS" if diff < 1e-1 else "FAIL"
            results[name] = (out, t, diff, status)

    return results


def main():
    print("=" * 80)
    print("fp8_fp4_mqa_logits Benchmark: vLLM vs FlagGems vs TLE")
    print(f"  H={H}, D={D}  |  config={_DEFAULT_CONFIG}")
    print(f"  Warmup={WARMUP_ITERS}, Bench={BENCH_ITERS}")
    print("=" * 80)

    # --- Performance ---
    perf_header = f"{'Shape':>10} | {'vLLM ms':>9} | {'FG ms':>8} | {'TLE ms':>8} | {'TLE/FG':>7}"
    print("--- Performance ---")
    print(perf_header)
    print("-" * len(perf_header))

    all_ok = True
    correctness_rows = []

    for M, N in BENCH_SHAPES:
        r = benchmark_shape(M, N)
        if not r:
            continue

        v_t = r.get("vLLM", (None, None))[1]
        f_t = r.get("FlagGems", (None, None))[1]
        t_t = r.get("TLE", (None, None))[1]

        def _s(v): return f"{v:.4f}" if v else "     N/A"
        speedup = f"{f_t/t_t:.2f}x" if (t_t and f_t) else "    N/A"

        parts = []
        for name in ["vLLM", "FlagGems", "TLE"]:
            e = r.get(name)
            if e and len(e) >= 4:
                parts.append(f"{name}:{e[3]}")
                if e[3] != "PASS":
                    all_ok = False
        corr = " | ".join(parts)

        print(f"{M}x{N:>5} | {_s(v_t):>9} | {_s(f_t):>8} | {_s(t_t):>8} | {speedup:>7}")
        correctness_rows.append((M, N, corr, parts))

    # --- Correctness ---
    print(f"\n--- Correctness (ref = {'vLLM' if VLLM_AVAILABLE else 'FlagGems'}) ---")
    for M, N, corr, parts in correctness_rows:
        failed = [p for p in parts if "FAIL" in p]
        flag = "  OK" if not failed else "FAIL: " + ", ".join(failed)
        print(f"  {M}x{N:<5}  {flag}")

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED — see above'}")
    print("FlagGems-ref = original two-kernel | TLE Step 1 = fused clean-logits")
    print("=" * 80)


if __name__ == "__main__":
    main()

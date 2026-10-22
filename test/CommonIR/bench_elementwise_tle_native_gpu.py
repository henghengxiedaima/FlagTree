"""B-group benchmark: native TLE path (no CommonIR) on the same triad shape.

In a CommonIR build this measures the CommonIR path; in a non-CommonIR build
(`FLAGTREE_COMMON_IR=0`) `tle.gpu.*` lowers to local_ptr + tl.load/store and the
backend's own TTGIR passes (for metax: mctle) optimise it. That is the "TLE
native" baseline the CommonIR path is compared against.

    python test/CommonIR/bench_elementwise_tle_native_gpu.py
    ... --rows 65536 --cols 1024
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _add_tle_native(a_ptr, b_ptr, c_ptr, N, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * XBLOCK + tl.arange(0, XBLOCK)
    cols = pid_n * YBLOCK + tl.arange(0, YBLOCK)

    the_a = a_ptr + rows[:, None] * N + cols[None, :]
    the_b = b_ptr + rows[:, None] * N + cols[None, :]
    the_c = c_ptr + rows[:, None] * N + cols[None, :]

    a_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    c_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)

    rid = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    cid = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_smem, (rid, cid))
    b_ptrs = tle.gpu.local_ptr(b_smem, (rid, cid))
    c_ptrs = tle.gpu.local_ptr(c_smem, (rid, cid))

    tle.gpu.copy(the_a, a_smem, [XBLOCK, YBLOCK])
    tle.gpu.copy(the_b, b_smem, [XBLOCK, YBLOCK])
    av = tl.load(a_ptrs)
    bv = tl.load(b_ptrs)
    tl.store(c_ptrs, av + bv)
    tle.gpu.copy(c_smem, the_c, [XBLOCK, YBLOCK])


@triton.jit
def _add_plain(a_ptr, b_ptr, c_ptr, N, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * XBLOCK + tl.arange(0, XBLOCK)
    cols = pid_n * YBLOCK + tl.arange(0, YBLOCK)
    offs = rows[:, None] * N + cols[None, :]
    tl.store(c_ptr + offs, tl.load(a_ptr + offs) + tl.load(b_ptr + offs))


def _build(rows, cols, xblock, yblock):
    a = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
    b = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
    c = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(rows, xblock), triton.cdiv(cols, yblock))
    return a, b, c, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--xblock", type=int, default=16)
    parser.add_argument("--yblock", type=int, default=64)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    try:
        import triton._common_ir as cir
        mode = "CommonIR" if cir.ENABLED else "native-TLE (no CommonIR)"
    except Exception:
        mode = "unknown"
    print(f"[mode] {mode}")

    a, b, c, grid = _build(args.rows, args.cols, args.xblock, args.yblock)
    kb = lambda fn: fn[grid](a, b, c, args.cols, XBLOCK=args.xblock, YBLOCK=args.yblock, num_warps=4)

    ck = kb(_add_tle_native)
    if not args.no_check:
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)
        print(f"[check] tle-native triad PASS  shared={getattr(ck.metadata, 'shared', 'n/a')} B")
    if args.no_bench:
        return

    t_tle = triton.testing.do_bench(lambda: kb(_add_tle_native), warmup=25, rep=100)
    t_pl = triton.testing.do_bench(lambda: kb(_add_plain), warmup=25, rep=100)
    gbytes = 3 * args.rows * args.cols * 4 / 1e9
    print(f"[bench] tle-native {t_tle:8.4f} ms {gbytes / (t_tle * 1e-3):7.1f} GB/s "
          f"shared={getattr(ck.metadata, 'shared', 'n/a')} B")
    print(f"[bench] plain      {t_pl:8.4f} ms {gbytes / (t_pl * 1e-3):7.1f} GB/s")
    print(f"[bench] ratio (tle-native/plain) = {t_tle / t_pl:.3f}x")


if __name__ == "__main__":
    main()

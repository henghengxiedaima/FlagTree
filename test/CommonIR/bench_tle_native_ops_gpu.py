"""B-group (native TLE, no CommonIR) benchmarks for reduction ops and matmul.

Same shapes as the CommonIR benchmarks so the four groups can be compared
directly. In a CommonIR build these kernels would take the tile.* path; run
them in a `FLAGTREE_COMMON_IR=0` build to measure the native path.
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _softmax_native(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = tl.arange(0, BLOCK)

    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    out_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)
    buf_p = tle.gpu.local_ptr(buf, (idx, ))
    out_p = tle.gpu.local_ptr(out_buf, (idx, ))

    tle.gpu.copy(a_ptr + offs, buf, [BLOCK])
    v = tl.load(buf_p)
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    s = tl.sum(e, 0)
    tl.store(out_p, e / s)
    tle.gpu.copy(out_buf, c_ptr + offs, [BLOCK])


@triton.jit
def _rmsnorm_native(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = tl.arange(0, BLOCK)

    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    out_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)
    buf_p = tle.gpu.local_ptr(buf, (idx, ))
    out_p = tle.gpu.local_ptr(out_buf, (idx, ))

    tle.gpu.copy(a_ptr + offs, buf, [BLOCK])
    v = tl.load(buf_p)
    ms = tl.sum(v * v, 0) / BLOCK
    tl.store(out_p, v / tl.sqrt(ms + 1e-6))
    tle.gpu.copy(out_buf, c_ptr + offs, [BLOCK])


@triton.jit
def _matmul_native(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_buf = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_buf = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    c_buf = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)

    rid_m = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    cid_k = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    rid_k = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
    cid_n = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
    a_p = tle.gpu.local_ptr(a_buf, (rid_m, cid_k))
    b_p = tle.gpu.local_ptr(b_buf, (rid_k, cid_n))

    rid_c = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_N))
    cid_c = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))
    c_p = tle.gpu.local_ptr(c_buf, (rid_c, cid_c))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a_tile = a_ptr + offs_m[:, None] * sam + (k0 + offs_k)[None, :] * sak
        b_tile = b_ptr + (k0 + offs_k)[:, None] * sbk + offs_n[None, :] * sbn
        tle.gpu.copy(a_tile, a_buf, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_tile, b_buf, [BLOCK_K, BLOCK_N])
        acc += tl.dot(tl.load(a_p), tl.load(b_p), input_precision="ieee")

    tl.store(c_p, acc)
    c_tile = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tle.gpu.copy(c_buf, c_tile, [BLOCK_M, BLOCK_N])


def _ref(op, a, block):
    rows = a.numel() // block
    x = a.view(rows, block).float()
    if op == "softmax":
        m = x.max(dim=1, keepdim=True).values
        e = (x - m).exp()
        return (e / e.sum(dim=1, keepdim=True)).view(-1)
    ms = (x * x).mean(dim=1, keepdim=True)
    return (x / torch.sqrt(ms + 1e-6)).view(-1)


def run_reduction(op, rows, block):
    n = rows * block
    torch.manual_seed(0)
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    grid = (rows, )
    kern = _softmax_native if op == "softmax" else _rmsnorm_native
    ck = kern[grid](a, c, BLOCK=block, num_warps=4)
    torch.testing.assert_close(c, _ref(op, a, block), atol=2e-5, rtol=2e-5)
    ms = triton.testing.do_bench(lambda: kern[grid](a, c, BLOCK=block, num_warps=4),
                                 warmup=25, rep=100)
    gbytes = 2 * n * 4 / 1e9
    print(f"[native] {op:<8} rows={rows:<6} BLOCK={block:<5} {ms:8.4f} ms "
          f"{gbytes / (ms * 1e-3):7.1f} GB/s shared={ck.metadata.shared} B")


def run_matmul(size, block=16):
    M = N = K = size
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, block), triton.cdiv(N, block))
    launch = lambda: _matmul_native[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLOCK_M=block, BLOCK_N=block, BLOCK_K=block)
    ck = launch()
    torch.testing.assert_close(c, torch.matmul(a.float(), b.float()), atol=2e-2, rtol=2e-2)
    ms = triton.testing.do_bench(launch, warmup=25, rep=100)
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[native] matmul   {size}^3  {ms:8.4f} ms {tf:6.3f} TFLOPS shared={ck.metadata.shared} B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["softmax", "rmsnorm", "matmul", "all"], default="all")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()

    try:
        import triton._common_ir as cir
        print(f"[mode] {'CommonIR' if cir.ENABLED else 'native-TLE (no CommonIR)'}")
    except Exception:
        pass

    if args.op in ("softmax", "rmsnorm", "all"):
        for op in ("softmax", "rmsnorm"):
            run_reduction(op, args.rows, args.block)
    if args.op in ("matmul", "all"):
        for size in (1024, 2048):
            run_matmul(size)


if __name__ == "__main__":
    main()

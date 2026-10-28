"""A-group baseline: the same 16x16-tiled matmul in plain Triton.

Used to fill the missing A column of the matmul table. Same tiling, same
accumulation precision, same strides -- only the buffer handling differs
(tl.load/tl.store instead of the TLE buffer API).

    CUDA_VISIBLE_DEVICES=0 python test/CommonIR/bench_matmul_plain_gpu.py
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_plain(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * sam + (k0 + offs_k)[None, :] * sak)
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * sbk + offs_n[None, :] * sbn)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc)


def run(size, block=16, check=True):
    M = N = K = size
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, block), triton.cdiv(N, block))
    launch = lambda: _matmul_plain[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLOCK_M=block, BLOCK_N=block, BLOCK_K=block)
    ck = launch()
    if check:
        torch.testing.assert_close(c, torch.matmul(a.float(), b.float()), atol=2e-2, rtol=2e-2)
    ms = triton.testing.do_bench(launch, warmup=25, rep=100)
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[plain] matmul {size}^3  {ms:9.4f} ms  {tf:6.3f} TFLOPS  "
          f"shared={getattr(ck.metadata, 'shared', 'n/a')} B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--block", type=int, default=16)
    args = parser.parse_args()
    for size in args.sizes:
        run(size, args.block)


if __name__ == "__main__":
    main()

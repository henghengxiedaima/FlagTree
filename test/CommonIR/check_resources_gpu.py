"""Resource footprint of the same kernels with and without the staging pass.

Run twice (separate processes, the switch is read at startup):
    CUDA_VISIBLE_DEVICES=0 python test/CommonIR/check_resources_gpu.py
    CUDA_VISIBLE_DEVICES=0 FLAGTREE_COMMONIR_NO_STAGING_ELIM=1 python test/CommonIR/check_resources_gpu.py

Reports shared memory, register count and spill count for the three families.
"""

import importlib.util
import pathlib

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ew = _load("b_ew", HERE / "bench_elementwise_gpu.py")
red = _load("b_red", HERE / "bench_softmax_rmsnorm_gpu.py")
mm = _load("b_mm", HERE / "native_matmul_gpu.py")


def _meta(ck):
    shared = getattr(ck.metadata, "shared", "n/a")
    regs = getattr(ck, "n_regs", None)
    if regs is None:
        regs = getattr(ck.metadata, "num_regs", "n/a")
    spills = getattr(ck, "n_spills", None)
    if spills is None:
        spills = getattr(ck.metadata, "num_spills", "n/a")
    return shared, regs, spills


def main():
    mode = "D (opt on)"
    try:
        import os
        if os.environ.get("FLAGTREE_COMMONIR_NO_STAGING_ELIM", "") not in ("", "0"):
            mode = "C (opt off)"
    except Exception:
        pass
    print(f"[mode] {mode}")

    n = 1 << 20
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(n, 1024), )
    ck = ew._elementwise_cm_gpu_kernel[grid](a, b, c, n, BLOCK=1024, num_warps=4)
    print(f"[res] elementwise triad  shared={_meta(ck)[0]:>6} B  regs={_meta(ck)[1]}  spills={_meta(ck)[2]}")

    n = 65536 * 1024
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    grid = (65536, )
    for name, kern in (("softmax", red._softmax_cm), ("rmsnorm", red._rmsnorm_cm)):
        ck = kern[grid](a, c, BLOCK=1024, num_warps=4)
        print(f"[res] {name:<17} shared={_meta(ck)[0]:>6} B  regs={_meta(ck)[1]}  spills={_meta(ck)[2]}")

    M = N = K = 512
    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    y = torch.randn((K, N), device="cuda", dtype=torch.float16)
    z = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    ck = mm._matmul_gpu_kernel[grid](
        x, y, z, M, N, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        z.stride(0), z.stride(1), BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    print(f"[res] matmul 512^3      shared={_meta(ck)[0]:>6} B  regs={_meta(ck)[1]}  spills={_meta(ck)[2]}")


if __name__ == "__main__":
    main()

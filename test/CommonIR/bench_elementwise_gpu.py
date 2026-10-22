"""Element-wise CommonIR (tile.*) benchmark: SMEM staging vs the plain path.

Companion to native_matmul_gpu.py. The CommonIR kernel keeps exactly the same
shape as the matmul seed (alloc -> copy -> to_tensor -> compute ->
store_tensor -> to_tensor -> global store), but the workload is memory bound:
if the CommonIR path materialises a *same-encoding* store/load round trip
through shared memory, there is no layout conversion to justify it and the
round trip is pure overhead.

Usage:
    python test/CommonIR/bench_elementwise_gpu.py
    python test/CommonIR/bench_elementwise_gpu.py --dump-tileir /tmp/e_tile.mlir \
        --dump-ttir /tmp/e_a.mlir --no-bench
"""

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _elementwise_cm_gpu_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    a_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    b_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    c_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)

    a_ptrs = a_ptr + offs
    b_ptrs = b_ptr + offs
    c_ptrs = c_ptr + offs

    tle.gpu.copy(a_ptrs, a_buf, [BLOCK])
    tle.gpu.copy(b_ptrs, b_buf, [BLOCK])

    av = tle.gpu.to_tensor(a_buf, writable=False)
    bv = tle.gpu.to_tensor(b_buf, writable=False)
    cv = av + bv

    tle.gpu.store_tensor(cv, c_buf)
    cvals = tle.gpu.to_tensor(c_buf, writable=False)
    tl.store(c_ptrs, cvals, mask=mask)


@triton.jit
def _elementwise_plain_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def _compile_options():
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 3})
    return target, backend, options


def _frontend_module(kernel, signature, constants):
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    target, backend, options = _compile_options()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(kernel, signature, constants)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    if not module.verify():
        raise RuntimeError("frontend module.verify() failed")
    return module


def _add_commonir_to_ttgir(pm, enable_async=False):
    lib = __import__("triton._C.libtriton", fromlist=["tle", "iluvatar", "nvidia"])
    for ns in ("tle", "iluvatar", "nvidia"):
        add = getattr(getattr(getattr(getattr(lib, ns, None), "passes", None), "commonir", None),
                      "add_to_ttgir", None)
        if add is not None:
            add(pm, enable_async)
            return ns
    raise RuntimeError("converted TTIR dump requires CommonIRToTTGIR (add_to_ttgir)")


def _default_ir_args(block=1024):
    constants = {"BLOCK": block}
    signature = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "n": "i32"}
    return signature, constants


def dump_tileir(path, block=1024):
    signature, constants = _default_ir_args(block)
    module = _frontend_module(_elementwise_cm_gpu_kernel, signature, constants)
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle not in text:
            raise RuntimeError(f"expected {needle} in TileIR dump")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-tileir] wrote {path}")


def dump_ttir(path, block=1024):
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes

    signature, constants = _default_ir_args(block)
    module = _frontend_module(_elementwise_cm_gpu_kernel, signature, constants)
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    pm.run(module, "bench_elementwise.inliner")

    pm = ir.pass_manager(module.context)
    _add_commonir_to_ttgir(pm, False)
    pm.run(module, "bench_elementwise.tileir_to_ttgir")

    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle in text:
            raise RuntimeError(f"converted TTIR dump still contains {needle}")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-ttir] wrote {path}")


def _device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_musa  # noqa: F401
        if torch.musa.is_available():
            return "musa"
    except Exception:
        pass
    return "gcu"


def _run(kernel, a, b, c, n, block, warp=4):
    grid = (triton.cdiv(n, block), )
    return kernel[grid](a, b, c, n, BLOCK=block, num_warps=warp)


def _shared_bytes(compiled):
    try:
        return compiled.metadata.shared
    except Exception:
        meta = getattr(compiled, "metadata", None)
        return getattr(meta, "shared", "n/a")


def run_check(n=1 << 20, block=1024):
    device = _device()
    torch.manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)
    c = torch.empty(n, device=device, dtype=torch.float32)
    ck = _run(_elementwise_cm_gpu_kernel, a, b, c, n, block)
    torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)
    print(f"[check] elementwise CommonIR precision PASS (n={n}, BLOCK={block}, "
          f"shared={_shared_bytes(ck)} B)")
    return ck


def run_bench(n=1 << 20, block=1024):
    device = _device()
    torch.manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)
    c = torch.empty(n, device=device, dtype=torch.float32)
    grid = (triton.cdiv(n, block), )

    ck_cm = _run(_elementwise_cm_gpu_kernel, a, b, c, n, block)
    ck_pl = _run(_elementwise_plain_kernel, a, b, c, n, block)

    t_cm = triton.testing.do_bench(lambda: _run(_elementwise_cm_gpu_kernel, a, b, c, n, block),
                                   warmup=25, rep=100)
    t_pl = triton.testing.do_bench(lambda: _run(_elementwise_plain_kernel, a, b, c, n, block),
                                   warmup=25, rep=100)

    # 3 arrays touched: read a, read b, write c
    gbytes = 3 * n * 4 / 1e9
    print(f"[bench] n={n} BLOCK={block} grid={grid[0]}")
    print(f"[bench] CommonIR : {t_cm:8.4f} ms  {gbytes / (t_cm * 1e-3):7.1f} GB/s  "
          f"shared={_shared_bytes(ck_cm)} B")
    print(f"[bench] plain    : {t_pl:8.4f} ms  {gbytes / (t_pl * 1e-3):7.1f} GB/s  "
          f"shared={_shared_bytes(ck_pl)} B")
    print(f"[bench] ratio (CommonIR/plain) = {t_cm / t_pl:.3f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-tileir", nargs="?", const="bench_elementwise_tileir.mlir")
    parser.add_argument("--dump-ttir", nargs="?", const="bench_elementwise_ttir.mlir")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--n", type=int, default=1 << 20)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()

    if args.dump_tileir:
        dump_tileir(args.dump_tileir, args.block)
    if args.dump_ttir:
        dump_ttir(args.dump_ttir, args.block)
    if not args.no_check:
        run_check(args.n, args.block)
    if not args.no_bench:
        run_bench(args.n, args.block)


if __name__ == "__main__":
    main()

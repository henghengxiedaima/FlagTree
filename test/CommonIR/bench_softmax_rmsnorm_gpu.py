"""Reduction-shaped CommonIR benchmark: softmax and RMSNorm per row.

Both kernels are written twice:
  * `*_cm` uses the TLE GPU (CommonIR) buffer API: alloc -> copy -> to_tensor ->
    reduce -> store_tensor -> to_tensor -> global store.
  * `*_plain` is the same math with plain tl.load/tl.store.

Unlike the element-wise seed these have a cross-thread reduction, so the staged
buffer is read after a reduction and the backend needs a real barrier; the point
of the benchmark is whether the CommonIR staging still costs something here.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test/CommonIR/bench_softmax_rmsnorm_gpu.py
    ... --op softmax --rows 4096 --block 1024
    ... --dump-tileir /tmp/s_tile.mlir --dump-ttir /tmp/s_a.mlir --op softmax --no-bench
"""

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

EPS = 1e-6


@triton.jit
def _softmax_cm(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    out_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)

    tle.gpu.copy(a_ptr + offs, buf, [BLOCK])
    v = tle.gpu.to_tensor(buf, writable=False)

    m = tl.max(v, 0)
    e = tl.exp(v - m)
    s = tl.sum(e, 0)
    y = e / s

    tle.gpu.store_tensor(y, out_buf)
    out = tle.gpu.to_tensor(out_buf, writable=False)
    tl.store(c_ptr + offs, out)


@triton.jit
def _softmax_plain(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(a_ptr + offs)
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    s = tl.sum(e, 0)
    tl.store(c_ptr + offs, e / s)


@triton.jit
def _rmsnorm_cm(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    out_buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)

    tle.gpu.copy(a_ptr + offs, buf, [BLOCK])
    v = tle.gpu.to_tensor(buf, writable=False)

    ms = tl.sum(v * v, 0) / BLOCK
    y = v / tl.sqrt(ms + 1e-6)

    tle.gpu.store_tensor(y, out_buf)
    out = tle.gpu.to_tensor(out_buf, writable=False)
    tl.store(c_ptr + offs, out)


@triton.jit
def _rmsnorm_plain(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(a_ptr + offs)
    ms = tl.sum(v * v, 0) / BLOCK
    tl.store(c_ptr + offs, v / tl.sqrt(ms + 1e-6))


OPS = {
    "softmax": (_softmax_cm, _softmax_plain),
    "rmsnorm": (_rmsnorm_cm, _rmsnorm_plain),
}


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


def dump_ir(op, kind, path, block):
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes

    cm_kernel, _ = OPS[op]
    signature = {"a_ptr": "*fp32", "c_ptr": "*fp32"}
    constants = {"BLOCK": block}
    module = _frontend_module(cm_kernel, signature, constants)

    if kind == "ttir":
        pm = ir.pass_manager(module.context)
        passes.common.add_inliner(pm)
        pm.run(module, "bench_reduction.inliner")
        pm = ir.pass_manager(module.context)
        _add_commonir_to_ttgir(pm, False)
        pm.run(module, "bench_reduction.tileir_to_ttgir")

    Path(path).write_text(str(module), encoding="utf-8")
    print(f"[dump-{kind}] {op}: wrote {path}")


def _device():
    return "cuda" if torch.cuda.is_available() else "gcu"


def _run(kernel, a, c, grid, block):
    return kernel[grid](a, c, BLOCK=block)


def reference(op, a, block):
    rows = a.numel() // block
    x = a.view(rows, block).float()
    if op == "softmax":
        m = x.max(dim=1, keepdim=True).values
        e = (x - m).exp()
        return (e / e.sum(dim=1, keepdim=True)).view(-1)
    ms = (x * x).mean(dim=1, keepdim=True)
    return (x / torch.sqrt(ms + 1e-6)).view(-1)


def run(op, rows, block, do_check=True, do_bench=True):
    cm_kernel, plain_kernel = OPS[op]
    device = _device()
    n = rows * block
    torch.manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float32)
    c = torch.empty(n, device=device, dtype=torch.float32)
    grid = (rows, )

    ck_cm = _run(cm_kernel, a, c, grid, block)
    if do_check:
        ref = reference(op, a, block)
        torch.testing.assert_close(c, ref, atol=2e-5, rtol=2e-5)
        print(f"[check] {op} CommonIR precision PASS (rows={rows}, BLOCK={block})")

    if not do_bench:
        return

    ck_pl = _run(plain_kernel, a, c, grid, block)
    t_cm = triton.testing.do_bench(lambda: _run(cm_kernel, a, c, grid, block),
                                   warmup=25, rep=100)
    t_pl = triton.testing.do_bench(lambda: _run(plain_kernel, a, c, grid, block),
                                   warmup=25, rep=100)
    gbytes = 2 * n * 4 / 1e9  # read a + write c
    shared = getattr(ck_cm.metadata, "shared", "n/a")
    print(f"[bench] {op:<8} rows={rows:<6} BLOCK={block:<5} "
          f"CommonIR {t_cm:8.4f} ms {gbytes / (t_cm * 1e-3):7.1f} GB/s shared={shared} B | "
          f"plain {t_pl:8.4f} ms {gbytes / (t_pl * 1e-3):7.1f} GB/s | "
          f"ratio {t_cm / t_pl:.3f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=sorted(OPS), default=None)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--dump-tileir", default=None)
    parser.add_argument("--dump-ttir", default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    ops = [args.op] if args.op else sorted(OPS)
    for op in ops:
        if args.dump_tileir:
            dump_ir(op, "tileir", args.dump_tileir, args.block)
        if args.dump_ttir:
            dump_ir(op, "ttir", args.dump_ttir, args.block)
        run(op, args.rows, args.block, not args.no_check, not args.no_bench)


if __name__ == "__main__":
    main()

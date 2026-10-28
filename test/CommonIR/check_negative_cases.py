"""Boundary cases for the staging-elimination pass.

Two groups:

  must KEEP (staging survives, shared > 0)
    A) two readers                        -> C3 fails
    B) written inside a loop, read after  -> C4 fails (different blocks)
    C) read before write in the same block-> C4 fails (order)

  must ELIMINATE (shared == 0) and stay numerically correct
    D) write + read inside the same loop iteration -> C1..C7 hold, so the
       staging is loop-local and is removed; the result must still be right.

Run in a CommonIR build:
    CUDA_VISIBLE_DEVICES=0 python test/CommonIR/check_negative_cases.py
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _alloc(block):
    return tle.gpu.alloc([block], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)


@triton.jit
def _two_readers(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    tle.gpu.copy(a_ptr + offs, buf, [BLOCK])
    v1 = tle.gpu.to_tensor(buf, writable=False)
    v2 = tle.gpu.to_tensor(buf, writable=False)
    tl.store(c_ptr + offs, v1 + v2)


@triton.jit
def _write_in_loop_read_after(a_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    for i in range(0, n, BLOCK):
        tle.gpu.copy(a_ptr + i + offs, buf, [BLOCK])
    v = tle.gpu.to_tensor(buf, writable=False)   # different block from the write
    tl.store(c_ptr + offs, v)


@triton.jit
def _read_then_write(a_ptr, c_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    before = tle.gpu.to_tensor(buf, writable=False)   # read precedes the write
    v = before + tl.load(a_ptr + offs)
    tle.gpu.store_tensor(v, buf)
    tl.store(c_ptr + offs, tle.gpu.to_tensor(buf, writable=False))


@triton.jit
def _loop_local(a_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n, BLOCK):
        tle.gpu.copy(a_ptr + i + offs, buf, [BLOCK])
        acc += tle.gpu.to_tensor(buf, writable=False)
    tl.store(c_ptr + offs, acc)


def _launch(kernel, args, block, grid=(8, )):
    ck = kernel[grid](*args, BLOCK=block, num_warps=4)
    return getattr(ck.metadata, "shared", None)


def main():
    nblocks = 8
    block = 1024
    n = nblocks * block
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    keep_cases = [
        ("two readers        (C3)", _two_readers, (a, c)),
        ("write in loop, read after (C4)", _write_in_loop_read_after, (a, c, n)),
        ("read then write    (C4)", _read_then_write, (a, c)),
    ]
    drop_cases = [
        ("loop-local write+read (safe to drop)", _loop_local, (a, c, n)),
    ]

    failures = 0
    for name, kern, args in keep_cases:
        shared = _launch(kern, args, block)
        ok = shared is not None and shared > 0
        failures += 0 if ok else 1
        print(f"[boundary] {name:<34} shared={shared:>6} B  -> {'KEPT (ok)' if ok else 'ELIMINATED (BUG)'}")

    for name, kern, args in drop_cases:
        c.zero_()
        shared = _launch(kern, args, block, grid=(1, ))
        ok_ir = shared == 0
        acc = a.view(nblocks, block).sum(dim=0)
        ok_num = torch.allclose(c[:block], acc, atol=1e-4, rtol=1e-4)
        failures += 0 if (ok_ir and ok_num) else 1
        print(f"[boundary] {name:<34} shared={shared:>6} B  -> "
              f"{'ELIMINATED (ok)' if ok_ir else 'KEPT (unexpected)'}, "
              f"numeric {'PASS' if ok_num else 'FAIL'}")

    if failures:
        raise SystemExit(f"{failures} boundary case(s) behaved unexpectedly")
    print("[boundary] all cases as expected")


if __name__ == "__main__":
    main()

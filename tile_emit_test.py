import torch, triton, triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def _roundtrip(src, dst, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.copy(src + idx, buf, [BLOCK])
    value = tle.gpu.to_tensor(buf, writable=False)
    tle.gpu.store_tensor(value + 1, buf)
    tle.gpu.copy(buf, dst + idx, [BLOCK])

def main():
    src = torch.arange(16, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    try:
        k = _roundtrip[(1,)](src, dst, 16)
        print(">>> LAUNCH OK 端到端跑通")
        for s in ("ttir", "ttgir", "llir"):
            if s in k.asm:
                print(f">>> [{s}] tile.count = {k.asm[s].count('tile.')}")
    except Exception:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()

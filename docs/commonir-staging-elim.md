# CommonIR 层“透传暂存消除”实验记录

> 状态：MetaX C550 上完成实现与评测；HCU / Enflame / Iluvatar / MThreads 待复现。
> 代码：分支 `test-little`（服务器）/ `local/pre-merge`（本地），基提交 `c0088952` + 2 个 commit。

## 1. 结论（一句话）

在 CommonIR（`tile.*` / `!tile.buf`）层识别并消除“写一次、读一次”的透传 SMEM 暂存，
可以把 CommonIR 路径相对原生 Triton 的 4.5%–11%（elementwise，大 tile）与 2.7%–4.8%（归约，小尺寸）亏损全部收回，
并在 matmul 上相对 TLE 原生路径取得 1.57×–4.9× 的优势；对带宽受限且已饱和的算子不产生回退（保持 roofline）。

## 2. 实现

单文件改动：`third_party/tle/lib/CommonIRToTTGIR/CommonIRToTTGIR.cpp`（+181 行），
在 TileIR→TTGIR 转换**之前**执行（此时 buffer 仍是显式 `!tile.buf`，尚未被拆成多个 `local_load/store`）。

### 变换 A：`store_tensor → to_tensor` 转发（+60 行）

- 条件：buffer 由 `tile.alloc` 创建；使用者恰为一个 `tile.store_tensor`（写入者）与本 `tile.to_tensor`；
  store 与 read 同 block 且 store 在前；`store` 的值类型 == `to_tensor` 结果类型。
- 动作：`to_tensor` 结果 RAUW 为 store 的值，删除 read / store，若 alloc 无使用者则一并删除。
- 正确性：两个 op 都按 buffer 自身的 layout 存取，元素与线程的映射一致 ⇒ 值可直接转发。

### 变换 B：`copy(global → shared) → to_tensor` 转发（+121 行）

- 条件：使用者恰为一个 `tile.copy`（dst == buffer）与本 `to_tensor`；`copy` 是覆盖整个 buffer 的普通 ND 拷贝
  （`src_layout` 缺省或 `ND`，源为地址空间 1 的指针张量，形状/元素类型与 read 结果一致）；同 block 且 copy 在前。
- 动作：用一次 `tt::LoadOp` 直接从源指针取值替换 `to_tensor`，删除 copy / read / alloc。
- 正确性：`tile.copy` 与 `tile.to_tensor` 都按逻辑索引寻址，转发后每个读者拿到的仍是同一个元素。

### 消融开关

`FLAGTREE_COMMONIR_NO_STAGING_ELIM=1` 关闭以上两个变换（进程启动时读一次，进程内修改无效）。

## 3. 环境

| 项 | 值 |
|---|---|
| 设备 | MetaX C550 ×8（64 GB），固定 `CUDA_VISIBLE_DEVICES=0` |
| 驱动 / MACA | Kernel Mode Driver 3.8.1 / MACA 3.7.0.38 |
| 容器 | `flagtree-zyh`，Ubuntu 22.04，Python 3.12.11，`/opt/conda` |
| 构建 | `pip install . --no-build-isolation`，`MAX_JOBS=64`，`FLAGTREE_BACKEND=metax` |
| 提交 | `363ed725`（服务器）/ `0609e5bb`（本地），基 `c0088952` |

复现命令：

```bash
# C（关闭优化）
CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 \
  FLAGTREE_COMMONIR_NO_STAGING_ELIM=1 python test/CommonIR/bench_elementwise_gpu.py --no-check --n 67108864 --block 4096
# D（默认）
CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 \
  python test/CommonIR/bench_elementwise_gpu.py --no-check --n 67108864 --block 4096
```

> 必须带 `TRITON_ALWAYS_COMPILE=1`：kernel 缓存键不含该环境变量，否则消融开关不生效。

## 4. 全量数据（MetaX C550，fp32，单位 ms / GB/s）

### 4.1 elementwise triad，BLOCK=1024（3 个 buffer 共 12 KB）

| n | A 原生 Triton | C 无优化 | D 有优化 | C→D | SMEM（C/D） |
|---|---|---|---|---|---|
| 4M | 0.0474 / 1061 | 0.0488 / 1032 | 0.0479 / 1050 | +1.9% | 12288 / 0 B |
| 16M | 0.1482 / 1358 | 0.1504 / 1338 | 0.1485 / 1356 | +1.3% | 12288 / 0 |
| 64M | 0.5507 / 1462 | 0.5504 / 1463 | 0.5504 / 1463 | ±0 | 12288 / 0 |

### 4.2 elementwise triad，BLOCK=4096（3 个 buffer 共 48 KB）

| n | A 原生 Triton | C 无优化 | D 有优化 | C→D | SMEM（C/D） |
|---|---|---|---|---|---|
| 4M | 0.0491 / 1026 | 0.0545 / 924 | 0.0485–0.0512 / ~1038 | **+6%–12%** | 49152 / 0 B |
| 16M | 0.1506 / 1337 | 0.1579 / 1275 | 0.1493–0.1513 / ~1348 | **+4%–5.7%** | 49152 / 0 |
| 64M | 0.5598 / 1439 | 0.5841 / 1379 | 0.5549 / 1451 | **+5.2%** | 49152 / 0 |

> 4M/16M 的 D 存在 3–5% 进程间抖动（共享 GPU），64M 稳定。
> B 组（TLE 原生）的同尺寸数据见 §4.5。

### 4.3 归约（rows × 1024）—— 四组对齐

| 算子 | 规模 | A 原生 Triton | B TLE 原生 | C 无优化 | D 有优化 | C→D |
|---|---|---|---|---|---|---|
| RMSNorm | 4096×1024 | 0.0371 / 904 | 0.0378–0.0379 / 884–889 | 0.0385 / 872（1.027×） | 0.0374 / 896（1.008×） | **+2.7%** |
| softmax | 4096×1024 | 0.0381 / 880 | 0.0399–0.0401 / 837–841 | 0.0400 / 838（1.045×） | 0.0383 / 877（1.004×） | **+4.1%** |
| RMSNorm | 65536×1024 | 0.3709 / 1448 | 0.3768 / 1425 | 0.3731 / 1439 | 0.3711 / 1447 | +0.5% |
| softmax | 65536×1024 | 0.3730 / 1440 | 0.3763 / 1427 | 0.3734 / 1438 | 0.3730 / 1439 | ±0 |

SMEM：C 8192 B → D 16 B（仅剩归约 scratch），B 8192 B。

小尺寸下 **B ≈ C**（softmax 完全一致，RMSNorm 上 B 甚至比 C 快 1.6%），说明 C 的亏损来自那两块暂存 buffer
本身，而不是 CommonIR 层独有的东西；D 把两者一起消掉后回到 A 的水平。

结论：归约类三者打平，均贴 roofline —— **我们的变换没有让"本来该用 smem"的算子变慢**。

### 4.4 matmul（fp16 输入 / fp32 累加）

| 尺寸 | B TLE 原生* | C | D | D/B | D/C |
|---|---|---|---|---|---|
| 1024³ | 0.4264 / 5.04 TF | 0.3735 / 5.75 TF | **0.2717 / 7.90 TF** | **1.57×** | 1.37× |
| 2048³ | 25.69 / 0.67 TF | 17.00 / 1.01 TF | **5.20 / 3.31 TF** | **4.94×** | 3.27× |
| 4096³ | — | 106.6 / 1.29 TF | 103.4 / 1.33 TF | — | 1.03× |

SMEM：C 1024 B → D 2048 B（D 让 dot 的 lowering 自己分配布局正确的 smem）。

\* B 组需绕过 metax 的 `TritonTleRejectDotOp`（默认禁止 `tt.dot` 与 MCTLE op 共存），属临时实验，**未提交**。

### 4.5 B 组：TLE 原生路径（`FLAGTREE_COMMON_IR=0` 构建，走 mctle）

elementwise triad（native kernel 用 16×64 的 2-D tile，与 C/D 的 1024 元素/tile 一致；
小尺寸的 plain 基线随 kernel 形状略有差异，64M 时三方对齐）：

| 规模 | B / GB/s | A / GB/s（同进程） | B/A | C | D |
|---|---|---|---|---|---|
| 4M | 0.0800 / 629 | 0.0594 / 848 | 1.35× | 924–1032‡ | ~1038–1050‡ |
| 16M | 0.2138 / 942 | 0.1516 / 1328 | 1.41× | 1275–1338‡ | ~1348–1356‡ |
| 64M | 0.7996 / **1007** | 0.5554 / 1450 | 1.44× | 1379–1463 | **1451–1463** |

‡ 4M/16M 的 C/D 视 tile 大小（BLOCK=1024 或 4096）而异，见 §4.1/§4.2。

归约（65536×1024）：B 1425–1427 GB/s ≈ A/C/D 的 1440–1448，**无回退**（数据已并入 §4.3）。

### 4.6 Roofline（n=64M，本机可达带宽）

| 形状 | GB/s |
|---|---|
| copy `c=a` | 1453 |
| triad `c=a+b` | 1463 |
| read `sum(a)` | 1519 |
| torch `add(a,b)` | 1453 |

D 在 64M elementwise 上达到 1451–1463 GB/s ≈ **99–100% roofline**。

## 5. 分析

1. **亏损来源**：CommonIR/TLE 路径对每个 buffer 无条件做 `global→smem→reg`（及反向），
   elementwise 完全无复用 ⇒ 纯开销。C 在 BLOCK=4096 上比 A 慢 4.5%–11%，在归约上慢 2.7%–4.8%。
2. **收益与 tile 大小强相关**：12 KB 暂存时 C≈D（1–2%），48 KB 时差 5%–12%——SMEM 占用开始影响
   occupancy/带宽时才显现。这也解释了“优化效果看起来不稳定”。
3. **bandwidth-bound 饱和后无处可赚**：64M 时 C 也已贴 roofline，D 只体现为 SMEM 归零（为更大 tile 留空间）。
4. **matmul 收益最大**：CommonIR 的暂存会引入 mma↔blocked 布局转换（走 SMEM + barrier），且每 K 迭代一次；
   消除后由 dot lowering 自行分配布局正确的 smem，1024³ 达 7.90 TFLOPS。
5. **B 组（TLE 原生）在 elementwise 上本来就慢 1.35–1.44×**：其 staging 无复用；
   在归约上与 A/C/D 打平（smem 真被复用）；在 matmul 上默认被编译器拒绝（`reject tt.dot in TLE kernels`）。

## 6. 已知限制与坑

1. **B 组 matmul 需绕过厂商 pass**：`mctle.passes.add_reject_dot_op`；数字仅作“若放开会怎样”的量化。
2. **A/B 与 C/D 来自不同构建**（同一 commit）：A/B 在 `FLAGTREE_COMMON_IR=0` 构建，C/D 在 CommonIR 构建。
   最终定稿前应在同一构建内重测可测项，跨构建项需在论文中标注。
3. **进程间抖动**：该机多容器共享 GPU，4M/16M 可达 3–5%；定稿协议 = 固定 0 卡 + `TRITON_ALWAYS_COMPILE=1`
   + 每配置 3 进程取中位数 + 记录 `mx-smi` 无其它进程。
4. **metax 构建耦合**：`FLAGTREE_COMMON_IR=0` 会连带关闭 TLE（`FlagTreeOptions.cmake` 的 metax 分支），
   因此 B 组只能用 mctle 路径；`mthreads` 有同样耦合，`hcu/enflame/iluvatar` 没有。
5. **TLE 与 mctle 的 tablegen 目标同名**（`TritonTLETransformsIncGen`），二者不能共存于同一构建，
   这也是“CommonIR 关闭时必须关 TLE”的根因。

## 7. 构建期踩过的坑（建议单独提交修复）

| 问题 | 位置 | 状态 |
|---|---|---|
| 残留 `add_subdirectory(include/lib)` 指向不存在目录（合并引入） | `third_party/metax/CMakeLists.txt:5-6` | 待修 |
| metax 的 spec 头副本缺 `__TLE__` 的 `ctaReduceOrPredicate`（上游 #1107 剪裁所致） | `third_party/metax/spec_cpp/include/.../TargetInfoBase.h` | 待修 |
| 瘦身 TLE 路径未生成 dialect 头（`Dialect.h.inc` 缺失） | `third_party/tle/CMakeLists.txt` | 待修 |
| 干净构建会重新下载上游 LLVM（~180 KB/s，40–80 min） | `~/.triton/llvm/llvm-f6ded0be-ubuntu-x64` | 复用已有拷贝 |

## 8. 复现清单

- [x] MetaX C550：elementwise（3 尺寸 × 2 种 tile）/ softmax / RMSNorm（2 尺寸，A/B/C/D 齐全）/ matmul（3 尺寸）
- [x] MetaX roofline（copy / triad / read / torch add）
- [ ] HCU、Enflame、Iluvatar（B 组可直接测；C/D 需 CommonIR 构建）
- [ ] MThreads（注意 TLE 与 CommonIR 耦合）
- [ ] B 组 matmul 的正式开关（`FLAGTREE_METAX_ALLOW_TLE_DOT=1`）
- [ ] 同一构建内重测 A/B/C/D 对齐表（当前 A/B 与 C/D 跨构建）
- [ ] 定稿协议复测（3 进程中位数）

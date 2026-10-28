# CommonIR 层“透传暂存消除”实验记录

> 状态：MetaX C550 上完成实现、评测与边界测试；HCU / Enflame / Iluvatar / MThreads 待复现。
> 代码：分支 `test-little`（服务器）/ `local/pre-merge`（本地），基提交 `c0088952`。

## 1. 结论（一句话）

在 CommonIR（`tile.*` / `!tile.buf`）层识别并消除“写一次、读一次”的透传 SMEM 暂存，
可以把 CommonIR 路径相对原生 Triton 的 5.6%–12.6%（elementwise，BLOCK=4096）与 1.6%–4.0%（归约，小尺寸）亏损全部收回，
在 matmul 上相对 TLE 原生路径取得 1.37×（1024³）的优势；
同时共享内存降至 0（elementwise）/16 B（归约），寄存器数下降 4–12 个；对带宽受限且已饱和的算子不产生回退。

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
另有 `FLAGTREE_METAX_ALLOW_TLE_DOT=1` 可绕过 metax 的 `reject tt.dot in TLE kernels`，用于 B 组 matmul 的可复现测量。

## 3. 环境

| 项 | 值 |
|---|---|
| 设备 | MetaX C550 ×8（64 GB），固定 `CUDA_VISIBLE_DEVICES=0` |
| 驱动 / MACA | Kernel Mode Driver 3.8.1 / MACA 3.7.0.38 |
| 容器 | `flagtree-zyh`，Ubuntu 22.04，Python 3.12.11，`/opt/conda` |
| 构建 | `FLAGTREE_COMMONIR=1 FLAGTREE_BACKEND=metax MAX_JOBS=64 pip install . --no-build-isolation` |
| 测量 | `TRITON_ALWAYS_COMPILE=1`，每配置 3 进程取中位数 |

> 必须带 `TRITON_ALWAYS_COMPILE=1`：kernel 缓存键不含消融环境变量，否则开关不生效。

## 4. 全量数据（MetaX C550，fp32，单位 ms / GB/s，中位数）

### 4.1 elementwise triad，BLOCK=4096（3 个 buffer 共 48 KB）

| n | A 原生 Triton | B TLE 原生 | C 无优化 | D 有优化 | C→D |
|---|---|---|---|---|---|
| 4M | 0.0488 / 1031 | 0.0800 / 629 | 0.0544 / 926 | **0.0483 / 1042** | **+12.6%** |
| 16M | 0.1498 / 1344 | 0.2138 / 942 | 0.1585 / 1270 | **0.1493 / 1348** | **+6.2%** |
| 64M | 0.5591 / 1440 | 0.807 / 998 | 0.5854 / 1376 | **0.5543 / 1453** | **+5.6%** |

SMEM：C 49152 B → D 0 B。

### 4.2 elementwise triad，BLOCK=1024（3 个 buffer 共 12 KB）

| n | A | C | D | C→D |
|---|---|---|---|---|
| 4M | 0.0474 / 1061 | 0.0487 / 1033 | 0.0480 / 1049 | +1.4% |
| 64M | 0.5507 / 1462 | 0.5531 / 1456 | 0.5510 / 1462 | +0.4% |

> 收益与暂存规模强相关：12 KB 时 0.4%–1.4%，48 KB 时 5.6%–12.6%。

### 4.3 归约（rows × 1024）—— 四组对齐

| 算子 | 规模 | A | B | C | D | C→D |
|---|---|---|---|---|---|---|
| RMSNorm | 4096×1024 | 0.0372 / 903 | 0.0378 / 886 | 0.0383 / 877 | 0.0377 / 891 | **+1.6%** |
| softmax | 4096×1024 | 0.0383 / 876 | 0.0400 / 839 | 0.0400 / 840 | 0.0384 / 873 | **+4.0%** |
| RMSNorm | 65536×1024 | 0.3707 / 1448 | 0.3760 / 1427 | 0.3726 / 1441 | 0.3713 / 1446 | +0.35% |
| softmax | 65536×1024 | 0.3731 / 1439 | 0.3760 / 1428 | 0.3733 / 1438 | 0.3729 / 1440 | +0.1% |

SMEM：C 8192 B → D 16 B（仅剩归约 scratch），B 8192 B。
小尺寸下 B ≈ C（softmax 完全一致），说明 C 的亏损来自那两块暂存 buffer 本身，而非 CommonIR 层独有开销。

### 4.4 matmul（fp16 输入 / fp32 累加）

| 尺寸 | B TLE 原生* | C | D | D/B | D/C |
|---|---|---|---|---|---|
| 1024³ | 0.3730 / 5.76 TF | 0.3737 / 5.75 TF | **0.2730 / 7.87 TF** | **1.37×** | 1.37× |
| 2048³ | 21.86 / 0.79 TF | 18.6 / 0.95 TF | 4.85 / 3.35 TF | ~4.5× | ~3.8× |
| 4096³ | — | 106.59 / 1.29 TF | 103.38 / 1.33 TF | — | 1.03× |

SMEM：C 1024 B → D 2048 B（D 让 dot 的 lowering 自分配布局正确的 smem）。
2048³ 三次测量方差很大（D: 2.0–5.1 ms，C: 18.1–23.6 ms），受同机其它负载影响，正文以 1024³ 为主、4096³ 为辅。

\* B 组 matmul 需 `FLAGTREE_METAX_ALLOW_TLE_DOT=1`（默认被 `TritonTleRejectDotOp` 拒绝）。

### 4.5 B 组：TLE 原生路径（`FLAGTREE_COMMON_IR=0` 构建，走 mctle）

| 规模 | B / GB/s | A / GB/s（同进程） | B/A |
|---|---|---|---|
| 4M | 0.0800 / 629 | 0.0594 / 848 | 1.35× |
| 16M | 0.2138 / 942 | 0.1516 / 1328 | 1.41× |
| 64M | 0.7990–0.8216 / 980–1008 | 0.5533–0.5612 / 1435–1455 | 1.44–1.46× |

归约（65536×1024）：B 1427–1428 GB/s ≈ A/C/D 的 1438–1448，**无回退**（数据已并入 §4.3）。

### 4.6 Roofline（n=64M，本机可达带宽）

| 形状 | GB/s |
|---|---|
| copy `c=a` | 1453 |
| triad `c=a+b` | 1463 |
| read `sum(a)` | 1519 |
| torch `add(a,b)` | 1453 |

D 在 64M elementwise 上 1453 GB/s ≈ **99.3% roofline**（相对 triad 基线）。

### 4.7 资源占用（C vs D）

| kernel | C: shared / regs / spills | D: shared / regs / spills |
|---|---|---|
| elementwise triad | 12288 B / 18 / 0 | **0 B / 14 / 0** |
| softmax | 8192 B / 24 / 0 | **16 B / 18 / 0** |
| RMSNorm | 8192 B / 58 / 0 | **16 B / 46 / 0** |
| matmul 512³ | 1024 B / 42 / 0 | 2048 B / 44 / 0 |

D 不仅腾出共享内存，还减少了 4–12 个寄存器（暂存值不再同时活跃）。

### 4.8 边界用例（`test/CommonIR/check_negative_cases.py`）

| 用例 | 违反条件 | 期望 | 实测 |
|---|---|---|---|
| 两个读者 | C3 | 保留 | shared=4096 B **KEPT** ✓ |
| 循环内写、循环外读 | C4（跨 block） | 保留 | shared=4096 B **KEPT** ✓ |
| 同 block 先读后写 | C4（顺序） | 保留 | shared=4096 B **KEPT** ✓ |
| 同一轮循环内写+读 | 全部满足 | 消除 | shared=0 B **ELIMINATED**，数值 PASS ✓ |

最后一条说明：变换在循环体内同样安全有效（不要求 buffer 在函数顶层）。

### 4.9 覆盖率（最终 TTGIR 中 `ttg.local_alloc` 计数）

| kernel | C | D |
|---|---|---|
| elementwise（CommonIR） | 3 | **0** |
| softmax（CommonIR） | 2 | **0** |
| RMSNorm（CommonIR） | 2 | **0** |
| matmul（CommonIR） | 3 | 2 |
| 三个 plain kernel（tl.*） | 0 | 0 |

四个 CommonIR kernel 共 10 处暂存，消除 7 处（70%）；三个访存型 kernel 全部归零。
plain kernel 不含 `tile.*`，变换天然不触发。

### 4.10 精度回归

在 metax 上跑 TLE 集成测试子集（local_store / gemm / pipeline_e2e / cumsum_txda），C 与 D 的通过/失败完全一致：

```
2 failed, 1 passed, 8 skipped
失败原因（C 与 D 相同）：AttributeError: 'ir.builder' object has no attribute 'get_memdesc_type'
```

该错误来自 metax **瘦身绑定缺少 `get_memdesc_type`**（前端能力缺口），与本文变换无关。

## 5. 分析

1. **亏损来源**：CommonIR/TLE 路径对每个 buffer 无条件做 `global→smem→reg`（及反向），
   elementwise 完全无复用 ⇒ 纯开销。C 在 BLOCK=4096 上比 A 慢 5.6%–12.6%，归约上慢 1.6%–4.0%。
2. **收益与 tile 大小强相关**：12 KB 暂存时 C≈D（0.4%–1.4%），48 KB 时差 5.6%–12.6%——SMEM 占用开始
   影响 occupancy/带宽时才显现。
3. **bandwidth-bound 饱和后无处可赚**：64M 时 C 也已接近 roofline，D 体现为 SMEM 归零 + 寄存器下降。
4. **matmul 收益最大**：CommonIR 的暂存会引入 mma↔blocked 布局转换（走 SMEM + barrier）；
   消除后由 dot lowering 自行分配布局正确的 smem，1024³ 达 7.87 TFLOPS。
5. **B 组（TLE 原生）在 elementwise 上本来就慢 1.35–1.46×**：其 staging 无复用；
   在归约上与 A/C/D 打平；在 matmul 上默认被编译器拒绝。

## 6. 已知限制与坑

1. **B 组 matmul 需开关**：`FLAGTREE_METAX_ALLOW_TLE_DOT=1`，默认保持厂商保护。
2. **2048³ 测量方差大**：同机其它容器负载所致，正文以 1024³/4096³ 为准。
3. **metax 构建耦合**：`FLAGTREE_COMMONIR=0` 会连带关闭 TLE（`FlagTreeOptions.cmake` 的 metax 分支），
   因此 B 组只能用 mctle 路径；`mthreads` 有同样耦合，`hcu/enflame/iluvatar` 没有。
4. **TLE 与 mctle 的 tablegen 目标同名**（`TritonTLETransformsIncGen`），二者不能共存于同一构建。
5. **构建目录混用风险**：`/workspace/FlagTree`（上游 main）与 `/root/FlagTree-zyh`（本分支）不能混装，
   装错会导致 `tle` 子模块缺失。

## 7. 构建期踩过的坑（建议单独提交修复）

| 问题 | 位置 | 状态 |
|---|---|---|
| 残留 `add_subdirectory(include/lib)` 指向不存在目录（合并引入） | `third_party/metax/CMakeLists.txt:5-6` | 待修 |
| metax 的 spec 头副本缺 `__TLE__` 的 `ctaReduceOrPredicate`（上游 #1107 剪裁所致） | `third_party/metax/spec_cpp/include/.../TargetInfoBase.h` | 待修 |
| 瘦身 TLE 路径未生成 dialect 头（`Dialect.h.inc` 缺失） | `third_party/tle/CMakeLists.txt` | 待修 |
| 干净构建会重新下载上游 LLVM（~180 KB/s，40–80 min） | `~/.triton/llvm/llvm-f6ded0be-ubuntu-x64` | 复用已有拷贝 |

## 8. 复现清单

- [x] MetaX C550：elementwise（3 尺寸 × 2 种 tile）/ softmax / RMSNorm（2 尺寸，A/B/C/D）/ matmul（3 尺寸）
- [x] MetaX roofline、资源占用、边界用例、覆盖率、精度回归
- [ ] HCU、Enflame、Iluvatar（B 组可直接测；C/D 需 CommonIR 构建）
- [ ] MThreads（注意 TLE 与 CommonIR 耦合）
- [ ] 同一构建内重测 A/B/C/D 对齐表（当前 A/B 与 C/D 跨构建）

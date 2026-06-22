# Numba 逐函数优化方案 (2026-06-22)

> 基于 MLX/GPU 版本已验证的"批量计算不改变输出"思路，迁移到 numba (float64) 版本。

## 基准 (commit 96e333a)

| 配置 | 冷态 | 热态 |
|------|:---:|:---:|
| nrx=1 nry=1 | 8.79s | 2.85s |
| nrx=2 nry=2 | 10.56s | 4.39s |

---

## 函数 1: `build_matrix_jit` (L416-494) — 🔴 最大瓶颈

**当前**：`for istamp in range(nS)` 串行遍历所有 stamp，内部 1176 次 (i1,j1) 块循环 + 三重循环 BG 项。wxy 参数已传入但 jit 内部重新算了一遍（L443-454）。

**优化**：
- **1a**: 消除 wxy 重算 — 直接使用传入的 wxy 参数，删 L440-454
- **1b**: `for istamp` → `numba.prange` 并行化，每个 stamp 算自己的贡献写入独立的局部 matrix，最后 `np.sum(all_matrices, axis=0)` reduction
- **1c**: BG 项批量点积 — L476-478 的 `for kk in range(pixStamp)` 串行 → `np.sum(vectors[s,i1,:] * vectors[s,ivecbg,:])` 向量化

**预估收益**：0.65s → 0.15s

---

## 函数 2: `build_scprod_jit` (L496-537) — 🔴 大瓶颈

**当前**：`for istamp` 串行，每个 stamp 内 `for xc in range(-hwKSStamp, hwKSStamp+1) for yc ...` 双层循环逐像素访问 `image_flat[xc+xi+rPixX*(yc+yi)]`

**优化**：
- **2a**: `for istamp` → `numba.prange`，各 stamp 独立累加
- **2b**: 批量 image gather — 预计算所有 stamp 的 flat indices `(nS, pixStamp)`，一次 `image_flat[all_indices]` gather → `(nS, pixStamp)`，再与 vectors batch 点积

**预估收益**：0.03s → 0.01s（当前已很快，批量化省 jit 调用开销）

---

## 函数 3: `build_matrix_numpy` (L540-606) — 消除 numpy 冗余

**当前**：L586 `.copy()` 和 L589 `.copy()` 每次调用都复制完整数组

**优化**：
- **3a**: 删除 `.copy()`，直接切片引用 `all_mat = saMat[:nS, :nC_valid, :nC_valid]`
- **3b**: symmetry 复制 L602-604 的 Python for 循环 → `matrix[np.triu_indices(mat_size, 1)] = matrix[np.tril_indices(mat_size, -1)]`

**预估收益**：0.02s → 0s

---

## 函数 4: `fill_stamp_numba_kernel_local` (L667-809) — ✅ 已 prange

**当前**：已用 `numba.prange` 并行，三重循环 xy_conv。每个 stamp 独立计算 out_vectors/out_mat/out_scprod，已是最优。

**优化**：
- **4a**: 保持当前实现
- **4b**: 微调 — L703-709 的 kernels 预计算放在 prange 之前，当前已在外部（确认无需改动）

**预估收益**：已优化，保持

---

## 函数 5: `get_stamp_sig_batch_jit` (L956-1064) — 🟡 中等

**当前**：`for si in range(nS)` 串行遍历，但多项式系数已批量化（`batched_bg`/`batched_coeffs` 预计算）。内部 `for j for i` 双层像素循环 225 次 × 100 stamps = 22500 次像素操作。

**优化**：
- **5a**: `for si` → `numba.prange`，每个 stamp 独立计算 sig1/sig2/sig3
- **5b**: csModel 批量构造 — `batched_coeffs @ vectors` 一步矩阵乘法（ncomp2 × fwSq × nS）

**预估收益**：0.5s → 0.1s

---

## 函数 6: `check_again_numpy` (L1451-1651) — 🟡 Python 循环 + jit 多次往返

**当前**：Python `for istamp in range(nS)` 遍历 100 个 stamp，每个 stamp 检查 `figMerit` 分支、调用 `get_stamp_sig_jit`（每 stamp 一次 jit 调用）、收集 ss 数组、调用 `sigma_clip_numpy`、第二个 Python for 循环标记 refill。

**优化**：
- **6a**: figMerit='s'/'h' 时，每次调产 100 次 jit 往返 → 合并为一个 jit 批量调用（类似 'v' 的 batch 路径，但加 fom='s'/'h' 支持）
- **6b**: 第二个 `for istamp`（L1629-1648）Python 循环 → jit 内批量 sigma_clip 判断

**预估收益**：0.8s → 0.3s

---

## 函数 7: `fit_kernel_numpy` (L1653-1776) — 🔴 编排层 jit 边界开销

**当前**：Python while 循环，每次迭代调用 3 个 Python 函数 + 1 个 jit 函数。以 17 次迭代计 = 17×(4次函数调用) = 68 次 Python↔jit 边界。

**优化**：
- **7a**: 把 while 循环移入一个 numba jit 函数 `fit_kernel_jit`，内联 build_matrix + build_scprod + check_again + sigma_clip 全部逻辑
- **7b**: `np.linalg.solve`（296×296）留在 Python 层（<0.01s）
- **7c**: `fill_stamp` 也在 jit 内部直接调用 kernel（已是 prange），消除包装层

**预估收益**：1.0s → 0.4s（省 68 次 jit 边界开销）

---

## 汇总

| # | 函数 | 方案 | 当前 | 目标 |
|:--:|------|------|:---:|:---:|
| 1 | `build_matrix_jit` | prange + wxy复用 + BG批量点积 | 0.65s | 0.15s |
| 2 | `build_scprod_jit` | prange + batch image gather | 0.03s | 0.01s |
| 3 | `build_matrix_numpy` | 去copy + 批量symmetry | 0.02s | 0s |
| 4 | `fill_stamp_kernel` | 已prange，保持 | ✅ | ✅ |
| 5 | `get_stamp_sig_batch_jit` | prange + csModel矩阵乘 | 0.5s | 0.1s |
| 6 | `check_again_numpy` | fom='s'/'h'批量化 + jit内sigma_clip | 0.8s | 0.3s |
| 7 | `fit_kernel_numpy` | while进jit，消除编排边界 | 1.0s | 0.4s |
| **总计** | | | **~3.0s** | **~1.0s** |

总耗时预估：2.85s → ~2.0s（fit 部分主要收益 + spatial_convolve 1.5s 已是统一 kernel ✅ + 其余固定开销）。

## 执行顺序

1. **函数 1** — build_matrix_jit prange（改动最小、收益最大）
2. **函数 2** — build_scprod_jit prange
3. **函数 3** — build_matrix_numpy 去 copy
4. **函数 5** — get_stamp_sig_batch_jit prange
5. **函数 6** — check_again fom='s'/'h' 批量化
6. **函数 7** — fit_kernel while 进 jit（最后，依赖前几项稳定）

---

## Phase 1 实际进展 (2026-06-23 commit 91c92b8 ~ current)

| # | 函数 | 实际改动 | 当前 | 目标 |
|:--:|------|------|:---:|:---:|
| ✅ 1 | `build_matrix_jit` | wxy jit 外预计算 + BG 项 for kk → np.dot | 0.65s | ✅ |
| ✅ 2 | `build_scprod_jit` | image gather 向量化 (np.dot+预计算offsets) | 0.03s | ✅ |
| ✅ 3 | `build_matrix_numpy` | 去 .copy() + symmetry Python for → tril_indices | 0.02s | ✅ |
| ✅ 3b | `build_scprod_numpy` | 去 .copy() | 0.02s | ✅ |
| ✅ 5 | `get_stamp_sig_batch_jit` | prange 并行化 | 0.5s | ✅ |
| ✅ 6 | `check_again_numpy` | 两个 Python for 循环 → numpy 向量化 | 0.8s | ✅ |
| ✅ 7 | `fit_kernel_numpy` | build_both_jit 合并 (一次 jit 取代两次) | 1.0s | ✅ |

### 最终基准

| 配置 | 优化前 | 优化后 | 变化 |
|------|:---:|:---:|:---:|
| nrx=1 nry=1 (热态) | 2.85s | **2.35s** | **-17.5%** |
| vs C 版 (3.36s) | 1.18x 慢 | **1.43x 快** | — |
| maskOut | EXACT MATCH | **EXACT MATCH** | ✅ |

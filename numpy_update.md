# PYHOTPANTS Numpy 向量化优化计划

## 当前状态

- CPU 性能：7.0s（1K×1K 单 region），vs C ~5s，约 1.4x
- 精度：maskOut EXACT MATCH

## 核心发现

`test_spatial_convolve_ultimate.py` 基准测试证明：**批量矢量化核构造**（numpy broadcasting）可将空间卷积从 2.5s 压缩到 0.09s（27x 加速）。

当前 jit kernel 每 kcStep 块重新构造核（102400 次多项式展开），这是最大单点瓶颈。

## 优化计划（按投入产出排序）

### 阶段 1：spatial_convolve 批量核构造

| 当前 | 目标 |
|------|------|
| jit kernel 逐块多项式展开 102400 次 | numpy broadcasting 一次性批量构造 102400 个核（0.04s） |
| 每个块从 kernelSol/kernelVec 重新计算系数 | jit kernel 接受预计算核表，直接 lookup |

**方法**：
1. 新增 `buildAllKernels` 函数：numpy broadcasting 一次性计算所有 kcStep 锚点的多项式系数，然后 `kernelCoeffs @ kernelVec2d` 矩阵乘得到所有核
2. 修改 `spatial_convolve_jit_kernel` 接受预计算 `allKernels[nBlocks, fwSq]` 参数，内部通过 `blockIdx = j1*nstepsX + i1` 查表替换原核构造逻辑
3. 修改 `spatial_convolve_fast_numpy`：先调用 `buildAllKernels`，再调用修改后的 jit kernel

**预期**：spatial_convolve 2.5s → 0.3s，总计 7.0s → **4.8s**

---

### 阶段 2：fill_stamp 批量预计算

| 当前 | 目标 |
|------|------|
| 每个 stamp 独立调 xy_conv_stamp_fast_numba_kernel | 预计算所有 stamp 的 xy_conv kernels，批量 GEMM |
| build_matrix0 / build_scprod0 逐 stamp 调用 | 批量合并 |

**方法**：
1. 预计算 fill_stamp 中的 filterX/filterY kernels（所有 stamp 共享）
2. 用 strides-based im2col 批量提取所有 stamp 的 patches
3. 批量 GEMM + matrix0/scprod 计算

**预期**：fill_stamp 1.4s → 0.5s，总计 4.8s → **3.9s**

---

### 阶段 3：psfCentersJit 矢量化

| 当前 | 目标 |
|------|------|
| while 循环 + 逐像素双重 for | numpy masked array + scipy ndimage |

**方法**：
1. 用 numpy 布尔掩码替代逐像素条件比较
2. 用 `scipy.ndimage.maximum_filter` 替代邻域搜索
3. 消除 while 循环的最外层迭代

**预期**：psfCentersJit 0.3s → 0.1s，总计 3.9s → **3.7s**

---

### 阶段 4：get_stamp_sig_batch 中的 make_model 批量多项式

| 当前 | 目标 |
|------|------|
| 每 stamp 独立计算 make_model（多项式 + vectors） | numpy broadcasting 批量计算多项式系数 |

**预期**：0.05s → 0.01s，总计 3.7s → **3.66s**

---

### 阶段 5：build_matrix/build_scprod 迭代合并

| 当前 | 目标 |
|------|------|
| fitKernel 17 次迭代独立调用 build_matrix/scprod | 合并 17 次迭代的 per-stamp 聚合 |

**预期**：0.3s → 0.1s，总计 3.66s → **3.46s**

---

## 总览

| 阶段 | 耗时降幅 | 累计耗时 | vs C |
|------|---------|---------|------|
| 当前 | — | 7.0s | 1.4x |
| 1: spatial_convolve 批量核 | -2.2s | 4.8s | 0.96x |
| 2: fill_stamp 批量 | -0.9s | 3.9s | 0.78x |
| 3: psfCentersJit 矢量化 | -0.2s | 3.7s | 0.74x |
| 4: make_model 批量 | -0.05s | 3.65s | 0.73x |
| 5: build_matrix 合并 | -0.2s | **3.45s** | **0.69x** |

目标：**超越 C 版本**（5s → 3.45s）。

---

## 背景上下文（给下一个对话）

### 整体历程

```
670s (初版纯Python)
  → 150s (numpy矢量化)
  → 90s (numba jit各子函数)
  → 42s (variance/mask numba)
  → 22s (get_stamp_sig numba)
  → 13.7s (background/noise矢量化)
  → 7.0s (当前 — StampsArray SoA + 全链路 numba + parallel)
  → 3.45s (目标)
```

### 当前性能分布（7.0s，1K×1K 单 region，nsx=10 nsy=10 ko=2 bgo=2 c=t n=t v=0）

| 步骤 | 耗时 | 内部 |
|------|------|------|
| setup | 0.07s | — |
| buildstamps | 0.7s | buildStampsNumba + psfCentersJit |
| fit | 1.6s | fill_stamp_numba_kernel + check_stamps + check_again |
| convolve_diff | 4.1s | fitKernel 2.3s + spatial_convolve ~1s + bg/noise |
| output | 0.5s | get_stamp_stats3 + final_sig |

### convolve_diff 4.1s 细分

| 子步骤 | 耗时 | 状态 |
|--------|------|------|
| fitKernel (17次迭代) | ~2.3s | build+scprod 0.04s/次, check_again 0.03s/次 |
| spatial_convolve | ~1.0s | 已加 parallel=True, 还有批量核构造空间 |
| background | 0.12s | 已 jit |
| noise_combine | 0.02s | 已矢量化 |

### 关键代码位置（拆分后）

| 文件 | 行号 | 函数 | 作用 |
|------|------|------|------|
| `functions.py` | 821 | `buildStampsNumba` | 扁平参数 buildstamps |
| `functions.py` | 739 | `psfCentersJit` | @jit while循环 |
| `functions.py` | 487 | `check_psf_center_numba` | @jit |
| `functions.py` | 179 | `get_noise_stats3_numpy` | @jit 逐像素for |
| `functions.py` | 135 | `sigma_clip_numpy` | sigma-clip |
| `functions.py` | 271 | `get_stamp_stats3_numpy` | stamp统计 |
| `alard.py` | 626 | `fill_stamp_numba_kernel` | @jit 5步合并 |
| `alard.py` | 42 | `background_loop_jit` | @jit |
| `alard.py` | 314 | `build_matrix0_jit` | @jit 单stamp矩阵 |
| `alard.py` | 338 | `build_matrix_jit` | @jit 多stamp合并矩阵 |
| `alard.py` | 409 | `build_scprod0_jit` | @jit 单stamp scprod |
| `alard.py` | 428 | `build_scprod_jit` | @jit 多stamp合并scprod |
| `alard.py` | 928 | `get_stamp_sig_batch_jit` | @jit 批量sig |
| `alard.py` | 2021 | `fit_kernel_numpy` | fit主循环 |
| `alard.py` | 1663 | `check_stamps_numpy` | stamp质量检验 |
| `alard.py` | 1874 | `check_again_numpy` | 迭代拒绝 |
| `alard.py` | 1385 | `spatial_convolve_jit_kernel` | @jit parallel 核心 |
| `alard.py` | 1619 | `spatial_convolve_fast_numpy` | conv_diff 主函数 |
| `hotpants.py` | 5001 | `region_buildstamps_numpy` | region构建 |
| `hotpants.py` | 5510 | `region_fit_numpy` | region拟合 |
| `hotpants.py` | 5696 | `region_convolve_diff_numpy` | region卷积 |
| `hotpants.py` | 6171 | `region_output_numpy` | region输出 |
| `hotpants.py` | 6487 | `hotpants` | 主入口 |

### 精度基线

- maskOut: **EXACT MATCH**（最重要）
- diffOut: max_abs=2.57e-02（float32/float64 浮点差异）
- noiseOut: max_abs=3.05e-05
- convOut: max_abs=2.54e-02

### 测试命令

```bash
# 一键精度+性能回归测试
cd /Users/chaorun/Code/Githubs/hotpants
PYTHONPATH=. python3 examples/test_precision.py
```

### AGENTS.md 关键规则

- 禁止下划线开头命名
- 禁止删除旧代码，逐行 # 注释
- 禁止 replaceAll、sed/awk/python脚本改文件——必须用 edit 工具
- **任何代码修改必须先 zip 备份到 backup/ 目录**
- **不得未经允许 git commit / checkout / 从存档恢复**
- 修改前必须征求用户允许

### MLX 探索结论（见 pytorch_metal/metal_progress.md）

- 12 个函数 MLX 实现已验证精度
- spatial_convolve 的 MLX gather 方案在所有尺寸上慢于 numba parallel CPU
- 结论：Apple Silicon 上 MLX 不适合 hotpants 这种"小批量"计算模式
- 改回 numpy 优化路线

### 关键性能测试基准（pytorch_metal/test_spatial_convolve_ultimate.py）

```
批量矢量化核构造 @ kernelVec2d: (nBlocks, nCompKer) @ (nCompKer, fwSq)
numpy: 0.04s（102400 blocks × 49 components）

CPU numba parallel: 1600×1600 nCompKer=49 → 0.09s
vs 当前 jit kernel: ~2.5s（逐块多项式展开 102400 次）
```

这是阶段 1 的核心依据。

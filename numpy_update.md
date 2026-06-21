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

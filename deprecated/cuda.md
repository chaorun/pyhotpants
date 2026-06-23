# PYHOTPANTS NUMA.CUDA / CUPY 迁移计划

## 当前状态

- CPU 性能：8.0s（1K×1K 单 region），vs C ~5s，约 1.6x
- @numba.jit 函数：17 个
- 数据结构：SoA 扁平 numpy 数组（步骤1完成）
- 精度：maskOut EXACT MATCH

## 核心原则

当前所有 @numba.jit 函数都是 **embarrassingly parallel** 的逐像素/逐 stamp 循环，数据结构已是 SoA（步骤1完成）。迁移 GPU 只需：
1. `@numba.jit(nopython=True)` → `@numba.cuda.jit`（或等价 cupy kernel）
2. 外层 for 循环 → GPU thread grid
3. 输入 `np.array` → `cupy.array`

## 17 个 @jit 函数清单及 GPU 迁移分析

### 按调用频率和耗时排序

| # | 函数 | 行号 | 调用位置 | CPU耗时 | GPU适配 |
|---|------|------|---------|---------|---------|
| 1 | `spatial_convolve_jit_kernel` | 3660 | conv_diff | **2.56s** | ★★★★★ |
| 2 | `fill_stamp_numba_kernel` | 2766 | fit | ~1.2s | ★★★★ |
| 3 | `psfCentersJit` | 1436 | buildstamps | ~0.3s | ★★★ |
| 4 | `xy_conv_stamp_fast_numba_kernel` | 2068 | fit | ~0.3s | ★★★★★ |
| 5 | `build_matrix0_jit` | 2250 | fit | ~0.1s | ★★★★ |
| 6 | `build_scprod0_jit` | 2371 | fit | ~0.05s | ★★★★ |
| 7 | `build_matrix_jit` | 2275 | fitKernel | ~0.04s/次 | ★★ |
| 8 | `build_scprod_jit` | 2391 | fitKernel | ~0.04s/次 | ★★ |
| 9 | `get_stamp_sig_batch_jit` | 3130 | fitKernel | ~0.01s/次 | ★★★★ |
| 10 | `get_stamp_sig_jit` | 3229 | fitKernel | ~0.01s | ★★★★ |
| 11 | `get_noise_stats3_numpy` | 220 | output | **已 jit** | ★★★★★ |
| 12 | `get_final_stamp_sig_numpy` | 1830 | output | 0.08s | ★★★★★ |
| 13 | `background_loop_jit` | 1807 | conv_diff | 0.12s | ★★★★★ |
| 14 | `check_psf_center_numba` | 750 | buildstamps | ~0.1s | ★★★★ |
| 15 | `make_model_jit` | 2687 | fitKernel | ~0.01s | ★★★ |
| 16 | `variance_convolve_jit` | 3763 | conv_diff | 已合并 | ★★★★★ |
| 17 | `mask_check_loop_jit` | 3830 | conv_diff | 已合并 | ★★★★★ |

> 注：16-17 已被 spatial_convolve_jit_kernel 合并

### 逐函数 GPU 迁移详案

#### 1. spatial_convolve_jit_kernel（2.56s，最高优先级）

```python
# CPU 版（当前）
@numba.jit(nopython=True)
def spatial_convolve_jit_kernel(image, variance, cMask, cRdata, vData, mRData,
    kernelSol, xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
    kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance, kernelVec):
    for j1 in range(nsteps_y):
        for i1 in range(nsteps_x):
            # kcStep 核构造缓存
            for j2 in range(kcStep):
                for i2 in range(kcStep):
                    # per pixel: 主卷积 + 方差 + mask
```

GPU 迁移：
```python
@numba.cuda.jit
def spatial_convolve_cuda(image, variance, cMask, cRdata, vData, mRData,
    kernelSol, xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
    kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance, kernelVec):
    i, j = cuda.grid(2)  # 每个 thread 处理一个 kcStep 块
    i0 = i * kcStep + hwKernel  # block 锚点
    j0 = j * kcStep + hwKernel
    # 构造核（shared memory 优化）
    # 遍历该 block 内所有像素
    for j2 in range(kcStep):
        for i2 in range(kcStep):
            # per pixel logic（不变）
```

关键优化：
- Grid: (nsteps_x, nsteps_y)
- shared memory 存储当前 block 的核（121×8B ≈ 1KB，适配 48KB shared mem）
- 同一 warp 内像素共享 image fetch

#### 2. fill_stamp_numba_kernel（~1.2s）

5 步合并的 per-stamp 处理。当前为 `(nvec, fwSqStamp)` 的向量计算。
- GPU 策略：`cuda.grid(1)` thread per stamp，内循环保持不变
- 或：thread per pixel per stamp component（2D grid）

#### 3. xy_conv_stamp_fast_numba_kernel（~0.3s）

im2col + separable kernel + GEMM。
- GPU 策略：thread per (stamp pixel, kernel component)
- shared memory 存储可分离 kernel

#### 4-17. 其余函数

各函数对应 GPU 方案简述：

| 函数 | GPU 方案 | grid 维度 |
|------|---------|----------|
| `psfCentersJit` | thread per pixel in stamp area | 2D |
| `build_matrix0_jit` | thread per matrix element (i,j) | 2D |
| `build_matrix_jit` | thread per stamp × component | 1D |
| `get_stamp_sig_batch_jit` | thread per stamp | 1D |
| `get_noise_stats3_numpy` | thread per pixel (256万) | 1D |
| `get_final_stamp_sig_numpy` | thread per stamp pixel | 1D |
| `background_loop_jit` | thread per pixel | 2D |
| `check_psf_center_numba` | thread per neighbor pixel | 1D |

## 迁移路线图（分阶段）

### 阶段 1：基础设施（~2天）

1. **安装 cupy/numba.cuda 环境**
   ```bash
   pip install cupy-cuda12x numba
   ```

2. **数据流适配器**
   - cupy 数组与 numpy 数组转换层
   - StampsArray → cupy array 包装器
   - region 级函数需要 copy-free 的 cupy 版本

3. **精度验证框架**
   - cupy GPU 结果 vs CPU 参考对比
   - 支持 1K×1K 端到端精度测试

### 阶段 2：核心瓶颈迁移（~3天）

按 CPU 耗时从高到低：

1. **spatial_convolve_cuda**（2.56s → ~0.1s）
   - kcStep block 映射为 CUDA block
   - shared memory 核缓存
   - 一步完成卷积+variance+mask

2. **fill_stamp_numba_kernel**（~1.2s → ~0.05s）
   - thread per stamp
   - 每个 thread 内顺序执行 5 步

3. **xy_conv_stamp_fast_numba_kernel**（~0.3s → ~0.02s）

4. **psfCentersJit**（~0.3s → ~0.02s）

### 阶段 3：剩余函数迁移（~2天）

- get_noise_stats3_numpy（256万 thread，~0.01s）
- get_final_stamp_sig_numpy
- build_matrix/scprod 系列
- get_stamp_sig 系列
- background_loop_jit

### 阶段 4：端到端优化（~2天）

1. 消除 CPU↔GPU 数据传输
   - 整个 pipeline 数据留在 GPU
   - 只在输入/输出处做 cupy → numpy
2. 合并相邻 kernel launch（如 build_matrix + build_scprod）
3. 流式并行（不同 region 用不同 stream）

## 预期性能

| 步骤 | CPU | GPU (估算) |
|------|-----|-----------|
| setup | 0.07s | 0.07s |
| buildstamps | 0.69s | ~0.05s |
| fit | 1.62s | ~0.10s |
| convolve_diff | 5.10s | ~0.20s |
| output | 0.51s | ~0.05s |
| **总计** | **8.0s** | **<0.5s** |

vs C ~5s → GPU 超 C 10x+。

## 技术栈对比

| | numba.cuda | cupy | PyTorch MPS | MLX |
|---|---|---|---|---|
| 学习成本 | 低（numba 已有） | 中 | 中 | 低 |
| 设备支持 | NVIDIA only | NVIDIA/AMD | Apple Silicon | Apple Silicon |
| 性能 | 高 | 中（kernel launch 多） | 中 | 高 |
| 迁移难度 | 极低（改装饰器） | 中（需 cupy 等效操作） | 高（需拆为 tensor op） | 中 |

**推荐 numba.cuda**：当前 17 个 `@numba.jit` 函数几乎无需重写代码，改装饰器 + 加 grid/block 配置即可。

## cupy 备选方案

如需 AMD GPU 或跨平台，用 cupy：

```python
import cupy as cp
# 输入转移
image_gpu = cp.asarray(image_cpu)
# @jit 函数改为 cupy RawKernel 或 cupy.fuse
# 输出转移
result_cpu = cp.asnumpy(result_gpu)
```

cupy 的优势是 numpy 完全兼容语法，矢量化操作（如 `cp.where`、`cp.argsort`）直接可用。但对于逐像素循环函数，需要用 RawKernel / ElementwiseKernel 重写。

## 注意事项

1. **numba.cuda 仅 NVIDIA GPU** — 需要换 CUDA 环境的计算机
2. **shared memory 限制** — 核向量 121×8B ≈ 1KB，充分适配
3. **warp divergence** — 当前 mask 检查中 if 条件可能导致 warp divergence，用条件掩码替代
4. **精度** — float32 vs float64：GPU float64 性能显著低于 float32。当前 numba CPU 用 float64，GPU 可能需降为 float32
5. **kcStep 缓存** — shared memory 完美适配 （每块 121 doubles ≈ 1KB）

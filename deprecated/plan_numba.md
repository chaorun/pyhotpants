# PYHOTPANTS NUMBA/GPU 重构方案

## 当前状态总览

| 指标 | 初版 | 当前 | vs C |
|------|------|------|------|
| 总耗时 | 670s | 13.7s (49x) | 2.7x |
| buildstamps | 4.2s | 2.4s | — |
| fit | 95s | 0.6s | — |
| convolve_diff | 556s | 6.5s | — |
| output | 14s | 4.1s | — |

### 当前架构特点

- 核心数据结构：`list[dict]`（stamps），每个 dict 22 字段
- 主卷积：FFT 可分离（49 次 fftconvolve）+ 系数场多项式展开
- variance：numba 空间域逐像素（variance_convolve_jit）
- mask：numba 空间域逐像素（mask_check_loop_jit）
- fitKernel：17 次迭代，各子函数已 numba 化但逐 stamp 独立调用

### 当前瓶颈分布 (13.7s)

| 步骤 | 耗时 | 内部 |
|------|------|------|
| buildstamps | 2.4s | dict 操作 + xy_conv_stamp + build_matrix0/scprod0 |
| fit | 0.6s | build_matrix0+scprod0 (jit) + check_stamps |
| convolve_diff | 6.5s | FFT 1.7s + coeff 0.7s + variance 1s + bad_count FFT + mask 0.3s + fitKernel 2.2s + bkg/noise/realloc |
| output | 4.1s | numpy 切片 + stats |

---

## 三步重构路线图

```
现在                  → 阶段1 (numba CPU)      → 阶段2 (GPU)
FFT分离式主卷积          纯空间域统一kernel      @numba.cuda.jit
list[dict] stamps      扁平numpy SoA           cupy.array
逐stamp独立调用         批量numba merged        单次launch
多步独立调用            一体式kernel            零调度开销
```

---

## 步骤1：stamps dict → 扁平 numpy SoA

### 1.1 当前问题

```
ctStamps = [dict with 22 keys for _ in range(nS)]  # nS≈100
```

每次访问 stamp 字段都是 dict lookup。所有 numba 包装函数都需要从 dict 逐字段提取 numpy 数组（如 `build_matrix_numpy` 中 30 行提取代码，`build_scprod_numpy` 中 30 行，`get_stamp_sig_numpy` 中 ~10 行）。

### 1.2 目标数据结构（SoA 布局）

```python
class StampsArray:
    """扁平 SoA 结构，所有 stamp 数据 pack 为连续内存 numpy 数组。
    GPU 友好：直接对应 cupy.array，支持合并内存访问。"""

    def __init__(self, nS, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC):
        fwSq = fwKSStamp * fwKSStamp
        nVec = nCompKer + nBGVectors

        # ---- 标量元数据 (nS,) ----
        self.sscnt   = np.zeros(nS, dtype=np.int32)  # 当前 substamp 索引
        self.nss     = np.zeros(nS, dtype=np.int32)  # substamp 数量
        self.x0      = np.zeros(nS, dtype=np.int32)  # stamp 起点 x
        self.y0      = np.zeros(nS, dtype=np.int32)  # stamp 起点 y
        self.x       = np.zeros(nS, dtype=np.int32)  # stamp 中心 x
        self.y       = np.zeros(nS, dtype=np.int32)  # stamp 中心 y

        # ---- 子stamp 坐标 (nS, nKSStamps) ----
        self.xss     = np.zeros((nS, nKSStamps), dtype=np.int32)
        self.yss     = np.zeros((nS, nKSStamps), dtype=np.int32)

        # ---- 核心计算数据 (3D/2D 压扁) ----
        self.vectors  = np.zeros((nS, nVec, fwSq), dtype=np.float64)  # (nS, nVec, fwSq)
        self.mat      = np.zeros((nS, nC, nC), dtype=np.float64)       # (nS, nC, nC)
        self.scprod   = np.zeros((nS, nC), dtype=np.float64)            # (nS, nC)
        self.krefArea = np.zeros((nS, fwSq), dtype=np.float64)          # (nS, fwSq)

        # ---- 统计量 (nS,) ----
        self.chi2    = np.zeros(nS, dtype=np.float64)
        self.norm    = np.zeros(nS, dtype=np.float64)
        self.diff    = np.zeros(nS, dtype=np.float64)
        self.sum_val = np.zeros(nS, dtype=np.float64)
        self.mean_val = np.zeros(nS, dtype=np.float64)
        self.median  = np.zeros(nS, dtype=np.float64)
        self.mode    = np.zeros(nS, dtype=np.float64)
        self.sd      = np.zeros(nS, dtype=np.float64)
        self.fwhm    = np.zeros(nS, dtype=np.float64)
        self.lfwhm   = np.zeros(nS, dtype=np.float64)

        # ---- 掩码：标记哪些 stamp 有效 (nss>0) ----
        self.valid   = np.zeros(nS, dtype=np.bool_)
        self.ntS     = 0  # 有效 stamp 数量

    # ---- 便捷访问器（保留 dict 风格兼容） ----
    def get_sscnt(self, si):
        return self.sscnt[si]

    def get_xc(self, si):
        """当前 substamp 的中心 x 坐标"""
        return self.xss[si, self.sscnt[si]]

    def get_yc(self, si):
        """当前 substamp 的中心 y 坐标"""
        return self.yss[si, self.sscnt[si]]

    def get_vectors(self, si):
        return self.vectors[si]

    def get_mat(self, si):
        return self.mat[si]

    def get_scprod(self, si):
        return self.scprod[si]

    def increment_sscnt(self, si):
        """子stamp 递进，返回是否还有更多"""
        self.sscnt[si] += 1
        return self.sscnt[si] < self.nss[si]
```

### 1.3 SoA vs AoS 对比

| | AoS (当前 dict) | SoA (目标) |
|---|---|---|
| 内存布局 | 分散 22 个独立对象/stamp | 每种字段连续内存 |
| 访问单 stamp | `d['vectors']` dict lookup | `sa.vectors[si]` 数组索引 |
| 批量访问 | for 循环 + dict | `sa.vectors[valid_mask]` 切片 |
| numba 传递 | 需要逐字段提取 | 直接传整个数组 |
| GPU 迁移 | 需要逐字段 extract→cupy | `cupy.array(sa.vectors)` 零拷贝 |
| GPU warp 访问 | 无法合并 | 连续内存合并访问 |

### 1.4 受影响的函数

以下函数需要从接收 `ctStamps`/`ciStamps`（`list[dict]`）改为接收 `StampsArray`：

| 函数 | 当前参数 | 改为 | 当前行号 |
|------|---------|------|---------|
| `fill_stamp_numpy` | `stamp_dict: dict` | `sa: StampsArray, si: int` | 1844 |
| `xy_conv_stamp_fast_numpy` | `stamp: dict` | `sa: StampsArray, si: int` | 1277 |
| `cut_sstamp_numpy` | `stamp: dict` | `sa: StampsArray, si: int` | 600 |
| `build_matrix0_numpy` | `stamp: dict` | `sa: StampsArray, si: int` | 1455 |
| `build_scprod0_numpy` | `stamp: dict` | `sa: StampsArray, si: int` | 1532 |
| `make_model_numpy` | `stamp: dict` | `sa: StampsArray, si: int` | 1797 |
| `get_stamp_sig_numpy` | `stamp_dict: dict` | `sa: StampsArray, si: int` | 1988 |
| `build_matrix_numpy` | `stamps_dicts: list[dict]` | `sa: StampsArray` | 1541 |
| `build_scprod_numpy` | `stamps_dicts: list[dict]` | `sa: StampsArray` | 1685 |
| `check_stamps_numpy` | `stamps_dicts: list[dict]` | `sa: StampsArray` | 2608 |
| `check_again_numpy` | `stamps: list[dict]` | `sa: StampsArray` | 2745 |
| `fit_kernel_numpy` | `stamps_dicts: list[dict]` | `sa: StampsArray` | 2812 |
| `region_buildstamps_numpy` | (创建 ctStamps/ciStamps) | 创建 StampsArray | 2910 |
| `region_fit_numpy` | `ctStamps: list, ciStamps: list` | `ctSa: StampsArray, ciSa: StampsArray` | 3115 |
| `region_convolve_diff_numpy` | `ctStamps: list, ciStamps: list` | `ctSa: StampsArray, ciSa: StampsArray` | 3224 |
| `region_output_numpy` | `ctStamps: list, ciStamps: list` | `ctSa: StampsArray, ciSa: StampsArray` | 3592 |

### 1.5 jit 函数改造

当前 jit 函数的包装层（如 `build_matrix_numpy`）需要从 dict 逐字段提取 numpy 数组再传给 jit。改造后：

**改造前** (build_matrix_numpy, ~50行):
```python
def build_matrix_numpy(stamps_dicts, ...):
    valid_mask = np.zeros(nS, dtype=np.int32)
    all_x = np.zeros(nS, dtype=np.int64)
    all_y = np.zeros(nS, dtype=np.int64)
    all_mat = np.zeros((nS, nC, nC), ...)
    all_vectors = np.zeros((nS, nVecTotal, fwSq), ...)
    for i in range(nS):
        stamp = stamps_dicts[i]
        if stamp['sscnt'] < stamp['nss']:
            valid_mask[i] = 1
            all_x[i] = stamp['xss'][stamp['sscnt']]
            all_y[i] = stamp['yss'][stamp['sscnt']]
            all_mat[i] = stamp['mat'][:nC_valid, :nC_valid]
            all_vectors[i] = stamp['vectors'][:nVecTotal]
    return build_matrix_jit(all_mat, all_vectors, valid_mask, ...)
```

**改造后** (~5行):
```python
def build_matrix_numpy(sa, ...):
    valid = sa.sscnt < sa.nss
    return build_matrix_jit(sa.mat, sa.vectors, valid, ...)
```

### 1.6 迁移步骤

1. **新建 `StampsArray` 类**（在 numutils.py 顶部，约 60 行）
2. **修改 `allocate_stamp_dict` → `allocate_stamps_array`**
   - 返回单个 StampsArray 对象而非 nS 个 dict
3. **修改 `region_buildstamps_numpy`**：创建 StampsArray 替代 `[allocate_stamp_dict() for _ in ...]`
4. **逐函数迁移**（按调用顺序）：
   - 先改叶子节点：`cut_sstamp_numpy`, `xy_conv_stamp_fast_numpy`, `build_matrix0_numpy`, `build_scprod0_numpy`, `make_model_numpy`
   - 再改中间层：`fill_stamp_numpy`, `get_stamp_sig_numpy`, `build_matrix_numpy`, `build_scprod_numpy`
   - 最后改顶层：`check_stamps_numpy`, `check_again_numpy`, `fit_kernel_numpy`
5. **修改 jit 包装函数**：删除 dict 提取代码，直接使用 StampsArray 切片
6. **修改 region 级函数**：`region_fit_numpy`, `region_convolve_diff_numpy`, `region_output_numpy`
7. **chotpants.pyx 中的测试函数**：`stamp_c_to_dict` → `stamp_c_to_stampsarray`（或为测试保留旧 dict 方式，用适配器转换）

### 1.7 预期收益

| 方面 | 预期 |
|------|------|
| buildstamps | 2.4s → ~1.5s（消除 dict 创建/访问开销） |
| fit build_matrix+scprod | 0.03s → ~0.01s（零提取开销） |
| check_again get_stamp_sig | 1s → ~0.5s（无 dict lookup） |
| GPU 直接性 | 数据已是 numpy SoA，cupy 零拷贝迁移 |

---

## 步骤2：spatial_convolve FFT 分离式 → 纯空间域统一 kernel

### 2.1 当前架构

```
spatial_convolve_fast_numpy (4 段独立计算, ~4.2s)
  │
  ├── A. FFT 卷积 (1.7s)
  │     49 次 fftconvolve(image, basis_i) → convMaps
  │     中间存储: 49 × 1601×1601 × 8B = 1GB
  │
  ├── B. 系数场 (0.7s)
  │    多项式展开 + coeffFields 构造 + 加权求和
  │     中间存储: 49 × 1601×1601 × 8B = 1GB
  │
  ├── C. 写入 cRdata (<0.01s)
  │
  ├── D. variance (1.0s) 【已 numba 空间域】
  │    variance_convolve_jit: 逐像素构造核 + 方差卷积
  │
  ├── E. bad_count FFT (~1.2s)
  │    fftconvolve(badFlag2d, ones_kernel) — 统计邻域坏像素
  │
  └── F. mask (0.3s) 【已 numba 空间域】
       mask_check_loop_jit: 逐像素构造核 + kerFracMask 检查
```

**核心问题**：
- 主卷积 A+B 用了 FFT，但 variance D 和 mask F 已用空间域 numba
- 3 次独立遍历（FFT频域、variance空间域、mask空间域各自遍历）
- 49 个 convMaps + 49 个 coeffFields 的巨大中间存储（~2GB）
- GPU 上 FFT 有全局同步开销，对小核 11×11 效率不如空间域

### 2.2 目标架构：单一空间域 kernel

```python
@numba.jit(nopython=True)
def spatial_convolve_jit(
    # 输入
    image2d,           # (ySize, xSize) float64 输入图像
    variance2d,        # (ySize, xSize) float64 或 zeros（dovar=False时）
    cMask2d,           # (ySize, xSize) int32 卷积掩码
    kernelSol,         # (nSol,) float64 核解系数
    kernel_vec_2d,     # (nCompKer, fwSq) float64 核基函数
    # 输出 (in-place)
    cRdata2d,          # (ySize, xSize) float64 输出卷积结果
    vData2d,           # (ySize, xSize) float64 输出方差
    mRData2d,          # (ySize, xSize) int32 输出掩码
    # 参数
    xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
    kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance):
    """
    一次遍历完成：构造核 + 主卷积 + 方差卷积 + mask 传播
    kcStep 量化缓存：同一 block 内像素共用核
    """

    fwSq = fwKernel * fwKernel
    halfX = 0.5 * rPixX
    halfY = 0.5 * rPixY
    nSolTotal = len(kernelSol)

    FLAG_INPUT_ISBAD = 0x80
    FLAG_OUTPUT_ISBAD = 0x8000
    FLAG_BAD_CONV = 0x10
    FLAG_OK_CONV = 0x40

    # 核缓存变量
    prev_i0 = -999
    prev_j0 = -999
    kernel = np.zeros(fwSq, dtype=np.float64)    # 当前块的核
    kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)

    # 边界：内部像素范围
    jStart = hwKernel
    jEnd = ySize - hwKernel
    iStart = hwKernel
    iEnd = xSize - hwKernel

    for j in range(jStart, jEnd):
        blockJ = (j - hwKernel) // kcStep
        j0 = blockJ * kcStep + hwKernel

        for i in range(iStart, iEnd):
            blockI = (i - hwKernel) // kcStep
            i0 = blockI * kcStep + hwKernel

            # ============================================
            # 核构造（kcStep 缓存：同一 block 只构造一次）
            # ============================================
            if i0 != prev_i0 or j0 != prev_j0:
                prev_i0 = i0
                prev_j0 = j0

                xi = i0 + hwKernel
                yi = j0 + hwKernel
                xf = (xi - halfX) / halfX
                yf = (yi - halfY) / halfY

                # 计算多项式系数 (同 make_kernel_numpy 逻辑)
                kernel_coeffs[0] = kernelSol[1]  # component 0 常数
                k = 2
                for i1 in range(1, nCompKer):
                    coeff = 0.0
                    k_local = k
                    for ix in range(kerOrder + 1):
                        xpow = xf ** ix
                        for iy in range(kerOrder - ix + 1):
                            ypow = yf ** iy
                            coeff += kernelSol[k_local] * xpow * ypow
                            k_local += 1
                    kernel_coeffs[i1] = coeff
                    k = k_local

                # 构造核 kernel[jk,ik] = Σ kernel_coeffs[c] × kernel_vec[c, kernel_idx]
                for idx in range(fwSq):
                    val = 0.0
                    for c in range(nCompKer):
                        val += kernel_coeffs[c] * kernel_vec_2d[c, idx]
                    kernel[idx] = val

            # ============================================
            # 1. 主卷积：cRdata[j,i] = Σ kernel[idx] × image[j+dy, i+dx]
            # ============================================
            conv_sum = 0.0
            for jc in range(j - hwKernel, j + hwKernel + 1):
                jk = j - jc + hwKernel
                for ic in range(i - hwKernel, i + hwKernel + 1):
                    ik = i - ic + hwKernel
                    k_idx = ik + jk * fwKernel
                    conv_sum += kernel[k_idx] * image2d[jc, ic]
            cRdata2d[j, i] = conv_sum

            # ============================================
            # 2. 方差卷积：仅当 dovar=True
            # ============================================
            if dovar:
                var_sum = 0.0
                for jc in range(j - hwKernel, j + hwKernel + 1):
                    jk = j - jc + hwKernel
                    for ic in range(i - hwKernel, i + hwKernel + 1):
                        ik = i - ic + hwKernel
                        k_idx = ik + jk * fwKernel
                        if convolveVariance:
                            var_sum += kernel[k_idx] * kernel[k_idx] * variance2d[jc, ic]
                        else:
                            var_sum += abs(kernel[k_idx]) * variance2d[jc, ic]
                vData2d[j, i] = var_sum

            # ============================================
            # 3. mask 传播
            # ============================================
            # 3a. 继承 cMask 自身
            inner_cmask = cMask2d[j, i]
            mRData2d[j, i] |= inner_cmask
            if (inner_cmask & FLAG_INPUT_ISBAD):
                mRData2d[j, i] |= FLAG_OUTPUT_ISBAD

            # 3b. 统计邻域 bad pixels
            bad_count = 0
            for jc in range(j - hwKernel, j + hwKernel + 1):
                for ic in range(i - hwKernel, i + hwKernel + 1):
                    if (cMask2d[jc, ic] & FLAG_INPUT_ISBAD):
                        bad_count += 1

            # 3c. kerFracMask 检查（仅当邻域有坏像素时）
            if bad_count > 0:
                # 计算核加权的坏像素比例
                aks = 0.0
                uks = 0.0
                for jc in range(j - hwKernel, j + hwKernel + 1):
                    jk = j - jc + hwKernel
                    for ic in range(i - hwKernel, i + hwKernel + 1):
                        ik = i - ic + hwKernel
                        k_idx = ik + jk * fwKernel
                        kk = abs(kernel[k_idx])
                        aks += kk
                        if not (cMask2d[jc, ic] & FLAG_INPUT_ISBAD):
                            uks += kk

                if aks > 0.0 and (uks / aks) < kerFracMask:
                    ni = i + xSize * j
                    mRData2d[j, i] |= (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
                else:
                    mRData2d[j, i] |= FLAG_OK_CONV
            else:
                mRData2d[j, i] |= FLAG_OK_CONV
```

### 2.3 复杂度分析

| 操作 | 遍历像素 | 每像素核操作 | 总运算量 |
|------|---------|-------------|---------|
| 主卷积 | 256 万 | 121 次乘法加法 | ~3.1 亿 MAC |
| 方差卷积 | 256 万 | 121 次乘法加法 | ~3.1 亿 MAC |
| mask bad_count | 256 万 | 121 次条件判断 | ~3.1 亿分支 |
| mask aks/uks | ~3 万 (bad_count>0) | 121 次乘法加法 | ~0.4 亿 MAC |
| **总计** | | | **~7 亿 次操作** |

**核构造缓存**：kcStep=5 时，约 320×320 = 10.2 万个块。每个块构造1次核（nCompKer×fwSq 次乘加）。核构造总量可忽略。

**numba CPU 预估**：7 亿次操作 ÷ 4GHz × SIMD(4x) ≈ 0.04s 纯运算。内存带宽为瓶颈：256万像素 × 121核元素 × 8B = 2.5GB 数据读取 / ~100GB/s 带宽 ≈ 0.025s。加上 Python/numba 开销 ≈ **1-2s**。

### 2.4 与 current 的对比

| | 当前 FFT 混合 | 空间域统一 kernel |
|---|---|---|
| 主卷积 | fftconvolve × 49 = 1.7s | 内联到统一遍历 |
| 系数场 | numpy outer 0.7s | 内联到核构造 |
| variance | numba 独立遍历 1.0s | 合并到主遍历 |
| bad_count | fftconvolve ~1.2s | 内联到统一遍历 (O(1)) |
| mask | numba 独立遍历 0.3s | 合并到主遍历 |
| 中间存储 | ~2GB (49×convMaps + 49×coeffFields) | **零中间存储** |
| **总计** | **~4.2s** | **~1-2s** |

### 2.5 GPU 直接性

逐像素遍历 = **embarrassingly parallel**。GPU 版本只需：
- `@numba.jit(nopython=True)` → `@numba.cuda.jit`
- 外层 `for j,i` → GPU thread per pixel
- 核构造缓存 → shared memory
- 预估 GPU：**~0.1s**（256万像素 ÷ 1000核心 ÷ 少量warp）

### 2.6 迁移步骤

1. **新增 `spatial_convolve_jit`**（numba CPU 版，约 150 行）
2. **在 spatial_convolve_fast_numpy 中**：
   - 注释掉当前 FFT+coeffFields+加权求和代码（A、B、C 步）
   - 注释掉 variance_convolve_jit 调用（D 步，逻辑已内联）
   - 注释掉 fftconvolve bad_count 代码（E 步）
   - 注释掉 mask_check_loop_jit 调用（F 步，逻辑已内联）
   - 替换为单次 `spatial_convolve_jit(...)` 调用
3. **精度验证**：test_spatial_convolve vs C 版本
4. **性能验证**：1K 测试

### 2.7 平滑过渡策略

由于改动较大（4段合并），可分两步：
1. **先做步骤2a**：主卷积加入统一 kernel（FFT 卷积转为空间域），variance/mask 仍独立调用
2. **再做步骤2b**：合并 variance 和 mask 到统一 kernel

或一步到位——因为 variance/mask 逻辑已经清楚（各自独立的 numba 函数可直接内联），一步合并风险可控。

---

## 步骤3：fitKernel 批量合并

### 3.1 当前问题

fitKernel 的 17 次迭代中，绝大部分时间在 `check_again` 的逐 stamp 循环：

```
fit_kernel_numpy (17次迭代)
  │
  ├── build_matrix_numpy  (0.03s)  ← dict提取→jit, 已优化
  ├── build_scprod_numpy  (0.03s)  ← dict提取→jit, 已优化
  ├── np.linalg.solve     (~0s)
  │
  └── check_again_numpy   (1.25s/次 × 17 = 21s → now 1.25s/次??)
       │
       ├── for si in range(nS):          ← Python for 循环 × 100
       │     ├── get_stamp_sig_numpy(...)  ← dict提取→jit (0.01s/stamp)
       │     └── fill_stamp_numpy(...)     ← 仅替换时 (1-2次/迭代)
       │
       └── sigma_clip_numpy(...)           ← 已矢量化
```

注意：步骤1完成后，dict 提取开销消失，但 `for si in range(nS)` 的 Python 循环 + jit 调用开销（100次 × 每次 jit 调用边界开销）仍然存在。

### 3.2 优化策略

#### 策略 A：批量 get_stamp_sig

```python
@numba.jit(nopython=True)
def get_stamp_sig_batch_jit(
    # StampsArray 的全部数据（SoA 扁平数组）
    sa_vectors,      # (nS, nVec, fwSq) float64
    sa_krefArea,     # (nS, fwSq) float64
    sa_sscnt,        # (nS,) int32
    sa_nss,          # (nS,) int32
    sa_xss,          # (nS, nKSStamps) int32
    sa_yss,          # (nS, nKSStamps) int32
    # 共享参数
    kernelSol,       # (nSol,) float64
    imNoise2d,       # (ySize, xSize) float64
    mRData2d,        # (ySize, xSize) int32
    # 参数
    nS, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp,
    rPixX, rPixY, figMerit_is_v, statSig,
    # 输出
    out_chi2,        # (nS,) float64 每个 stamp 的 chi2
    out_sig1,        # (nS,) float64 信噪比
):
    for si in range(nS):
        scnt = sa_sscnt[si]
        if scnt >= sa_nss[si]:
            out_chi2[si] = -1.0  # 标记无效
            continue

        xi = sa_xss[si, scnt]
        yi = sa_yss[si, scnt]

        vectors = sa_vectors[si]       # (nVec, fwSq) 视图
        im = sa_krefArea[si]           # (fwSq,) 视图

        # ---- 内联 make_model 逻辑 ----
        csModel = np.zeros(fwSq, dtype=np.float64)
        # kernel component 部分
        for ipix in range(fwSq):
            val = kernelSol[1] * vectors[0, ipix]
            for jv in range(1, nCompKer):
                val += kernelSol[k_idx] * ...  # 多项式展开 + vectors[jv, ipix]
            csModel[ipix] = val
        # background component 部分
        # ...

        # ---- 内联 sig 计算 ----
        chi2 = 0.0; nEff = 0
        for ipix in range(fwSq):
            ix = xi + ipix % fwKSStamp - hwKSStamp
            iy = yi + ipix // fwKSStamp - hwKSStamp
            if 0 <= ix < rPixX and 0 <= iy < rPixY:
                ni = ix + rPixX * iy
                if not (mRData2d[iy, ix] & FLAG_OUTPUT_ISBAD):
                    diff = (im[ipix] - csModel[ipix]) / imNoise2d[iy, ix]
                    chi2 += diff * diff
                    nEff += 1
        out_chi2[si] = chi2 / nEff if nEff > 0 else -1.0
```

#### 策略 B：批量 fill_stamp

fill_stamp 只在 stamp 需要替换时调用（1-2次/迭代）。其内部调用 `xy_conv_stamp_fast_numpy`（im2col+GEMM，已优化）、`build_matrix0_jit`、`build_scprod0_jit`。将这些步骤也合并为一个 numba 函数：

```python
@numba.jit(nopython=True)
def fill_stamp_jit(sa_vectors, sa_mat, sa_scprod, sa_krefArea,
                   sa_sscnt, sa_xss, sa_yss,
                   image2d, si, ...):
    """对一个 stamp 填充所有数据，替代 fill_stamp_numpy"""
```

但 fill_stamp 调用频率低（~1-2次/迭代 × 17次 = ~30次），优化优先级更低。

### 3.3 预期收益

| 项目 | 当前 (步骤1后) | 步骤3后 |
|------|---------------|--------|
| get_stamp_sig (100 stamps) | ~0.5s (100 jit 调用) | ~0.01s (1 jit 调用) |
| build_matrix (dict提取) | ~0s (步骤1消除) | ~0s |
| fill_stamp (~2次) | ~0.1s | ~0.05s |
| check_again 总计 | ~0.5s/迭代 | ~0.01s/迭代 |
| fitKernel 总计 (17次) | ~1.2s | **~0.2s** |

### 3.4 迁移步骤

1. **新增 `get_stamp_sig_batch_jit`**（约 80 行）
2. **修改 `check_again_numpy`**：用单次 jit 调用替代 for 循环中的逐 stamp get_stamp_sig
3. **（可选）新增 `fill_stamp_batch_jit`**：批量 fill_stamp
4. **精度验证**：test_check_again, test_fit_kernel
5. **性能验证**：1K 测试

---

## 综合路线图

```
阶段 0 (当前)          阶段 1 (步骤1)           阶段 2 (步骤2+3)        GPU 就绪
═══════════════       ═══════════════         ════════════════       ════════════
                                                                    
list[dict] stamps  →  StampsArray SoA     →  StampsArray SoA     →  cupy.array
FFT 主卷积 1.7s    →  FFT 主卷积 1.7s     →  空间域统一 1.5s     →  cuda 0.1s
coeff fields 0.7s  →  coeff fields 0.7s   →  (合并)              →  (合并)
var jit 1.0s       →  var jit 1.0s        →  (合并)              →  (合并)
mask jit 0.3s      →  mask jit 0.3s       →  (合并)              →  (合并)
逐stamp sig 1.0s   →  逐stamp sig 0.5s    →  批量 sig 0.01s      →  (批量)
dict提取 build 0.03s → 切片 build 0.01s   →  切片 build 0.01s    →  (无变化)
                                                                    
总计 ~13.7s           总计 ~11s              总计 ~5s               总计 <1s
vs C 2.7x             vs C 2.2x              vs C 1.0x             vs C 0.1x
```

---

## 文件变更清单

| 文件 | 变更内容 | 规模估计 |
|------|---------|---------|
| `numutils.py` 顶部 | 新增 `StampsArray` 类 | +60 行 |
| `numutils.py` 中部 | 修改 15 个函数签名 + 删除 dict 提取代码 | ~200 行变更 |
| `numutils.py` 中部 | 新增 `spatial_convolve_jit` | +150 行 |
| `numutils.py` 中部 | 新增 `get_stamp_sig_batch_jit` | +80 行 |
| `numutils.py` 中部 | 修改 spatial_convolve_fast_numpy | ~100 行注释 +20 行新 |
| `numutils.py` 中部 | 修改 check_again_numpy | ~30 行变更 |
| `chotpants.pyx` | 测试函数适配（stamp_c_to_dict 保留） | ~20 行 |

总计新增约 300 行，修改约 350 行。

---

## 风险与注意事项

1. **精度验证**：每一步改动后必须 vs C 版本验证精度（maskOut EXACT MATCH 是最重要的验收标准）
2. **noiseOut 回归**：session-ses_11b1 发现的 noiseOut 精度问题需优先修复
3. **numba JIT 冷启动**：首次运行编译时间可能达到 30-60s，可通过 `@jit(cache=True)` 减轻
4. **chotpants.pyx 测试**：`test_*` 函数依赖 `stamp_c_to_dict`，需保留向后兼容适配器
5. **kcStep 量化精度**：空间域 kernel 中 kcStep 量化引入了近似（同一 block 内共用核），需确认与 C 版本的精度误差在可接受范围

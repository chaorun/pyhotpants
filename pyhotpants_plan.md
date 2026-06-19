# pyhotpants 移植计划

## 项目目标

将 C 语言的 hotpants 图像差分核心逐步迁移到 Python/numpy，最终消除对 C 代码的依赖。

## 当前架构

```
Python (.pyx):
  hotpants_init(&ctx, ...)
  for ri in range(nR):
      region_setup(ctx, p, ri, &rs)
      region_buildstamps(ctx, p, &rs, ...)
      region_fit(ctx, p, &rs, ...)
      region_convolve_diff(ctx, p, ri, &rs, ...)
      region_output(ctx, p, ri, &rs)
      region_cleanup_local(&rs, convTmpl)
  hotpants_cleanup(&ctx)
```

Python 掌控主循环，6 个子步骤仍为 C 函数。

## 已完成

### Phase 1a — C 层重构
- 从 hotpants_compute 中提取 hotpants_init / hotpants_cleanup
- 定义 hotpants_context struct，暴露到头文件
- EXACT MATCH 验证通过

### Phase 1b — Python 主循环
- 定义 hotpants_params struct
- 提取 hotpants_process_region (~900行)
- .pyx 中用 Python 写主循环 (init + for process_region + cleanup)
- INPUT + OUTPUT EXACT MATCH

### Phase 2a — process_region 拆分
- 定义 region_state struct
- 将 process_region 拆成 6 个子步骤函数：
  1. region_setup — 边界 + 数组分配 + extract_subregion + noise + mask
  2. region_buildstamps — stamp 构建 + getKernelVec
  3. region_fit — fillStamp + check_stamps + 决定 convTmpl
  4. region_convolve_diff — fitKernel + convolve + noise合成 + mask
  5. region_output — stats统计 + insert_subregion + 写出
  6. region_cleanup_local — free 所有 region 数组
- 每步编译 + replay EXACT MATCH

## Phase 3 — 逐步替换为 numpy

### 核心原则：影子执行验证

**不替换旧函数**。新的 numpy 函数与旧的 C 函数同步执行，逐数组对比输出：

```python
for ri in range(nR):
    # === C 版本（主路径，保证结果不变） ===
    region_setup(ctx, p, ri, &rs)

    # === numpy 版本（影子路径） ===
    tRData_py, iRData_py, ... = region_setup_numpy(...)

    # === 逐数组对比 ===
    c_view = <float[:n]>rs.tRData
    assert np.array_equal(np.asarray(c_view), tRData_py.ravel())
    # ... 对比所有输出数组 ...

    # === 确认一致后，继续用 C 结果走后续步骤 ===
    region_buildstamps(ctx, p, &rs, ...)
```

对于会被修改的输入数组：先 `.copy()` 再传给 numpy 版本，互不干扰。

验证通过后，删掉 C 调用和对比代码，只留 numpy 版本。

### 替换优先级

| 优先级 | 子步骤 | numpy 替换难度 | 关键替换 |
|--------|--------|----------------|----------|
| 1 | region_setup | 低 | extract_subregion→切片, makeNoiseImage4→numpy, fset→赋值 |
| 2 | region_output | 低 | insert_subregion→切片, getStampStats3→numpy统计 |
| 3 | region_cleanup_local | 无 | numpy 自动 GC，函数消失 |
| 4 | region_buildstamps | 中 | buildStamps 逻辑复杂 |
| 5 | region_fit | 高 | fillStamp + check_stamps 涉及矩阵运算 |
| 6 | region_convolve_diff | 最高 | fitKernel + spatial_convolve 核心线性代数 |

### Phase 3a：region_setup → numpy

region_setup 中的 C 函数替换对照：

| C 函数 | numpy 替换 |
|--------|-----------|
| `calloc + fset` | `np.full((rPixY, rPixX), fillVal, dtype=np.float32)` |
| `extract_subregion_flt` | `full_data[ymin:ymax+1, xmin:xmax+1].copy()` |
| `extract_subregion_int` | 同上，dtype=np.int32 |
| `makeNoiseImage4(data, 1/gain, rdnoise/gain, nx, ny)` | `data / gain + (rdnoise/gain)**2` |
| `makeInputMask` | numpy 条件 + 位运算 |
| `fset(arr, val, nx, ny)` | `arr[:] = val` |

必须保留的所有条件分支：
- `nR > 1` vs `nR == 1` — buffer 计算方式不同
- `iNoiseFullData` 存在与否 — extract vs makeNoiseImage4
- `tNoiseFullData` 存在与否 — 同上
- `iMaskFullData` 存在与否 — 是否提取 mask
- `tMaskFullData` 存在与否 — 同上
- `tPedestal/iPedestal != 0` — 是否减去 pedestal
- `localForceConvolve` 为 `"i"`/`"t"`/`"b"` — 控制 ctStamps/ciStamps 分配

### Phase 3b：region_output → numpy

| C 函数 | numpy 替换 |
|--------|-----------|
| `insert_subregion_flt` | `out[y1:y2, x1:x2] = sub[...]` |
| `insert_subregion_int` | 同上 |
| `getStampStats3` | numpy mean/median/std + sigma clip |
| `getNoiseStats3` | numpy 运算 |
| `sigma_clip` | numpy 实现 |
| `getFinalStampSig` | numpy 实现 |

### Phase 3c：region_cleanup_local 自动消失

numpy 数组由 Python GC 管理，不再需要手动 free。

## Git 工作流

- 所有 commit 到 dev 分支
- main 保持 v1.0 稳定版
- 用户认可后才合并到 main

## 验证方式

- C 端：replay 工具验证 EXACT MATCH
- Python 端：dump py_input.bin + py_output.bin，与 C 端 cmp 对比
- 影子执行：numpy 版本与 C 版本逐数组 assert_array_equal

## Phase 3a 完成记录

region_setup 已切换为 numpy 主路径：
- numpy 计算 7 个数据数组 → malloc + memcpy 到 C 内存
- stamps/kerSol 由 Cython calloc + C allocateStamps 分配
- C 的 region_setup 作为影子验证（已验证逐字节一致）
- INPUT + OUTPUT EXACT MATCH

## Phase 3b — region_output numpy 化

### 架构原则

严格的 Python/C 边界分离：

```
Cython 胶水层:
  C struct → Python dict/array    (输入转换)
  纯 Python/numpy 函数()           (零 C 调用)
  Python result → C struct         (输出转换)
  freeStampMem 等 C 清理           (外部执行)
```

每个 numpy 化函数内部只有纯 Python/numpy 代码，不调用任何 C 函数。

### 前置任务

#### Step 0: 读取源码（已完成）

stamp_struct 完整定义（hotpants_globals.h）：
```c
typedef struct {
   int       x0, y0;       // stamp 在 region 中的起点
   int       x, y;         // stamp 在 region 中的中心
   int       nx, ny;       // stamp 大小
   int       *xss;         // substamp 中心 x 坐标数组
   int       *yss;         // substamp 中心 y 坐标数组
   int       nss;          // substamp 数量 (1..nss)
   int       sscnt;        // 当前使用的 substamp 索引 (0..nss-1)
   double    **vectors;    // 卷积图像数据
   double    *krefArea;    // kernel substamp 数据
   double    **mat;        // 拟合矩阵
   double    *scprod;      // kernel sum solution
   double    sum, mean, median, mode, sd, fwhm, lfwhm;  // 统计量
   double    chi2;         // 拟合残差
   double    norm;         // kernel sum
   double    diff;         // (norm - mean_ksum) * sqrt(sum)
} stamp_struct;
```

ran1 常量（Numerical Recipes 三线性同余生成器）：
```
M1=259200  IA1=7141  IC1=54773  RM1=1/M1
M2=134456  IA2=8121  IC2=28411  RM2=1/M2
M3=243000  IA3=4561  IC3=51349
```

getStampStats3 中每次调用 idum=-666 重新初始化 ran1，序列完全确定性。

#### Step 1: ran1 Python 实现

翻译 C 的 ran1 为纯 Python 类，精确模拟 static 状态和整数运算。
单元测试：调用 100 次，逐值对比 C 输出。

#### Step 2: stamp_struct .pxd 完整声明 + C→Python 转换

在 .pxd 中声明 stamp_struct 所有字段。
编写 stamp_to_dict(stamp_struct *s) 转换函数。
region_output 只读 stamp 字段（xss, yss, sscnt, nss），暂不需要回写。

#### Step 3: region_output numpy 化

翻译以下 C 函数为纯 Python/numpy：
- insert_subregion_flt/int → numpy 切片赋值
- getStampStats3 → numpy 直方图 + sigma_clip + ran1
- getNoiseStats3 → numpy 运算
- getFinalStampSig → numpy 运算（需要 stamp 的 xss/yss/sscnt）
- sigma_clip → numpy 实现

freeStampMem 在函数外由 Cython 胶水层调用 C 执行。

#### Step 4: 影子验证

numpy 为主路径，C 为影子，逐字节对比所有输出（diffOut, noiseOut, convOut, maskOut, stats）。

#### Step 5: 切换

确认一致后，注释掉 C 影子代码。

### 已知难点

1. getStampStats3 (275行)：直方图+随机采样+迭代，需要 ran1 一致性
2. 数值精度：C float/double 隐式转换 vs numpy float32/float64
3. 后续 buildstamps/fit/convolve_diff 的 stamp 双向转换更复杂（远期）

## 完整路线图

### Phase 1 — C 层重构 + Python 主循环 ✅

| 子阶段 | 内容 | 状态 |
|--------|------|------|
| 1a | 从 hotpants_compute 提取 init/cleanup，暴露 context/params 到头文件 | ✅ |
| 1b | 提取 process_region (~900行)，.pyx 中 Python 掌控主循环 | ✅ |

### Phase 2 — process_region 拆分 ✅

| 子阶段 | 内容 | 状态 |
|--------|------|------|
| 2a | 拆成 6 个子步骤函数 (setup/buildstamps/fit/convolve_diff/output/cleanup) | ✅ |

### Phase 3 — 逐步 numpy 化（当前阶段）

架构原则：纯 Python 函数 + Cython 胶水层（C↔Python 转换在函数外，函数内零 C 调用）

| 子阶段 | 内容 | 核心替换 | 状态 |
|--------|------|----------|------|
| 3a | region_setup → numpy | extract_subregion→切片, makeNoiseImage4→numpy, makeInputMask→numpy | ✅ |
| 3b | region_output → numpy | insert_subregion→切片, getStampStats3→numpy, sigma_clip→numpy | 进行中 |
| 3c | region_cleanup_local | numpy 数组由 GC 管理，C free 逐步消失 | 待做 |
| 3d | region_buildstamps → numpy | buildStamps 逻辑（stamp 中心检测、substamp 构建） | 待做 |
| 3e | region_fit → numpy | fillStamp→numpy 卷积, check_stamps→numpy 矩阵运算 | 待做 |
| 3f | region_convolve_diff → numpy | fitKernel→numpy 线性代数, spatial_convolve→numpy 卷积 | 待做 |

### Phase 4 — init/cleanup numpy 化

| 内容 | 说明 |
|------|------|
| hotpants_init | 整数计算→Python，calloc→numpy，kernel_vec/check_mat 指针数组→numpy+指针 |
| hotpants_cleanup | numpy 数组自动 GC，手动 free 消失 |

### Phase 5 — 完全脱离 C

| 内容 | 说明 |
|------|------|
| 移除 csrc/ 目录 | 所有 C 代码不再需要 |
| 移除 .pxd 中的 C 声明 | 所有接口纯 Python |
| setup.py 简化 | 不再编译 C 扩展（或保留可选的 Cython 加速层） |
| 纯 pip install | 无需 C 编译器 |

### 各 Phase 难度评估

| Phase | 难度 | 关键挑战 |
|-------|------|----------|
| 3a | 低 | ✅ 已完成 |
| 3b | 中 | getStampStats3 (275行, ran1+直方图+迭代), 数值精度 |
| 3c | 无 | numpy GC 自动处理 |
| 3d | 中高 | buildStamps 的 PSF 中心检测, stamp 双向转换已就绪 |
| 3e | 高 | fillStamp 的基函数卷积, check_stamps 的矩阵求解 (ludcmp/lubksb) |
| 3f | 最高 | fitKernel 的迭代最小二乘, spatial_convolve 的空间变化卷积 |
| 4 | 低 | init 只是整数计算+数组分配 |
| 5 | 低 | 删除 C 代码，清理构建系统 |

### 已完成的基础设施

| 组件 | 文件 | 说明 |
|------|------|------|
| Ran1 类 | numutils.py | 与 C ran1 完全一致的 Python 随机数生成器 |
| stamp_c_to_dict | chotpants.pyx | C stamp_struct → Python dict（完整 22 字段） |
| dict_to_stamp_c | chotpants.pyx | Python dict → C stamp_struct（完整双向，roundtrip 测试通过） |
| region_setup_numpy | chotpants.pyx | 纯 numpy 的 region_setup 实现 |
| 影子执行框架 | chotpants.pyx | numpy 主路径 + C 影子验证模式 |

### Convert 函数计划（C↔Python 转换工具箱）

在做底层函数 numpy 化之前，先准备完整的 C↔Python 转换函数。所有 convert 函数为 cdef（Cython 胶水层），放在 chotpants.pyx 中。

#### 已完成

| 函数 | 方向 | 说明 |
|------|------|------|
| stamp_c_to_dict | C→Py | stamp_struct → dict，完整 22 字段 + vectors/mat/krefArea/scprod |
| dict_to_stamp_c | Py→C | dict → stamp_struct，malloc + 完整数据复制 |
| test_stamp_roundtrip | 测试 | 随机数据往返 C→Py→C 逐字节一致 |

#### 待实现：基本指针转换

| 函数 | 方向 | 签名 | 说明 |
|------|------|------|------|
| float_ptr_to_numpy | C→Py | (float *ptr, int n) → np.ndarray | 复制 float* 为 float32 数组 |
| int_ptr_to_numpy | C→Py | (int *ptr, int n) → np.ndarray | 复制 int* 为 int32 数组 |
| double_ptr_to_numpy | C→Py | (double *ptr, int n) → np.ndarray | 复制 double* 为 float64 数组 |
| double_pp_to_numpy2d | C→Py | (double **ptr, int rows, int cols) → np.ndarray | 复制 double** 为 2D float64 数组 |
| numpy_to_float_ptr | Py→C | (np.ndarray) → float* | malloc + memcpy，调用者负责 free |
| numpy_to_int_ptr | Py→C | (np.ndarray) → int* | 同上 |
| numpy_to_double_ptr | Py→C | (np.ndarray) → double* | 同上 |
| numpy2d_to_double_pp | Py→C | (np.ndarray) → double** | malloc 行指针 + 每行 malloc + 复制 |
| free_double_pp | 清理 | (double **ptr, int rows) | 释放 double** 的每行和行指针 |

#### 验证方式

每个 convert 函数都做 roundtrip 测试：
1. 用 numpy 随机数创建测试数据
2. numpy_to_xxx_ptr → C 指针
3. xxx_ptr_to_numpy → 新 numpy 数组
4. 对比原始数据与往返数据，逐字节一致
5. free C 指针

#### 使用场景

这些 convert 函数在两个地方使用：
1. **底层函数单元测试**：随机输入 → convert 为 C 类型 → 调用 C 函数 → convert 输出为 Python → 对比 numpy 版本
2. **region 子步骤替换**：主循环胶水层中，把 C 数据转为 Python 传给纯 numpy 函数，结果转回 C

### 底层函数 numpy 化计划

在 convert 函数就绪后，按先易后难的顺序 numpy 化所有底层 C 函数。

#### 函数所在文件

| 文件 | 函数 |
|------|------|
| functions.c | sigma_clip, getStampStats3, getNoiseStats3, insert_subregion_flt/int, buildStamps, getPsfCenters, cutSStamp, makeNoiseImage4, makeInputMask, spreadMask, fset, ran1, allocateStamps, freeStampMem |
| alard.c | getKernelVec, fillStamp, check_stamps, fitKernel, spatial_convolve, make_kernel, get_background, getFinalStampSig, getStampSig, check_again, build_matrix/scprod, build_matrix0/scprod0, ludcmp, lubksb, xy_conv_stamp, make_model |

#### 执行批次

| 批次 | 函数 | 难度 | 说明 |
|------|------|------|------|
| 第一批 | sigma_clip, getNoiseStats3, insert_subregion, getFinalStampSig, get_background, make_kernel | 低 | 短小直接，纯数组运算 |
| 第二批 | getStampStats3, getKernelVec, ludcmp/lubksb | 中 | getStampStats3 需要 ran1；ludcmp/lubksb 可用 np.linalg.solve |
| 第三批 | fillStamp, build_matrix/scprod, check_stamps, buildStamps, getPsfCenters | 中高 | 基函数卷积 + 矩阵组装 |
| 第四批 | fitKernel, spatial_convolve | 最高 | 迭代最小二乘 + 空间变化卷积 |

#### 每个函数的测试方式

1. 用随机数据生成输入（numpy 随机数）
2. 通过 convert 函数转为 C 类型
3. 调用 C 函数获取参考输出
4. 调用 numpy 版本获取输出
5. 逐值对比（float 允许 eps 误差或要求逐字节一致）
6. 释放 C 内存

### alard.c numpy 化计划

alard.c 包含 Alard & Lupton 算法的核心实现，共 21 个函数约 1700 行。
自底向上分四批执行，每批完成后影子测试验证。

#### 精度要求

- ludcmp/lubksb：逐行翻译（不用 np.linalg.solve 替代），对比差异 < 1e-5 可接受
- 其他函数：逐字节一致或浮点精度内差异可接受

#### 函数清单与依赖关系

| 函数 | 行数 | 难度 | 被谁调用 | 调用谁 |
|------|------|------|----------|--------|
| get_background | 29 | 低 | region_output | 无 |
| getFinalStampSig | 34 | 低 | region_output | 无 |
| make_kernel | 40 | 低 | region_convolve_diff | 无 |
| lubksb | 20 | 低 | check_stamps, fitKernel | 无 |
| kernel_vector_PCA | 30 | 低 | fillStamp(PCA模式) | 无 |
| kernel_vector | 64 | 中 | fillStamp, xy_conv_stamp | 无 |
| getKernelVec | 18 | 低-中 | region_buildstamps | kernel_vector |
| ludcmp | 67 | 中 | check_stamps, fitKernel | 无 |
| xy_conv_stamp | 50 | 中 | fillStamp | kernel_vector |
| xy_conv_stamp_PCA | 34 | 中 | fillStamp(PCA模式) | 无 |
| build_matrix0 | 51 | 中 | fitKernel | 无 |
| build_scprod0 | 46 | 中 | fitKernel | 无 |
| make_model | 47 | 中 | getStampSig | 无 |
| build_matrix | 123 | 中高 | check_stamps | 无 |
| build_scprod | 59 | 中高 | check_stamps | 无 |
| fillStamp | 83 | 中高 | region_fit, check_again | xy_conv_stamp, kernel_vector |
| getStampSig | 79 | 中高 | check_stamps, check_again | make_model |
| spatial_convolve | 96 | 高 | region_convolve_diff | 无 |
| fitKernel | 66 | 高 | region_convolve_diff | build_matrix0, build_scprod0, ludcmp, lubksb, check_again |
| check_stamps | 242 | 高 | region_fit | build_matrix, build_scprod, ludcmp, lubksb, getStampSig |
| check_again | 105 | 高 | fitKernel | fillStamp, getStampSig, xy_conv_stamp |

#### 执行批次

| 批次 | 函数 | 状态 |
|------|------|------|
| 第一批（无依赖，低难度） | get_background, getFinalStampSig, make_kernel, lubksb, kernel_vector_PCA | 待做 |
| 第二批（底层，中等） | kernel_vector, getKernelVec, ludcmp, xy_conv_stamp, xy_conv_stamp_PCA, build_matrix0, build_scprod0, make_model | 待做 |
| 第三批（中间层） | build_matrix, build_scprod, fillStamp, getStampSig | 待做 |
| 第四批（顶层，最难） | check_stamps, fitKernel, check_again, spatial_convolve | 待做 |

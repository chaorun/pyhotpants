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

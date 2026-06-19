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

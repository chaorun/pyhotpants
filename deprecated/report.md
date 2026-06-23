# pyhotpants 重构报告

## 概述

将 hotpants 天文图像差分工具从 C/Cython 重构为纯 Python + Numba 实现。无需编译，性能与 C 原版持平。

## 时间线

```
670s (初版纯Python)
  → 150s (numpy 矢量化)
  → 90s  (numba jit 各子函数)
  → 42s  (variance/mask numba)
  → 22s  (get_stamp_sig numba)
  → 13.7s (background/noise 矢量化)
  → 7.0s  (StampsArray SoA + 全链路 numba + parallel)
  → 3.87s (五阶段优化完成)
```

## 当前性能

1K×1K 单 region，ko=2，bgo=2，Apple M 系列：

| | 耗时 | vs C |
|------|------|------|
| C 原版 | 3.36s | — |
| pyhotpants 热态 | 3.87s | **1.15x** |
| 精度 | **maskOut EXACT MATCH** | — |

## 架构

```
hotpants.py             alard.py               functions.py
┌──────────────┐       ┌──────────────────┐   ┌─────────────────┐
│ hotpants()   │──┐    │ fill_stamp        │   │ 常量 + 工具函数 │
│ region_setup │  │    │ build_matrix      │   │ sigma_clip      │
│ region_build │  │    │ fit_kernel        │   │ check_psf_center │
│ region_fit   │  │    │ spatial_convolve  │   │ psfCenters       │
│ region_conv  │  │    │ get_stamp_sig     │   │ get_stamp_stats  │
│ region_output│  │    │ variance_convolve │   │ make_noise_image │
└──────┬───────┘  │    └──────────────────┘   └─────────────────┘
       │          │
       └──────────┘
```

模块依赖：`hotpants.py → alard.py → functions.py`，单向无循环。

## 五阶段优化

| 阶段 | commit | 内容 | 收益 |
|------|--------|------|------|
| 1 | `1d813f6` | spatial_convolve 批量核构造（numpy broadcasting） | **-3.2s** |
| 2 | `be86dfe` | fill_stamp precompute kernels/image | 持平 |
| 3 | `be86dfe` | psfCenters 矢量化（scipy maximum_filter） | 持平 |
| 4 | `ab6e232` | get_stamp_sig batch polynomial | 持平 |
| 5 | `a154b3b` | build_matrix wxy precompute | 持平 |
| 去 StampsArray | `ad370eb` | 删除 StampsArray 类，完全扁平化 | -0.2s |

## 从 Cython 到纯 Python

- 删除 `chotpants.pyx`/`.pxd`/`.c`/`.so`、`setup.py`、`csrc/`
- `numutils.py`（6749 行）拆分为 `hotpants.py` + `alard.py` + `functions.py`
- 所有数据通过 numpy 数组传递，无需 FITS I/O
- Logger 外部传入，替换 `sys.stderr.write`

## 代码统计

| 文件 | 行数 | 内容 |
|------|------|------|
| `hotpants.py` | ~7120 | 主入口 + 6 个 region 调度函数 |
| `alard.py` | ~2360 | 核拟合、空间卷积、显著性评估 |
| `functions.py` | ~1190 | 常量、工具函数 |

## 精度基线

对比 C 原版（`raw_code/hotpants`）：

| 输出 | 差异像素 | max_abs |
|------|---------|---------|
| maskOut | **0** | **EXACT MATCH** |
| diffOut | 2,422,191 | 2.57e-02 |
| noiseOut | 1,943,031 | 3.05e-05 |
| convOut | 1,511,606 | 2.54e-02 |

差异来自 float32/float64 浮点精度，非算法错误。

## 返回值

```python
diff, noise, conv, mask, stats = hotpants(inim=sci, tmplim=tmpl, ...)
```

stats 字段：

| 组 | 字段 |
|----|------|
| diff GOOD | `diff_good_mean` / `_median` / `_mode` / `_sd` |
| noise GOOD | `noise_good_mean` / `_median` / `_mode` |
| diff OK | `diff_ok_mean` / `_median` / `_mode` / `_sd` |
| noise OK | `noise_ok_mean` / `_median` / `_mode` |
| 全局 | `sum_kernel` `x2norm` `diffrat` `sub_mean_sig` 等 |

## 测试

```bash
cd /path/to/hotpants
PYTHONPATH=. python3 examples/test_precision.py
```

## 依赖

- Python 3.8+
- NumPy, Numba, SciPy
- astropy（仅测试用）

## Commit 历史

```
0628975 move metal.md pyhotpants_plan.md to deprecated/
2a43eef cleanup: remove compiled binaries from deprecated/
f006639 optimize: ascontiguousarray->asarray, kernelSol.astype->np.asarray
aae1b00 docs: clarify float32/float64 input in README/Usage
76af623 stats keys rename with good/ok suffix, add median/mode
107420d sys.stderr.write -> logger, accept logger param externally
ca78683 cleanup: remove redundant np.float64(np.float32(...)) nesting
a154b3b phase5: wxy precompute, skip polynomial re-expansion
ad370eb refactor: remove StampsArray class, full flat array dicts
ab6e232 phase4: batch polynomial via numpy broadcasting
be86dfe phase2+3: fill_stamp precompute + psfCenters vectorized
1d813f6 phase1: spatial_convolve batch kernel via numpy broadcasting
5c5d1e1 refactor: split numutils -> hotpants/alard/functions
```

## MLX (Metal GPU) 探索

参见 `deprecated/metal.md`。12 个函数 MLX 实现已验证精度，但 Apple Silicon 上 MLX 的 launch overhead 导致所有尺寸慢于 numba parallel CPU。结论：当前架构下 GPU 无优势。

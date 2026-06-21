# pyhotpants — hotpants 图像差分的纯 Python 实现

## 项目概述

hotpants 天文图像差分算法的纯 Python 实现。核心计算基于 numba JIT 编译，性能与 C 原版持平（热态 3.87s，C 3.36s）。无需编译，即装即用。

## 文件结构

```
pyhotpants/
  __init__.py              from .hotpants import hotpants
  hotpants.py              主入口 + region 调度函数
  alard.py                 核拟合、空间卷积、显著性评估核心算法
  functions.py             常量、工具函数（StampsArray、sigma-clip 等）
  README.md                本文件
  Usage.md                 详细 API 文档
  COPIYNG / LICENSE
  deprecated/              弃用文件（Cython 时代残留）
    chotpants.pyx / .c / .so
    numutils.py
    csrc/                  C 源码（历史参考）
```

## 依赖

- Python 3.8+
- NumPy
- Numba
- SciPy

## 安装

```bash
cd pyhotpants
pip install -e .
```

或直接使用（无需安装）：

```bash
export PYTHONPATH=/path/to/hotpants:$PYTHONPATH
```

## 快速开始

```python
import logging
from astropy.io import fits
import numpy as np
from pyhotpants import hotpants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('hotpants')
logging.getLogger('numba').setLevel(logging.WARNING)

tmpl = fits.getdata('template.fits').astype(np.float32)
sci = fits.getdata('science.fits').astype(np.float32)

diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    logger=logger,
    c='t', n='t', r=10, ko=2, bgo=1,
)

fits.writeto('diff.fits', diff, overwrite=True)
```

### 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `diff` | float32 2D | 差分图像 |
| `noise` | float32 2D | 噪声图像 |
| `conv` | float32 2D | 卷积图像 |
| `mask` | int32 2D | 掩码图像 |
| `stats` | list[dict] | 每 region 统计信息 |

stats 字段参见 [Usage.md](Usage.md)。

## 性能

1K×1K 单 region，ko=2，bgo=2，Apple M 系列：

| | 热态 | vs C |
|------|------|------|
| C 原版 | 3.36s | — |
| pyhotpants | 3.87s | 1.15x |

精度：maskOut EXACT MATCH。

## 测试

```bash
cd /path/to/hotpants
PYTHONPATH=. python3 examples/test_precision.py
```

## 架构

```
hotpants.py             functions.py
┌──────────────┐       ┌─────────────┐
│ hotpants()   │──┐    │ 常量        │
│ region_*()   │  │    │ sigma_clip  │
└──────┬───────┘  │    │ check_psf   │
       │          │    │ psfCenters  │
       ▼          │    └─────────────┘
alard.py           │
┌──────────────┐   │
│ fill_stamp   │◄──┘
│ build_matrix │
│ fit_kernel   │
│ spatial_conv │
│ get_stamp_sig│
│ optimize_all │      optimize_all

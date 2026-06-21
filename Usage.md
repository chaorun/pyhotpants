# pyhotpants 使用指南

## 安装

```bash
cd pyhotpants && pip install -e .
```

或直接 `PYTHONPATH=. python` 使用。

## 函数签名

```python
from pyhotpants import hotpants

diff, noise, conv, mask, stats = hotpants(
    inim, tmplim,              # 必需：科学图像、模板图像（numpy float32 二维数组）
    logger=None,               # logging.Logger 实例，None 时使用模块级 logger

    # 可选输入
    tni=None, ini=None,        # 模板/科学噪声图像
    tmi=None, imi=None,        # 模板/科学掩膜图像

    # 阈值与增益
    tu=25000., tuk=None, tl=0., tg=1., tr=0., tp=0.,
    iu=25000., iuk=None, il=0., ig=1., ir=0., ip=0.,

    # 核参数
    r=10, ko=2, bgo=1,
    ng=3, ng_deg=None, ng_sig=None,
    pca=None,

    # 区域与 stamp
    nrx=1, nry=1, rf=None,
    nsx=10, nsy=10, ssf=None,
    afssc=1, nss=3, rss=15, uss=0,

    # 拟合控制
    ft=20.0, sft=0.5, nft=0.1,
    ssig=3.0, ks=2.0, kfm=0.99,
    mins=1.0, mous=1.0,
    fi=1e-30, fin=0.,

    # 卷积控制
    c='b', n='t', fom='v',
    sconv=0, okn=0, convvar=0,

    # 杂项
    v=1, kcs=0, savexy=0,
    dump_dir=None,
)
```

## 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `diff` | float32 2D | 差分图像 |
| `noise` | float32 2D | 噪声图像 |
| `conv` | float32 2D | 卷积图像 |
| `mask` | int32 2D | 掩膜图像 |
| `stats` | list[dict] | 每个 region 的统计信息 |

## stats 字段

```python
for i, s in enumerate(stats):
    print(f"Region {i}:")
    print(f"  卷积方向: {'模板' if s['conv_tmpl'] else '科学'}")
    print(f"  核总和: {s['sum_kernel']:.4f}")
    print(f"  Sub-stamp 均值 sigma: {s['sub_mean_sig']:.4f}")
    print(f"  Sub-stamp 散度 sigma: {s['sub_scatter_sig']:.4f}")
    print(f"  Sub-stamp 最终均值 sigma: {s['sub_final_mean_sig']:.4f}")
    print(f"  X2NORM: {s['x2norm']:.4f}")
    print(f"  --- GOOD pixels ---")
    print(f"    diff  mean/median/mode/sd: {s['diff_good_mean']:.2f}/{s['diff_good_median']:.2f}/{s['diff_good_mode']:.2f}/{s['diff_good_sd']:.2f}")
    print(f"    noise mean/median/mode: {s['noise_good_mean']:.2f}/{s['noise_good_median']:.2f}/{s['noise_good_mode']:.2f}")
    print(f"  --- OK pixels ---")
    print(f"    diff  mean/median/mode/sd: {s['diff_ok_mean']:.2f}/{s['diff_ok_median']:.2f}/{s['diff_ok_mode']:.2f}/{s['diff_ok_sd']:.2f}")
    print(f"    noise mean/median/mode: {s['noise_ok_mean']:.2f}/{s['noise_ok_median']:.2f}/{s['noise_ok_mode']:.2f}")
    print(f"  diffrat: {s['diffrat']:.4f}")
```

完整 stats 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `conv_tmpl` | int | 卷积方向（1=模板，0=科学） |
| `sum_kernel` | float | 核总和 |
| `sub_mean_sig` | float | Sub-stamp 均值 sigma |
| `sub_scatter_sig` | float | Sub-stamp 散度 sigma |
| `sub_final_mean_sig` | float | Sub-stamp 最终均值 sigma |
| `sub_final_scatter_sig` | float | Sub-stamp 最终散度 sigma |
| `x2norm` | float | 卡方归一化值 |
| `nx2norm` | int | 卡方归一化像素数 |
| `diff_good_mean` | float | GOOD 像素差分均值 |
| `diff_good_median` | float | GOOD 像素差分中位数 |
| `diff_good_mode` | float | GOOD 像素差分众数 |
| `diff_good_sd` | float | GOOD 像素差分标准差 |
| `noise_good_mean` | float | GOOD 像素噪声均值 |
| `noise_good_median` | float | GOOD 像素噪声中位数 |
| `noise_good_mode` | float | GOOD 像素噪声众数 |
| `diff_ok_mean` | float | OK 像素差分均值 |
| `diff_ok_median` | float | OK 像素差分中位数 |
| `diff_ok_mode` | float | OK 像素差分众数 |
| `diff_ok_sd` | float | OK 像素差分标准差 |
| `noise_ok_mean` | float | OK 像素噪声均值 |
| `noise_ok_median` | float | OK 像素噪声中位数 |
| `noise_ok_mode` | float | OK 像素噪声众数 |
| `diffrat` | float | Emperical/Expected noise ratio |

## C 命令行参数对照表

| C 参数 | Python 参数 | 默认值 | 说明 |
|--------|-------------|--------|------|
| `-inim` | `inim` | 必需 | 科学图像 |
| `-tmplim` | `tmplim` | 必需 | 模板图像 |
| `-tni` | `tni` | None | 模板噪声图像 |
| `-ini` | `ini` | None | 科学噪声图像 |
| `-tmi` | `tmi` | None | 模板掩膜 |
| `-imi` | `imi` | None | 科学掩膜 |
| `-tu` | `tu` | 25000 | 模板上阈值 |
| `-tuk` | `tuk` | None(=tu) | 模板核上阈值 |
| `-tl` | `tl` | 0 | 模板下阈值 |
| `-tg` | `tg` | 1 | 模板增益 |
| `-tr` | `tr` | 0 | 模板读噪声 |
| `-tp` | `tp` | 0 | 模板基底 |
| `-iu` | `iu` | 25000 | 科学上阈值 |
| `-iuk` | `iuk` | None(=iu) | 科学核上阈值 |
| `-il` | `il` | 0 | 科学下阈值 |
| `-ig` | `ig` | 1 | 科学增益 |
| `-ir` | `ir` | 0 | 科学读噪声 |
| `-ip` | `ip` | 0 | 科学基底 |
| `-r` | `r` | 10 | 核半宽 |
| `-ko` | `ko` | 2 | 核空间阶数 |
| `-bgo` | `bgo` | 1 | 背景空间阶数 |
| `-ng` | `ng` | 3 | 高斯基函数个数 |
| `-nrx` | `nrx` | 1 | x 方向 region 数 |
| `-nry` | `nry` | 1 | y 方向 region 数 |
| `-nsx` | `nsx` | 10 | x 方向 stamp 数 |
| `-nsy` | `nsy` | 10 | y 方向 stamp 数 |
| `-nss` | `nss` | 3 | 每 stamp 子 stamp 数 |
| `-rss` | `rss` | 15 | 子 stamp 半宽 |
| `-afssc` | `afssc` | 1 | 自动寻找子 stamp 中心 |
| `-ft` | `ft` | 20.0 | 核拟合阈值 |
| `-sft` | `sft` | 0.5 | 缩放拟合阈值 |
| `-nft` | `nft` | 0.1 | 最小好 stamp 比例 |
| `-ssig` | `ssig` | 3.0 | 统计 sigma 裁剪 |
| `-ks` | `ks` | 2.0 | 核 sigma 裁剪 |
| `-kfm` | `kfm` | 0.99 | 核比例掩膜 |
| `-fi` | `fi` | 1e-30 | 填充值 |
| `-fin` | `fin` | 0 | 噪声填充值 |
| `-c` | `c` | 'b' | 卷积方向 (t/i/b) |
| `-n` | `n` | 't' | 归一化 (t/i/u) |
| `-v` | `v` | 1 | 详细程度 |
| `-savexy` | `savexy` | 0 | 保存 stamp 坐标 |

## 示例

### 基本用法

```python
from astropy.io import fits
import numpy as np
from pyhotpants import hotpants

tmpl = fits.getdata('template.fits').astype(np.float32)
sci = fits.getdata('science.fits').astype(np.float32)

diff, noise, conv, mask, stats = hotpants(inim=sci, tmplim=tmpl)
```

### 带 logger

```python
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('hotpants')
logging.getLogger('numba').setLevel(logging.WARNING)

diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    tg=2.3, tr=5.0, ig=1.8, ir=4.5,
    logger=logger,
)
```

### 调整核参数

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    r=12, ko=2, bgo=2, c='t', n='t',
)
```

### 多 region 处理

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    nrx=2, nry=2, nsx=15, nsy=15,
)
```

### 自定义 region

```python
regions = [(100, 500, 100, 500), (500, 900, 500, 900)]
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl, rf=regions,
)
```

## 测试

```bash
cd /path/to/hotpants
PYTHONPATH=. python3 examples/test_precision.py
```

## 注意事项

1. 输入图像必须为 `float32` 类型的二维 numpy 数组
2. 模板和科学图像尺寸可以不同，输出尺寸取两者的最大值
3. `c='b'` 时程序自动选择卷积方向

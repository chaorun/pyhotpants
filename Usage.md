# pyhotpants 使用指南

## 函数签名

```python
diff, noise, conv, mask, stats = hotpants(
    inim, tmplim,              # 必需：科学图像、模板图像（numpy float32 二维数组）
    # 可选输入
    tni=None, ini=None,        # 模板/科学噪声图像
    tmi=None, imi=None,        # 模板/科学掩膜图像
    # 阈值与增益
    tu=25000., tuk=None, tl=0., tg=1., tr=0., tp=0.,   # 模板：上阈值/核上阈值/下阈值/增益/读噪声/基底
    iu=25000., iuk=None, il=0., ig=1., ir=0., ip=0.,   # 科学：同上
    # 核参数
    r=10, ko=2, bgo=1,         # 核半宽/核阶数/背景阶数
    ng=3, ng_deg=None, ng_sig=None,  # 高斯个数/各高斯阶数/各高斯sigma
    pca=None,                  # PCA 基函数
    # 区域与stamp
    nrx=1, nry=1, rf=None,    # x/y 方向 region 数/自定义 region 列表
    nsx=10, nsy=10,            # x/y 方向 stamp 数
    ssf=None,                  # 自定义 stamp 中心坐标列表
    afssc=1, nss=3, rss=15,   # 自动寻找 stamp/每 stamp 子stamp数/子stamp半宽
    uss=0,                     # 使用整个 stamp（非子stamp）
    # 拟合控制
    ft=20.0, sft=0.5, nft=0.1,     # 核拟合阈值/缩放拟合阈值/最小好stamp比例
    ssig=3.0, ks=2.0, kfm=0.99,    # 统计sigma/核sigma裁剪/核比例掩膜
    mins=1.0, mous=1.0,             # 核传播掩膜参数
    fi=1e-30, fin=0.,               # 填充值/噪声填充值
    # 卷积控制
    c='b', n='t', fom='v',    # 卷积方向/归一化方式/质量评估方法
    sconv=0, okn=0, convvar=0, # 同向卷积/重缩放/卷积方差
    # 杂项
    v=1, kcs=0,                # 详细程度/核中心步长
    savexy=0,                  # 保存stamp坐标到stats
    dump_dir=None,             # dump 目录（输入/输出 dump，None=不 dump）
)
```

## 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `diff` | float32 二维数组 | 差分图像 |
| `noise` | float32 二维数组 | 噪声图像 |
| `conv` | float32 二维数组 | 卷积图像 |
| `mask` | int32 二维数组 | 掩膜图像 |
| `stats` | list[dict] | 每个 region 的统计信息 |

## C 命令行参数对照表

| C 参数 | Python 参数 | 默认值 | 说明 |
|--------|-------------|--------|------|
| `-inim` | `inim` | (必需) | 科学图像 |
| `-tmplim` | `tmplim` | (必需) | 模板图像 |
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
| `-r` | `r` | 10 | 核半宽（像素） |
| `-ko` | `ko` | 2 | 核空间阶数 |
| `-bgo` | `bgo` | 1 | 背景空间阶数 |
| `-ng` | `ng` | 3 | 高斯基函数个数 |
| `-nrx` | `nrx` | 1 | x 方向 region 数 |
| `-nry` | `nry` | 1 | y 方向 region 数 |
| `-nsx` | `nsx` | 10 | x 方向 stamp 数 |
| `-nsy` | `nsy` | 10 | y 方向 stamp 数 |
| `-nss` | `nss` | 3 | 每 stamp 子stamp 数 |
| `-rss` | `rss` | 15 | 子stamp 半宽 |
| `-afssc` | `afssc` | 1 | 自动寻找子stamp中心 |
| `-ft` | `ft` | 20.0 | 核拟合阈值 |
| `-sft` | `sft` | 0.5 | 缩放拟合阈值 |
| `-nft` | `nft` | 0.1 | 最小好stamp比例 |
| `-ssig` | `ssig` | 3.0 | 统计sigma裁剪 |
| `-ks` | `ks` | 2.0 | 核sigma裁剪 |
| `-kfm` | `kfm` | 0.99 | 核比例掩膜 |
| `-fi` | `fi` | 1e-30 | 填充值 |
| `-fin` | `fin` | 0 | 噪声填充值 |
| `-c` | `c` | 'b' | 卷积方向 (t/i/b) |
| `-n` | `n` | 't' | 归一化 (t/i/u) |
| `-v` | `v` | 1 | 详细程度 |
| `-savexy` | `savexy` | 0 | 保存stamp坐标 |
| `-uss` | `uss` | 0 | 使用整个stamp |
| `-sconv` | `sconv` | 0 | 同向卷积 |
| `-okn` | `okn` | 0 | 重缩放 |

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

### 指定增益和读噪声

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    tg=2.3, tr=5.0,     # 模板增益=2.3, 读噪声=5.0
    ig=1.8, ir=4.5,      # 科学增益=1.8, 读噪声=4.5
)
```

### 调整核参数

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    r=12,       # 核半宽 12 像素
    ko=2,       # 核空间阶数 2
    bgo=2,      # 背景空间阶数 2
    c='t',      # 强制卷积模板
    n='t',      # 按模板归一化
)
```

### 多 region 处理

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    nrx=2, nry=2,   # 2x2 = 4 个 region
    nsx=15, nsy=15,  # 每 region 15x15 个 stamp
)
```

### 自定义 region

```python
regions = [
    (100, 500, 100, 500),   # (xmin, xmax, ymin, ymax)
    (500, 900, 500, 900),
]
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    rf=regions,
)
```

### 使用输入/输出 dump

```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    dump_dir='debug_dump',  # 在此目录生成 py_input.bin 和 py_output.bin
)
```

dump 文件为二进制格式，每个变量按 `[name_len(int32)][name][data_len(int64)][data]` 格式顺序存储，可用于与 C 端输出交叉对比。

## stats 字段说明

```python
for i, s in enumerate(stats):
    print(f"Region {i}:")
    print(f"  卷积方向: {'模板' if s['conv_tmpl'] else '科学'}")
    print(f"  核总和: {s['sum_kernel']:.4f}")
    print(f"  均值sigma: {s['mean_sig']:.4f}")
    print(f"  散度sigma: {s['scatter_sig']:.4f}")
    print(f"  最终均值sigma: {s['final_mean_sig']:.4f}")
    print(f"  差分均值: {s['diff_mean']:.4f}")
    print(f"  差分标准差: {s['diff_sd']:.4f}")
```

## 注意事项

1. 输入图像必须为 `float32` 类型的二维 numpy 数组
2. 模板和科学图像尺寸可以不同，输出尺寸取两者的最大值
3. `nrx`/`nry` 不宜过大（region 太小会导致拟合不稳定）
4. `c='b'` 时程序自动选择卷积方向，`c='t'` 强制卷积模板，`c='i'` 强制卷积科学图像

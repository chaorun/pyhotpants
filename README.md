# pyhotpants — hotpants 图像差分的 Python/Cython 封装

## 项目概述

将 hotpants（天文图像差分工具）从纯 C 命令行程序重构为可通过 Python 直接调用的库。核心计算代码完全保留原始 C 实现，通过 Cython 封装为 Python 扩展模块。所有数据通过 numpy 数组在 Python 层传入传出，无需文件 I/O。

## 重构历程

| 阶段 | 内容 |
|------|------|
| Phase 1-6 | 43 个 C 函数消除全局变量，改为显式参数传递 |
| Phase 7 | I/O 函数（getKernelInfo/readKernel）参数化 |
| Phase 8 | vargs 改为 84 个指针参数输出 |
| Phase 9 | main 中所有全局变量改为局部变量 |
| main 拆分 | main.c（FITS I/O）+ hotpants_compute.c（纯计算） |
| globals 分离 | hotpants_globals.h（纯 struct）替代 globals.h |
| Cython 化 | .pxd + .pyx + setup.py，numpy 数组直接传入 C 层 |

## 文件结构

```
pyhotpants/                  Python 包
  __init__.py                from .chotpants import hotpants
  chotpants.pxd              C 结构体/函数声明
  chotpants.pyx              Python 包装层
  setup.py                   Cython 编译配置
  README.md                  本文件
  Usage.md                   详细 API 文档
  csrc/                      C 核心源码
    hotpants_compute.c/h     纯计算函数（无 FITS I/O）
    alard.c                  核心算法（kernel/stamp/convolve）
    functions.c/h            辅助函数
    hotpants_globals.h       struct 定义（无全局变量）
    defaults.h               宏定义/默认值

raw_code/                    C 源代码（重构后，含命令行入口）
  main.c                     命令行入口（FITS I/O）
  vargs.c                    命令行参数解析
  replay.c                   从 bin 文件重放 hotpants_compute（调试用）
  globals.h                  全局变量（仅 extractkern/maskim 使用）
  Makefile                   C 编译配置
  （其余 .c/.h 与 csrc/ 同步）
```

## 依赖

- Python 3.8+
- Cython
- NumPy
- cfitsio（头文件 + 库）
- gcc

## 编译

### 编译 Python 扩展

```bash
cd pyhotpants
python setup.py build_ext --inplace
```

编译成功后会生成 `chotpants.cpython-*-darwin.so`（macOS）或 `chotpants.cpython-*-linux.so`（Linux）。

macOS (Homebrew) 下 cfitsio 路径已在 setup.py 中配置为 `/opt/homebrew`。其他平台需修改 `include_dirs` 和 `library_dirs`。

### 编译 C 命令行版本

```bash
cd raw_code
make clean && make
```

## 快速开始

```python
from astropy.io import fits
import numpy as np
from pyhotpants import hotpants

tmpl = fits.getdata('template.fits').astype(np.float32)
sci = fits.getdata('science.fits').astype(np.float32)

diff, noise, conv, mask, stats = hotpants(
    inim=sci,
    tmplim=tmpl,
    c='t',      # 卷积模板
    n='t',      # 归一化到模板
    r=10,       # 核半宽
    ko=2,       # 核空间阶数
    bgo=1,      # 背景阶数
)

fits.writeto('diff.fits', diff, overwrite=True)
```

### 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| diff | ndarray float32 | 差分图像 |
| noise | ndarray float32 | 噪声图像 |
| conv | ndarray float32 | 卷积图像 |
| mask | ndarray int32 | 掩码图像 |
| stats | list[dict] | 每个 region 的统计信息 |

stats 中每个 dict 包含：conv_tmpl, sum_kernel, mean_sig, scatter_sig, final_mean_sig, final_scatter_sig, x2norm, nx2norm, diff_mean, diff_sd, noise_mean, diff_mean_ok, diff_sd_ok, noise_mean_ok, diffrat

## 详细 API 文档

完整参数对照表、示例代码参见 [Usage.md](Usage.md)。

## 测试

```bash
python -c "
from astropy.io import fits
from pyhotpants import hotpants
import numpy as np
t = fits.getdata('testdata/templ1K.fit').astype('f4')
s = fits.getdata('testdata/input1K.fit').astype('f4')
d,n,c,m,st = hotpants(inim=s, tmplim=t, c='t', n='t', v=0)
print(f'diff: {d.shape} range=[{d.min():.1f}, {d.max():.1f}]')
print(f'kernel_sum={st[0][\"sum_kernel\"]:.4f}')
"
```

## 调试工具

### 输入/输出 dump

Python 端通过 `dump_dir` 参数开启：
```python
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    dump_dir='debug_dump',  # 生成 py_input.bin 和 py_output.bin
)
```

C 端通过环境变量 `HOTPANTS_IO_DUMP` 开启：
```bash
HOTPANTS_IO_DUMP=debug_dump raw_code/hotpants -inim sci.fits -tmplim tmpl.fits -outim diff.fits
# 生成 c_input.bin 和 c_output.bin
```

dump 文件为二进制格式，每个变量按 `[name_len(int32)][name][data_len(int64)][data]` 顺序存储，两端格式一致可直接 `cmp` 对比。

### replay 工具

从 bin 文件读入参数直接调用 hotpants_compute，可复现任意输入：
```bash
cd raw_code && make && gcc -funroll-loops -O3 -ansi -std=c99 -I/opt/homebrew/include -c replay.c && gcc replay.o alard.o functions.o hotpants_compute.o -o replay -L/opt/homebrew/lib -lm -lcfitsio
./replay input.bin output.bin
```

## 已知限制

1. 编译时仍需链接 cfitsio（functions.c 中有未使用的 FITS 相关函数残留），运行时不依赖 FITS I/O
2. -sht（16bit 输出）在 Python 端不适用（Python 直接操作 float32 数组）
3. -ki（读入已有 kernel 解）功能在 compute 中未启用
4. PCA 模式未经充分测试
5. fwStamp 计算中 nRegX/nRegY 用 sqrt(nR) 近似（对称 region 正确，非对称可能有偏差）
6. setuptools 自动注入的 `-fno-strict-overflow` 编译参数在极端参数（大量微小 region）下可能导致浮点累积差异；setup.py 已通过 extra_compile_args 统一编译参数规避此问题

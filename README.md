# pyhotpants — hotpants 图像差分的 Python 实现

## 项目概述

hotpants 天文图像差分算法的 Python 实现，支持多种计算后端。核心计算代码完全保留 C 实现，numba JIT 编译提供纯 Python 路径，Cython 封装提供 C 原生路径。所有数据通过 numpy 数组传入传出，无需文件 I/O。

## 后端

| 后端 | 精度 | 平台 | 说明 |
|------|:---:|------|------|
| `numba` | float64 | CPU 通用 | 默认后端，零编译依赖 |
| `float32` | float32 | CPU 通用 | 全程 float32，适合 GPU 移植基准 |
| `metal` | float32 | Apple Silicon GPU | 需 macOS + MLX 包 |
| `cython` | float64 | CPU (调用 C 库) | 调用 C 原版核心，需编译 |
| `cuda64` | — | NVIDIA GPU | 🚧 待开发 |
| `agx_orin` | — | Jetson AGX Orin | 🚧 待开发 |

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
| numba 化 | 纯 Python 实现，numba JIT 编译，零编译依赖 |
| Metal 化 | Apple MLX GPU 加速 |
| Backend 统一 | 多后端统一路由（numba/float32/metal/cython/cuda64/agx_orin） |

## 文件结构

```
pyhotpants/                   Python 包
  __init__.py                 from .interface import hotpants
  interface.py                backend 路由
  functions.py                共用工具函数（常量、sigma-clip、StampsArray 等）
  setup.py                    安装脚本
  README.md                   本文件
  Usage.md                    详细 API 文档
  COPIYNG / LICENSE

  numba/                      Numba CPU float64 后端
    hotpants.py               hotpants 入口 + region 调度函数
    alard.py                  核拟合、空间卷积、显著性评估核心算法
  float32/                    Numba CPU float32 后端
    hotpants.py
    alard.py
  metal/                      Apple MLX GPU 后端
    hotpants.py               patching 编排层
    hotpants_mlx.py           工具函数 MLX 实现
    hotpants_mlx_v2.py        fill_stamp + spatial_convolve GPU
    hotpants_mlx_v3.py        build_matrix + fit_kernel GPU
  cython/                     Cython 后端（需编译）
    chotpants.pxd             C 结构体/函数声明
    chotpants.pyx             Python 包装层
    setup.py                  Cython 编译配置
    csrc/                     C 核心源码
      hotpants_compute.c/h    纯计算函数（无 FITS I/O）
      alard.c                 核心算法（kernel/stamp/convolve）
      functions.c/h           辅助函数
      hotpants_globals.h      struct 定义（无全局变量）
      defaults.h              宏定义/默认值
  cuda64/                     CUDA — 待开发
  agx_orin_cuda/              Jetson AGX Orin — 待开发

  deprecated/                 弃用文件（Cython 时代残留）
    chotpants.pyx / .c / .so
    numutils.py
    csrc/                     C 源码（历史参考）

raw_code/                     C 源代码（参考用，重构后）
  main.c                      命令行入口（FITS I/O）
  vargs.c                     命令行参数解析
  replay.c                    从 bin 文件重放 hotpants_compute（调试用）
  globals.h                   全局变量（仅 extractkern/maskim 使用）
  Makefile                    C 编译配置
```

## 依赖

- Python 3.8+
- NumPy
- Numba
- SciPy

`cython` 后端额外需要：Cython、cfitsio（头文件 + 库）、gcc。
`metal` 后端额外需要：mlx。
`cuda64` 和 `agx_orin` 后端待开发，暂无依赖。

## 安装

### 纯 Python 后端（numba / float32）

```bash
python3 setup.py install
```

### Metal GPU 后端

```bash
python3 setup.py install --gpu=metal
```

### Cython 后端

需先安装 cfitsio（macOS: `brew install cfitsio`），再编译：

```bash
# 方式 A：通过主 setup.py
python3 setup.py install --cython

# 方式 B：直接编译 cython 子包
cd cython && CFITSIO_LIB=/opt/homebrew/lib python3 setup.py build_ext --inplace
```

`CFITSIO_LIB` 和 `CFITSIO_INC` 环境变量可指定 cfitsio 路径，不设置时默认为 `/usr/local/lib` 和 `/usr/local/include`。

### 组合安装

```bash
python3 setup.py install --cython --gpu=cuda64
```

### 直接使用（无需安装）

```bash
export PYTHONPATH=/path/to/hotpants:$PYTHONPATH
```

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

# 默认 numba 后端
diff, noise, conv, mask, stats = hotpants(
    inim=sci, tmplim=tmpl,
    c='t', n='t', r=10, ko=2, bgo=1,
)

# 切换后端
diff, noise, conv, mask, stats = hotpants(
    backend='float32',        # 'numba' | 'float32' | 'metal' | 'cython'
    inim=sci, tmplim=tmpl,
    c='t', n='t', r=10, ko=2, bgo=1,
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

## 性能

1K×1K 单 region，ko=2，bgo=2，nsx=10，nsy=10，Apple M 系列：

| 后端 | nrx=1 | nrx=2 | nrx=3 |
|------|:---:|:---:|:---:|
| C 原版 | 3.39s | 4.97s | — |
| numba (f64) | 7.88s | 3.16s | 9.08s |
| float32 | ~8.0s | — | — |
| metal | ~8.8s | — | — |

> nrx=2 时 Python 耗时异常偏快（< C），可能与 numba jit 函数中 build_matrix_jit/build_scprod_jit 的 `@numba.jit` 装饰器被注释有关，待排查。

## C vs Python 精度对比

### 1K 测试图像（1601×1601）

| 参数 | diffOut max | convOut max | noiseOut max | maskOut |
|------|:--:|:--:|:--:|:--:|
| nrx=1 | 0.15 | 0.15 | 0.0005 | **EXACT MATCH** |
| nrx=2 | 478 | 737 | 75.7 | 5 pixels |
| nrx=3 | 17,200 | 14,300 | 167 | 85 pixels |

### 14K 测试图像（10654×14206）

| 参数 | 偏差分布 |
|------|---------|
| nrx=2 | 右下角 region 偏差 ~710，其余 3 个正常 |
| nrx=3 | 6/9 个 region 有偏差（90~1520），3 个正常（~2-5） |

### 根因分析

通过 fitKernel 逐迭代 dump 对比（C vs Python iter 0 的 matrix/scprod/kernelSol），确认：

1. **fill_stamp → build_matrix 阶段已存在 ~1e-5 的相对矩阵差异**（C float32 卷积累积器 vs Python float64、numba parallel 累加顺序、不同编译器的浮点优化路径）
2. 差异量级极小（1e-5），但**受 region 矩阵条件数放大**——条件数差的 region 放大至数百~数千的偏差，条件数好的 region 保持个位数偏差
3. 这不是可修复的特定 bug，而是跨语言（C vs numba-Python）复刻的数值精度天花板
4. **14K 大图同样存在此问题**，排除了"1K 图样本不足"的早期假说

### 结论

- **nrx=1（全图单 region）精度可接受，可用于科学计算**
- **多 region 模式下存在不可预测的区域性偏差，精度不可保证**
- 偏差根因为跨语言浮点路径差异，无法通过局部修改消除

## 详细 API 文档

完整参数对照表、示例代码参见 [Usage.md](Usage.md)。

## 测试

```python
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

或：

```bash
cd /path/to/hotpants
PYTHONPATH=. python3 tmp/precision_test.py
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


## 已知限制

1. 编译时仍需链接 cfitsio（functions.c 中有未使用的 FITS 相关函数残留），运行时不依赖 FITS I/O
2. -sht（16bit 输出）在 Python 端不适用（Python 直接操作 float32 数组）
3. -ki（读入已有 kernel 解）功能在 compute 中未启用
4. PCA 模式未经充分测试
5. fwStamp 计算中 nRegX/nRegY 用 sqrt(nR) 近似（对称 region 正确，非对称可能有偏差）
6. setuptools 自动注入的 `-fno-strict-overflow` 编译参数在极端参数（大量微小 region）下可能导致浮点累积差异；setup.py 已通过 extra_compile_args 统一编译参数规避此问题

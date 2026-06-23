"""
hotpants_mlx_full.py — 全 MLX 计算 + 显式 patching 编排层
策略：patching origin_f32 的计算函数为 MLX 版本，编排层保留 origin。
这不是"monkey-patch hack"，而是显式的模块级替换。Python 的 import 语义保证了
hotpants 模块内通过全局命名空间调用的函数会使用 patched 版本。
"""
import numpy as np
import sys, time, logging
from typing import List, Dict, Tuple, Optional

# === 导入所有 MLX 计算函数 ===
from .hotpants_mlx_v2 import fill_stamp_kernel_mlx_v2, spatial_convolve_mlx_v2
from .hotpants_mlx_v3 import fit_kernel_mlx_v3 as _fit_kernel_mlx
from .hotpants_mlx import (sigma_clip_mlx, get_noise_stats3_mlx, make_noise_image4_mlx,
                           buildAllKernels_mlx, get_stamp_sig_mlx, get_stamp_sig_batch_mlx,
                           background_loop_mlx, check_again_mlx,
                           get_final_stamp_sig_mlx)

# === 导入 origin 编排层模块 ===
# 先导入 float32 backend 的 alard 和 functions（此时还没 patch）
from ..float32 import alard as _al
from .. import functions as _fn

# === 显式 patching：替换计算函数 ===
# alard 层
_al.fill_stamp_numba_kernel_local = fill_stamp_kernel_mlx_v2
_al.spatial_convolve_jit_kernel = spatial_convolve_mlx_v2
_al.fit_kernel_numpy = _fit_kernel_mlx
_al.background_loop_jit = background_loop_mlx
_al.buildAllKernels = buildAllKernels_mlx
_al.get_stamp_sig_jit = get_stamp_sig_mlx
_al.get_stamp_sig_batch_jit = get_stamp_sig_batch_mlx
_al.get_final_stamp_sig_numpy = get_final_stamp_sig_mlx
_al.check_again_numpy = check_again_mlx
# 注意：build_matrix_numpy, build_scprod_numpy, make_kernel_numpy 等
# 由 fit_kernel_mlx_v3 内部直接调用 MLX 版本，不需要 patching

# functions 层
_fn.sigma_clip_numpy = sigma_clip_mlx
_fn.get_noise_stats3_numpy = get_noise_stats3_mlx
_fn.make_noise_image4_numpy = make_noise_image4_mlx

# === 导入 hotpants 模块（此时 alard/functions 已被 patched） ===
import importlib
_hp = importlib.import_module('..float32.hotpants', package=__package__)

# 同步 hotpants 模块内的引用（from .alard import xxx 的局部变量）
_hp.fill_stamp_numba_kernel_local = fill_stamp_kernel_mlx_v2
_hp.background_loop_jit = background_loop_mlx
_hp.get_final_stamp_sig_numpy = get_final_stamp_sig_mlx
_hp.get_stamp_sig_jit = get_stamp_sig_mlx
_hp.get_stamp_sig_batch_jit = get_stamp_sig_batch_mlx
_hp.make_noise_image4_numpy = make_noise_image4_mlx
_hp.get_noise_stats3_numpy = get_noise_stats3_mlx
_hp.sigma_clip_numpy = sigma_clip_mlx
_hp.fit_kernel_numpy = _al.fit_kernel_numpy
_hp.fill_stamp_numba = _al.fill_stamp_numba
_hp.spatial_convolve_fast_numpy = _al.spatial_convolve_fast_numpy
_hp.check_stamps_numpy = _al.check_stamps_numpy
_hp.make_kernel_numpy = _al.make_kernel_numpy
_hp.get_kernel_vec_numpy = _al.get_kernel_vec_numpy

# === 公开的 hotpants 入口 ===
hotpants = _hp.hotpants

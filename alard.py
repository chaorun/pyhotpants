import numpy as np
import numba
from typing import List, Dict, Tuple, Optional

from .functions import MAXVAL, sigma_clip_numpy, get_stamp_stats3_numpy


# 预计算所有 kcStep 块锚点的空间可变卷积核，返回 (allKernels, nstepsX, nstepsY)
def buildAllKernels(kernelSol: np.ndarray, kernelVec2d: np.ndarray, nCompKer: int, kerOrder: int, fwKernel: int, hwKernel: int, kcStep: int, rPixX: int, rPixY: int, xSize: int, ySize: int) -> Tuple[np.ndarray, int, int]:
    """预计算所有 kcStep 块的卷积核矩阵。
    kernelSol: 核多项式系数向量 (长度 nCompKer+2)
    kernelVec2d: 核基函数 2D 矩阵 (nCompKer × fwSq)
    nCompKer: 核分量数
    kerOrder: 空间可变多项式阶数
    fwKernel: 核全宽; hwKernel: 核半宽
    kcStep: 核缓存步长
    rPixX, rPixY: region 半宽归一化分母
    xSize, ySize: 图像尺寸
    返回 (allKernels, nstepsX, nstepsY): 预计算的所有块核矩阵、X/Y方向块数
    """
    # fwSq = fwKernel * fwKernel
    halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
    nstepsX = int(np.ceil(xSize / kcStep))
    nstepsY = int(np.ceil(ySize / kcStep))
    nBlocks = nstepsX * nstepsY

    i0Arr = np.arange(nstepsX) * kcStep + hwKernel
    j0Arr = np.arange(nstepsY) * kcStep + hwKernel
    iGrid, jGrid = np.meshgrid(i0Arr, j0Arr, indexing='xy')
    xi = (iGrid + hwKernel).ravel().astype(np.float64)
    yi = (jGrid + hwKernel).ravel().astype(np.float64)
    xf = (xi - halfX) / halfX
    yf = (yi - halfY) / halfY

    kernelCoeffs = np.zeros((nBlocks, nCompKer), dtype=np.float64)
    kernelCoeffs[:, 0] = kernelSol[1]

    k = 2
    for ig in range(1, nCompKer):
        coeff = np.zeros(nBlocks, dtype=np.float64)
        ax = np.ones(nBlocks, dtype=np.float64)
        for ix in range(kerOrder + 1):
            ay = np.ones(nBlocks, dtype=np.float64)
            for iy in range(kerOrder - ix + 1):
                coeff += kernelSol[k] * ax * ay
                k += 1
                ay *= yf
            ax *= xf
        kernelCoeffs[:, ig] = coeff

    allKernels = kernelCoeffs @ kernelVec2d

    return allKernels, nstepsX, nstepsY

# numba jit 空间域卷积核，按 kcStep 块遍历像素做卷积 + variance + mask 传播
@numba.jit(nopython=True, parallel=True)
def spatial_convolve_jit_kernel(
    image: np.ndarray, variance: np.ndarray, cMask: np.ndarray,
    cRdata: np.ndarray, vData: np.ndarray, mRData: np.ndarray,
    kernelSol: np.ndarray,
    xSize: int, ySize: int, nCompKer: int, kerOrder: int, fwKernel: int, hwKernel: int,
    kcStep: int, rPixX: int, rPixY: int, kerFracMask: float, dovar: int, convolveVariance: int,
    kernel_vec_2d: np.ndarray,
    allKernels: Optional[np.ndarray] = None,
    nstepsX_in: Optional[int] = None) -> None:
    """按 kcStep 块遍历像素，对每个锚点构造空间可变卷积核（或使用预计算 allKernels），完成空间域卷积、variance 计算和 mask 标记传播。结果原地写入 cRdata/vData/mRData。支持 numba 并行。

    image: 输入图像 1D 数组 (xSize × ySize)
    variance: 输入 variance 1D 数组，供卷积方差计算
    cMask: 输入 mask 1D 数组，用于标记坏像素
    cRdata: 输出卷积结果 1D 数组，原地写入
    vData: 输出 variance 1D 数组，原地写入
    mRData: 输出 mask 1D 数组，原地更新
    kernelSol: 核多项式系数向量
    xSize, ySize: 图像尺寸
    nCompKer: 核分量数
    kerOrder: 核多项式阶数
    fwKernel, hwKernel: 核全宽和半宽
    kcStep: 核缓存步长（块大小）
    rPixX, rPixY: region 半宽归一化分母
    kerFracMask: mask 分数阈值（低于此值的核像素被标记）
    dovar: 是否计算 variance
    convolveVariance: 是否卷积 variance 本身（vs 平方卷积）
    kernel_vec_2d: 核基函数 2D 矩阵
    allKernels: 预计算的所有块核矩阵，None 时在线计算
    nstepsX_in: X 方向块数（allKernels 不为 None 时必须传入）
    """

    fwSq = fwKernel * fwKernel
    FLAG_INPUT_ISBAD = np.int32(0x80)
    FLAG_OUTPUT_ISBAD = np.int32(0x8000)
    FLAG_BAD_CONV = np.int32(0x10)
    FLAG_OK_CONV = np.int32(0x40)
    halfX = 0.5 * rPixX
    halfY = 0.5 * rPixY

    # kernel = np.zeros(fwSq, dtype=np.float64)  # moved inside prange
    # kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)  # moved inside prange

    nsteps_x = int(np.ceil(xSize / kcStep))
    nsteps_y = int(np.ceil(ySize / kcStep))

    for j1 in numba.prange(nsteps_y):
        j0 = j1 * kcStep + hwKernel
        for i1 in range(nsteps_x):
            i0 = i1 * kcStep + hwKernel

            if allKernels is not None:
                kernel = allKernels[j1 * nstepsX_in + i1]
            else:
                # ---- make_kernel(i0+hwKernel, j0+hwKernel) ----
                kernel = np.zeros(fwSq, dtype=np.float64)
                kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
                xi = i0 + hwKernel
                yi = j0 + hwKernel
                xf = (xi - halfX) / halfX
                yf = (yi - halfY) / halfY

                k = 2
                for i1k in range(1, nCompKer):
                    coeff = 0.0
                    ax = 1.0
                    for ix in range(kerOrder + 1):
                        ay = 1.0
                        for iy in range(kerOrder - ix + 1):
                            coeff += kernelSol[k] * ax * ay
                            k += 1
                            ay *= yf
                        ax *= xf
                    kernel_coeffs[i1k] = coeff
                kernel_coeffs[0] = kernelSol[1]

                for ii in range(fwSq):
                    for c in range(nCompKer):
                        kernel[ii] += kernel_coeffs[c] * kernel_vec_2d[c, ii]
                # ---- end make_kernel ----

            for j2 in range(kcStep):
                j = j0 + j2
                if j >= ySize - hwKernel:
                    break
                for i2 in range(kcStep):
                    i = i0 + i2
                    if i >= xSize - hwKernel:
                        break

                    ni = i + xSize * j
                    q = 0.0
                    qv = 0.0
                    aks = 0.0
                    uks = 0.0
                    mbit = np.int32(0)

                    for jc in range(j - hwKernel, j + hwKernel + 1):
                        jk = j - jc + hwKernel
                        for ic in range(i - hwKernel, i + hwKernel + 1):
                            ik = i - ic + hwKernel
                            nc = ic + xSize * jc
                            kk = kernel[ik + jk * fwKernel]

                            q += image[nc] * kk
                            if dovar:
                                if convolveVariance:
                                    qv += variance[nc] * kk * kk
                                else:
                                    qv += variance[nc] * kk * kk
                            mbit |= cMask[nc]
                            aks += abs(kk)
                            if not (cMask[nc] & FLAG_INPUT_ISBAD):
                                uks += abs(kk)

                    cRdata[ni] = q
                    if dovar:
                        vData[ni] = qv

                    mRData[ni] = mRData[ni] | cMask[ni]
                    mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD * np.int32((cMask[ni] & FLAG_INPUT_ISBAD) > 0))

                    if mbit:
                        if aks > 0.0 and (uks / aks) < kerFracMask:
                            mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
                        else:
                            mRData[ni] = mRData[ni] | FLAG_OK_CONV

# 将多项式背景叠加到输出图像上（原地修改 oRData1d）
@numba.jit(nopython=True)
def background_loop_jit(oRData1d: np.ndarray, kernelSol: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int, rPixX: int, rPixY: int, hwKernel: int) -> None:
    """遍历图像有效区域（跳过核半宽的边界像素），将多项式背景叠加到输出图像上。背景系数从 kernelSol 中按 nCompKer 和 kerOrder 定位，对区域内每个像素计算归一化坐标的多项式展开并累加到 oRData1d。

    oRData1d: 输出图像 1D 数组，原地修改
    kernelSol: 核拟合解向量，背景多项式系数存储在其后半段
    nCompKer: 核分量数，用于定位背景系数偏移
    kerOrder: 核多项式阶数，决定背景系数起始位置的计算
    bgOrder: 背景多项式阶数
    rPixX, rPixY: region 像素尺寸，用于像素坐标的归一化
    hwKernel: 核半宽，边界跳过的像素数
    """

    nCompForBG = nCompKer - 1
    ncompBG = nCompForBG * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    halfX = np.float64(0.5 * rPixX)
    halfY = np.float64(0.5 * rPixY)
    for j in range(hwKernel, rPixY - hwKernel):
        yf = (j - halfY) / halfY
        for i in range(hwKernel, rPixX - hwKernel):
            xf = (i - halfX) / halfX
            bg = np.float64(0.0)
            k = 1
            ax = np.float64(1.0)
            for idegx in range(bgOrder + 1):
                ay = np.float64(1.0)
                for idegy in range(bgOrder - idegx + 1):
                    bg += kernelSol[ncompBG + k] * ax * ay
                    k += 1
                    ay *= yf
                ax *= xf
            oRData1d[i + rPixX * j] += bg

@numba.jit(nopython=True)
# 获取单个 stamp 在最终输出图像上的信噪比
def get_final_stamp_sig_numpy(saXss: np.ndarray, saYss: np.ndarray, saSscnt: np.ndarray, si: int, imDiff: np.ndarray, imNoise: np.ndarray, fwKSStamp: int, hwKSStamp: int, rPixX: int, mRData: np.ndarray) -> float:
    """计算单个 stamp 在最终输出差图像上的信噪比。
    saXss, saYss: stamps 的 X/Y 坐标数组 (nS × nKSStamps)
    saSscnt: stamps 的已填充子 stamp 计数
    si: stamp 索引
    imDiff: 差图像 1D 数组
    imNoise: 噪声图像 1D 数组
    fwKSStamp: stamp 子区域全宽; hwKSStamp: 半宽
    rPixX: region 的 X 像素数（用于 1D 索引计算）
    mRData: mask 数据 1D 数组
    返回 sig 值（方差加权平均），若 stamp 无效则返回 -1.0
    """
    
    FLAG = np.int32(0x80)
    xRegion = saXss[si, saSscnt[si]]
    yRegion = saYss[si, saSscnt[si]]
    sig = 0.0
    nsig = 0
    for j in range(fwKSStamp):
        yRegion2 = yRegion - hwKSStamp + j
        for i in range(fwKSStamp):
            xRegion2 = xRegion - hwKSStamp + i
            idx = xRegion2 + rPixX * yRegion2
            idat = imDiff[idx]
            ndat = imNoise[idx]
            if (mRData[idx] & FLAG) != 0:
                continue
            nsig += 1
            sig += (idat * idat) / (ndat * ndat)
    if nsig > 0:
        sig /= nsig
    else:
        sig = -1.0
    return sig

# 在给定像素位置 (xi,yi) 构造空间可变卷积核，返回核元素之和
def make_kernel_numpy(xi: int, yi: int, kernelSol: np.ndarray, rPixX: int, rPixY: int, nCompKer: int, kerOrder: int, fwKernel: int,
                      kernel_vec: np.ndarray, kernel_coeffs: np.ndarray, kernel: np.ndarray) -> float:
    """在给定像素位置 (xi,yi) 构造空间可变卷积核。计算该位置的归一化坐标，对每个核分量展开多项式并乘以 kernelSol 中的系数得到核多项式系数，再将系数与核基函数 kernel_vec 做线性组合得到完整卷积核。

    xi, yi: 像素坐标
    kernelSol: 核多项式系数向量
    rPixX, rPixY: region 半宽归一化分母
    nCompKer: 核分量数
    kerOrder: 空间可变多项式阶数
    fwKernel: 核全宽
    kernel_vec: 核基函数列表 (nCompKer × fwSq)，用于基叠加
    kernel_coeffs: 核系数数组 (nCompKer)，输出此像素处的多项式系数
    kernel: 核数组 (fwSq)，输出此像素处的完整卷积核

    返回 sumKernel: 核元素之和，用于归一化
    """
    k = 2
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
    for i1 in range(1, nCompKer):
        coeff = 0.0
        ax = 1.0
        for ix in range(kerOrder + 1):
            ay = 1.0
            for iy in range(kerOrder - ix + 1):
                coeff += float(kernelSol[k]) * ax * ay
                k += 1
                ay *= yf
            ax *= xf
        kernel_coeffs[i1] = coeff
    kernel_coeffs[0] = float(kernelSol[1])
    fwSq = fwKernel * fwKernel
    for i in range(fwSq):
        kernel[i] = 0.0
    sum_kernel = 0.0
    for i in range(fwSq):
        val = 0.0
        for i1 in range(nCompKer):
            val += float(kernel_coeffs[i1]) * float(kernel_vec[i1][i])
        kernel[i] = val
        sum_kernel += val
    return sum_kernel

# PCA 模式下从 PCA 基生成核向量
def kernel_vector_pca_numpy(n: int, fwKernel: int, PCA: np.ndarray, kernel_vec: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """PCA 模式下从 PCA 基矩阵生成核向量。n=0 时直接取 PCA 基，n>0 时取 PCA 基并减去 kernel_vec[0] 做正交化。

    n: 高斯分量索引（决定使用 PCA 的哪一行）
    fwKernel: 核全宽，也等于 PCA 每行的长度
    PCA: PCA 基矩阵，每行为一个基向量
    kernel_vec: 已生成的核向量列表，n>0 时取其第一个元素做正交化

    返回 vector: 核向量 (fwSq 长度)
    返回 ren: 是否需要后续正交化（n=0 时 0，否则 1）
    """
    vector = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    for i in range(fwKernel):
        for j in range(fwKernel):
            vector[i + fwKernel * j] = float(PCA[n][i + fwKernel * j])
    ren = 0
    if n > 0:
        kernel0 = kernel_vec[0]
        for i in range(fwKernel * fwKernel):
            vector[i] -= kernel0[i]
    return vector, ren

# 为单个高斯分量生成核向量基函数（高斯包络 × 多项式）
def kernel_vector_numpy(n: int, deg_x: int, deg_y: int, ig: int, usePCA: int, fwKernel: int, hwKernel: int,
                        sigma_gauss: List[float], filter_x: np.ndarray, filter_y: np.ndarray, kernel_vec: List[np.ndarray], PCA: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
    """为单个高斯分量生成核向量基函数。对 fwKernel 个像素位置计算高斯包络乘以 X/Y 方向的多项式展开，得到可分离的 X/Y 滤波器，再做外积得到完整核向量。n>0 且 dx=dy=0 时对核向量做正交化。

    n: 高斯分量序号，同时用作 filter_x/filter_y 的写入偏移
    deg_x, deg_y: X/Y 方向多项式的阶数
    ig: 高斯分量索引，用于取 sigma_gauss[ig]
    usePCA: 是否使用 PCA 基（为真时直接调用 kernel_vector_pca_numpy）
    fwKernel, hwKernel: 核全宽和半宽
    sigma_gauss: 各高斯分量的 σ 值列表
    filter_x, filter_y: 预分配的 X/Y 方向滤波器数组，原地写入当前分量的滤波器值
    kernel_vec: 已生成核向量的列表，n>0 时用于正交化
    PCA: PCA 基矩阵，usePCA 为真时传入

    返回 vector: 核向量 (fwSq 长度)
    返回 ren: 是否已做正交化（dx=dy=0 且 n>0 时为 1，否则 0）
    """
    if usePCA:
        vec, ren = kernel_vector_pca_numpy(n, fwKernel, PCA, kernel_vec)
        return vec, ren

    vector = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    dx = (deg_x // 2) * 2 - deg_x
    dy = (deg_y // 2) * 2 - deg_y
    sum_x = 0.0
    sum_y = 0.0
    ren = 0

    sig = float(sigma_gauss[ig])
    for ix in range(fwKernel):
        x = float(ix - hwKernel)
        k = ix + n * fwKernel
        qe = np.exp(-x * x * sig)
        filter_x[k] = qe * (x ** deg_x)
        filter_y[k] = qe * (x ** deg_y)
        sum_x += filter_x[k]
        sum_y += filter_y[k]

    kernel0 = None
    if n > 0:
        kernel0 = kernel_vec[0].copy()

    sum_x = 1.0 / sum_x
    sum_y = 1.0 / sum_y

    if dx == 0 and dy == 0:
        for ix in range(fwKernel):
            filter_x[ix + n * fwKernel] *= sum_x
            filter_y[ix + n * fwKernel] *= sum_y

        for i in range(fwKernel):
            for j in range(fwKernel):
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]

        if n > 0:
            for i in range(fwKernel * fwKernel):
                vector[i] -= kernel0[i]
            ren = 1
    else:
        for i in range(fwKernel):
            for j in range(fwKernel):
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]

    return vector, ren

# 遍历所有高斯分量和多项式阶数，生成完整核向量列表
def get_kernel_vec_numpy(ngauss: int, deg_fixe: List[int], usePCA: int, fwKernel: int, hwKernel: int,
                         sigma_gauss: List[float], filter_x: np.ndarray, filter_y: np.ndarray, PCA: Optional[np.ndarray]) -> List[np.ndarray]:
    """遍历所有高斯分量和多项式阶数组合（嵌套三重循环），对每个组合调用 kernel_vector_numpy 生成核向量，收集为完整列表供后续卷积计算使用。

    ngauss: 高斯分量个数
    deg_fixe: 每个高斯分量的多项式阶数列表
    usePCA: 是否使用 PCA 基
    fwKernel, hwKernel: 核全宽和半宽
    sigma_gauss: 各高斯分量的 σ 值列表，传入 kernel_vector_numpy
    filter_x, filter_y: 预分配的滤波器数组，传入 kernel_vector_numpy 原地写入
    PCA: PCA 基矩阵，usePCA 为真时传入

    返回 kernel_vec: 核向量列表，长度为 nCompKer，每个元素为 fwSq 长度的 np.ndarray
    """
    kernel_vec = []
    nvec = 0
    for ig in range(ngauss):
        for idegx in range(int(deg_fixe[ig]) + 1):
            for idegy in range(int(deg_fixe[ig]) - idegx + 1):
                vec, ren = kernel_vector_numpy(nvec, idegx, idegy, ig, usePCA,
                                                fwKernel, hwKernel, sigma_gauss, filter_x, filter_y, kernel_vec, PCA)
                kernel_vec.append(vec)
                nvec += 1
    return kernel_vec

# 合并构建 matrix + kernelSol（一次 jit 调用完成两个操作，保留用于 check_stamps）
@numba.jit(nopython=True)
def build_both_jit(all_mat: np.ndarray, all_vectors: np.ndarray, all_scprod: np.ndarray,
                    valid_mask: np.ndarray, all_x: np.ndarray, all_y: np.ndarray,
                    wxy: np.ndarray, matrix: np.ndarray, kernelSol: np.ndarray,
                    image_flat: np.ndarray,
                    nS: int, kerOrder: int, fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
                    ncomp: int, ncomp1: int, ncomp2: int, nbg_vec: int, pixStamp: int) -> None:
    """一次 jit 调用完成 build_matrix + build_scprod。先累加 matrix，再累加 kernelSol。"""
    # === build_matrix ===
    for istamp in range(nS):
        if valid_mask[istamp] == 0:
            continue

        for i in range(ncomp):
            i1 = i // ncomp2
            i2 = i - i1 * ncomp2
            for j in range(i + 1):
                j1 = j // ncomp2
                j2 = j - j1 * ncomp2
                matrix[i + 2, j + 2] += wxy[istamp, i2] * wxy[istamp, j2] * all_mat[istamp, i1 + 2, j1 + 2]

        matrix[1, 1] += all_mat[istamp, 1, 1]
        for i in range(ncomp):
            i1 = i // ncomp2
            i2 = i - i1 * ncomp2
            matrix[i + 2, 1] += wxy[istamp, i2] * all_mat[istamp, i1 + 2, 1]

        for ibg in range(nbg_vec):
            ii = ncomp + ibg + 1
            ivecbg = ncomp1 + ibg + 1
            for i1 in range(1, ncomp1 + 1):
                p0 = np.dot(all_vectors[istamp, i1, :], all_vectors[istamp, ivecbg, :])
                for i2 in range(ncomp2):
                    jj = (i1 - 1) * ncomp2 + i2 + 1
                    matrix[ii + 1, jj + 1] += p0 * wxy[istamp, i2]
            p0 = np.dot(all_vectors[istamp, 0, :], all_vectors[istamp, ivecbg, :])
            matrix[ii + 1, 1] += p0
            for jbg in range(ibg + 1):
                q = np.dot(all_vectors[istamp, ivecbg, :], all_vectors[istamp, ncomp1 + jbg + 1, :])
                matrix[ii + 1, ncomp + jbg + 2] += q

    # === build_scprod ===
    so_x = np.empty(pixStamp, dtype=np.int64)
    so_y = np.empty(pixStamp, dtype=np.int64)
    idx = 0
    for xc in range(-hwKSStamp, hwKSStamp + 1):
        for yc in range(-hwKSStamp, hwKSStamp + 1):
            so_x[idx] = xc
            so_y[idx] = yc
            idx += 1

    img_patch = np.empty(pixStamp, dtype=np.float64)
    for istamp in range(nS):
        if valid_mask[istamp] == 0:
            continue

        xi = all_x[istamp]
        yi = all_y[istamp]

        p0 = all_scprod[istamp, 1]
        kernelSol[1] += p0

        for i1 in range(1, ncomp1 + 1):
            p0 = all_scprod[istamp, i1 + 1]
            for i2 in range(ncomp2):
                ii = (i1 - 1) * ncomp2 + i2 + 1
                kernelSol[ii + 1] += p0 * wxy[istamp, i2]

        for k in range(pixStamp):
            img_patch[k] = image_flat[so_x[k] + xi + rPixX * (so_y[k] + yi)]
        for ibg in range(nbg_vec):
            kernelSol[ncomp + ibg + 2] += np.dot(all_vectors[istamp, ncomp1 + ibg + 1, :], img_patch)

# numba jit 构建标量积向量（逐 stamp 累加图像与核向量的点积）
@numba.jit(nopython=True)
def build_scprod_jit(all_vectors: np.ndarray, all_scprod: np.ndarray, valid_mask: np.ndarray, all_x: np.ndarray, all_y: np.ndarray,
                     wxy: np.ndarray, image_flat: np.ndarray, kernelSol: np.ndarray,
                     nS: int, fwKSStamp: int, hwKSStamp: int, rPixX: int,
                     ncomp: int, ncomp1: int, ncomp2: int, nbg_vec: int) -> None:
    """逐 stamp 累加构建标量积向量 kernelSol。遍历所有有效 stamp，对每个 stamp 计算图像像素值与 stamp 向量的加权点积，累加到 kernelSol 中作为最小二乘拟合的右端项。

    all_vectors: 每个 stamp 的向量 (nS × ncomp × fwSq) 或等效切片
    all_scprod: 每个 stamp 的标量积 (nS × nC)，此处仅用于类型编译
    valid_mask: 有效 stamp 掩码 (nS 长度 int32)
    all_x, all_y: 每个 stamp 的当前子 stamp 中心坐标
    wxy: 预计算的 wxy 矩阵 (nS × ncomp2)
    image_flat: 展平的图像 1D 数组 (rPixX × rPixY)
    kernelSol: 输出标量积向量，原地累加修改
    nS: stamps 总数
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX: region 的 X 像素数
    ncomp, ncomp1, ncomp2, nbg_vec: 编译时维度参数（numba jit 要求）
    """
    pixStamp = fwKSStamp * fwKSStamp
    so_x = np.empty(pixStamp, dtype=np.int64)
    so_y = np.empty(pixStamp, dtype=np.int64)
    idx = 0
    for xc in range(-hwKSStamp, hwKSStamp + 1):
        for yc in range(-hwKSStamp, hwKSStamp + 1):
            so_x[idx] = xc
            so_y[idx] = yc
            idx += 1

    img_patch = np.empty(pixStamp, dtype=np.float64)
    for istamp in range(nS):
        if valid_mask[istamp] == 0:
            continue

        xi = all_x[istamp]
        yi = all_y[istamp]

        p0 = all_scprod[istamp, 1]
        kernelSol[1] += p0

        for i1 in range(1, ncomp1 + 1):
            p0 = all_scprod[istamp, i1 + 1]
            for i2 in range(ncomp2):
                ii = (i1 - 1) * ncomp2 + i2 + 1
                kernelSol[ii + 1] += p0 * wxy[istamp, i2]

        for k in range(pixStamp):
            img_patch[k] = image_flat[so_x[k] + xi + rPixX * (so_y[k] + yi)]
        for ibg in range(nbg_vec):
            kernelSol[ncomp + ibg + 2] += np.dot(all_vectors[istamp, ncomp1 + ibg + 1, :], img_patch)

# 打包 stamps 数据，调用 build_matrix_jit 构建拟合矩阵
def build_matrix_numpy(saMat: np.ndarray, saVectors: np.ndarray, saSscnt: np.ndarray, saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray, nS: int, nCompKer: int, kerOrder: int, bgOrder: int, fwKSStamp: int, rPixX: int, rPixY: int, nKSStamps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """打包 stamps 数据并预计算 wxy 矩阵，调用 build_matrix_jit 构建最小二乘拟合矩阵。从 stamp 数组中提取有效 stamp 的向量和矩阵，计算 valid_mask，预计算每个 stamp 的空间多项式 wxy，然后调用 numba jit 函数完成累加。

    saMat: stamps 的局部矩阵数组 (nS × nC × nC)
    saVectors: stamps 的向量数组 (nS × nComp × fwSq)
    saSscnt: stamps 的已填充子 stamp 计数
    saNss: stamps 的子 stamp 总数上限
    saXss, saYss: stamps 的 X/Y 坐标数组 (nS × nKSStamps)
    nS: stamps 总数
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    fwKSStamp: stamp 子区域全宽
    rPixX, rPixY: region 像素尺寸
    nKSStamps: 每个 stamp 的核测试子 stamp 数

    返回 matrix: 拟合矩阵 (mat_size+1 × mat_size+1)
    返回 wxy: 预计算的 wxy 矩阵 (nS × ncomp2)
    """
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)
    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        return matrix, wxy
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)

    # if n_valid == 0:
    #     for i in range(nS):
    #         for j in range(ncomp2):
    #             wxy[i, j] = 0.0
    #     matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
    #     return matrix

    # nC = stamps_dicts[0]['mat'].shape[0]
    # nC = sa.nC
    nC_valid = nCompKer + 1
    # all_mat = np.zeros((nS, nC_valid, nC_valid), dtype=np.float64)
    all_mat = saMat[:nS, :nC_valid, :nC_valid]
    nvec_total = nCompKer + nbg_vec
    all_vectors = saVectors[:nS, :nvec_total, :]
    # for i in range(nS):
    #     if valid_mask[i]:
    #         all_mat[i] = stamps_dicts[i]['mat'][:nC_valid, :nC_valid]
    #         all_vectors[i] = np.asarray(stamps_dicts[i]['vectors'])[:nvec_total]

    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)

    rPixX2 = np.float64(0.5 * rPixX); rPixY2 = np.float64(0.5 * rPixY)
    for s in range(nS):
        if valid_mask[s] == 0:
            continue
        fx = (np.float64(all_x[s]) - rPixX2) / rPixX2
        fy = (np.float64(all_y[s]) - rPixY2) / rPixY2
        kk = 0; a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2; kk += 1; a2 *= fy
            a1 *= fx

    build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
                     wxy, matrix, nS, kerOrder, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec,
                     pixStamp)

    tri_i, tri_j = np.tril_indices(mat_size, -1)
    matrix[tri_j + 1, tri_i + 1] = matrix[tri_i + 1, tri_j + 1]

    return matrix, wxy

# 打包 stamps 数据，调用 build_scprod_jit 构建标量积
def build_scprod_numpy(saScprod: np.ndarray, saVectors: np.ndarray, saSscnt: np.ndarray, saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray, nS: int, image: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int, fwKSStamp: int, hwKSStamp: int, rPixX: int, wxy: np.ndarray, nKSStamps: Optional[int] = None) -> np.ndarray:
    """打包 stamps 数据和图像，调用 build_scprod_jit 构建标量积向量 kernelSol。从 stamp 数组中提取有效 stamp 的向量和标量积，将图像展平为 1D 数组，预计算 valid_mask 和坐标，然后调用 numba jit 函数累加标量积。

    saScprod: stamps 的局部标量积数组 (nS × nC)
    saVectors: stamps 的向量数组 (nS × nComp × fwSq)
    saSscnt, saNss: stamps 的已填充/总子 stamp 计数
    saXss, saYss: stamps 的 X/Y 坐标数组 (nS × nKSStamps)
    nS: stamps 总数
    image: 输入图像 2D 数组 (rPixY × rPixX)
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX: region 的 X 像素数
    wxy: 预计算的 wxy 矩阵 (nS × ncomp2)
    nKSStamps: 每个 stamp 的核测试子 stamp 数

    返回 kernelSol: 标量积向量 (ncomp + nbg_vec + 2 长度)
    """
    # def build_scprod_numpy(stamps_dicts, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp


    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)

    # if n_valid == 0:
    #     return np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)

    # all_scprod = np.zeros((nS, nCompKer + 1), dtype=np.float64)
    all_scprod = saScprod[:nS, :nCompKer + 1]
    nvec_total = nCompKer + nbg_vec
    all_vectors = saVectors[:nS, :nvec_total, :]
    # for i in range(nS):
    #     if valid_mask[i]:
    #         all_scprod[i] = stamps_dicts[i]['scprod'][:nCompKer + 1]
    #         all_vectors[i] = np.asarray(stamps_dicts[i]['vectors'])[:nvec_total]

    image_arr = np.asarray(image, dtype=np.float64)
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)

    build_scprod_jit(all_vectors, all_scprod, valid_mask, all_x, all_y,
                     wxy, image_arr, kernelSol,
                     nS, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec)

    return kernelSol

# 合并打包：一次准备数据，一次 jit 调用完成 matrix + kernelSol
def build_both_numpy(saMat: np.ndarray, saVectors: np.ndarray, saScprod: np.ndarray,
                     saSscnt: np.ndarray, saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray,
                     nS: int, image: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int,
                     fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int, nKSStamps: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)
    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
        return matrix, kernelSol, wxy
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)

    rPixX2 = np.float64(0.5 * rPixX); rPixY2 = np.float64(0.5 * rPixY)
    for s in range(nS):
        if valid_mask[s] == 0:
            continue
        fx = (np.float64(all_x[s]) - rPixX2) / rPixX2
        fy = (np.float64(all_y[s]) - rPixY2) / rPixY2
        kk = 0; a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2; kk += 1; a2 *= fy
            a1 *= fx

    nC_valid = nCompKer + 1
    all_mat = saMat[:nS, :nC_valid, :nC_valid]
    nvec_total = nCompKer + nbg_vec
    all_vectors = saVectors[:nS, :nvec_total, :]
    all_scprod = saScprod[:nS, :nCompKer + 1]

    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
    image_arr = np.asarray(image, dtype=np.float64)

    build_both_jit(all_mat, all_vectors, all_scprod, valid_mask, all_x, all_y,
                   wxy, matrix, kernelSol, image_arr,
                   nS, kerOrder, fwKSStamp, hwKSStamp, rPixX, rPixY,
                   ncomp, ncomp1, ncomp2, nbg_vec, pixStamp)

    tri_i, tri_j = np.tril_indices(mat_size, -1)
    matrix[tri_j + 1, tri_i + 1] = matrix[tri_i + 1, tri_j + 1]

    return matrix, kernelSol, wxy

# numba jit 批量填充 stamps 的向量/矩阵/标量积（kernel 级实现，原地写入）
@numba.jit(nopython=True, parallel=True)
def fill_stamp_numba_kernel_local(
    image: np.ndarray, imRef: np.ndarray, filterX: np.ndarray, filterY: np.ndarray,
    xi: np.ndarray, yi: np.ndarray, fwKSStamp: int, hwKSStamp: int, fwKernel: int, hwKernel: int,
    rPixX: int, rPixY: int, nCompKer: int, bgOrder: int,
    nvec: int, renFlags_arr: np.ndarray, fillVal: float, mRData1d: np.ndarray,
    out_vectors: np.ndarray, out_krefArea: np.ndarray, out_mat: np.ndarray, out_scprod: np.ndarray, out_sum_val: np.ndarray,
    n_stamps: int) -> None:
    """批量填充 stamps 的向量、矩阵和标量积（numba jit kernel 实现）。对每个有效 stamp，做 xy_conv_stamp 计算（图像与核滤波器的卷积），内联构建背景向量和核向量，计算局部矩阵和标量积，结果写入 out_vectors/out_mat/out_scprod 等输出数组。

    image: 输入图像 1D 数组（供卷积用）
    imRef: 参考图像 1D 数组（供卷积用）
    filterX, filterY: 预计算的 X/Y 方向核滤波器数组
    xi, yi: stamps 的 X/Y 坐标数组 (n_stamps 长度)
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    fwKernel, hwKernel: 核全宽和半宽
    rPixX, rPixY: region 像素尺寸
    nCompKer: 核分量数
    bgOrder: 背景多项式阶数
    nvec: 向量维度 (nCompKer + nBGVectors)
    renFlags_arr: 正交化标志数组
    fillVal: 填充值（用于无效像素）
    mRData1d: mask 数据 1D 数组
    out_vectors: 输出向量数组 (n_stamps × nvec × fwSqStamp)
    out_krefArea: 输出参考区域数组 (n_stamps × fwSqStamp)
    out_mat: 输出局部矩阵数组 (n_stamps × nC+1 × nC+1)
    out_scprod: 输出局部标量积数组 (n_stamps × nC+1)
    out_sum_val: 输出求和值数组 (n_stamps)
    n_stamps: 本批次 stamp 数量
    """

    LOCAL_FLAG_INPUT_ISBAD = 0x80
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel
    # nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2

    kernels = np.zeros((nvec, fwSqKernel), dtype=np.float64)
    for n in range(nvec):
        for jc in range(fwKernel):
            for ic in range(fwKernel):
                fy_idx = fwKernel - 1 - jc
                fx_idx = fwKernel - 1 - ic
                kernels[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]

    for si in numba.prange(n_stamps):
        xi_s = xi[si]
        yi_s = yi[si]
# 
        for n in range(nvec):
            for fsi in range(fwSqStamp):
                out_vectors[si, n, fsi] = 0.0

        for fsi in range(fwSqStamp):
            fsi_x = fsi % fwKSStamp
            fsi_y = fsi // fwKSStamp
            img_base_x = xi_s - hwKSStamp + fsi_x - hwKernel
            img_base_y = yi_s - hwKSStamp + fsi_y - hwKernel
            for fki in range(fwSqKernel):
                kx = fki % fwKernel
                ky = fki // fwKernel
                img_x = img_base_x + kx
                img_y = img_base_y + ky
                img_val = image[img_x + rPixX * img_y]
                for n in range(nvec):
                    out_vectors[si, n, fsi] += kernels[n, fki] * img_val

        for n in range(nvec):
            if renFlags_arr[n]:
                for fsi in range(fwSqStamp):
                    out_vectors[si, n, fsi] -= out_vectors[si, 0, fsi]
# 
        for fsi in range(fwSqStamp):
            out_krefArea[si, fsi] = fillVal

        sumVal = 0.0
        for y_offset in range(fwKSStamp):
            img_y = yi_s - hwKSStamp + y_offset
            for x_offset in range(fwKSStamp):
                img_x = xi_s - hwKSStamp + x_offset
                k = img_x + rPixX * img_y
                dpt = imRef[k]
                out_krefArea[si, x_offset + y_offset * fwKSStamp] = dpt
                if (mRData1d[k] & LOCAL_FLAG_INPUT_ISBAD) == 0:
                    sumVal += abs(dpt)

        out_sum_val[si, 0] = sumVal

        rPixX2 = np.float64(0.5 * rPixX)
        rPixY2 = np.float64(0.5 * rPixY)
        for y_offset in range(fwKSStamp):
            j = yi_s - hwKSStamp + y_offset
            yf = (j - rPixY2) / rPixY2
            for x_offset in range(fwKSStamp):
                i = xi_s - hwKSStamp + x_offset
                xf = (i - rPixX2) / rPixX2
                ipix = x_offset + y_offset * fwKSStamp
                ax = 1.0
                nv = nvec
                for idegx in range(bgOrder + 1):
                    ay = 1.0
                    for idegy in range(bgOrder - idegx + 1):
                        out_vectors[si, nv, ipix] = ax * ay
                        ay *= yf
                        nv += 1
                    ax *= xf

        ncomp1 = nCompKer
        pixStamp = fwSqStamp

        for i in range(ncomp1):
            for j in range(i + 1):
                q = 0.0
                for k in range(pixStamp):
                    q += out_vectors[si, i, k] * out_vectors[si, j, k]
                out_mat[si, i + 1, j + 1] = q

        ivecbg = ncomp1
        for i1 in range(ncomp1):
            p0 = 0.0
            for k in range(pixStamp):
                p0 += out_vectors[si, i1, k] * out_vectors[si, ivecbg, k]
            out_mat[si, ncomp1 + 1, i1 + 1] = p0

        q = 0.0
        for k in range(pixStamp):
            q += out_vectors[si, ivecbg, k] * out_vectors[si, ncomp1, k]
        out_mat[si, ncomp1 + 1, ncomp1 + 1] = q

        for i1 in range(ncomp1):
            p0 = 0.0
            for xc in range(-hwKSStamp, hwKSStamp + 1):
                for yc in range(-hwKSStamp, hwKSStamp + 1):
                    k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                    p0 += out_vectors[si, i1, k] * out_krefArea[si, k]
            out_scprod[si, i1 + 1] = p0

        q = 0.0
        for xc in range(-hwKSStamp, hwKSStamp + 1):
            for yc in range(-hwKSStamp, hwKSStamp + 1):
                k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                q += out_vectors[si, ncomp1, k] * out_krefArea[si, k]
        out_scprod[si, ncomp1 + 1] = q

    return 0

# 填充单个 stamp 的向量、矩阵和标量积，返回 0 成功 / 1 跳过
def fill_stamp_numpy(saVectors: np.ndarray, saMat: np.ndarray, saScprod: np.ndarray, saXss: np.ndarray, saYss: np.ndarray, saSscnt: np.ndarray, saNss: np.ndarray, saKrefArea: np.ndarray, saSumVal: np.ndarray, si: int, imConv: np.ndarray, imRef: np.ndarray, rPixX: int, rPixY: int, ngauss: int, deg_fixe: List[int],
                     hwKSStamp: int, fwKSStamp: int, hwKernel: int, fwKernel: int, bgOrder: int, nCompKer: int,
                     filter_x: np.ndarray, filter_y: np.ndarray, fillVal: float, mRData: np.ndarray) -> int:
    """填充单个 stamp 的向量、矩阵和标量积。首先检查 stamp 是否有效（sscnt < nss），无效则跳过。有效时调用 fill_stamp_numba 批量填充（参数包装为单元素列表），并将结果写回 saVectors/saMat/saScprod 等数组。

    saVectors, saMat, saScprod: stamps 的向量/矩阵/标量积数组，原地写入
    saXss, saYss: stamps 的 X/Y 坐标数组
    saSscnt, saNss: stamps 的已填充/总子 stamp 计数
    saKrefArea: stamps 的参考区域数组
    saSumVal: stamps 的求和值数组
    si: stamp 索引
    imConv, imRef: 卷积/参考图像 1D 数组
    rPixX, rPixY: region 像素尺寸
    ngauss: 高斯分量数
    deg_fixe: 每个高斯分量的多项式阶数列表
    hwKSStamp, fwKSStamp: stamp 子区域半宽和全宽
    hwKernel, fwKernel: 核半宽和全宽
    bgOrder: 背景多项式阶数
    nCompKer: 核分量数
    filter_x, filter_y: 预计算的核滤波器数组
    fillVal: 填充值
    mRData: mask 数据 1D 数组

    返回 0 成功填充，返回 1 无效 stamp 跳过
    """
    # 无效 stamp 快速返回
    if saSscnt[si] >= saNss[si]:
        return 1



    # ===== 批量调用 fill_stamp_numba =====
    result = fill_stamp_numba(saXss, saYss, saSscnt, saNss, [si], imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                              hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                              filter_x, filter_y, fillVal, mRData)
    out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val = result
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
    fwSqStamp = fwKSStamp * fwKSStamp
    nC = nCompKer + 1
    saVectors[si, :nCompKer + nbg, :fwSqStamp] = out_vectors[0]
    saKrefArea[si, :] = out_krefArea[0]
    saMat[si, :nC + 1, :nC + 1] = out_mat[0]
    saScprod[si, :nC + 1] = out_scprod[0]
    saSumVal[si] = out_sum_val[0]
    return 0

# 批量 stamps 的填充入口，调用 fill_stamp_numba_kernel_local
def fill_stamp_numba(saXss: np.ndarray, saYss: np.ndarray, saSscnt: np.ndarray, saNss: np.ndarray, si_list: List[int], imConv: np.ndarray, imRef: np.ndarray, rPixX: int, rPixY: int, ngauss: int, deg_fixe: List[int],
                     hwKSStamp: int, fwKSStamp: int, hwKernel: int, fwKernel: int, bgOrder: int, nCompKer: int,
                     filter_x: np.ndarray, filter_y: np.ndarray, fillVal: float, mRData: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量 stamps 的填充入口，收集有效 stamp 的坐标和图像数据，计算 renFlags 和核滤波器预计算，调用 fill_stamp_numba_kernel_local 完成批量填充。

    saXss, saYss: stamps 的 X/Y 坐标数组 (nS × nKSStamps)
    saSscnt, saNss: stamps 的已填充/总子 stamp 计数
    si_list: 本批次 stamp 索引列表
    imConv, imRef: 卷积/参考图像 1D 数组
    rPixX, rPixY: region 像素尺寸
    ngauss: 高斯分量数
    deg_fixe: 每个高斯分量的多项式阶数列表
    hwKSStamp, fwKSStamp: stamp 子区域半宽和全宽
    hwKernel, fwKernel: 核半宽和全宽
    bgOrder: 背景多项式阶数
    nCompKer: 核分量数
    filter_x, filter_y: 预计算的核滤波器数组
    fillVal: 填充值
    mRData: mask 数据 1D 数组

    返回 (out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val)
    """
    # si_list: 批次 stamp 索引列表
    n_stamps = len(si_list)

    # 预计算 renFlags
    nvec = 0
    renFlags = []
    for ig in range(ngauss):
        for idegx in range(int(deg_fixe[ig]) + 1):
            for idegy in range(int(deg_fixe[ig]) - idegx + 1):
                ren = 0
                dx = (idegx // 2) * 2 - idegx
                dy = (idegy // 2) * 2 - idegy
                if dx == 0 and dy == 0 and nvec > 0:
                    ren = 1
                renFlags.append(ren)
                nvec += 1

    # 收集所有有效 stamp 的 xi/yi（跳过 sscnt >= nss 的）
    xi_arr = np.zeros(n_stamps, dtype=np.int32)
    yi_arr = np.zeros(n_stamps, dtype=np.int32)
    valid_mask = np.zeros(n_stamps, dtype=np.int32)
    n_valid = 0
    for idx in range(n_stamps):
        si = si_list[idx]
        if saSscnt[si] < saNss[si]:
            xi_arr[n_valid] = int(saXss[si, saSscnt[si]])
            yi_arr[n_valid] = int(saYss[si, saSscnt[si]])
            valid_mask[idx] = 1
            n_valid += 1
    if n_valid == 0:
        return out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val[:, 0]

    img_flat = np.asarray(imConv, dtype=np.float64).ravel()
    imRef_flat = np.asarray(imRef, dtype=np.float64).ravel()
    fx = np.asarray(filter_x, dtype=np.float64)
    fy = np.asarray(filter_y, dtype=np.float64)
    rflags = np.array(renFlags, dtype=np.int32)
    mRData1d = mRData.ravel()

    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
    fwSqStamp = fwKSStamp * fwKSStamp
    nC = nCompKer + 1

    out_vectors = np.zeros((n_stamps, nvec + nbg, fwSqStamp), dtype=np.float64)
    out_krefArea = np.zeros((n_stamps, fwSqStamp), dtype=np.float64)
    out_mat = np.zeros((n_stamps, nC + 1, nC + 1), dtype=np.float64)
    out_scprod = np.zeros((n_stamps, nC + 1), dtype=np.float64)
    out_sum_val = np.zeros((n_stamps, 1), dtype=np.float64)

    fill_stamp_numba_kernel_local(
        img_flat, imRef_flat, fx, fy,
        xi_arr, yi_arr, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
        rPixX, rPixY, nCompKer, bgOrder,
        nvec, rflags, fillVal, mRData1d,
        out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val,
        n_valid)

    return out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val[:, 0]

# numba jit 批量计算 sig + sigma_clip + mark refill（一次 jit 替代三个步骤）
@numba.jit(nopython=True)
def get_sig_and_clip_jit(
    sa_vectors: np.ndarray, sa_krefArea: np.ndarray,
    sa_sscnt: np.ndarray, sa_nss: np.ndarray, sa_xss: np.ndarray, sa_yss: np.ndarray,
    kernelSol: np.ndarray, imNoise: np.ndarray, mRData1d: np.ndarray,
    fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
    nCompKer: int, kerOrder: int, bgOrder: int, nS: int,
    sscnt_local: np.ndarray, chi2_local: np.ndarray,
    refill_flags: np.ndarray, ss: np.ndarray,
    batched_bg: np.ndarray, batched_coeffs: np.ndarray,
    kerSigReject: float, statSig: float,
) -> Tuple[float, float, int]:
    """一次 jit 调用完成：batch sig → sigma_clip → mark refill。
    refill_flags 和 sscnt_local 原地修改。返回 (mean, stdev, ncheck)。"""
    LOCAL_ZVAL = 1e-10; LOCAL_MAXVAL = 1e10
    LOCAL_IBAD = 0x80; LOCAL_ISNAN = 0x08
    fwSq = fwKSStamp * fwKSStamp

    nss = 0
    for si in range(nS):
        scnt = sa_sscnt[si]
        if scnt >= sa_nss[si]:
            chi2_local[si] = -1.0
            continue
        bg_val = batched_bg[si]
        csModel = np.empty(fwSq, dtype=np.float64)
        coeff0 = kernelSol[1]
        for i in range(fwSq):
            csModel[i] = coeff0 * sa_vectors[si, 0, i]
        for i1j in range(1, nCompKer):
            coeff2 = batched_coeffs[si, i1j]
            for i in range(fwSq):
                csModel[i] += coeff2 * sa_vectors[si, i1j, i]
        im = sa_krefArea[si]
        nsig = 0; sig1 = 0.0
        xi = sa_xss[si, scnt]; yi = sa_yss[si, scnt]
        for j in range(fwKSStamp):
            yR = yi - hwKSStamp + j
            for i in range(fwKSStamp):
                xR = xi - hwKSStamp + i
                idk = i + j * fwKSStamp
                tdat = csModel[idk]; idat = im[idk]
                ndat = imNoise[xR + rPixX * yR]
                diff_val = tdat - idat + bg_val
                mr_idx = xR + rPixX * yR
                if (mRData1d[mr_idx] & LOCAL_IBAD) or (abs(idat) <= LOCAL_ZVAL):
                    continue
                if np.isnan(tdat) or np.isnan(idat):
                    mRData1d[mr_idx] = mRData1d[mr_idx] | (LOCAL_IBAD | LOCAL_ISNAN)
                    continue
                nsig += 1
                sig1 += diff_val * diff_val / ndat
        if nsig > 0:
            sig1 /= nsig
            if sig1 >= LOCAL_MAXVAL:
                sig1 = -1.0
        else:
            sig1 = -1.0
        chi2_local[si] = sig1
        if sig1 != -1.0:
            ss[nss] = sig1
            nss += 1
        else:
            sscnt_local[si] += 1
            refill_flags[si] = 1

    if nss == 0:
        return (0.0, float(MAXVAL), 0)

    arr = np.asarray(ss[:nss], dtype=np.float32).astype(np.float64)
    count = nss; mask = np.zeros(count, dtype=np.int32)
    cnt = 0; ncnt = count; iternum = 0
    mean_val = 0.0; stdev_val = 0.0
    maxiter = 10
    while (ncnt != cnt) and (iternum < maxiter):
        cnt = ncnt
        good = arr[mask == 0]; ncnt = len(good)
        if ncnt == 0:
            return (0.0, float(MAXVAL), 0)
        mean_val = float(good.mean())
        if ncnt == 1:
            return (mean_val, float(MAXVAL), 0)
        sg = good - mean_val
        stdev_val = float(np.sqrt(np.sum(sg * sg) / (ncnt - 1)))
        istdev = 1.0 / stdev_val
        deviations = np.abs(sg) * istdev
        new_outliers = deviations > statSig
        good_indices = np.where(mask == 0)[0]
        mask[good_indices[new_outliers]] = 1
        ncnt = count - int(np.sum(mask))
        iternum += 1

    ncheck = 0
    for si in range(nS):
        if sscnt_local[si] < sa_nss[si] and chi2_local[si] != -1.0:
            if (chi2_local[si] - mean_val) > kerSigReject * stdev_val:
                sscnt_local[si] += 1
                refill_flags[si] = 1
                ncheck = 1
    return (mean_val, stdev_val, ncheck)

# numba jit 批量计算所有 stamps 的信噪比（figMerit="v" 模式），结果写入 out_sig 数组
@numba.jit(nopython=True, parallel=True)
def get_stamp_sig_batch_jit(
    sa_vectors: np.ndarray, sa_krefArea: np.ndarray, sa_sscnt: np.ndarray, sa_nss: np.ndarray, sa_xss: np.ndarray, sa_yss: np.ndarray,
    kernelSol: np.ndarray, imNoise: np.ndarray, mRData1d: np.ndarray,
    fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
    nCompKer: int, kerOrder: int, bgOrder: int, nS: int,
    out_sig1: np.ndarray, out_sig2: np.ndarray, out_sig3: np.ndarray,
    batched_bg: Optional[np.ndarray] = None, batched_coeffs: Optional[np.ndarray] = None) -> None:
    """批量计算所有 stamps 的信噪比（figMerit="v" 方差模式）。遍历所有有效 stamp，对每个计算重建模型与图像的残差方差加权平均，结果写入 out_sig1/out_sig2/out_sig3 数组。支持传入预计算的 batched_bg 和 batched_coeffs 加速。

    sa_vectors: stamps 的向量数组 (nS × nCompTotal × fwSqStamp)
    sa_krefArea: stamps 的参考区域数组 (nS × fwSqStamp)
    sa_sscnt, sa_nss: stamps 的已填充/总子 stamp 计数
    sa_xss, sa_yss: stamps 的 X/Y 坐标数组
    kernelSol: 核拟合解向量
    imNoise: 噪声图像 1D 数组
    mRData1d: mask 数据 1D 数组
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX, rPixY: region 像素尺寸
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    nS: stamps 总数
    out_sig1, out_sig2, out_sig3: 输出信噪比数组 (nS 长度)，原地写入
    batched_bg: 预计算的所有 stamp 背景值，None 时在线计算
    batched_coeffs: 预计算的核系数 (nS × nCompKer)，None 时在线计算
    """

    LOCAL_ZEROVAL = 1e-10
    LOCAL_MAXVAL = 1e10
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    LOCAL_FLAG_ISNAN = 0x08
    fwSq = fwKSStamp * fwKSStamp

    for si in numba.prange(nS):
        scnt = sa_sscnt[si]
        if scnt >= sa_nss[si]:
            out_sig1[si] = -1.0
            out_sig2[si] = -1.0
            out_sig3[si] = -1.0
            continue

        xi = sa_xss[si, scnt]
        yi = sa_yss[si, scnt]

        if batched_bg is not None:
            background = batched_bg[si]
        else:
            ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
            background = 0.0
            k = 1
            xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
            yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
            ax = 1.0
            for i in range(bgOrder + 1):
                ay = 1.0
                for j in range(bgOrder - i + 1):
                    background += kernelSol[ncompBG + k] * ax * ay
                    k += 1
                    ay *= yf
                ax *= xf

        csModel = np.zeros(fwSq, dtype=np.float64)
        coeff = kernelSol[1]
        for i in range(fwSq):
            csModel[i] = coeff * sa_vectors[si, 0, i]

        if batched_coeffs is not None:
            for i1 in range(1, nCompKer):
                for i in range(fwSq):
                    csModel[i] += batched_coeffs[si, i1] * sa_vectors[si, i1, i]
        else:
            kk = 2
            for i1 in range(1, nCompKer):
                coeff2 = 0.0
                ax = 1.0
                for ix in range(kerOrder + 1):
                    ay = 1.0
                    for iy in range(kerOrder - ix + 1):
                        coeff2 += kernelSol[kk] * ax * ay
                        kk += 1
                        ay *= yf
                    ax *= xf
                for i in range(fwSq):
                    csModel[i] += coeff2 * sa_vectors[si, i1, i]

        im = sa_krefArea[si]

        nsig = 0
        sig1 = 0.0
        temp = np.zeros(fwSq, dtype=np.float64)
        for j in range(fwKSStamp):
            yRegion2 = yi - hwKSStamp + j
            for i in range(fwKSStamp):
                xRegion2 = xi - hwKSStamp + i
                idx = i + j * fwKSStamp

                tdat = csModel[idx]
                idat = im[idx]
                ndat = imNoise[xRegion2 + rPixX * yRegion2]
                diff = tdat - idat + background

                mr_idx = xRegion2 + rPixX * yRegion2
                if (mRData1d[mr_idx] & LOCAL_FLAG_INPUT_ISBAD) or (abs(idat) <= LOCAL_ZEROVAL):
                    continue

                temp[idx] = diff
                if np.isnan(tdat) or np.isnan(idat):
                    mRData1d[mr_idx] = mRData1d[mr_idx] | (LOCAL_FLAG_INPUT_ISBAD | LOCAL_FLAG_ISNAN)
                    continue

                nsig += 1
                sig1 += diff * diff / ndat

        if nsig > 0:
            sig1 /= nsig
            if sig1 >= LOCAL_MAXVAL:
                sig1 = -1.0
        else:
            sig1 = -1.0

        out_sig1[si] = sig1
        out_sig2[si] = -1.0
        out_sig3[si] = -1.0

# numba jit 单个 stamp 的信噪比计算，返回 (sig1, sig2, sig3, nsig)
@numba.jit(nopython=True)
def get_stamp_sig_jit(vectors: np.ndarray, kernelSol: np.ndarray, imNoise: np.ndarray, mRData1d: np.ndarray, im: np.ndarray,
                       fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
                       nCompKer: int, kerOrder: int, bgOrder: int, xi: int, yi: int,
                       temp: np.ndarray) -> Tuple[float, float, float, int]:
    """计算单个 stamp 的信噪比。根据 figMerit 模式（方差 v/信号 s/直方图 h），在 stamp 子区域内遍历像素，计算模型重建值与数据的残差，返回不同定义下的信噪比和标志位。结果也写入 temp 数组供进一步统计用。

    vectors: stamp 的向量 (nCompTotal × fwSqStamp)
    kernelSol: 核拟合解向量
    imNoise: 噪声图像 1D 数组
    mRData1d: mask 数据 1D 数组
    im: 图像 1D 数组（模板或科学图像）
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX, rPixY: region 像素尺寸
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    xi, yi: stamp 子区域的中心坐标
    temp: 临时数组 (fwSqStamp 长度)，用于写入每个像素的残差值

    返回 sig1: 方差加权信噪比
    返回 sig2: 信号加权信噪比（figMerit="s" 模式）
    返回 sig3: 直方图信噪比（figMerit="h" 模式）
    返回 nsig: 有效像素数
    """
    LOCAL_ZEROVAL = 1e-10
    LOCAL_MAXVAL = 1e10
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    LOCAL_FLAG_ISNAN = 0x08

    ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    background = 0.0
    k = 1
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
    ax = 1.0
    for i in range(bgOrder + 1):
        ay = 1.0
        for j in range(bgOrder - i + 1):
            background += kernelSol[ncompBG + k] * ax * ay
            k += 1
            ay *= yf
        ax *= xf

    fwSq = fwKSStamp * fwKSStamp
    csModel = np.zeros(fwSq, dtype=np.float64)

    coeff = kernelSol[1]
    for i in range(fwSq):
        csModel[i] = coeff * vectors[0, i]

    kk = 2
    for i1 in range(1, nCompKer):
        coeff = 0.0
        ax = 1.0
        for ix in range(kerOrder + 1):
            ay = 1.0
            for iy in range(kerOrder - ix + 1):
                coeff += kernelSol[kk] * ax * ay
                kk += 1
                ay *= yf
            ax *= xf

        for i in range(fwSq):
            csModel[i] += coeff * vectors[i1, i]

    nsig = 0
    sig1 = 0.0
    sig2 = -1.0
    sig3 = -1.0

    for j in range(fwKSStamp):
        yRegion2 = yi - hwKSStamp + j
        for i in range(fwKSStamp):
            xRegion2 = xi - hwKSStamp + i
            idx = i + j * fwKSStamp

            tdat = csModel[idx]
            idat = im[idx]
            ndat = imNoise[xRegion2 + rPixX * yRegion2]

            diff = tdat - idat + background

            mr_idx = xRegion2 + rPixX * yRegion2
            if (mRData1d[mr_idx] & LOCAL_FLAG_INPUT_ISBAD) or (abs(idat) <= LOCAL_ZEROVAL):
                continue
            else:
                temp[idx] = diff

            if np.isnan(tdat) or np.isnan(idat):
                mRData1d[mr_idx] = mRData1d[mr_idx] | (LOCAL_FLAG_INPUT_ISBAD | LOCAL_FLAG_ISNAN)
                continue

            nsig += 1
            sig1 += diff * diff / ndat

    if nsig > 0:
        sig1 /= nsig
        if sig1 >= LOCAL_MAXVAL:
            sig1 = -1.0
    else:
        sig1 = -1.0

    return (sig1, sig2, sig3, nsig)

# 快速空间域卷积：预计算所有 kcStep 块的核，调用 jit kernel 一次遍历完成卷积+variance+mask
def spatial_convolve_fast_numpy(image: np.ndarray, variance: np.ndarray, xSize: int, ySize: int, kernelSol: np.ndarray, cMask: np.ndarray, kcStep: int,
                                hwKernel: int, fwKernel: int,
                                convolveVariance: int, kerFracMask: float,
                                rPixX: int, rPixY: int, nCompKer: int, kerOrder: int, kernel_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """快速空间域卷积：预计算所有 kcStep 块的核，调用 spatial_convolve_jit_kernel 一次遍历完成卷积、variance 计算和 mask 标记。将图像和 variance 转换为 1D float64 数组，调用 buildAllKernels 预计算块核，然后启动 numba jit kernel 并行处理。

    image: 输入图像 1D 或 2D 数组
    variance: 输入 variance 数组，None 表示不计算 variance
    xSize, ySize: 图像尺寸
    kernelSol: 核多项式系数向量
    cMask: 输入卷积 mask 1D 数组
    kcStep: 核缓存步长
    hwKernel, fwKernel: 核半宽和全宽
    convolveVariance: 是否卷积 variance 本身
    kerFracMask: mask 分数阈值
    rPixX, rPixY: region 像素尺寸
    nCompKer: 核分量数
    kerOrder: 核多项式阶数
    kernel_vec: 核基函数列表，传入 buildAllKernels

    返回 vData: variance 输出数组（dovar 为 False 时返回 None）
    返回 cRdata_out: 卷积结果输出数组 (float32)
    返回 mRData_out: mask 输出数组
    """

    fwSq = fwKernel * fwKernel
    dovar = variance is not None

    image1d = image.astype(np.float64).ravel()
    cMask1d = cMask.astype(np.int32).ravel()

    cRdata64 = np.zeros(xSize * ySize, dtype=np.float64)
    mRData64 = np.zeros(xSize * ySize, dtype=np.int32)

    if dovar:
        var1d = variance.astype(np.float64).ravel()
    else:
        var1d = np.zeros(1, dtype=np.float64)

    vData = None
    vData64 = np.zeros(xSize * ySize, dtype=np.float64)

    kernel_vec_2d = np.array([np.asarray(kernel_vec[idx][:fwSq], dtype=np.float64)
                              for idx in range(nCompKer)])


    allKernels, nstepsX, nstepsY = buildAllKernels(
        kernelSol.astype(np.float64), kernel_vec_2d,
        nCompKer, kerOrder, fwKernel, hwKernel, kcStep,
        rPixX, rPixY, xSize, ySize)

    spatial_convolve_jit_kernel(image1d, var1d, cMask1d,
        cRdata64, vData64, mRData64, kernelSol.astype(np.float64),
        xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
        kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
        kernel_vec_2d, allKernels=allKernels, nstepsX_in=nstepsX)

    if dovar:
        vData = vData64.astype(np.float32)

    cRdata_out = cRdata64.astype(np.float32)
    mRData_out = mRData64
    return vData, cRdata_out, mRData_out

# 初始 stamps 质量检查：计算 merit 值，筛选合格 stamps 用于构建拟合矩阵
def check_stamps_numpy(saScprod: np.ndarray, saMat: np.ndarray, saNorm: np.ndarray, saDiff: np.ndarray, saSscnt: np.ndarray, saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray,
                       saVectors: np.ndarray, saKrefArea: np.ndarray, nS: int, imRef: np.ndarray, imNoise: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int,
                       forceConvolve: str, figMerit: str,
                       kerSigReject: float, statSig: float, fwKSStamp: int, hwKSStamp: int,
                       rPixX: int, rPixY: int, fwKernel: int, kernel_vec: np.ndarray, mRData: np.ndarray,
                       nKSStamps: Optional[int] = None) -> float:
    """初始 stamps 质量检查。对每个 stamp 计算 merit 值（基于标量积和矩阵重建的核均值与 stamp 自身值的偏差），筛选出偏差在 kerSigReject 倍标准差以内的合格 stamps。合格 stamps 的局部矩阵和标量积被用于构建核拟合矩阵。

    saScprod, saMat: stamps 的标量积和矩阵数组
    saNorm, saDiff: stamps 的归一化和偏差值数组，输出
    saSscnt, saNss: stamps 的已填充/总子 stamp 计数
    saXss, saYss: stamps 的 X/Y 坐标数组
    saVectors, saKrefArea: stamps 的向量和参考区域数组
    nS: stamps 总数
    imRef, imNoise: 参考图像和噪声图像数组
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    forceConvolve: 强制卷积方向 ("t"/"i")
    figMerit: 品质指标类型 ("v"/"s"/"h")
    kerSigReject: 核拟合的 sigma 拒绝阈值
    statSig: sigma-clip 的 sigma 阈值
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX, rPixY: region 像素尺寸
    fwKernel: 核全宽
    kernel_vec: 核基函数列表
    mRData: mask 数据数组
    nKSStamps: 每个 stamp 的核测试子 stamp 数

    返回 merit: 当前品质指标值（用于方向选择）
    """

    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    ks = np.zeros(nS, dtype=np.float32)
    nks = 0

    nComps = nCompKer + 1

    for i in range(nS):
        check_mat = np.zeros((nComps + 1, nComps + 1), dtype=np.float64)
        check_vec = np.zeros(nComps + 1, dtype=np.float64)
        # indx = np.zeros(nComps + 1, dtype=np.int32)

        for im in range(1, nComps + 1):
            # check_vec[im] = stamps_dicts[i]['scprod'][im]
            check_vec[im] = saScprod[i, im]
            for jm in range(1, im + 1):
                # check_mat[im, jm] = stamps_dicts[i]['mat'][im, jm]
                check_mat[im, jm] = saMat[i, im, jm]
                check_mat[jm, im] = check_mat[im, jm]

        check_vec[1:nComps+1] = np.linalg.solve(
            check_mat[1:nComps+1, 1:nComps+1], check_vec[1:nComps+1])

        sum_val = check_vec[1]
        # stamps_dicts[i]['norm'] = sum_val
        saNorm[i] = sum_val
        ks[nks] = sum_val
        nks += 1

    kmean, kstdev, sc_rc = sigma_clip_numpy(ks[:nks], maxiter=10, stat_sig=statSig)

    for i in range(nS):
        # stamps_dicts[i]['diff'] = abs((stamps_dicts[i]['norm'] - kmean) / kstdev)
        saDiff[i] = abs((saNorm[i] - kmean) / kstdev)

    if forceConvolve[0:1] == "b":
        # ntestStamps = 0
        # for i in range(nS):
        #     if stamps_dicts[i]['diff'] < kerSigReject:
        #         ntestStamps += 1
        # testStamps = []
        # for i in range(nS):
        #     if stamps_dicts[i]['diff'] < kerSigReject:
        #         testStamps.append(stamps_dicts[i])
        testMask = saDiff < kerSigReject
        ntestStamps = testMask.sum()
        if ntestStamps == 0:
            return 0.0

        testScprod   = saScprod[testMask]
        testMat      = saMat[testMask]
        testVectors  = saVectors[testMask]
        testKrefArea = saKrefArea[testMask]
        testSscnt    = saSscnt[testMask]
        testNss      = saNss[testMask]
        testXss      = saXss[testMask]
        testYss      = saYss[testMask]
        # testSa = StampsArray.subset(sa, testMask)
        # if testSa is None:
        #     return 0.0
        # ntestStamps = testSa.nS

        testKerSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)

        wxy = np.zeros((ntestStamps, ncomp2), dtype=np.float64)

        matrix, testKerSol, wxy_unused = build_both_numpy(testMat, testVectors, testScprod, testSscnt, testNss, testXss, testYss,
                                                   ntestStamps, imRef, nCompKer, kerOrder, bgOrder,
                                                   fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)

        # indx2 = np.zeros(mat_size + 1, dtype=np.int32)
        testKerSol[1:mat_size+1] = np.linalg.solve(
            matrix[1:mat_size+1, 1:mat_size+1], testKerSol[1:mat_size+1])

        kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
        kernel_arr = np.zeros(fwKernel * fwKernel, dtype=np.float64)
        kmean = make_kernel_numpy(0, 0, testKerSol, rPixX, rPixY, nCompKer, kerOrder,
                                  fwKernel, kernel_vec, kernel_coeffs, kernel_arr)

        m1 = np.zeros(ntestStamps, dtype=np.float32)
        m2 = np.zeros(ntestStamps, dtype=np.float32)
        m3 = np.zeros(ntestStamps, dtype=np.float32)
        mcnt1 = 0
        mcnt2 = 0
        mcnt3 = 0

        for i in range(ntestStamps):

            sscnt = testSscnt[i]
            xRegion = int(testXss[i, sscnt])
            yRegion = int(testYss[i, sscnt])
            im = testKrefArea[i]
            vectors = np.asarray(testVectors[i], dtype=np.float64)
            kSol = np.asarray(testKerSol, dtype=np.float64)
            imNoiseArr = np.asarray(imNoise, dtype=np.float64)
            mRDataArr = np.asarray(mRData, dtype=np.int64)
            imArr = np.asarray(im, dtype=np.float64)

            # figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
            temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)

            sig1, sig2, sig3, nsig = get_stamp_sig_jit(
                vectors, kSol, imNoiseArr, mRDataArr, imArr,
                fwKSStamp, hwKSStamp, rPixX, rPixY,
                nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                temp)

            if figMerit[0:1] != "v":
                temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                mRData_2d = mRData.reshape(-1, rPixX)
                result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                                fwKSStamp, fwKSStamp,
                                                0x0, 0xffff, 5, mRData_2d, statSig)
                if result[7] != 0:
                    sig2 = -1.0
                    sig3 = -1.0
                else:
                    sig2 = result[4]
                    sig3 = result[5]
                    if sig2 < 0 or sig2 >= MAXVAL:
                        sig2 = -1.0
                    elif sig3 < 0 or sig3 >= MAXVAL:
                        sig3 = -1.0

            if sig1 != -1 and sig1 <= MAXVAL:
                m1[mcnt1] = sig1
                mcnt1 += 1
            if sig2 != -1 and sig2 <= MAXVAL:
                m2[mcnt2] = sig2
                mcnt2 += 1
            if sig3 != -1 and sig3 <= MAXVAL:
                m3[mcnt3] = sig3
                mcnt3 += 1

        merit1, sig1sc, rc1 = sigma_clip_numpy(m1[:mcnt1], maxiter=10, stat_sig=statSig)
        merit2, sig2sc, rc2 = sigma_clip_numpy(m2[:mcnt2], maxiter=10, stat_sig=statSig)
        merit3, sig3sc, rc3 = sigma_clip_numpy(m3[:mcnt3], maxiter=10, stat_sig=statSig)

        merit1 /= kmean
        merit2 /= kmean
        merit3 /= kmean

        if figMerit[0:1] == "v":
            if mcnt1 > 0:
                return merit1
            elif mcnt2 > 0:
                return merit2
            elif mcnt3 > 0:
                return merit3
            else:
                return 666.0
        elif figMerit[0:1] == "s":
            if mcnt2 > 0:
                return merit2
            elif mcnt1 > 0:
                return merit1
            elif mcnt3 > 0:
                return merit3
            else:
                return 666.0
        elif figMerit[0:1] == "h":
            if mcnt3 > 0:
                return merit3
            elif mcnt1 > 0:
                return merit1
            elif mcnt2 > 0:
                return merit2
            else:
                return 666.0
    else:
        return 0.0

    return 0.0

# 迭代核拟合中的 stamps 再检查：计算信噪比，sigma-clip 剔除异常 stamp，生成重填列表
def check_again_numpy(saSscnt: np.ndarray, saNss: np.ndarray, saChi2: np.ndarray, saXss: np.ndarray, saYss: np.ndarray, saVectors: np.ndarray, saKrefArea: np.ndarray, kernelSol: np.ndarray,
                      imNoise: np.ndarray,
                      nS: int, figMerit: str, kerSigReject: float, statSig: float,
                      fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int, mRData: np.ndarray,
                      nCompKer: int, kerOrder: int, bgOrder: int) -> Tuple[int, float, float, int, List[int], np.ndarray, np.ndarray]:
    """迭代核拟合中的 stamps 再检查和重填。对当前核拟合解计算每个 stamp 的信噪比，用 sigma-clip 剔除异常 stamp（chi2 偏离均值超 kerSigReject 倍标准差），生成需要重填的 stamp 索引列表和更新的 sscnt/chi2 数组供 fill_stamp_numpy 重填。

    saSscnt, saNss: stamps 的已填充/总子 stamp 计数
    saChi2: stamps 的卡方值数组，更新
    saXss, saYss: stamps 的 X/Y 坐标数组
    saVectors, saKrefArea: stamps 的向量和参考区域数组
    kernelSol: 当前核拟合解向量
    imNoise: 噪声图像数组
    nS: stamps 总数
    figMerit: 品质指标类型
    kerSigReject: 核拟合的 sigma 拒绝阈值
    statSig: sigma-clip 的 sigma 阈值
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX, rPixY: region 像素尺寸
    mRData: mask 数据数组
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数

    返回 check: 是否有 stamp 被重填（0/1）
    返回 meansigSubstamps: 所有 stamps 的平均信噪比
    返回 scatterSubstamps: 信噪比的弥散度
    返回 nskippedSubstamps: 被跳过的 stamp 数
    返回 refill_indices: 需要重填的 stamp 索引列表
    返回 sscnt_update: 更新后的 sscnt 数组 (nS 长度)
    返回 chi2_update: 更新后的 chi2 数组 (nS 长度)
    """

    ss = np.zeros(nS, dtype=np.float32)
    nss = 0

    sig = 0.0
    check = 0
    mean = 0.0
    stdev = 0.0
    nskippedSubstamps = 0
    refill_indices = []

    sscnt_local = saSscnt[:nS].copy()
    chi2_local = saChi2[:nS].copy()

    # 批量计算所有 stamps 的 sig (figMerit="v" 模式)
    batch_sig1 = None
    if figMerit[0:1] == "v":
        batch_sig1 = np.zeros(nS, dtype=np.float64)
        batch_sig2 = np.zeros(nS, dtype=np.float64)
        batch_sig3 = np.zeros(nS, dtype=np.float64)

        # batch polynomial pre-computation for all stamps
        halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
        xf_batch = np.zeros(nS, dtype=np.float64)
        yf_batch = np.zeros(nS, dtype=np.float64)
        for si in range(nS):
            if sscnt_local[si] < saNss[si]:
                xi = float(saXss[si, sscnt_local[si]])
                yi = float(saYss[si, sscnt_local[si]])
                xf_batch[si] = (xi - halfX) / halfX
                yf_batch[si] = (yi - halfY) / halfY

        ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
        batched_bg = np.zeros(nS, dtype=np.float64)
        kb = 1
        ax_arr = np.ones(nS, dtype=np.float64)
        for i in range(bgOrder + 1):
            ay_arr = np.ones(nS, dtype=np.float64)
            for j in range(bgOrder - i + 1):
                batched_bg += kernelSol[ncompBG + kb] * ax_arr * ay_arr
                kb += 1
                ay_arr *= yf_batch
            ax_arr *= xf_batch

        batched_coeffs4 = np.zeros((nS, nCompKer), dtype=np.float64)
        kk = 2
        for i1 in range(1, nCompKer):
            coeff_arr = np.zeros(nS, dtype=np.float64)
            ax_arr = np.ones(nS, dtype=np.float64)
            for ix in range(kerOrder + 1):
                ay_arr = np.ones(nS, dtype=np.float64)
                for iy in range(kerOrder - ix + 1):
                    coeff_arr += kernelSol[kk] * ax_arr * ay_arr
                    kk += 1
                    ay_arr *= yf_batch
                ax_arr *= xf_batch
            batched_coeffs4[:, i1] = coeff_arr

        refill_flags = np.zeros(nS, dtype=np.int32)
        mean, stdev, ncheck_j = get_sig_and_clip_jit(
            np.asarray(saVectors, dtype=np.float64), np.asarray(saKrefArea, dtype=np.float64),
            saSscnt, saNss, saXss, saYss,
            np.asarray(kernelSol, dtype=np.float64),
            np.asarray(imNoise, dtype=np.float64),
            np.asarray(mRData, dtype=np.int32).ravel(),
            fwKSStamp, hwKSStamp, rPixX, rPixY,
            nCompKer, kerOrder, bgOrder, nS,
            sscnt_local, chi2_local, refill_flags, ss,
            batched_bg=batched_bg, batched_coeffs=batched_coeffs4,
            kerSigReject=kerSigReject, statSig=statSig)
        refill_indices.extend(np.where(refill_flags > 0)[0].tolist())
        check = ncheck_j
        meansigSubstamps = mean
        scatterSubstamps = stdev
    else:
        for istamp in range(nS):
            if sscnt_local[istamp] < saNss[istamp]:
                # 内联 get_stamp_sig_numpy 逻辑，使用扁平数组
                sscnt = sscnt_local[istamp]
                xRegion = int(saXss[istamp, sscnt])
                yRegion = int(saYss[istamp, sscnt])
                im = saKrefArea[istamp]
                vectors = np.asarray(saVectors[istamp], dtype=np.float64)
                kSol = np.asarray(kernelSol, dtype=np.float64)
                imNoiseArr = np.asarray(imNoise, dtype=np.float64)
                mRDataArr = np.asarray(mRData, dtype=np.int64)
                imArr = np.asarray(im, dtype=np.float64)

                # figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
                temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)

                sig1, sig2, sig3, nsig = get_stamp_sig_jit(
                    vectors, kSol, imNoiseArr, mRDataArr, imArr,
                    fwKSStamp, hwKSStamp, rPixX, rPixY,
                    nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                    temp)

                if figMerit[0:1] != "v":
                    temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                    mRData_2d = mRData.reshape(-1, rPixX)
                    result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                                    fwKSStamp, fwKSStamp,
                                                    0x0, 0xffff, 5, mRData_2d, statSig)
                    if result[7] != 0:
                        sig2 = -1.0
                        sig3 = -1.0
                    else:
                        sig2 = result[4]
                        sig3 = result[5]
                        if sig2 < 0 or sig2 >= MAXVAL:
                            sig2 = -1.0
                        elif sig3 < 0 or sig3 >= MAXVAL:
                            sig3 = -1.0

                if (figMerit[0:1] == "v" and sig1 == -1) or \
                   (figMerit[0:1] == "s" and sig2 == -1) or \
                   (figMerit[0:1] == "h" and sig3 == -1):
                    sscnt_local[istamp] += 1
                    refill_indices.append(istamp)
                    check = 1
                else:
                    if figMerit[0:1] == "v":
                        sig = sig1
                    elif figMerit[0:1] == "s":
                        sig = sig2
                    elif figMerit[0:1] == "h":
                        sig = sig3

                    chi2_local[istamp] = sig
                    ss[nss] = sig
                    nss += 1
            else:
                nskippedSubstamps += 1

        mean, stdev, retcode = sigma_clip_numpy(ss[:nss], maxiter=10, stat_sig=statSig)

        meansigSubstamps = mean
        scatterSubstamps = stdev

        mask = (sscnt_local < saNss[:nS]) & ((chi2_local - mean) > kerSigReject * stdev)
        sscnt_local[mask] += 1
        refill_indices.extend(np.where(mask)[0].tolist())
        if np.any(mask):
            check = 1

    return (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
            refill_indices, sscnt_local, chi2_local)

# 迭代核拟合主循环：构建矩阵→求解→检查→重填，直到收敛或达到最大迭代
def fit_kernel_numpy(sa: Dict, imRef: np.ndarray, imConv: np.ndarray, imNoise: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int,
                     nS: int, fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int, figMerit: str,
                     kerSigReject: float, statSig: float, mRData: np.ndarray, ngauss: int, deg_fixe: List[int],
                     hwKernel: int, fwKernel: int, filter_x: np.ndarray, filter_y: np.ndarray, fillVal: float,
                     logger: Optional = None) -> Dict:
    """迭代核拟合主循环。迭代过程：从 stamps 数组中提取数据 → build_matrix_numpy 构建拟合矩阵 → build_scprod_numpy 构建标量积 → np.linalg.solve 求解 → check_again_numpy 检查并重填异常的 stamp。重复直到所有 stamp 收敛或达到最大迭代次数。

    sa: stamps 字典数组，包含 vectors/mat/scprod/xss/yss 等字段
    imRef, imConv, imNoise: 参考图像、卷积图像、噪声图像数组
    nCompKer: 核分量数
    kerOrder, bgOrder: 核和背景的多项式阶数
    nS: stamps 总数
    fwKSStamp, hwKSStamp: stamp 子区域全宽和半宽
    rPixX, rPixY: region 像素尺寸
    figMerit: 品质指标类型 ("v"/"s"/"h")
    kerSigReject: 核拟合的 sigma 拒绝阈值
    statSig: sigma-clip 的 sigma 阈值
    mRData: mask 数据数组
    ngauss: 高斯分量数
    deg_fixe: 每个高斯分量的多项式阶数列表
    hwKernel, fwKernel: 核半宽和全宽
    filter_x, filter_y: 预计算的核滤波器数组
    fillVal: 填充值
    logger: 日志记录器

    返回字典包含 kernelSol/meansigSubstamps/scatterSubstamps/NskippedSubstamps/stamps
    """
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)

    # 从 sa dict 提取扁平数组（用于后续扁平化的函数调用）
    saMat      = sa['mat']
    saVectors  = sa['vectors']
    saSscnt    = sa['sscnt']
    saNss      = sa['nss']
    saXss      = sa['xss']
    saYss      = sa['yss']
    saScprod   = sa['scprod']
    saKrefArea = sa['krefArea']
    saSumVal   = sa['sum_val']
    # saX0       = sa['x0']
    # saY0       = sa['y0']
    saChi2     = sa['chi2']
    # nC         = sa['nC']
    nKSStamps  = sa['nKSStamps']

    # import time
    iter_count = 0
    # t_start = time.time()
    logger.debug("  fitKernel: iteration %d start", iter_count)
    matrix, kernelSol, wxy = build_both_numpy(saMat, saVectors, saScprod, saSscnt, saNss, saXss, saYss,
                                               nS, imRef, nCompKer, kerOrder, bgOrder,
                                               fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
    logger.debug("  fitKernel: build_matrix+scprod done")
    # tm_bm = time.time(); logger.debug("  fitKernel: build+scprod %.3fs", tm_bm - tm)

    # indx = np.zeros(mat_size + 1, dtype=np.int32)
    kernelSol[1:mat_size+1] = np.linalg.solve(
        matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
    logger.debug("  fitKernel: solve done")
    # tm_slv = time.time(); logger.debug("  fitKernel: solve %.3fs", tm_slv - tm_bm)

    (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
     refill_indices, sscnt_update, chi2_update) = check_again_numpy(
        saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea, kernelSol,
        imNoise, nS, figMerit, kerSigReject, statSig,
        fwKSStamp, hwKSStamp, rPixX, rPixY, mRData, nCompKer, kerOrder, bgOrder)
    saSscnt[:nS] = sscnt_update
    saChi2[:nS] = chi2_update

    for idx in refill_indices:
        fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, idx, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                         hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                         bgOrder, nCompKer, filter_x, filter_y,
                         fillVal, mRData)
    logger.debug("  fitKernel: check_again done, check=%s", check)
    # tm_ca = time.time(); logger.debug("  fitKernel: check_again %.3fs", tm_ca - tm_slv)

    while check:
        iter_count += 1
        logger.debug("  fitKernel: iteration %d start", iter_count)

        matrix, kernelSol, wxy = build_both_numpy(saMat, saVectors, saScprod, saSscnt, saNss, saXss, saYss,
                                                   nS, imRef, nCompKer, kerOrder, bgOrder,
                                                   fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
        logger.debug("  fitKernel: build_matrix+scprod done")

        kernelSol[1:mat_size+1] = np.linalg.solve(
            matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
        logger.debug("  fitKernel: solve done")

        (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
         refill_indices, sscnt_update, chi2_update) = check_again_numpy(
            saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea, kernelSol,
            imNoise, nS, figMerit, kerSigReject, statSig,
            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData, nCompKer, kerOrder, bgOrder)
        saSscnt[:nS] = sscnt_update
        saChi2[:nS] = chi2_update

        for idx in refill_indices:
            fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, idx, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                             hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, filter_x, filter_y,
                             fillVal, mRData)
        logger.debug("  fitKernel: check_again done, check=%s", check)
        # tm_ca = time.time(); logger.debug("  fitKernel: check_again %.3fs", tm_ca - tm_slv)

    return {'kernelSol': kernelSol,
        'meansigSubstamps': meansigSubstamps, 'scatterSubstamps': scatterSubstamps,
        'NskippedSubstamps': nskippedSubstamps, 'stamps': sa}


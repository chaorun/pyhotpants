import numpy as np
# 导入numpy科学计算库
import numba
# 导入numba JIT编译器
from typing import List, Dict, Tuple, Optional
# 从typing模块导入类型标注工具

from ..functions import MAXVAL, sigma_clip_numpy, get_stamp_stats3_numpy
# 从functions子模块导入最大值常量、Sigma裁剪函数、Stamp统计量计算函数


def buildAllKernels(kernelSol: np.ndarray, kernelVec2d: np.ndarray, nCompKer: int, kerOrder: int, fwKernel: int, hwKernel: int, kcStep: int, rPixX: int, rPixY: int, xSize: int, ySize: int) -> Tuple[np.ndarray, int, int]:
# 预计算所有kcStep块锚点的空间可变卷积核矩阵。kernelSol为核多项式系数，kernelVec2d为核基函数2D矩阵
    halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
    # 计算X和Y方向半宽归一化因子（用于将坐标映射到[-1,1]范围）
    nstepsX = int(np.ceil(xSize / kcStep))
    # 计算X方向核缓存块数（向上取整）
    nstepsY = int(np.ceil(ySize / kcStep))
    # 计算Y方向核缓存块数
    nBlocks = nstepsX * nstepsY
    # 总块数

    i0Arr = np.arange(nstepsX) * kcStep + hwKernel
    # 所有块的X方向锚点i0坐标（偏移hwKernel）
    j0Arr = np.arange(nstepsY) * kcStep + hwKernel
    # 所有块的Y方向锚点j0坐标
    iGrid, jGrid = np.meshgrid(i0Arr, j0Arr, indexing='xy')
    # 生成所有块的网格坐标
    xi = (iGrid + hwKernel).ravel().astype(np.float64)
    # 计算每个块的核中心X坐标（i0+hwKernel），展平为1D
    yi = (jGrid + hwKernel).ravel().astype(np.float64)
    # 计算每个块的核中心Y坐标
    xf = (xi - halfX) / halfX
    # X坐标归一化到[-1,1]范围
    yf = (yi - halfY) / halfY
    # Y坐标归一化到[-1,1]范围

    kernelCoeffs = np.zeros((nBlocks, nCompKer), dtype=np.float64)
    # 分配所有块的核系数矩阵 (nBlocks × nCompKer)
    kernelCoeffs[:, 0] = kernelSol[1]
    # 第0个系数固定为kernelSol[1]（常数核项，对所有块相同）

    k = 2
    # kernelSol索引从2开始（跳过0和1）
    for ig in range(1, nCompKer):
    # 遍历每个核分量（从1开始，0已处理）
        coeff = np.zeros(nBlocks, dtype=np.float64)
        # 当前分量的系数数组
        ax = np.ones(nBlocks, dtype=np.float64)
        # X方向多项式系数积初值
        for ix in range(kerOrder + 1):
        # 遍历X方向多项式阶数
            ay = np.ones(nBlocks, dtype=np.float64)
            # Y方向多项式系数积初值
            for iy in range(kerOrder - ix + 1):
            # 遍历Y方向多项式阶数（受kerOrder限制）
                coeff += kernelSol[k] * ax * ay
                # 累加系数：kernelSol[k] × x^ix × y^iy
                k += 1
                # 索引前移
                ay *= yf
                # 更新y^iy系数
            ax *= xf
            # 更新x^ix系数
        kernelCoeffs[:, ig] = coeff
        # 保存当前分量的系数

    allKernels = kernelCoeffs @ kernelVec2d
    # 矩阵乘法：系数矩阵(nBlocks×nCompKer) × 核基向量(nCompKer×fwSq) → 所有块的卷积核(nBlocks×fwSq)

    return allKernels, nstepsX, nstepsY
    # 返回所有块核矩阵和方向块数

@numba.jit(nopython=True, parallel=True)
def spatial_convolve_jit_kernel(
# 空间域卷积JIT核函数：逐块遍历像素，用预计算或在线构造的核做卷积、计算variance、传播mask
    image: np.ndarray, variance: np.ndarray, cMask: np.ndarray,
    # 输入：图像1D、方差1D、输入遮罩1D
    cRdata: np.ndarray, vData: np.ndarray, mRData: np.ndarray,
    # 输出：卷积结果、方差结果、更新后的遮罩（均原地写入）
    kernelSol: np.ndarray,
    # 核多项式系数向量
    xSize: int, ySize: int, nCompKer: int, kerOrder: int, bgOrder: int, fwKernel: int, hwKernel: int,
    # 图像尺寸和核参数
    kcStep: int, rPixX: int, rPixY: int, kerFracMask: float, dovar: int, convolveVariance: int,
    # 核步长、区域尺寸归一化、核分数遮罩阈值、方差计算标志
    kernel_vec_2d: np.ndarray,
    # 核基函数2D矩阵（在线构造核时使用）
    allKernels: Optional[np.ndarray] = None,
    # 预计算的所有块核矩阵（为None时在线构造）
    nstepsX_in: Optional[int] = None) -> None:
    # X方向块数（allKernels不为None时必须传入）
    fwSq = fwKernel * fwKernel
    # 核的总像素数
    FLAG_INPUT_ISBAD = np.int32(0x80)
    # 输入坏像素标记常量（128）
    FLAG_OUTPUT_ISBAD = np.int32(0x8000)
    # 输出坏像素标记常量（32768）
    FLAG_BAD_CONV = np.int32(0x10)
    # 坏卷积标记常量（16）
    FLAG_OK_CONV = np.int32(0x40)
    # 正常卷积标记常量（64）
    halfX = 0.5 * rPixX
    # X方向半宽（用于坐标归一化）
    halfY = 0.5 * rPixY
    # Y方向半宽

    nsteps_x = int(np.ceil(xSize / kcStep))
    # X方向核块数
    nsteps_y = int(np.ceil(ySize / kcStep))
    # Y方向核块数

    for j1 in numba.prange(nsteps_y):
    # 并行遍历Y方向核块
        j0 = j1 * kcStep + hwKernel
        # 当前核块的Y起始坐标
        for i1 in range(nsteps_x):
        # 遍历X方向核块
            i0 = i1 * kcStep + hwKernel
            # 当前核块的X起始坐标

            if allKernels is not None:
            # 使用预计算的核矩阵
                kernel = allKernels[j1 * nstepsX_in + i1]
                # 从预计算数组中取出对应块核
            else:
            # 在线构造核（调用内联make_kernel）
                kernel = np.zeros(fwSq, dtype=np.float64)
                # 分配核数组
                kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
                # 分配核系数数组
                xi = i0 + hwKernel
                # 核中心X坐标
                yi = j0 + hwKernel
                # 核中心Y坐标
                xf = (xi - halfX) / halfX
                # X坐标归一化到[-1,1]
                yf = (yi - halfY) / halfY
                # Y坐标归一化

                k = 2
                # kernelSol索引从2开始
                for i1k in range(1, nCompKer):
                # 遍历核分量
                    coeff = 0.0; ax = 1.0
                    # 初始化系数和X方向多项式
                    for ix in range(kerOrder + 1):
                    # X多项式阶数
                        ay = 1.0
                        # Y多项式初值
                        for iy in range(kerOrder - ix + 1):
                        # Y多项式阶数
                            coeff += kernelSol[k] * ax * ay
                            # 累加加权系数
                            k += 1
                            # 索引前移
                            ay *= yf
                            # 更新Y幂
                        ax *= xf
                        # 更新X幂
                    kernel_coeffs[i1k] = coeff
                    # 保存分量系数
                kernel_coeffs[0] = kernelSol[1]
                # 第0个系数为常数核项

                for ii in range(fwSq):
                # 遍历核的每个像素
                    for c in range(nCompKer):
                    # 遍历核分量
                        kernel[ii] += kernel_coeffs[c] * kernel_vec_2d[c, ii]
                        # 系数×基向量 → 最终核值
                # ---- end make_kernel ----

            for j2 in range(kcStep):
            # 遍历当前块内Y方向像素
                j = j0 + j2
                # 当前像素Y坐标
                if j >= ySize - hwKernel:
                    break
                    # 超出有效范围
                for i2 in range(kcStep):
                # 遍历X方向像素
                    i = i0 + i2
                    # 当前像素X坐标
                    if i >= xSize - hwKernel:
                        break
                        # 超出有效范围

                    ni = i + xSize * j
                    # 当前像素1D索引
                    q = 0.0; qv = 0.0; aks = 0.0; uks = 0.0; mbit = np.int32(0)
                    # 初始化：卷积结果、方差结果、全部核和、有效核和、遮罩位

                    for jc in range(j - hwKernel, j + hwKernel + 1):
                    # 遍历核在Y方向的范围
                        jk = j - jc + hwKernel
                        # 核的Y索引
                        for ic in range(i - hwKernel, i + hwKernel + 1):
                        # 遍历核在X方向的范围
                            ik = i - ic + hwKernel
                            # 核的X索引
                            nc = ic + xSize * jc
                            # 图像像素1D索引
                            kk = kernel[ik + jk * fwKernel]
                            # 核值

                            q += image[nc] * kk
                            # 累加卷积结果
                            if dovar:
                            # 如果需要计算方差
                                if convolveVariance:
                                # 方差卷积模式（卷积variance本身）
                                    qv += variance[nc] * kk * kk
                                else:
                                    qv += variance[nc] * kk * kk
                                # 累加方差（当前两种模式相同）
                            mbit |= cMask[nc]
                            # 累积输入遮罩位
                            aks += abs(kk)
                            # 累加核绝对值和（全像素）
                            if not (cMask[nc] & FLAG_INPUT_ISBAD):
                                uks += abs(kk)
                                # 累加有效像素的核绝对值和

                    cRdata[ni] = q
                    # 输出卷积结果
                    if dovar:
                        vData[ni] = qv
                        # 输出方差结果

                    mRData[ni] = mRData[ni] | cMask[ni]
                    # 传播输入遮罩到输出遮罩
                    mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD * np.int32((cMask[ni] & FLAG_INPUT_ISBAD) > 0))
                    # 如果输入像素为坏，标记输出也为坏

                    if mbit:
                    # 如果核覆盖区域有任何标记
                        if aks > 0.0 and (uks / aks) < kerFracMask:
                        # 有效核占比低于阈值：核覆盖太多坏像素
                            mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
                            # 标记为坏卷积
                        else:
                            mRData[ni] = mRData[ni] | FLAG_OK_CONV
                            # 标记为正常卷积

    nCompForBG = nCompKer - 1
    ncompBG = nCompForBG * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    halfX = np.float64(0.5 * rPixX)
    halfY = np.float64(0.5 * rPixY)
    for j in range(hwKernel, ySize - hwKernel):
        yf = (j - halfY) / halfY
        for i in range(hwKernel, xSize - hwKernel):
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
            cRdata[i + xSize * j] += bg

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
    # 计算核分量减1（用于背景系数偏移）
    ncompBG = nCompForBG * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    # 背景系数的起始索引
    halfX = np.float64(0.5 * rPixX)
    # X半宽归一化
    halfY = np.float64(0.5 * rPixY)
    # Y半宽归一化
    for j in range(hwKernel, rPixY - hwKernel):
    # 遍历有效Y范围
        yf = (j - halfY) / halfY
        # Y坐标归一化
        for i in range(hwKernel, rPixX - hwKernel):
        # 遍历有效X范围
            xf = (i - halfX) / halfX
            # X坐标归一化
            bg = np.float64(0.0); k = 1; ax = np.float64(1.0)
            # 初始化背景值、系数索引、X多项式
            for idegx in range(bgOrder + 1):
            # X多项式阶数
                ay = np.float64(1.0)
                # Y多项式初值
                for idegy in range(bgOrder - idegx + 1):
                # Y多项式阶数
                    bg += kernelSol[ncompBG + k] * ax * ay
                    # 累加背景系数
                    k += 1; ay *= yf
                    # 索引前移，更新Y幂
                ax *= xf
                # 更新X幂
            oRData1d[i + rPixX * j] += bg
            # 将背景叠加到输出像素

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
    # 信噪比累加器
    nsig = 0
    for j in range(fwKSStamp):
    # 遍历子Stamp Y方向
        yRegion2 = yRegion - hwKSStamp + j
        # 绝对Y坐标
        for i in range(fwKSStamp):
        # 遍历X方向
            xRegion2 = xRegion - hwKSStamp + i
            # 绝对X坐标
            idx = xRegion2 + rPixX * yRegion2
            # 1D索引
            idat = imDiff[idx]
            # 差图像值
            ndat = imNoise[idx]
            # 噪声值
            if (mRData[idx] & FLAG) != 0:
            # 坏像素则跳过
                continue
            nsig += 1
            # 有效像素计数加1
            sig += (idat * idat) / (ndat * ndat)
            # 累加方差加权和：(diff²/noise²)
    if nsig > 0:
    # 有效像素数大于0
        sig /= nsig
        # 平均信噪比
    else:
        sig = -1.0
        # 无有效像素，返回-1
    return sig
    # 返回信噪比

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
    # kernelSol索引从2开始（跳过0和1号元素）
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    # X坐标归一化到[-1,1]范围
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
    # Y坐标归一化
    for i1 in range(1, nCompKer):
    # 遍历核分量（从1开始）
        coeff = 0.0; ax = 1.0
        # 初始化系数和X多项式
        for ix in range(kerOrder + 1):
        # X多项式阶数
            ay = 1.0
            # Y多项式初值
            for iy in range(kerOrder - ix + 1):
            # Y多项式阶数
                coeff += float(kernelSol[k]) * ax * ay
                # 累加空间多项式加权值
                k += 1; ay *= yf
                # 索引前移，更新Y幂
            ax *= xf
            # 更新X幂
        kernel_coeffs[i1] = coeff
        # 保存当前分量系数
    kernel_coeffs[0] = float(kernelSol[1])
    # 第0个系数为常数核项
    fwSq = fwKernel * fwKernel
    # 核总像素数
    for i in range(fwSq):
        kernel[i] = 0.0
        # 清零核数组
    sum_kernel = 0.0
    # 核总和累加器
    for i in range(fwSq):
    # 遍历核像素
        val = 0.0
        # 当前像素核值
        for i1 in range(nCompKer):
        # 遍历核分量
            val += float(kernel_coeffs[i1]) * float(kernel_vec[i1][i])
            # 系数×基向量（float强制转换确保精度）
        kernel[i] = val
        # 写入核数组
        sum_kernel += val
        # 累加总和
    return sum_kernel
    # 返回核元素之和

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
    # 分配核向量内存（fwSq个double）
    for i in range(fwKernel):
    # 遍历核Y方向
        for j in range(fwKernel):
        # 遍历核X方向
            vector[i + fwKernel * j] = float(PCA[n][i + fwKernel * j])
            # 从PCA矩阵第n行复制为核向量
    ren = 0
    # 正交化标志初始为0
    if n > 0:
    # 非第0个分量时需要正交化
        kernel0 = kernel_vec[0]
        # 取第0个核向量作为参考
        for i in range(fwKernel * fwKernel):
        # 遍历每个像素
            vector[i] -= kernel0[i]
            # 减去参考向量实现正交化
    return vector, ren
    # 返回核向量和正交化标志

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
    # 如果使用PCA模式
        vec, ren = kernel_vector_pca_numpy(n, fwKernel, PCA, kernel_vec)
        # 委托给PCA版本的函数
        return vec, ren
        # 直接返回

    vector = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    # 分配核向量内存
    dx = (deg_x // 2) * 2 - deg_x
    # 计算deg_x的奇偶性（用于判断是否需要归一化）：偶数→0，奇数→±1
    dy = (deg_y // 2) * 2 - deg_y
    # 计算deg_y的奇偶性
    sum_x = 0.0; sum_y = 0.0; ren = 0
    # X/Y滤波器求和变量、正交化标志

    sig = float(sigma_gauss[ig])
    # 取当前高斯分量的σ值（实际存的是1/(2σ²)）
    for ix in range(fwKernel):
    # 遍历核的X方向
        x = float(ix - hwKernel)
        # 像素偏移量（-hwKernel到+hwKernel）
        k = ix + n * fwKernel
        # 滤波器数组中的索引（n决定写入哪个分量对应的位置）
        qe = np.exp(-x * x * sig)
        # 高斯包络值：exp(-x²×σ)
        filter_x[k] = qe * (x ** deg_x)
        # X方向滤波器：高斯×x^deg_x
        filter_y[k] = qe * (x ** deg_y)
        # Y方向滤波器：高斯×x^deg_y
        sum_x += filter_x[k]
        # 累加X滤波器总和（用于归一化）
        sum_y += filter_y[k]
        # 累加Y滤波器总和

    kernel0 = None
    # 第0个核向量引用
    if n > 0:
        kernel0 = kernel_vec[0].copy()
        # 复制第0个核向量（用于正交化）

    sum_x = 1.0 / sum_x; sum_y = 1.0 / sum_y
    # 计算归一化系数（使滤波器总和为1）

    if dx == 0 and dy == 0:
    # 如果X和Y阶数均为偶数（包括0）：需要归一化
        for ix in range(fwKernel):
            filter_x[ix + n * fwKernel] *= sum_x
            # 归一化X滤波器
            filter_y[ix + n * fwKernel] *= sum_y
            # 归一化Y滤波器

        for i in range(fwKernel):
        # 遍历核Y方向
            for j in range(fwKernel):
            # 遍历核X方向
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]
                # 可分离滤波器外积：fx(x)×fy(y) → 2D核向量

        if n > 0:
        # 非第0个分量：正交化
            for i in range(fwKernel * fwKernel):
                vector[i] -= kernel0[i]
                # 减去第0个核向量
            ren = 1
            # 设置正交化标志
    else:
    # dx或dy非零（奇数次幂）：不需要归一化
        for i in range(fwKernel):
            for j in range(fwKernel):
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]
                # 外积（无归一化）
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]
                # 外积（无归一化）

    return vector, ren
    # 返回核向量和正交化标志

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
    # 核向量列表初始化为空
    nvec = 0
    # 核分量计数器
    for ig in range(ngauss):
    # 遍历高斯分量
        for idegx in range(int(deg_fixe[ig]) + 1):
        # 遍历X多项式阶数
            for idegy in range(int(deg_fixe[ig]) - idegx + 1):
            # 遍历Y多项式阶数
                vec, ren = kernel_vector_numpy(nvec, idegx, idegy, ig, usePCA,
                # 调用kernel_vector_numpy生成单个核基向量
                                                fwKernel, hwKernel, sigma_gauss, filter_x, filter_y, kernel_vec, PCA)
                # 传入核尺寸、σ值、滤波器数组（原地写入）、已生成核向量（用于正交化）
                kernel_vec.append(vec)
                # 添加新生成的核向量到列表
                nvec += 1
                # 计数器加1
    return kernel_vec
    # 返回完整核基向量列表

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
    # 遍历所有Stamp
        if valid_mask[istamp] == 0:
            continue
            # 跳过无效Stamp

        for i in range(ncomp):
        # 遍历核分量组合数
            i1 = i // ncomp2
            # 第1个核分量索引
            i2 = i - i1 * ncomp2
            # 空间变化分量索引
            for j in range(i + 1):
            # 遍历对称部分（含对角线）
                j1 = j // ncomp2; j2 = j - j1 * ncomp2
                # 第2个核分量和空间变化分量索引
                matrix[i + 2, j + 2] += wxy[istamp, i2] * wxy[istamp, j2] * all_mat[istamp, i1 + 2, j1 + 2]
                # 空间加权的局部矩阵累加：w(i2)×w(j2)×local_mat(i1,j1)

        matrix[1, 1] += all_mat[istamp, 1, 1]
        # 累加常数项（无空间变化权重）
        for i in range(ncomp):
            i1 = i // ncomp2; i2 = i - i1 * ncomp2
            matrix[i + 2, 1] += wxy[istamp, i2] * all_mat[istamp, i1 + 2, 1]
            # 空间变化分量与常数项的交叉项

        for ibg in range(nbg_vec):
        # 遍历背景多项式项
            ii = ncomp + ibg + 1
            # 背景项在矩阵中的行索引
            ivecbg = ncomp1 + ibg + 1
            # 背景向量在vectors中的索引
            for i1 in range(1, ncomp1 + 1):
            # 遍历核分量
                p0 = np.dot(all_vectors[istamp, i1, :], all_vectors[istamp, ivecbg, :])
                # 核分量向量与背景向量的点积
                for i2 in range(ncomp2):
                    jj = (i1 - 1) * ncomp2 + i2 + 1
                    matrix[ii + 1, jj + 1] += p0 * wxy[istamp, i2]
                    # 空间加权的核-背景交叉项
            p0 = np.dot(all_vectors[istamp, 0, :], all_vectors[istamp, ivecbg, :])
            # 第0个核向量与背景向量的点积
            matrix[ii + 1, 1] += p0
            # 常数核项与背景的交叉项
            for jbg in range(ibg + 1):
            # 背景-背景自交叉
                q = np.dot(all_vectors[istamp, ivecbg, :], all_vectors[istamp, ncomp1 + jbg + 1, :])
                # 两个背景向量的点积
                matrix[ii + 1, ncomp + jbg + 2] += q
                # 累加背景-背景项

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
            # 分量标量积
            for i2 in range(ncomp2):
                ii = (i1 - 1) * ncomp2 + i2 + 1
                kernelSol[ii + 1] += p0 * wxy[istamp, i2]
                # 空间加权标量积

        for k in range(pixStamp):
            img_patch[k] = image_flat[so_x[k] + xi + rPixX * (so_y[k] + yi)]
            # 提取图像参考像素值
        for ibg in range(nbg_vec):
            kernelSol[ncomp + ibg + 2] += np.dot(all_vectors[istamp, ncomp1 + ibg + 1, :], img_patch)
            # 背景向量与图像点积

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
    ncomp1 = nCompKer - 1; ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2; ncomp = ncomp1 * ncomp2
    # 核分量数和空间变化组合数
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2; pixStamp = fwKSStamp * fwKSStamp; mat_size = ncomp1 * ncomp2 + nbg_vec + 1
    # 背景项数、子Stamp像素数、矩阵尺寸

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)
    # 分配空间权重矩阵
    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    # 有效stamp掩码
    n_valid = valid_mask.sum()
    if n_valid == 0:
        matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        return matrix, wxy
        # 无有效stamp返回零矩阵
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    # 安全索引
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp当前子中心X坐标
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp当前子中心Y坐标

    nC_valid = nCompKer + 1; all_mat = saMat[:nS, :nC_valid, :nC_valid]
    # 提取局部矩阵子数组
    nvec_total = nCompKer + nbg_vec; all_vectors = saVectors[:nS, :nvec_total, :]
    # 提取向量子数组

    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
    # 分配聚合矩阵

    rPixX2 = np.float64(0.5 * rPixX); rPixY2 = np.float64(0.5 * rPixY)
    # 半宽归一化
    for s in range(nS):
    # 计算每个stamp的空间多项式权重
        if valid_mask[s] == 0: continue
        fx = (np.float64(all_x[s]) - rPixX2) / rPixX2
        # X归一化
        fy = (np.float64(all_y[s]) - rPixY2) / rPixY2
        # Y归一化
        kk = 0; a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2; kk += 1; a2 *= fy
                # 空间多项式权重
            a1 *= fx

    build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
    # 调用JIT构建聚合拟合矩阵
                     wxy, matrix, nS, kerOrder, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec,
                     pixStamp)

    tri_i, tri_j = np.tril_indices(mat_size, -1)
    # 获取下三角索引（不含对角线）
    matrix[tri_j + 1, tri_i + 1] = matrix[tri_i + 1, tri_j + 1]
    # 对称填充：将下三角复制到上三角

    return matrix, wxy
    # 返回聚合矩阵和空间权重矩阵

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
    ncomp1 = nCompKer - 1; ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    # 核分量数和空间多项式项数
    ncomp = ncomp1 * ncomp2; nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2; pixStamp = fwKSStamp * fwKSStamp
    # 组合数、背景项数、子Stamp像素数

    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    # 有效stamp掩码
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
        # 无有效stamp返回零向量
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    # 安全索引（防止越界）
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp当前子中心X坐标
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp当前子中心Y坐标

    all_scprod = saScprod[:nS, :nCompKer + 1]
    # 提取标量积子数组
    nvec_total = nCompKer + nbg_vec
    # 向量维度
    all_vectors = saVectors[:nS, :nvec_total, :]
    # 提取向量子数组

    image_arr = np.asarray(image, dtype=np.float64)
    # 参考图像转float64
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
    # 分配标量积向量

    build_scprod_jit(all_vectors, all_scprod, valid_mask, all_x, all_y,
    # 调用JIT构建标量积
                     wxy, image_arr, kernelSol,
                     nS, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec)

    return kernelSol
    # 返回标量积向量

# 合并打包：一次准备数据，一次 jit 调用完成 matrix + kernelSol
def build_both_numpy(saMat: np.ndarray, saVectors: np.ndarray, saScprod: np.ndarray,
                     saSscnt: np.ndarray, saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray,
                     nS: int, image: np.ndarray, nCompKer: int, kerOrder: int, bgOrder: int,
                     fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int, nKSStamps: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ncomp1 = nCompKer - 1; ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2; ncomp = ncomp1 * ncomp2
    # 核分量数和空间变化组合数
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2; pixStamp = fwKSStamp * fwKSStamp; mat_size = ncomp1 * ncomp2 + nbg_vec + 1
    # 背景项数、子Stamp像素数、拟合矩阵尺寸

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)
    # 空间权重矩阵
    valid_mask = (saSscnt[:nS] < saNss[:nS]).astype(np.int32)
    # 有效stamp掩码
    n_valid = valid_mask.sum()
    if n_valid == 0:
        matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
        return matrix, kernelSol, wxy
        # 无有效stamp时返回零矩阵和零向量
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp子中心X坐标
    all_y = np.where(valid_mask, saYss[:nS][np.arange(nS), safe_sscnt], 0).astype(np.int64)
    # 有效stamp子中心Y坐标

    rPixX2 = np.float64(0.5 * rPixX); rPixY2 = np.float64(0.5 * rPixY)
    # 半宽归一化
    for s in range(nS):
    # 计算每个stamp的空间多项式权重
        if valid_mask[s] == 0: continue
        fx = (np.float64(all_x[s]) - rPixX2) / rPixX2
        # X坐标归一化到[-1,1]
        fy = (np.float64(all_y[s]) - rPixY2) / rPixY2
        # Y坐标归一化
        kk = 0; a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2; kk += 1; a2 *= fy
                # 写入空间多项式权重
            a1 *= fx

    nC_valid = nCompKer + 1; all_mat = saMat[:nS, :nC_valid, :nC_valid]
    # 提取局部矩阵子数组
    nvec_total = nCompKer + nbg_vec; all_vectors = saVectors[:nS, :nvec_total, :]
    # 提取向量子数组
    all_scprod = saScprod[:nS, :nCompKer + 1]
    # 提取标量积子数组

    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
    # 分配聚合矩阵
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
    # 分配标量积向量
    image_arr = np.asarray(image, dtype=np.float64)
    # 参考图像转float64

    build_both_jit(all_mat, all_vectors, all_scprod, valid_mask, all_x, all_y,
    # 调用JIT函数一次性完成矩阵构建和标量积累加
                   wxy, matrix, kernelSol, image_arr,
                   nS, kerOrder, fwKSStamp, hwKSStamp, rPixX, rPixY,
                   ncomp, ncomp1, ncomp2, nbg_vec, pixStamp)

    tri_i, tri_j = np.tril_indices(mat_size, -1)
    matrix[tri_j + 1, tri_i + 1] = matrix[tri_i + 1, tri_j + 1]
    # 对称填充：将下三角复制到上三角（JIT仅计算了下三角）

    return matrix, kernelSol, wxy
    # 返回聚合矩阵、标量积向量、空间权重矩阵

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
    # 输入坏像素标记常量
    fwSqStamp = fwKSStamp * fwKSStamp
    # 子Stamp正方形像素数
    fwSqKernel = fwKernel * fwKernel
    # 卷积核正方形像素数

    kernels = np.zeros((nvec, fwSqKernel), dtype=np.float64)
    # 分配预计算核数组：nvec个核分量×(fwKernel²)个像素
    for n in range(nvec):
    # 遍历每个核向量分量
        for jc in range(fwKernel):
        # 核Y方向
            for ic in range(fwKernel):
            # 核X方向
                fy_idx = fwKernel - 1 - jc
                # Y方向滤波器索引（反转）
                fx_idx = fwKernel - 1 - ic
                # X方向滤波器索引（反转）
                kernels[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]
                # 可分离滤波器外积：fy(y_index)×fx(x_index) → 2D核

    for si in numba.prange(n_stamps):
    # 并行遍历每个Stamp
        xi_s = xi[si]; yi_s = yi[si]
        # 当前Stamp子中心坐标

        for n in range(nvec):
        # 遍历核向量分量
            for fsi in range(fwSqStamp):
            # 遍历子Stamp每个像素
                out_vectors[si, n, fsi] = 0.0
                # 清零输出向量

        for fsi in range(fwSqStamp):
        # 遍历子Stamp像素
            fsi_x = fsi % fwKSStamp
            # 像素X位置（0..fwKSStamp-1）
            fsi_y = fsi // fwKSStamp
            # 像素Y位置
            img_base_x = xi_s - hwKSStamp + fsi_x - hwKernel
            # 卷积区域左上角X坐标
            img_base_y = yi_s - hwKSStamp + fsi_y - hwKernel
            # 卷积区域左上角Y坐标
            for fki in range(fwSqKernel):
            # 遍历卷积核像素
                kx = fki % fwKernel; ky = fki // fwKernel
                # 核像素的X和Y位置
                img_x = img_base_x + kx; img_y = img_base_y + ky
                # 对应的图像像素坐标
                img_val = image[img_x + rPixX * img_y]
                # 取图像像素值
                for n in range(nvec):
                    out_vectors[si, n, fsi] += kernels[n, fki] * img_val
                    # 累加卷积：向量[n] × 核 × 图像

        for n in range(nvec):
        # 正交化处理
            if renFlags_arr[n]:
            # 如果当前分量需要正交化
                for fsi in range(fwSqStamp):
                    out_vectors[si, n, fsi] -= out_vectors[si, 0, fsi]
                    # 减去第0个分量（正交化）

        for fsi in range(fwSqStamp):
            out_krefArea[si, fsi] = fillVal
            # 初始填充参考区域为fillVal

        sumVal = 0.0
        # 像素绝对值之和
        for y_offset in range(fwKSStamp):
        # 遍历子Stamp Y方向
            img_y = yi_s - hwKSStamp + y_offset
            # 绝对Y坐标
            for x_offset in range(fwKSStamp):
            # 遍历X方向
                img_x = xi_s - hwKSStamp + x_offset
                # 绝对X坐标
                k = img_x + rPixX * img_y
                # 1D索引
                dpt = imRef[k]
                # 参考图像像素值
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
                ax = 1.0; nv = nvec
                # 初始化X多项式 = 1，背景向量起始索引
                for idegx in range(bgOrder + 1):
                # X方向多项式阶数
                    ay = 1.0
                    # Y多项式初值
                    for idegy in range(bgOrder - idegx + 1):
                    # Y方向多项式阶数
                        out_vectors[si, nv, ipix] = ax * ay
                        # 写入背景向量：x^i × y^j
                        ay *= yf
                        # 更新Y幂
                        nv += 1
                        # 向量索引前移
                    ax *= xf
                    # 更新X幂

        ncomp1 = nCompKer; pixStamp = fwSqStamp
        # 核分量数和子Stamp像素数

        for i in range(ncomp1):
        # 构建局部矩阵：遍历核分量
            for j in range(i + 1):
            # 对称矩阵（下三角）
                q = 0.0
                for k in range(pixStamp):
                    q += out_vectors[si, i, k] * out_vectors[si, j, k]
                    # 向量i和向量j的点积
                out_mat[si, i + 1, j + 1] = q
                # 写入局部矩阵

        ivecbg = ncomp1
        # 第一个背景向量索引
        for i1 in range(ncomp1):
            p0 = 0.0
            for k in range(pixStamp):
                p0 += out_vectors[si, i1, k] * out_vectors[si, ivecbg, k]
                # 核分量向量与背景向量的点积
            out_mat[si, ncomp1 + 1, i1 + 1] = p0
            # 核-背景交叉项

        q = 0.0
        for k in range(pixStamp):
            q += out_vectors[si, ivecbg, k] * out_vectors[si, ncomp1, k]
            # 背景向量自点积
        out_mat[si, ncomp1 + 1, ncomp1 + 1] = q
        # 背景-背景项

        for i1 in range(ncomp1):
        # 构建局部标量积：遍历核分量
            p0 = 0.0
            for xc in range(-hwKSStamp, hwKSStamp + 1):
            # 遍历子Stamp X偏移
                for yc in range(-hwKSStamp, hwKSStamp + 1):
                # 遍历子Stamp Y偏移
                    k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                    # 1D索引
                    p0 += out_vectors[si, i1, k] * out_krefArea[si, k]
                    # 向量×参考区域
            out_scprod[si, i1 + 1] = p0
            # 写入局部标量积

        q = 0.0
        for xc in range(-hwKSStamp, hwKSStamp + 1):
            for yc in range(-hwKSStamp, hwKSStamp + 1):
                k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                q += out_vectors[si, ncomp1, k] * out_krefArea[si, k]
                # 背景向量×参考区域
        out_scprod[si, ncomp1 + 1] = q
        # 背景标量积

    return 0
    # 填充成功返回0

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
        # 子Stamp计数器已耗尽，返回失败

    result = fill_stamp_numba(saXss, saYss, saSscnt, saNss, [si], imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
    # 调用批量填充函数（包装为单元素列表）
                              hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                              filter_x, filter_y, fillVal, mRData)
    out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val = result
    # 解包返回的5个数组
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2; fwSqStamp = fwKSStamp * fwKSStamp; nC = nCompKer + 1
    # 计算维度：背景项数、子Stamp像素数、拟合矩阵列数
    saVectors[si, :nCompKer + nbg, :fwSqStamp] = out_vectors[0]
    # 将卷积向量写回saVectors对应位置
    saKrefArea[si, :] = out_krefArea[0]
    # 将参考区域写回saKrefArea
    saMat[si, :nC + 1, :nC + 1] = out_mat[0]
    # 将局部矩阵写回saMat
    saScprod[si, :nC + 1] = out_scprod[0]
    # 将标量积写回saScprod
    saSumVal[si] = out_sum_val[0]
    # 将像素和写回saSumVal
    return 0
    # 填充成功

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
    # 本批次stamp数量

    nvec = 0; renFlags = []
    # 核向量分量计数和正交化标志列表
    for ig in range(ngauss):
    # 遍历高斯分量
        for idegx in range(int(deg_fixe[ig]) + 1):
        # X多项式阶数
            for idegy in range(int(deg_fixe[ig]) - idegx + 1):
            # Y多项式阶数
                ren = 0; dx = (idegx // 2) * 2 - idegx; dy = (idegy // 2) * 2 - idegy
                # 判断是否需要正交化：dx==0且dy==0且非首个分量时需要
                if dx == 0 and dy == 0 and nvec > 0: ren = 1
                renFlags.append(ren); nvec += 1
                # 记录正交化标志，计数加1

    xi_arr = np.zeros(n_stamps, dtype=np.int32)
    yi_arr = np.zeros(n_stamps, dtype=np.int32)
    valid_mask = np.zeros(n_stamps, dtype=np.int32); n_valid = 0
    # 预分配坐标数组、有效掩码和计数器
    for idx in range(n_stamps):
    # 收集有效stamp的坐标
        si = si_list[idx]
        if saSscnt[si] < saNss[si]:
        # stamp有效（仍有未使用的子Stamp）
            xi_arr[n_valid] = int(saXss[si, saSscnt[si]])
            # 取当前子中心X坐标
            yi_arr[n_valid] = int(saYss[si, saSscnt[si]])
            # 取当前子中心Y坐标
            valid_mask[idx] = 1; n_valid += 1
            # 标记有效
    if n_valid == 0:
        return out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val[:, 0]
        # 无有效stamp，提前返回空数组

    img_flat = np.asarray(imConv, dtype=np.float64).ravel()
    # 卷积图像展平
    imRef_flat = np.asarray(imRef, dtype=np.float64).ravel()
    # 参考图像展平
    fx = np.asarray(filter_x, dtype=np.float64); fy = np.asarray(filter_y, dtype=np.float64)
    # X/Y滤波器转float64
    rflags = np.array(renFlags, dtype=np.int32)
    # 正交化标志转int32
    mRData1d = mRData.ravel()
    # 遮罩展平

    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2; fwSqStamp = fwKSStamp * fwKSStamp; nC = nCompKer + 1
    # 计算维度参数

    out_vectors = np.zeros((n_stamps, nvec + nbg, fwSqStamp), dtype=np.float64)
    # 分配输出向量数组
    out_krefArea = np.zeros((n_stamps, fwSqStamp), dtype=np.float64)
    # 分配输出参考区域数组
    out_mat = np.zeros((n_stamps, nC + 1, nC + 1), dtype=np.float64)
    # 分配输出局部矩阵数组
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
    LOCAL_ZVAL = 1e-10; LOCAL_MAXVAL = 1e10; LOCAL_IBAD = 0x80; LOCAL_ISNAN = 0x08
    # 局部常量：零值阈值、最大值阈值、坏像素标记、NaN标记
    fwSq = fwKSStamp * fwKSStamp
    # 子Stamp像素数

    nss = 0
    # 有效信噪比计数器
    for si in range(nS):
    # 遍历所有Stamp
        scnt = sa_sscnt[si]
        # 当前子Stamp索引
        if scnt >= sa_nss[si]:
        # 无有效子Stamp
            chi2_local[si] = -1.0; continue
            # 标记为无效并跳过
        bg_val = batched_bg[si]
        # 取预计算的背景值
        csModel = np.empty(fwSq, dtype=np.float64)
        # 分配卷积模型缓存
        coeff0 = kernelSol[1]
        # 常数核系数
        for i in range(fwSq):
            csModel[i] = coeff0 * sa_vectors[si, 0, i]
            # 第0个核向量×常数系数
        for i1j in range(1, nCompKer):
        # 遍历高阶核分量
            coeff2 = batched_coeffs[si, i1j]
            # 取预计算的空间变化核系数
            for i in range(fwSq):
                csModel[i] += coeff2 * sa_vectors[si, i1j, i]
                # 高阶分量累加
        im = sa_krefArea[si]
        # 取参考区域数据
        nsig = 0; sig1 = 0.0
        # 有效像素计数和信噪比累加器
        xi = sa_xss[si, scnt]; yi = sa_yss[si, scnt]
        # 当前子Stamp中心坐标
        for j in range(fwKSStamp):
        # Y方向
            yR = yi - hwKSStamp + j
            for i in range(fwKSStamp):
            # X方向
                xR = xi - hwKSStamp + i
                idk = i + j * fwKSStamp
                # 1D索引
                tdat = csModel[idk]; idat = im[idk]
                # 模型值和参考值
                ndat = imNoise[xR + rPixX * yR]
                # 噪声值
                diff_val = tdat - idat + bg_val
                # 残差 = 模型 - 参考 + 背景
                mr_idx = xR + rPixX * yR
                # 遮罩1D索引
                if (mRData1d[mr_idx] & LOCAL_IBAD) or (abs(idat) <= LOCAL_ZVAL):
                    continue
                    # 跳过坏像素或近零像素
                if np.isnan(tdat) or np.isnan(idat):
                    mRData1d[mr_idx] = mRData1d[mr_idx] | (LOCAL_IBAD | LOCAL_ISNAN)
                    # 标记NaN像素
                    continue
                nsig += 1
                sig1 += diff_val * diff_val / ndat
                # 累加卡方贡献：(residual²/noise)
        if nsig > 0:
        # 至少有一个有效像素
            sig1 /= nsig
            # 平均卡方
            if sig1 >= LOCAL_MAXVAL:
                sig1 = -1.0
                # 超标记为无效
        else:
            sig1 = -1.0
            # 无有效像素
        chi2_local[si] = sig1
        # 写入卡方值
        if sig1 != -1.0:
            ss[nss] = sig1; nss += 1
            # 有效卡方值收集到ss数组
        else:
            sscnt_local[si] += 1; refill_flags[si] = 1
            # 标记需要refill（前进到下一个子Stamp）

    if nss == 0:
        return (0.0, float(MAXVAL), 0)
        # 无有效Stamp，返回失败

    arr = np.asarray(ss[:nss], dtype=np.float32).astype(np.float64)
    # 有效卡方值数组
    count = nss; mask = np.zeros(count, dtype=np.int32)
    cnt = 0; ncnt = count; iternum = 0; mean_val = 0.0; stdev_val = 0.0; maxiter = 10
    # Sigma-clip迭代参数
    while (ncnt != cnt) and (iternum < maxiter):
    # 迭代剔除离群值
        cnt = ncnt; good = arr[mask == 0]; ncnt = len(good)
        if ncnt == 0: return (0.0, float(MAXVAL), 0)
        mean_val = float(good.mean())
        # 均值
        if ncnt == 1: return (mean_val, float(MAXVAL), 0)
        sg = good - mean_val
        stdev_val = float(np.sqrt(np.sum(sg * sg) / (ncnt - 1)))
        # 标准差
        istdev = 1.0 / stdev_val
        deviations = np.abs(sg) * istdev
        # 标准化偏差
        new_outliers = deviations > statSig
        # 标记离群值
        good_indices = np.where(mask == 0)[0]
        mask[good_indices[new_outliers]] = 1
        # 剔除
        ncnt = count - int(np.sum(mask)); iternum += 1

    ncheck = 0
    for si in range(nS):
    # 遍历所有Stamp，标记需要refill的
        if sscnt_local[si] < sa_nss[si] and chi2_local[si] != -1.0:
            if (chi2_local[si] - mean_val) > kerSigReject * stdev_val:
            # 卡方值显著高于均值+拒绝阈值×标准差
                sscnt_local[si] += 1; refill_flags[si] = 1; ncheck = 1
                # 前进子Stamp计数器并标记refill
    return (mean_val, stdev_val, ncheck)
    # 返回均值、标准差、是否有refill

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

    LOCAL_ZEROVAL = 1e-10; LOCAL_MAXVAL = 1e10; LOCAL_FLAG_INPUT_ISBAD = 0x80; LOCAL_FLAG_ISNAN = 0x08
    # 局部常量：零值阈值、最大值、输入坏像素标记、NaN标记
    fwSq = fwKSStamp * fwKSStamp
    # 子Stamp像素数

    for si in numba.prange(nS):
    # 并行遍历所有Stamp
        scnt = sa_sscnt[si]
        # 当前子Stamp索引
        if scnt >= sa_nss[si]:
        # 子Stamp耗尽
            out_sig1[si] = -1.0; out_sig2[si] = -1.0; out_sig3[si] = -1.0; continue
            # 标记全部为无效

        xi = sa_xss[si, scnt]; yi = sa_yss[si, scnt]
        # 当前子中心坐标

        if batched_bg is not None:
        # 使用预计算的背景值
            background = batched_bg[si]
        else:
        # 在线计算背景值
            ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
            # 背景系数起始索引
            background = 0.0; k = 1
            xf = (xi - 0.5 * rPixX) / (0.5 * rPixX); yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
            # 归一化坐标
            ax = 1.0
            for i in range(bgOrder + 1):
                ay = 1.0
                for j in range(bgOrder - i + 1):
                    background += kernelSol[ncompBG + k] * ax * ay
                    # 累加背景多项式值
                    k += 1; ay *= yf
                ax *= xf

        csModel = np.zeros(fwSq, dtype=np.float64)
        # 分配卷积模型缓存
        coeff = kernelSol[1]
        for i in range(fwSq):
            csModel[i] = coeff * sa_vectors[si, 0, i]
            # 常数核项×第0个核向量

        if batched_coeffs is not None:
        # 使用预计算的核系数
            for i1 in range(1, nCompKer):
                for i in range(fwSq):
                    csModel[i] += batched_coeffs[si, i1] * sa_vectors[si, i1, i]
                    # 高阶分量累加
        else:
        # 在线计算核系数
            kk = 2
            for i1 in range(1, nCompKer):
                coeff2 = 0.0; ax = 1.0
                for ix in range(kerOrder + 1):
                    ay = 1.0
                    for iy in range(kerOrder - ix + 1):
                        coeff2 += kernelSol[kk] * ax * ay
                        # 空间多项式加权
                        kk += 1; ay *= yf
                    ax *= xf
                for i in range(fwSq):
                    csModel[i] += coeff2 * sa_vectors[si, i1, i]
                    # 分量累加到模型

        im = sa_krefArea[si]
        # 参考区域数据

        nsig = 0; sig1 = 0.0; temp = np.zeros(fwSq, dtype=np.float64)
        # 有效像素计数、卡方累加、差异图像缓存
        for j in range(fwKSStamp):
        # 遍历子Stamp Y方向
            yRegion2 = yi - hwKSStamp + j
            # 绝对Y坐标
            for i in range(fwKSStamp):
            # X方向
                xRegion2 = xi - hwKSStamp + i
                # 绝对X坐标
                idx = i + j * fwKSStamp
                # 1D索引
                tdat = csModel[idx]
                # 模型值
                idat = im[idx]
                # 参考值
                ndat = imNoise[xRegion2 + rPixX * yRegion2]
                # 噪声值
                diff = tdat - idat + background
                # 残差 = 模型 - 参考 + 背景

                mr_idx = xRegion2 + rPixX * yRegion2
                # 遮罩索引
                if (mRData1d[mr_idx] & LOCAL_FLAG_INPUT_ISBAD) or (abs(idat) <= LOCAL_ZEROVAL):
                    continue
                    # 跳过坏像素或近零像素

                temp[idx] = diff
                # 存储残差
                nsig += 1
                sig1 += diff * diff / ndat
                # 累加卡方贡献

        if nsig > 0:
        # 有有效像素
            sig1 /= nsig
            # 平均卡方
            if sig1 >= LOCAL_MAXVAL:
                sig1 = -1.0
                # 超阈值标记无效
        else:
            sig1 = -1.0
            # 无有效像素

        out_sig1[si] = sig1
        # 输出方差型信噪比
        out_sig2[si] = -1.0; out_sig3[si] = -1.0
        # 偏度和峰度型信号在此模式未实现，标记-1

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
    LOCAL_ZEROVAL = 1e-10; LOCAL_MAXVAL = 1e10; LOCAL_FLAG_INPUT_ISBAD = 0x80; LOCAL_FLAG_ISNAN = 0x08
    # 局部常量

    ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    # 背景系数起始索引
    background = 0.0; k = 1
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX); yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
    # 归一化坐标
    ax = 1.0
    for i in range(bgOrder + 1):
        ay = 1.0
        for j in range(bgOrder - i + 1):
            background += kernelSol[ncompBG + k] * ax * ay
            # 累加背景多项式值
            k += 1; ay *= yf
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

    nsig = 0; sig1 = 0.0; sig2 = -1.0; sig3 = -1.0
    # 初始化：有效像素数、方差信噪比、偏度(-1未实现)、峰度(-1未实现)

    for j in range(fwKSStamp):
    # 遍历子Stamp Y方向
        yRegion2 = yi - hwKSStamp + j
        for i in range(fwKSStamp):
        # X方向
            xRegion2 = xi - hwKSStamp + i
            idx = i + j * fwKSStamp
            # 1D索引
            tdat = csModel[idx]
            # 模型值
            idat = im[idx]
            # 参考值
            ndat = imNoise[xRegion2 + rPixX * yRegion2]
            # 噪声值
            diff = tdat - idat + background
            # 残差
            mr_idx = xRegion2 + rPixX * yRegion2
            # 遮罩索引
            if (mRData1d[mr_idx] & LOCAL_FLAG_INPUT_ISBAD) or (abs(idat) <= LOCAL_ZEROVAL):
                continue
                # 跳过坏像素或近零像素
            else:
                temp[idx] = diff
                # 存储残差到缓存数组
            if np.isnan(tdat) or np.isnan(idat):
                mRData1d[mr_idx] = mRData1d[mr_idx] | (LOCAL_FLAG_INPUT_ISBAD | LOCAL_FLAG_ISNAN)
                # 标记NaN像素
                continue
            nsig += 1; sig1 += diff * diff / ndat
            # 计数值，累加卡方

    if nsig > 0:
        sig1 /= nsig
        # 平均卡方
        if sig1 >= LOCAL_MAXVAL: sig1 = -1.0
        # 超阈值标记无效
    else:
        sig1 = -1.0
        # 无有效像素
    return (sig1, sig2, sig3, nsig)
    # 返回信噪比和有效像素数

# 快速空间域卷积：预计算所有 kcStep 块的核，调用 jit kernel 一次遍历完成卷积+variance+mask
def spatial_convolve_fast_numpy(image: np.ndarray, variance: np.ndarray, xSize: int, ySize: int, kernelSol: np.ndarray, cMask: np.ndarray, kcStep: int,
                                hwKernel: int, fwKernel: int,
                                convolveVariance: int, kerFracMask: float,
                                rPixX: int, rPixY: int, nCompKer: int, kerOrder: int, bgOrder: int, kernel_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # 核像素数
    dovar = variance is not None
    # 是否计算方差

    image1d = image.astype(np.float64).ravel()
    # 输入图像转float64并展平
    cMask1d = cMask.astype(np.int32).ravel()
    # 输入遮罩转int32并展平

    cRdata64 = np.zeros(xSize * ySize, dtype=np.float64)
    # 分配卷积结果缓存（float64）
    mRData64 = np.zeros(xSize * ySize, dtype=np.int32)
    # 分配输出遮罩缓存（int32）

    if dovar:
        var1d = variance.astype(np.float64).ravel()
        # 方差图像转float64
    else:
        var1d = np.zeros(1, dtype=np.float64)
        # 无方差时分配占位数组

    vData = None; vData64 = np.zeros(xSize * ySize, dtype=np.float64)
    # 方差输出缓存

    kernel_vec_2d = np.array([np.asarray(kernel_vec[idx][:fwSq], dtype=np.float64)
    # 将核基向量列表转换为2D矩阵 (nCompKer × fwSq)
                              for idx in range(nCompKer)])

    allKernels, nstepsX, nstepsY = buildAllKernels(
    # 预计算所有块锚点的卷积核
        kernelSol.astype(np.float64), kernel_vec_2d,
        nCompKer, kerOrder, fwKernel, hwKernel, kcStep,
        rPixX, rPixY, xSize, ySize)

    spatial_convolve_jit_kernel(image1d, var1d, cMask1d,
    # 调用JIT核函数执行并行空间卷积
        cRdata64, vData64, mRData64, kernelSol.astype(np.float64),
        xSize, ySize, nCompKer, kerOrder, bgOrder, fwKernel, hwKernel,
        kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
        kernel_vec_2d, allKernels=allKernels, nstepsX_in=nstepsX)
        # 传入预计算的所有块核

    if dovar:
        vData = vData64.astype(np.float32)
        # 方差输出转float32
    cRdata_out = cRdata64.astype(np.float32)
    # 卷积结果转float32
    mRData_out = mRData64
    # 遮罩结果
    return vData, cRdata_out, mRData_out
    # 返回方差、卷积结果、遮罩

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

    ncomp1 = nCompKer - 1; ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    # 核分量数和空间多项式项数
    ncomp = ncomp1 * ncomp2; nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2; mat_size = ncomp1 * ncomp2 + nbg_vec + 1
    # 组合数、背景项数、矩阵尺寸

    ks = np.zeros(nS, dtype=np.float32); nks = 0
    # 核总和解数组和计数器
    nComps = nCompKer + 1
    # 独立拟合用的分量数

    for i in range(nS):
    # 对每个Stamp做独立拟合
        check_mat = np.zeros((nComps + 1, nComps + 1), dtype=np.float64)
        # 独立拟合矩阵
        check_vec = np.zeros(nComps + 1, dtype=np.float64)
        # 独立拟合右端项

        for im in range(1, nComps + 1):
            check_vec[im] = saScprod[i, im]
            # 复制标量积到右端项
            for jm in range(1, im + 1):
                check_mat[im, jm] = saMat[i, im, jm]
                # 复制局部矩阵（下三角）
                check_mat[jm, im] = check_mat[im, jm]
                # 对称填充上三角

        check_vec[1:nComps+1] = np.linalg.solve(
        # 独立求解每个Stamp的核
            check_mat[1:nComps+1, 1:nComps+1], check_vec[1:nComps+1])

        sum_val = check_vec[1]
        # 核总和解（第一个元素）
        saNorm[i] = sum_val; ks[nks] = sum_val; nks += 1
        # 存储每个Stamp的核总和

    kmean, kstdev, sc_rc = sigma_clip_numpy(ks[:nks], maxiter=10, stat_sig=statSig)
    # Sigma-clip裁剪核总和异常值

    for i in range(nS):
        saDiff[i] = abs((saNorm[i] - kmean) / kstdev)
        # 计算每个Stamp核总和的标准偏差

    if forceConvolve[0:1] == "b":
    # 双向卷积模式：需要比较模板→图像和图像→模板的品质指标
        testMask = saDiff < kerSigReject
        # 筛选核总和不显著偏离均值的Stamps（合格Stamp）
        ntestStamps = testMask.sum()
        if ntestStamps == 0:
            return 0.0
            # 无合格Stamp

        testScprod   = saScprod[testMask]; testMat = saMat[testMask]
        # 筛选合格Stamp的标量积和矩阵
        testVectors  = saVectors[testMask]; testKrefArea = saKrefArea[testMask]
        # 筛选向量和参考区域
        testSscnt    = saSscnt[testMask]; testNss = saNss[testMask]
        testXss      = saXss[testMask]; testYss = saYss[testMask]
        # 筛选子Stamp计数器、总数、坐标

        testKerSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
        # 测试核解向量
        wxy = np.zeros((ntestStamps, ncomp2), dtype=np.float64)
        # 空间权重矩阵

        matrix, testKerSol, wxy_unused = build_both_numpy(testMat, testVectors, testScprod, testSscnt, testNss, testXss, testYss,
        # 对合格Stamp构建聚合矩阵和scprod
                                                   ntestStamps, imRef, nCompKer, kerOrder, bgOrder,
                                                   fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)

        # indx2 = np.zeros(mat_size + 1, dtype=np.int32)
        testKerSol[1:mat_size+1] = np.linalg.solve(
            matrix[1:mat_size+1, 1:mat_size+1], testKerSol[1:mat_size+1])

        kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
        kernel_arr = np.zeros(fwKernel * fwKernel, dtype=np.float64)
        kmean = make_kernel_numpy(0, 0, testKerSol, rPixX, rPixY, nCompKer, kerOrder,
                                  fwKernel, kernel_vec, kernel_coeffs, kernel_arr)

        m1 = np.zeros(ntestStamps, dtype=np.float32); m2 = np.zeros(ntestStamps, dtype=np.float32); m3 = np.zeros(ntestStamps, dtype=np.float32)
        # 三种品质指标的数组
        mcnt1 = 0; mcnt2 = 0; mcnt3 = 0
        # 各自的有效计数

        for i in range(ntestStamps):
        # 遍历合格Stamp
            sscnt = testSscnt[i]; xRegion = int(testXss[i, sscnt]); yRegion = int(testYss[i, sscnt])
            # 当前子中心坐标
            im = testKrefArea[i]; vectors = np.asarray(testVectors[i], dtype=np.float64)
            # 参考区域和向量
            kSol = np.asarray(testKerSol, dtype=np.float64); imNoiseArr = np.asarray(imNoise, dtype=np.float64)
            mRDataArr = np.asarray(mRData, dtype=np.int64); imArr = np.asarray(im, dtype=np.float64)
            # 转为合适类型传给JIT

            temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)
            # 差异图像缓存

            sig1, sig2, sig3, nsig = get_stamp_sig_jit(
            # 调用单个Stamp信噪比计算
                vectors, kSol, imNoiseArr, mRDataArr, imArr,
                fwKSStamp, hwKSStamp, rPixX, rPixY,
                nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                temp)

            if figMerit[0:1] != "v":
            # 非方差模式需要额外计算sig2/sig3
                temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                mRData_2d = mRData.reshape(-1, rPixX)
                result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                # 计算差异图像的统计量
                                                fwKSStamp, fwKSStamp,
                                                0x0, 0xffff, 5, mRData_2d, statSig)
                if result[7] != 0: sig2 = -1.0; sig3 = -1.0
                # 统计计算失败
                else:
                    sig2 = result[4]; sig3 = result[5]
                    # 标准差赋值给sig2，FWHM赋值给sig3
                    if sig2 < 0 or sig2 >= MAXVAL: sig2 = -1.0
                    elif sig3 < 0 or sig3 >= MAXVAL: sig3 = -1.0
                    # 验证有效性

            if sig1 != -1 and sig1 <= MAXVAL: m1[mcnt1] = sig1; mcnt1 += 1
            # 记录方差型品质指标
            if sig2 != -1 and sig2 <= MAXVAL:
                m2[mcnt2] = sig2
                mcnt2 += 1
            if sig3 != -1 and sig3 <= MAXVAL:
                m3[mcnt3] = sig3
                mcnt3 += 1

        merit1, sig1sc, rc1 = sigma_clip_numpy(m1[:mcnt1], maxiter=10, stat_sig=statSig)
        # 对方差指标做sigma-clip得均值和弥散度
        merit2, sig2sc, rc2 = sigma_clip_numpy(m2[:mcnt2], maxiter=10, stat_sig=statSig)
        # 对偏度指标做sigma-clip
        merit3, sig3sc, rc3 = sigma_clip_numpy(m3[:mcnt3], maxiter=10, stat_sig=statSig)
        # 对峰度指标做sigma-clip

        merit1 /= kmean; merit2 /= kmean; merit3 /= kmean
        # 按核总和归一化品质指标

        if figMerit[0:1] == "v":
        # 方差模式：优先方差指标
            if mcnt1 > 0: return merit1
            # 有方差指标
            elif mcnt2 > 0: return merit2
            # 退而求偏度
            elif mcnt3 > 0: return merit3
            # 再退而求峰度
            else: return 666.0
            # 全部无值
        elif figMerit[0:1] == "s":
            if mcnt2 > 0: return merit2
            elif mcnt1 > 0: return merit1
            elif mcnt3 > 0: return merit3
            else: return 666.0
        elif figMerit[0:1] == "h":
            if mcnt3 > 0: return merit3
            elif mcnt1 > 0: return merit1
            elif mcnt2 > 0: return merit2
            else: return 666.0
    else:
        return 0.0
        # 非双向模式返回0
    return 0.0
    # 兜底返回

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
    # 有效卡方值数组
    nss = 0
    # 有效计数
    sig = 0.0; check = 0; mean = 0.0; stdev = 0.0; nskippedSubstamps = 0; refill_indices = []
    # 初始化：信噪比、检查标志、均值、标准差、跳过计数、refill列表

    sscnt_local = saSscnt[:nS].copy()
    # 复制子Stamp计数器（本地可修改）
    chi2_local = saChi2[:nS].copy()
    # 复制卡方数组

    batch_sig1 = None
    if figMerit[0:1] == "v":
    # 方差模式：批量计算
        batch_sig1 = np.zeros(nS, dtype=np.float64)
        batch_sig2 = np.zeros(nS, dtype=np.float64)
        batch_sig3 = np.zeros(nS, dtype=np.float64)

        halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
        # 半宽归一化
        xf_batch = np.zeros(nS, dtype=np.float64)
        yf_batch = np.zeros(nS, dtype=np.float64)
        # 预计算各Stamp的归一化坐标
        for si in range(nS):
            if sscnt_local[si] < saNss[si]:
                xi = float(saXss[si, sscnt_local[si]])
                yi = float(saYss[si, sscnt_local[si]])
                # 当前子中心坐标
                xf_batch[si] = (xi - halfX) / halfX
                # 归一化到[-1,1]
                yf_batch[si] = (yi - halfY) / halfY

        ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
        # 背景系数在kernelSol中的起始索引
        batched_bg = np.zeros(nS, dtype=np.float64)
        # 预计算每个Stamp的背景值
        kb = 1; ax_arr = np.ones(nS, dtype=np.float64)
        for i in range(bgOrder + 1):
            ay_arr = np.ones(nS, dtype=np.float64)
            for j in range(bgOrder - i + 1):
                batched_bg += kernelSol[ncompBG + kb] * ax_arr * ay_arr
                # 向量化背景值计算：bg_coeff × x^i × y^j
                kb += 1; ay_arr *= yf_batch
            ax_arr *= xf_batch

        batched_coeffs4 = np.zeros((nS, nCompKer), dtype=np.float64)
        # 预计算核空间变化系数
        kk = 2
        for i1 in range(1, nCompKer):
            coeff_arr = np.zeros(nS, dtype=np.float64)
            ax_arr = np.ones(nS, dtype=np.float64)
            for ix in range(kerOrder + 1):
                ay_arr = np.ones(nS, dtype=np.float64)
                for iy in range(kerOrder - ix + 1):
                    coeff_arr += kernelSol[kk] * ax_arr * ay_arr
                    # 空间多项式加权
                    kk += 1; ay_arr *= yf_batch
                ax_arr *= xf_batch
            batched_coeffs4[:, i1] = coeff_arr
            # 保存每个Stamp的核系数

        refill_flags = np.zeros(nS, dtype=np.int32)
        # Refill标记数组
        mean, stdev, ncheck_j = get_sig_and_clip_jit(
        # 调用JIT函数：批量sig计算+sigma_clip+refill标记
            np.asarray(saVectors, dtype=np.float64), np.asarray(saKrefArea, dtype=np.float64),
            # 确保float64类型传入JIT
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
        # 收集被标记为refill的stamp索引
        check = ncheck_j
        # 是否有stamp需要refill
        meansigSubstamps = mean
        scatterSubstamps = stdev
    else:
    # 非方差模式（figMerit!="v"）：逐stamp调用get_stamp_sig_jit
        for istamp in range(nS):
            if sscnt_local[istamp] < saNss[istamp]:
            # stamp有效
                sscnt = sscnt_local[istamp]
                xRegion = int(saXss[istamp, sscnt]); yRegion = int(saYss[istamp, sscnt])
                # 当前子中心坐标
                im = saKrefArea[istamp]; vectors = np.asarray(saVectors[istamp], dtype=np.float64)
                # 参考区域和向量
                kSol = np.asarray(kernelSol, dtype=np.float64); imNoiseArr = np.asarray(imNoise, dtype=np.float64)
                mRDataArr = np.asarray(mRData, dtype=np.int64); imArr = np.asarray(im, dtype=np.float64)
                # 转为合适类型

                temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)
                # 差异缓存

                sig1, sig2, sig3, nsig = get_stamp_sig_jit(
                # 调用JIT计算单个stamp的信噪比
                    vectors, kSol, imNoiseArr, mRDataArr, imArr,
                    fwKSStamp, hwKSStamp, rPixX, rPixY,
                    nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                    temp)

                if figMerit[0:1] != "v":
                # 非方差模式需额外计算sig2/sig3
                    temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                    mRData_2d = mRData.reshape(-1, rPixX)
                    result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                    # 计算差异图像统计量
                                                    fwKSStamp, fwKSStamp,
                                                    0x0, 0xffff, 5, mRData_2d, statSig)
                    if result[7] != 0: sig2 = -1.0; sig3 = -1.0
                    # 统计失败
                    else:
                        sig2 = result[4]; sig3 = result[5]
                        # sd→sig2, fwhm→sig3
                        if sig2 < 0 or sig2 >= MAXVAL: sig2 = -1.0
                        elif sig3 < 0 or sig3 >= MAXVAL: sig3 = -1.0
                        # 验证有效性

                if (figMerit[0:1] == "v" and sig1 == -1) or \
                   (figMerit[0:1] == "s" and sig2 == -1) or \
                   (figMerit[0:1] == "h" and sig3 == -1):
                # 当前品质指标类型的信噪比无效（方差/偏度/峰度任一匹配即refill）
                    sscnt_local[istamp] += 1; refill_indices.append(istamp); check = 1
                    # 标记需要refill：前进sscnt并标记
                else:
                # 信噪比有效
                    if figMerit[0:1] == "v": sig = sig1
                    elif figMerit[0:1] == "s": sig = sig2
                    elif figMerit[0:1] == "h": sig = sig3
                    # 根据品质指标类型选择对应的信噪比
                    chi2_local[istamp] = sig; ss[nss] = sig; nss += 1
                    # 记录卡方值并收集到ss数组
            else:
                nskippedSubstamps += 1
                # 跳过已耗尽子Stamp的stamp

        mean, stdev, retcode = sigma_clip_numpy(ss[:nss], maxiter=10, stat_sig=statSig)
        # Sigma-clip裁剪有效信噪比
        meansigSubstamps = mean; scatterSubstamps = stdev
        # 记录均值和弥散度

        mask = (sscnt_local < saNss[:nS]) & ((chi2_local - mean) > kerSigReject * stdev)
        # 找出卡方偏离均值超过阈值的stamp
        sscnt_local[mask] += 1; refill_indices.extend(np.where(mask)[0].tolist())
        # 标记refill
        if np.any(mask): check = 1
        # 有stamp需要refill则设置check标志

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
    # 核分量数减1（除去常数项）
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    # 空间变化多项式的项数
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    # 背景多项式的项数
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1
    # 拟合矩阵的尺寸

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)
    # 空间权重矩阵

    saMat      = sa['mat']
    # Stamp局部矩阵
    saVectors  = sa['vectors']
    # Stamp卷积向量
    saSscnt    = sa['sscnt']
    # 子Stamp计数器
    saNss      = sa['nss']
    # 子Stamp总数
    saXss      = sa['xss']
    # 子中心X坐标
    saYss      = sa['yss']
    # 子中心Y坐标
    saScprod   = sa['scprod']
    # Stamp局部标量积
    saKrefArea = sa['krefArea']
    # Stamp参考区域
    saSumVal   = sa['sum_val']
    # Stamp像素和
    saChi2     = sa['chi2']
    # Stamp卡方值
    nKSStamps  = sa['nKSStamps']
    # 最大子Stamp数

    # import time
    iter_count = 0
    # 迭代计数器
    logger.debug("  fitKernel: iteration %d start", iter_count)
    matrix, kernelSol, wxy = build_both_numpy(saMat, saVectors, saScprod, saSscnt, saNss, saXss, saYss,
    # 首次构建：聚合matrix和累加scprod
                                               nS, imRef, nCompKer, kerOrder, bgOrder,
                                               fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
    logger.debug("  fitKernel: build_matrix+scprod done")
    # 构建完成

    kernelSol[1:mat_size+1] = np.linalg.solve(
    # 求解线性方程组：matrix × x = kernelSol（仅求解mat_size维）
        matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
    logger.debug("  fitKernel: solve done")
    # 首次求解完成

    (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
    # 首次check_again：计算每个stamp的信噪比，标记异常stamp
     refill_indices, sscnt_update, chi2_update) = check_again_numpy(
        saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea, kernelSol,
        imNoise, nS, figMerit, kerSigReject, statSig,
        fwKSStamp, hwKSStamp, rPixX, rPixY, mRData, nCompKer, kerOrder, bgOrder)
    saSscnt[:nS] = sscnt_update; saChi2[:nS] = chi2_update
    # 更新sscnt和chi2

    for idx in refill_indices:
    # 对每个需要refill的stamp，重新填充
        fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, idx, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                         hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                         bgOrder, nCompKer, filter_x, filter_y,
                         fillVal, mRData)
    logger.debug("  fitKernel: check_again done, check=%s", check)

    while check:
    # 迭代循环：有stamp被refill时继续
        iter_count += 1
        # 迭代计数
        logger.debug("  fitKernel: iteration %d start", iter_count)

        matrix, kernelSol, wxy = build_both_numpy(saMat, saVectors, saScprod, saSscnt, saNss, saXss, saYss,
        # 重新构建矩阵和scprod（使用更新后的stamp数据）
                                                   nS, imRef, nCompKer, kerOrder, bgOrder,
                                                   fwKSStamp, hwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
        logger.debug("  fitKernel: build_matrix+scprod done")

        kernelSol[1:mat_size+1] = np.linalg.solve(
        # 重新求解
            matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
        logger.debug("  fitKernel: solve done")

        (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
        # 重新检查
         refill_indices, sscnt_update, chi2_update) = check_again_numpy(
            saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea, kernelSol,
            imNoise, nS, figMerit, kerSigReject, statSig,
            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData, nCompKer, kerOrder, bgOrder)
        saSscnt[:nS] = sscnt_update; saChi2[:nS] = chi2_update
        # 更新sscnt和chi2

        for idx in refill_indices:
        # 对refill的stamp重新填充
            fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, idx, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                             hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, filter_x, filter_y,
                             fillVal, mRData)
        logger.debug("  fitKernel: check_again done, check=%s", check)

    return {'kernelSol': kernelSol,
    # 返回核解向量
        'meansigSubstamps': meansigSubstamps, 'scatterSubstamps': scatterSubstamps,
        # 平均和弥散度
        'NskippedSubstamps': nskippedSubstamps, 'stamps': sa}
        # 跳过数和更新后的stamp字典


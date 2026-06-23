import numpy as np
# 导入numpy科学计算库，别名np
import sys, os, struct
# 导入系统调用、文件路径操作、二进制结构体打包模块
import time as tm
# 导入时间模块，用于计时，别名tm
import logging
# 导入日志模块，用于分级输出调试信息
from typing import List, Dict, Tuple, Optional
# 从typing模块导入类型标注工具：列表、字典、元组、可选类型
from .functions import FLAG_BAD_PIXVAL, FLAG_SAT_PIXEL, FLAG_LOW_PIXEL
# 从functions子模块导入像素质量标记常量：坏值(0x01)、饱和(0x02)、过低(0x04)
from .functions import FLAG_INPUT_ISBAD, FLAG_OUTPUT_ISBAD
# 导入输入坏像素标记(0x80)和输出坏像素标记(0x8000)
from .functions import sigma_clip_numpy, get_noise_stats3_numpy
# 导入Sigma裁剪函数（迭代剔除离群值）和噪声统计函数（均值/中位数/众数/标准差/X2NORM）
from .functions import insert_subregion_flt_numpy, insert_subregion_int_numpy
# 导入子区域插入函数：将浮点/整数区域图像写入全图指定位置
from .functions import get_stamp_stats3_numpy, buildStampsNumba
# 导入Stamp统计量计算函数和Numba加速的批量Stamp构建函数
from .functions import make_noise_image4_numpy
# 导入噪声图像生成函数：基于增益和读出噪声计算方差图像
# ZEROVAL, MAXVAL,
# 以下为历史遗留注释：对应函数已被合并、内联或删除
# bin_quartile_numpy,
#    get_stamp_stats3_fast_numpy, cut_sstamp_numpy,
#    check_psf_center_numba, quick_sort_recurse,
#    psfCentersJit, cut_stamp_numpy, Ran1,
# BUILD_STAMP_FLAT_CACHE, LAST_SIGMA_CLIP, LAST_N, LAST_MEDIAN,

from .alard import get_final_stamp_sig_numpy  # background_loop_jit merged into spatial_convolve_jit_kernel
# 从alard子模块导入最终Stamp信噪比计算函数（背景循环已合入空间卷积核函数）
from .alard import make_kernel_numpy, get_kernel_vec_numpy
# 导入在指定坐标构造空间可变卷积核的函数、生成全部核基向量列表的函数
from .alard import fill_stamp_numba, spatial_convolve_fast_numpy
# 导入Numba加速的Stamp数据填充函数、预计算块核的快速空间卷积函数
from .alard import check_stamps_numpy, fit_kernel_numpy
# 导入独立Stamp质量检验函数（fit+sigma_clip+outlier标记）、迭代核拟合函数
#kernel_vector_pca_numpy, kernel_vector_numpy,  build_matrix_jit, build_scprod_jit, build_matrix_numpy, build_scprod_numpy,
# 以下注释的函数已被合并到其他函数中或不再使用
#    fill_stamp_numba_kernel_local, fill_stamp_numpy, get_stamp_sig_batch_jit, get_stamp_sig_jit,spatial_convolve_jit_kernel, buildAllKernels,
#check_again_numpy,

def compute_ctx_info(hwKernel: int, ngauss: int, deg_fixe: List[int], kerOrder: int, bgOrder: int,
# 定义上下文信息计算函数：接收核半宽、高斯分量数、每个高斯分量的多项式阶数列表、核/背景多项式阶数
                     nStampX_in: int, nStampY_in: int, hwKSStamp: int,
# 接收输入的X/Y方向Stamp数、Stamp子区域半宽
                     useFullSS: int, kerFitThresh: float, kcStep_in: int,
# 接收是否使用全子Stamp模式标志、核拟合阈值、核缓存步长输入值
                     tNx: int, tNy: int, iNx: int, iNy: int, nR: int) -> Dict:
# 接收模板和科学图像的X/Y像素数、区域划分数，返回上下文字典
    nCompKer = 0
    # 初始化核分量计数器为0
    for i in range(ngauss):
    # 遍历每个高斯分量
        nCompKer += ((deg_fixe[i] + 1) * (deg_fixe[i] + 2)) // 2
        # 累加当前高斯分量的多项式展开项数：组合数 = (deg+1)*(deg+2)/2

    nComp      = ((kerOrder + 1) * (kerOrder + 2)) // 2
    # 计算单个核分量内空间变化多项式的项数：组合数 = (kO+1)*(kO+2)/2
    nC         = nCompKer + 2
    # 拟合矩阵列数：核分量总数 + 常数项(1) + 第一个背景项(1)
    nCompBG    = (nCompKer - 1) * nComp + 1
    # 背景分量在kernelSol中的起始索引：从1开始计数
    nBGVectors = ((bgOrder + 1) * (bgOrder + 2)) // 2
    # 背景多项式展开的总项数：组合数 = (bgO+1)*(bgO+2)/2
    nCompTotal = nCompKer * nComp + nBGVectors
    # 核解向量的总长度：空间变化核部分 + 背景多项式部分

    fwKernel   = hwKernel * 2 + 1
    # 卷积核全宽：半宽乘2再加1（保证奇数）
    nStampX    = nStampX_in
    # X方向Stamp数：从输入参数赋值
    nStampY    = nStampY_in
    # Y方向Stamp数：从输入参数赋值

    if useFullSS:
    # 全子Stamp模式：每个像素都是一个子Stamp中心
        fwKSStamp = fwKernel
        # 子Stamp窗口全宽设为核全宽
        fwStamp   = fwKernel
        # Stamp全宽设为核全宽
        nStampX   = int(min(tNx, iNx) / nR / fwStamp)
        # 根据图像尺寸和区域数重新计算X方向Stamp数
        nStampY   = int(min(tNy, iNy) / nR / fwStamp)
        # 根据图像尺寸和区域数重新计算Y方向Stamp数
    else:
    # 标准模式：Stamp间距由fwStamp决定
        fwKSStamp = hwKSStamp * 2 + 1
        # 子Stamp窗口全宽：半宽乘2加1
        fwStamp = min(min(tNx, iNx) // int(np.sqrt(nR)) // nStampX,
        # 从图像较小维度除以区域数平方根再除以Stamp数反推Stamp全宽（X方向）
                        min(tNy, iNy) // int(np.sqrt(nR)) // nStampY)
        # 同时在Y方向计算，取两者中较小值
        fwStamp -= fwKernel
        # 减去核宽度，得到实际Stamp区域净宽
        if fwStamp % 2 == 0:
        # 确保Stamp全宽为奇数（便于定义中心像素）
            fwStamp -= 1
            # 偶数则减1变为奇数

        if fwStamp < fwKSStamp:
        # 如果计算的Stamp宽小于子Stamp窗口宽（图像太小或Stamp过多）
            fwStamp  = fwKSStamp + fwKernel
            # Stamp宽设为子Stamp宽加核宽（最小可用值）
            if fwStamp % 2 == 0:
            # 再次确保奇数
                fwStamp -= 1
                # 偶数减1
            nStampX = min(tNx, iNx) // int(np.sqrt(nR)) // fwStamp
            # 根据新的Stamp宽重新计算X方向Stamp数
            nStampY = min(tNy, iNy) // int(np.sqrt(nR)) // fwStamp
            # 根据新的Stamp宽重新计算Y方向Stamp数

    kcStep  = kcStep_in if kcStep_in else fwKernel
    # 核缓存步长：用户输入值非零则使用输入值，否则默认取核全宽
    nStamps = nStampX * nStampY
    # Stamp总数：X方向数乘以Y方向数
    sBorder = hwKSStamp + hwKernel
    # 区域边界宽度：子Stamp半宽加核半宽（卷积所需的额外padding）

    xMin = 0
    # 全局最小X坐标设为0
    yMin = 0
    # 全局最小Y坐标设为0
    xMax = min(tNx, iNx) - 1
    # 全局最大X坐标：模板和科学图像中较小宽度减1（0索引）
    yMax = min(tNy, iNy) - 1
    # 全局最大Y坐标：模板和科学图像中较小高度减1
    fitThresh = kerFitThresh
    # 保存核拟合阈值

    return {'nCompKer': nCompKer, 'nComp': nComp, 'nC': nC,
    # 返回上下文字典：核分量数、单分量空间项数、矩阵列数
        'nCompBG': nCompBG, 'nBGVectors': nBGVectors, 'nCompTotal': nCompTotal,
        # 背景起始索引、背景项数、核解向量总长度
        'fwKernel': fwKernel, 'fwStamp': fwStamp, 'fwKSStamp': fwKSStamp,
        # 核全宽、Stamp全宽、子Stamp窗口全宽
        'sBorder': sBorder, 'nStamps': nStamps,
        # 区域边界宽度、Stamp总数
        'nStampX': nStampX, 'nStampY': nStampY,
        # X/Y方向Stamp数
        'xMin': xMin, 'yMin': yMin, 'xMax': xMax, 'yMax': yMax,
        # 全局图像坐标范围（0索引）
        'fitThresh': fitThresh, 'kcStep': kcStep}
        # 核拟合阈值、核缓存步长

def region_setup_numpy(tmpl_2d: np.ndarray, sci_2d: np.ndarray, tnoise_2d: Optional[np.ndarray], inoise_2d: Optional[np.ndarray], tmask_2d: Optional[np.ndarray], imask_2d: Optional[np.ndarray],
# 区域初始化函数：为指定区域索引构建带padding的子图像、噪声和遮罩。接收模板2D图像、科学2D图像、
                        ri: int, rxmins_np: np.ndarray, rxmaxs_np: np.ndarray, rymins_np: np.ndarray, rymaxs_np: np.ndarray, nR: int,
# 区域索引（0起始）、各区域的X/Y起止坐标数组、区域总数
                        hwKernel: int, fwStamp: int, sBorder: int,
# 核半宽、Stamp全宽、边界宽度（用于padding计算）
                        xMin: int, yMin: int, xMax: int, yMax: int,
# 全局图像坐标范围（0索引）
                        fillVal: float, fillValNoise: float,
# 无效像素填充值、噪声图像填充值
                        tPedestal: float, iPedestal: float,
# 模板Pedestal（天光背景扣除值）、科学图像Pedestal
                        tGain: float, tRdnoise: float, iGain: float, iRdnoise: float,
# 模板增益(e-/ADU)、模板读出噪声(e-)、科学图像增益、科学图像读出噪声
                        tUThresh: float, tLThresh: float, iUThresh: float, iLThresh: float,
# 模板上下限阈值（超过上限为饱和，低于下限为过低）、科学图像上下限阈值
                        kfSpreadMask1: float, logger: Optional[logging.Logger] = None) -> Dict:
# 遮罩扩散因子（坏像素标记向周围扩张的比例）、日志记录器，返回区域设置结果字典
    if logger is None:
    # 如果未提供日志记录器
        logger = logging.getLogger('hotpants')
        # 获取默认的hotpants日志记录器实例
    logger.debug("  region_setup: ri=%d rXMin=%d rXMax=%d rYMin=%d rYMax=%d",
    # 输出调试日志：当前区域索引和原始坐标范围
                 ri, rxmins_np[ri], rxmaxs_np[ri], rymins_np[ri], rymaxs_np[ri])
                 # 区域索引、该区域在全局图像中的X起止和Y起止坐标
    rXMin = int(rxmins_np[ri])
    # 提取当前区域的原始X起始坐标（全局坐标系，未加padding）
    rXMax = int(rxmaxs_np[ri])
    # 提取当前区域的原始X结束坐标
    rYMin = int(rymins_np[ri])
    # 提取当前区域的原始Y起始坐标
    rYMax = int(rymaxs_np[ri])
    # 提取当前区域的原始Y结束坐标

    fwStamp = int(fwStamp)
    # 确保Stamp全宽为整数类型
    hwKernel = int(hwKernel)
    # 确保核半宽为整数类型
    sBorder = int(sBorder)
    # 确保边界宽度为整数类型
    xMin = int(xMin); yMin = int(yMin); xMax = int(xMax); yMax = int(yMax)
    # 全部坐标参数转换为整数类型，确保后续整数运算

    if nR > 1:
    # 多区域模式：使用fwStamp/2作为padding（比单区域hwKernel更大的缓冲）
        rXBMin = max(xMin, rXMin - fwStamp // 2)
        # 缓冲后X起始：区域左边界外扩半个Stamp宽度，但不低于图像左边界
        rYBMin = max(yMin, rYMin - fwStamp // 2)
        # 缓冲后Y起始：区域上边界外扩半个Stamp宽度，但不低于图像上边界
        rXBMax = min(xMax, rXMax + fwStamp // 2)
        # 缓冲后X结束：区域右边界外扩半个Stamp宽度，但不超出图像右边界
        rYBMax = min(yMax, rYMax + fwStamp // 2)
        # 缓冲后Y结束：区域下边界外扩半个Stamp宽度，但不超出图像下边界
    else:
    # 单区域模式：使用hwKernel作为padding（较小的缓冲）
        rXBMin = max(xMin, rXMin - hwKernel)
        # 缓冲后X起始：外扩核半宽
        rYBMin = max(yMin, rYMin - hwKernel)
        # 缓冲后Y起始：外扩核半宽
        rXBMax = min(xMax, rXMax + hwKernel)
        # 缓冲后X结束：外扩核半宽
        rYBMax = min(yMax, rYMax + hwKernel)
        # 缓冲后Y结束：外扩核半宽

    rPixX = rXBMax - rXBMin + 1
    # 缓冲后区域的X方向像素数（包含padding）
    rPixY = rYBMax - rYBMin + 1
    # 缓冲后区域的Y方向像素数

    fillVal_f = np.float32(fillVal)
    # 将无效值转换为float32类型（与图像数据类型一致）
    fillValNoise_f = np.float32(fillValNoise)
    # 将噪声填充值转换为float32类型

    tRData_py = np.full((rPixY, rPixX), fillVal_f, dtype=np.float32)
    # 创建模板区域数组，全部填充为无效值（padding区域保持fillVal）
    iRData_py = np.full((rPixY, rPixX), fillVal_f, dtype=np.float32)
    # 创建科学图像区域数组，全部填充为无效值
    oRData_py = np.full((rPixY, rPixX), fillValNoise_f, dtype=np.float32)
    # 创建输出差异图像区域数组，用噪声填充值初始化
    eRData_py = np.full((rPixY, rPixX), fillValNoise_f, dtype=np.float32)
    # 创建噪声方差图像区域数组，用噪声填充值初始化
    mRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)
    # 创建综合遮罩数组，初始全0（无任何标记位）
    misRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)
    # 创建科学图像输入遮罩数组，初始全0
    mtsRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)
    # 创建模板图像输入遮罩数组，初始全0

    tRData_py[:, :] = tmpl_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
    # 从完整模板图像中提取缓冲区域的数据，填充到tRData（覆盖padding区域的fillVal）
    iRData_py[:, :] = sci_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
    # 从完整科学图像中提取缓冲区域的数据，填充到iRData

    tPedestal_f = np.float32(tPedestal)
    # 将模板Pedestal转换为float32
    iPedestal_f = np.float32(iPedestal)
    # 将科学图像Pedestal转换为float32
    if tPedestal_f != 0. or iPedestal_f != 0.:
    # 如果模板或科学图像有非零的背景扣除值
        tRData_py -= tPedestal_f
        # 从模板区域数据中减去天光背景
        iRData_py -= iPedestal_f
        # 从科学图像区域数据中减去天光背景

    if inoise_2d is not None:
    # 如果提供了科学图像的噪声方差数组（外部传入）
        oRData_py[:, :] = inoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        # 从完整噪声方差图像中提取对应区域
        oRData_py *= oRData_py
        # 噪声值自身平方得到方差（外部数据可能存的是σ值而非σ²）
    else:
    # 未提供噪声数组，根据增益和读出噪声公式计算
        invGain_f = np.float32(1.0 / float(iGain))
        # 计算科学图像增益的倒数（1/gain，用于光子噪声计算）
        quad_f = np.float32(float(iRdnoise) / float(iGain))
        # 计算等效读出噪声除以增益（单位：电子/ADU → 电子）
        qquad = np.float64(quad_f) * np.float64(quad_f)
        # 计算等效读出噪声的平方（使用float64提高精度防止溢出）
        oRData_py = (np.abs(iRData_py.astype(np.float64)) * np.float64(invGain_f) + qquad).astype(np.float32)
        # 方差 = |信号|/增益 + (读出噪声/增益)^2，转回float32

    if tnoise_2d is not None:
    # 如果提供了模板的噪声方差数组
        eRData_py[:, :] = tnoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        # 从完整模板噪声方差图像中提取对应区域
        eRData_py *= eRData_py
        # 噪声值自身平方得到方差
    else:
    # 未提供模板噪声数组，同上根据增益和读出噪声计算
        invGain_t = np.float32(1.0 / float(tGain))
        # 计算模板增益的倒数
        quad_t = np.float32(float(tRdnoise) / float(tGain))
        # 计算等效模板读出噪声除以增益
        qquad_t = np.float64(quad_t) * np.float64(quad_t)
        # 计算模板等效读出噪声平方（float64精度）
        eRData_py = (np.abs(tRData_py.astype(np.float64)) * np.float64(invGain_t) + qquad_t).astype(np.float32)
        # 模板方差 = |信号|/增益 + (读出噪声/增益)^2，转回float32

    oRData_py += eRData_py
    # 总噪声方差 = 科学图像方差 + 模板方差（卷积后噪声传播需要两者之和）

    if imask_2d is not None:
    # 如果提供了科学图像的外部遮罩
        misRData_py[:, :] = imask_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        # 从完整科学图像遮罩中提取对应区域
        misRData_py |= np.int32(0x20) * (misRData_py > 0).astype(np.int32)
        # 对已有非零标记的像素额外设置0x20位（表示"已标记"）
        mRData_py |= misRData_py
        # 将科学图像遮罩按位或合并到综合遮罩中

    if tmask_2d is not None:
    # 如果提供了模板的外部遮罩
        mtsRData_py[:, :] = tmask_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        # 从完整模板遮罩中提取对应区域
        mtsRData_py |= np.int32(0x20) * (mtsRData_py > 0).astype(np.int32)
        # 对已有非零标记的像素额外设置0x20位
        mRData_py |= mtsRData_py
        # 将模板遮罩按位或合并到综合遮罩中

    tUThresh_f = np.float32(tUThresh)
    # 模板上限阈值转为float32
    tLThresh_f = np.float32(tLThresh)
    # 模板下限阈值转为float32
    iUThresh_f = np.float32(iUThresh)
    # 科学图像上限阈值转为float32
    iLThresh_f = np.float32(iLThresh)
    # 科学图像下限阈值转为float32
    mRData_py |= np.int32(0x80 | 0x01) * ((tRData_py == fillVal_f) | (iRData_py == fillVal_f)).astype(np.int32)
    # 标记填充值像素：模板或科学图像中等于fillVal的像素设置INPUT_ISBAD(0x80)|BAD_PIXVAL(0x01)
    mRData_py |= np.int32(0x80 | 0x02) * ((tRData_py >= tUThresh_f) | (iRData_py >= iUThresh_f)).astype(np.int32)
    # 标记饱和像素：模板或科学图像中超过上限的像素设置INPUT_ISBAD(0x80)|SAT_PIXEL(0x02)
    mRData_py |= np.int32(0x80 | 0x04) * ((tRData_py <= tLThresh_f) | (iRData_py <= iLThresh_f)).astype(np.int32)
    # 标记过低像素：模板或科学图像中低于下限的像素设置INPUT_ISBAD(0x80)|LOW_PIXEL(0x04)

    width = int(hwKernel * float(kfSpreadMask1))
    # 计算遮罩扩散宽度：核半宽乘以扩散因子（将坏像素标记向周围扩展）
    if width > 0:
    # 如果扩散宽度大于0，执行遮罩扩散
        w2 = width // 2
        # 计算扩散半宽（一次扩展的距离）
        bad = (mRData_py & 0x80) != 0
        # 找出当前所有标记为INPUT_ISBAD(0x80)的像素（坏像素）
        spread = np.zeros((rPixY, rPixX), dtype=np.bool_)
        # 创建扩散累积缓冲区，初始全False
        for dy in range(-w2, w2 + 1):
        # 遍历Y方向偏移：从负半宽到正半宽
            for dx in range(-w2, w2 + 1):
            # 遍历X方向偏移
                shifted = np.zeros((rPixY, rPixX), dtype=np.bool_)
                # 创建单次偏移扩散的目标数组
                sy1, sy2 = max(0, -dy), min(rPixY, rPixY - dy)
                # Y方向：目标区域的起止行索引（被位移后仍有效的区域）
                dy1_s, dy2_s = max(0, dy), min(rPixY, rPixY + dy)
                # Y方向：源区域的起止行索引（位移前的位置）
                sx1, sx2 = max(0, -dx), min(rPixX, rPixX - dx)
                # X方向：目标区域的起止列索引
                dx1_s, dx2_s = max(0, dx), min(rPixX, rPixX + dx)
                # X方向：源区域的起止列索引
                shifted[dy1_s:dy2_s, dx1_s:dx2_s] = bad[sy1:sy2, sx1:sx2]
                # 将bad像素沿(dx,dy)方向平移后写入shifted数组
                spread |= shifted
                # 用按位或累积所有偏移方向上的扩散结果
        mRData_py[spread & ~bad] |= np.int32(0x40)
        # 对扩散影响到的像素（原来不是bad的）设置0x40标记（OK_CONV相关）

    if sBorder > 0:
    # 如果边界宽度大于0，标记区域边界为坏像素
        mRData_py[:, :sBorder] |= np.int32(0x100 | 0x400)
        # 标记左边界列：左侧sBorder列设置T_BAD(0x100)|I_BAD(0x400)
        mRData_py[:, rPixX-sBorder:] |= np.int32(0x100 | 0x400)
        # 标记右边界列：右侧sBorder列设置T_BAD|I_BAD
        mRData_py[:sBorder, sBorder:rPixX-sBorder] |= np.int32(0x100 | 0x400)
        # 标记上边界行：顶部sBorder行（排除左右边界列以避免冗余）设置T_BAD|I_BAD
        mRData_py[rPixY-sBorder:, sBorder:rPixX-sBorder] |= np.int32(0x100 | 0x400)
        # 标记下边界行：底部sBorder行设置T_BAD|I_BAD

    xBufLo = rXMin - rXBMin
    # 计算左缓冲宽度：原始区域起始X减缓冲后起始X
    xBufHi = rXBMax - rXMax
    # 计算右缓冲宽度：缓冲后结束X减原始区域结束X
    yBufLo = rYMin - rYBMin
    # 计算上缓冲高度
    yBufHi = rYBMax - rYMax
    # 计算下缓冲高度
    fpixelOutX = rXBMin + xBufLo + 1
    # 计算输出图像中该区域的起始X像素（FITS格式1起始）
    fpixelOutY = rYBMin + yBufLo + 1
    # 计算输出图像中该区域的起始Y像素
    lpixelOutX = fpixelOutX + (rPixX - xBufHi - xBufLo - 1)
    # 计算输出图像中该区域的结束X像素：起始值+有效像素数（排除左右缓冲）
    lpixelOutY = fpixelOutY + (rPixY - yBufHi - yBufLo - 1)
    # 计算输出图像中该区域的结束Y像素

    logger.debug("  region_setup: done rPixX=%d rPixY=%d fpixelOut=(%d,%d) lpixelOut=(%d,%d)",
    # 输出调试日志：完成区域设置
                 rPixX, rPixY, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY)
                 # 缓冲后尺寸、输出起始像素、输出结束像素
    return {'tRData': tRData_py, 'iRData': iRData_py,
    # 返回模板区域数组和科学图像区域数组
            'oRData': oRData_py, 'eRData': eRData_py,
            # 返回输出差异区域数组和噪声方差区域数组
            'mRData': mRData_py, 'misRData': misRData_py, 'mtsRData': mtsRData_py,
            # 返回综合遮罩、科学图像遮罩、模板遮罩
            'rXMin': rXMin, 'rYMin': rYMin, 'rXMax': rXMax, 'rYMax': rYMax,
            # 返回原始区域坐标（无padding）
            'rXBMin': rXBMin, 'rYBMin': rYBMin, 'rXBMax': rXBMax, 'rYBMax': rYBMax,
            # 返回缓冲后区域坐标（含padding）
            'xBufLo': xBufLo, 'xBufHi': xBufHi, 'yBufLo': yBufLo, 'yBufHi': yBufHi,
            # 返回四边缓冲宽度
            'fpixelOutX': fpixelOutX, 'fpixelOutY': fpixelOutY,
            # 返回输出起始像素坐标
            'lpixelOutX': lpixelOutX, 'lpixelOutY': lpixelOutY,
            # 返回输出结束像素坐标
            'rPixX': rPixX, 'rPixY': rPixY}
            # 返回缓冲后区域尺寸


def region_buildstamps_numpy(setup_result: Dict, ctx_info: Dict, params_info: Dict, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:
# 区域Stamp构建函数：在区域内构建模板和图像的Stamp数组，含子Stamp中心检测。返回build结果字典
    logger.debug("region_buildstamps_numpy start")
    # 输出调试日志：开始构建区域Stamp

    nCompKer = ctx_info['nCompKer']
    # 从上下文取核分量数（含所有高斯分量及其多项式展开）
    nBGVectors = ctx_info['nBGVectors']
    # 取背景多项式向量数
    nC = ctx_info['nC']
    # 取拟合矩阵列数（核分量数+2）
    fwStamp = ctx_info['fwStamp']
    # 取Stamp全宽（含padding）
    fwKSStamp = ctx_info['fwKSStamp']
    # 取子Stamp窗口全宽
    fwKernel = ctx_info['fwKernel']
    # 取卷积核全宽
    nStamps = ctx_info['nStamps']
    # 取Stamp总数
    nStampX = ctx_info['nStampX']
    # 取X方向Stamp数
    nStampY = ctx_info['nStampY']
    # 取Y方向Stamp数
    fitThresh = ctx_info['fitThresh']
    # 取核拟合阈值（用于PsfCenter检测时的峰值筛选）

    hwKernel = params_info['hwKernel']
    # 从参数信息取核半宽
    ngauss = params_info['ngauss']
    # 取高斯分量个数
    deg_fixe = params_info['deg_fixe']
    # 取每个高斯分量的多项式阶数列表
    sigma_gauss = params_info['sigma_gauss']
    # 取每个高斯分量的sigma值（实际存的是1/(2σ^2)）
    hwKSStamp = params_info['hwKSStamp']
    # 取子Stamp窗口半宽
    nKSStamps = params_info['nKSStamps']
    # 取每个Stamp中允许的最大子Stamp数量
    scaleFitThresh = params_info['scaleFitThresh']
    # 取核拟合阈值缩放因子
    minFracGoodStamps = params_info['minFracGoodStamps']
    # 取最少有效Stamp比例阈值
    tUKThresh = params_info['tUKThresh']
    # 取模板PsfCenter检测的上限阈值
    iUKThresh = params_info['iUKThresh']
    # 取科学图像PsfCenter检测的上限阈值
    verbose = params_info.get('verbose', 0)
    # 取详细输出级别，默认0
    usePCA = params_info.get('usePCA', 0)
    # 取是否使用PCA基，默认0
    PCA = params_info.get('PCA', None)
    # 取PCA基矩阵，默认None
    xcmp = params_info.get('xcmp', None)
    # 取手动指定中心的X坐标数组，默认None
    ycmp = params_info.get('ycmp', None)
    # 取手动指定中心的Y坐标数组
    Ncmp = params_info.get('Ncmp', 0)
    # 取手动指定中心的数量
    statSig = params_info['statSig']
    # 取Sigma-clip的Sigma阈值
    findSSC = params_info.get('findSSC', 0)
    # 取是否自动搜索子Stamp中心，默认0

    tRData = setup_result['tRData']
    # 取区域模板图像数据
    iRData = setup_result['iRData']
    # 取区域科学图像数据
    mRData = setup_result['mRData']
    # 取区域综合遮罩数据
    rXBMin = setup_result['rXBMin']
    # 取缓冲后区域X起始坐标
    rYBMin = setup_result['rYBMin']
    # 取缓冲后区域Y起始坐标
    rXBMax = setup_result['rXBMax']
    # 取缓冲后区域X结束坐标
    rYBMax = setup_result['rYBMax']
    # 取缓冲后区域Y结束坐标
    rPixX = setup_result['rPixX']
    # 取缓冲后区域X像素数
    rPixY = setup_result['rPixY']
    # 取缓冲后区域Y像素数

    tRData1d = tRData.ravel()
    # 将模板2D数组展平为1D数组（用于传给buildStampsNumba）
    iRData1d = iRData.ravel()
    # 将科学图像2D数组展平为1D数组
    mRData1d = mRData.ravel()
    # 将遮罩2D数组展平为1D数组

    useFullSS = 0
    # 不使用全子Stamp模式（每个像素都是一个子中心）
    status = 1
    # 状态标志：1=首次构建Stamp，2=缩放阈值后重试
    flag = 1
    # 循环控制标志：非零则继续
    kerFitThresh = fitThresh
    # 初始化核拟合阈值为原始值
    ctSa = None
    # 模板Stamp数组初始为None
    ciSa = None
    # 图像Stamp数组初始为None

    while (status <= 2) and flag:
    # 最多执行两次（首次用原始阈值，第二次用缩放后的阈值）
        flag = 0
        # 先假设本次能成功（后续若Stamp不够会置1重新循环）
        niS = 0
        # 图像Stamp计数器归零
        ntS = 0
        # 模板Stamp计数器归零

        if localForceConvolve != "i":
        # 如果不需要强制使用图像（即需要构建模板Stamp）

            ct = {}  # template stamps flat arrays dict
            # 创建模板Stamp扁平数组字典（所有Stamp数据用numpy数组存储）
            nVec = nCompKer + nBGVectors
            # 向量总数：核分量向量数 + 背景多项式向量数
            fwSq = fwKSStamp * fwKSStamp
            # 子Stamp正方形像素数
            ct['nss'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的子Stamp数量数组
            ct['sscnt'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的当前子Stamp计数器数组
            ct['x0'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的起始X坐标数组
            ct['y0'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的起始Y坐标数组
            ct['x'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的中心X坐标数组
            ct['y'] = np.zeros(nStamps, dtype=np.int32)
            # 每个Stamp的中心Y坐标数组
            ct['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            # 每个Stamp的各子Stamp中心X坐标数组 (nStamps×max_sub)
            ct['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            # 每个Stamp的各子Stamp中心Y坐标数组
            ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
            # 每个Stamp的卷积向量数组 (nStamps×nVec×fwSq)
            ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
            # 每个Stamp的局部拟合矩阵数组 (nStamps×nC×nC)
            ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
            # 每个Stamp的局部标量积数组 (nStamps×nC)
            ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
            # 每个Stamp的参考区域像素值数组 (nStamps×fwSq)
            ct['chi2'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的卡方拟合残差数组
            ct['norm'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的核归一化系数数组
            ct['diff'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的偏差值数组
            ct['sum_val'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的像素绝对值之和数组
            ct['mean_val'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的像素均值数组
            ct['median'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的像素中位数数组
            ct['mode'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的像素众数数组（天光估计值）
            ct['sd'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的像素标准差数组
            ct['fwhm'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的半高全宽数组
            ct['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
            # 每个Stamp的大尺度半高全宽数组
            ct['valid'] = np.zeros(nStamps, dtype=np.bool_)
            # 每个Stamp的有效标志数组
            ct['nKSStamps'] = nKSStamps
            # 保存最大子Stamp数
            ct['fwKSStamp'] = fwKSStamp
            # 保存子Stamp窗口全宽
            ct['nCompKer'] = nCompKer
            # 保存核分量数
            ct['nBGVectors'] = nBGVectors
            # 保存背景向量数
            ct['nC'] = nC
            # 保存拟合矩阵列数
            ct['fwSq'] = fwSq
            # 保存子Stamp像素数
            ct['nVec'] = nVec
            # 保存向量总数
            ctSa = ct
            # 将模板Stamp字典赋值给ctSa变量
        if localForceConvolve != "t":
        # 如果不需要强制使用模板（即需要构建图像Stamp）
            ci = {}  # image stamps flat arrays dict
            # 创建图像Stamp扁平数组字典
            nVec = nCompKer + nBGVectors
            # 向量总数
            fwSq = fwKSStamp * fwKSStamp
            # 子Stamp像素数
            ci['nss'] = np.zeros(nStamps, dtype=np.int32); ci['sscnt'] = np.zeros(nStamps, dtype=np.int32)
            # 子Stamp数量和当前计数器
            ci['x0'] = np.zeros(nStamps, dtype=np.int32); ci['y0'] = np.zeros(nStamps, dtype=np.int32)
            # Stamp起始坐标
            ci['x'] = np.zeros(nStamps, dtype=np.int32); ci['y'] = np.zeros(nStamps, dtype=np.int32)
            # Stamp中心坐标
            ci['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32); ci['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            # 子Stamp中心坐标
            ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
            # 卷积向量数组
            ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
            # 局部拟合矩阵
            ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
            # 局部标量积
            ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
            # 参考区域像素值
            ci['chi2'] = np.zeros(nStamps, dtype=np.float64); ci['norm'] = np.zeros(nStamps, dtype=np.float64)
            # 卡方残差和归一化系数
            ci['diff'] = np.zeros(nStamps, dtype=np.float64)
            # 偏差值
            ci['sum_val'] = np.zeros(nStamps, dtype=np.float64); ci['mean_val'] = np.zeros(nStamps, dtype=np.float64)
            # 像素和、像素均值
            ci['median'] = np.zeros(nStamps, dtype=np.float64); ci['mode'] = np.zeros(nStamps, dtype=np.float64)
            # 中位数、众数
            ci['sd'] = np.zeros(nStamps, dtype=np.float64); ci['fwhm'] = np.zeros(nStamps, dtype=np.float64); ci['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
            # 标准差、半高全宽、大尺度半高全宽
            ci['valid'] = np.zeros(nStamps, dtype=np.bool_)
            # 有效标志
            ci['nKSStamps'] = nKSStamps; ci['fwKSStamp'] = fwKSStamp
            # 最大子Stamp数和窗口全宽
            ci['nCompKer'] = nCompKer; ci['nBGVectors'] = nBGVectors; ci['nC'] = nC
            # 核分量数、背景向量数、矩阵列数
            ci['fwSq'] = fwSq; ci['nVec'] = nVec
            # 子Stamp像素数、向量总数
            ciSa = ci
            # 将图像Stamp字典赋值给ciSa变量

        # 建立None安全的别名：当forceConvolve==\"t\"时ciSa为None，==\"i\"时ctSa为None
        if ctSa is not None:
        # 如果模板Stamp存在
            ctNss = ctSa['nss']; ctX0 = ctSa['x0']; ctY0 = ctSa['y0']; ctX = ctSa['x']; ctY = ctSa['y']
            # 子Stamp数、起始/中心坐标
            ctSumVal = ctSa['sum_val']; ctMeanVal = ctSa['mean_val']; ctMedian = ctSa['median']; ctMode = ctSa['mode']
            # 像素和、均值、中位数、众数
            ctSd = ctSa['sd']; ctFwhm = ctSa['fwhm']; ctLfwhm = ctSa['lfwhm']; ctXss = ctSa['xss']; ctYss = ctSa['yss']
            # 标准差、半高全宽、大尺度半高全宽、子中心坐标
        else:
        # 模板Stamp不存在时的占位数组
            ctNss = np.zeros(1, dtype=np.int32); ctX0 = np.zeros(1, dtype=np.int32); ctY0 = np.zeros(1, dtype=np.int32)
            # 分配大小为1的占位数组
            ctX = np.zeros(1, dtype=np.int32); ctY = np.zeros(1, dtype=np.int32)
            # 占位中心坐标
            ctSumVal = np.zeros(1, dtype=np.float64); ctMeanVal = np.zeros(1, dtype=np.float64); ctMedian = np.zeros(1, dtype=np.float64)
            # 占位统计量
            ctMode = np.zeros(1, dtype=np.float64); ctSd = np.zeros(1, dtype=np.float64); ctFwhm = np.zeros(1, dtype=np.float64)
            # 占位众数、标准差、半高全宽
            ctLfwhm = np.zeros(1, dtype=np.float64); ctXss = np.zeros((1, 1), dtype=np.int32); ctYss = np.zeros((1, 1), dtype=np.int32)
            # 占位大尺度半高全宽、子中心坐标
            ctSscnt = np.zeros(1, dtype=np.int32)
            # 占位子Stamp计数器
        if ciSa is not None:
        # 如果图像Stamp存在
            ciNss = ciSa['nss']; ciX0 = ciSa['x0']; ciY0 = ciSa['y0']; ciX = ciSa['x']; ciY = ciSa['y']
            # 子Stamp数、起始/中心坐标
            ciSumVal = ciSa['sum_val']; ciMeanVal = ciSa['mean_val']; ciMedian = ciSa['median']; ciMode = ciSa['mode']
            # 像素和、均值、中位数、众数
            ciSd = ciSa['sd']; ciFwhm = ciSa['fwhm']; ciLfwhm = ciSa['lfwhm']; ciXss = ciSa['xss']; ciYss = ciSa['yss']
            # 标准差、半高全宽、大尺度半高全宽、子中心坐标
        else:
        # 图像Stamp不存在时的占位数组
            ciNss = np.zeros(1, dtype=np.int32); ciX0 = np.zeros(1, dtype=np.int32); ciY0 = np.zeros(1, dtype=np.int32)
            # 占位子Stamp数和坐标
            ciX = np.zeros(1, dtype=np.int32); ciY = np.zeros(1, dtype=np.int32)
            # 占位中心坐标
            ciSumVal = np.zeros(1, dtype=np.float64); ciMeanVal = np.zeros(1, dtype=np.float64); ciMedian = np.zeros(1, dtype=np.float64)
            # 占位统计量
            ciMode = np.zeros(1, dtype=np.float64); ciSd = np.zeros(1, dtype=np.float64); ciFwhm = np.zeros(1, dtype=np.float64)
            # 占位众数、标准差、半高全宽
            ciLfwhm = np.zeros(1, dtype=np.float64); ciXss = np.zeros((1, 1), dtype=np.int32); ciYss = np.zeros((1, 1), dtype=np.int32)
            # 占位大尺度半高全宽、子中心坐标
            ciSscnt = np.zeros(1, dtype=np.int32)
            # 占位子Stamp计数器

        for l in range(nStampY):
        # 遍历Y方向所有Stamp行
            for k in range(nStampX):
            # 遍历X方向所有Stamp列
                logger.info("Build stamp  : t %4d i %4d (grid coord %2d %2d)", ntS, niS, k, l)
                # 输出每个Stamp的构建信息

                sXMin = rXBMin + k * rPixX // nStampX
                # 当前Stamp的X起始坐标（缓冲后区域的等分位置）
                sYMin = rYBMin + l * rPixY // nStampY
                # 当前Stamp的Y起始坐标
                sXMax = min(sXMin + fwStamp - 1, rXBMax)
                # 当前Stamp的X结束坐标（不超过缓冲后区域边界）
                sYMax = min(sYMin + fwStamp - 1, rYBMax)
                # 当前Stamp的Y结束坐标

                if localForceConvolve != "i":
                # 需要模板Stamp时初始化
                    ctSa['sscnt'][ntS] = 0
                    # 重置当前模板Stamp的子Stamp计数器
                    ctSa['nss'][ntS] = 0
                    # 重置当前模板Stamp的子Stamp总数
                    ctSa['chi2'][ntS] = 0.0
                    # 重置当前模板Stamp的卡方值
                if localForceConvolve != "t":
                # 需要图像Stamp时初始化
                    ciSa['sscnt'][niS] = 0
                    # 重置当前图像Stamp的子Stamp计数器
                    ciSa['nss'][niS] = 0
                    # 重置当前图像Stamp的子Stamp总数
                    ciSa['chi2'][niS] = 0.0
                    # 重置当前图像Stamp的卡方值

                if xcmp is not None and Ncmp > 0:
                # 如果提供了手动指定的Stamp中心坐标列表
                    if verbose >= 2:
                        logger.info("Adding centers manually")
                        # 输出手动添加中心的日志
                    for m in range(Ncmp):
                    # 遍历每个手动指定的中心点
                        if (xcmp[m] > sXMin + hwKernel + 1) and (xcmp[m] < sXMax - hwKernel - 1) and \
                           (ycmp[m] > sYMin + hwKernel + 1) and (ycmp[m] < sYMax - hwKernel - 1):
                        # 手动中心坐标必须在Stamp范围内（预留核半宽边界）

                            (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                            # 接收buildStampsNumba返回的模板Stamp各字段：起始/中心坐标、统计量、FWHM、子Stamp数、子中心坐标
                             lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                            # 接收buildStampsNumba返回的图像Stamp各字段
                             lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                            # 调用buildStampsNumba构建Stamp：传入区域坐标范围
                                                          sYMax, 0, rXBMin,
                                                          # getCenters=0表示使用手动指定中心(非自动检测)，rXBMin为区域全局X起点
                                                          rYBMin, ctNss[ntS],
                                                          # 区域Y起点，当前模板Stamp已有子Stamp数
                                                          ctX0[ntS], ctY0[ntS],
                                                          # 当前模板Stamp的起始X/Y坐标
                                                          ctX[ntS], ctY[ntS],
                                                          # 当前模板Stamp的中心X/Y坐标
                                                          ctSumVal[ntS],
                                                          # 当前模板Stamp的像素绝对值之和
                                                          ctMeanVal[ntS],
                                                          # 当前模板Stamp的像素均值
                                                          ctMedian[ntS],
                                                          # 当前模板Stamp的像素中位数
                                                          ctMode[ntS],
                                                          # 当前模板Stamp的像素众数（天空背景估计）
                                                          ctSd[ntS],
                                                          # 当前模板Stamp的像素标准差
                                                          ctFwhm[ntS],
                                                          # 当前模板Stamp的半高全宽
                                                          ctLfwhm[ntS],
                                                          # 当前模板Stamp的大尺度半高全宽
                                                          ctXss[ntS].copy(),
                                                          # 当前模板Stamp的子中心X坐标数组副本
                                                          ctYss[ntS].copy(),
                                                          # 当前模板Stamp的子中心Y坐标数组副本
                                                          ciNss[niS],
                                                          # 当前图像Stamp的子Stamp数
                                                          ciX0[niS],
                                                          # 当前图像Stamp的起始X坐标
                                                          ciY0[niS],
                                                          # 当前图像Stamp的起始Y坐标
                                                          ciX[niS],
                                                          # 当前图像Stamp的中心X坐标
                                                          ciY[niS],
                                                          # 当前图像Stamp的中心Y坐标
                                                          ciSumVal[niS],
                                                          # 当前图像Stamp的像素和
                                                          ciMeanVal[niS],
                                                          # 当前图像Stamp的均值
                                                          ciMedian[niS],
                                                          # 当前图像Stamp的中位数
                                                          ciMode[niS],
                                                          # 当前图像Stamp的众数
                                                          ciSd[niS],
                                                          # 当前图像Stamp的标准差
                                                          ciFwhm[niS],
                                                          # 当前图像Stamp的半高全宽
                                                          ciLfwhm[niS],
                                                          # 当前图像Stamp的大尺度半高全宽
                                                          ciXss[niS].copy(),
                                                          # 当前图像Stamp的子中心X坐标副本
                                                          ciYss[niS].copy(),
                                                          # 当前图像Stamp的子中心Y坐标副本
                                                          iRData1d, tRData1d,
                                                          # 图像和模板的1D扁平数据
                                                          xcmp[m] - rXBMin,
                                                          # 手动指定中心的X坐标（转为区域内相对坐标）
                                                          ycmp[m] - rYBMin,
                                                          # 手动指定中心的Y坐标（转为区域内相对坐标）
                                                          localForceConvolve,
                                                          # 卷积方向：t/i/b（模板/图像/双向）
                                                          rPixX, rPixY,
                                                          # 区域X/Y像素数
                                                          tUKThresh, iUKThresh,
                                                          # 模板/图像PsfCenter检测的上限阈值
                                                          hwKSStamp, fwStamp,
                                                          # 子Stamp半宽、Stamp全宽
                                                          nKSStamps,
                                                          # 最大子Stamp数
                                                          kerFitThresh,
                                                          # 核拟合阈值
                                                          mRData1d.copy(),
                                                          # 遮罩数据副本（内部可能修改）
                                                          statSig)
                                                          # Sigma-clip显著性阈值
                            ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                            # 将返回的模板Stamp起始坐标写回ctSa
                            ctX[ntS], ctY[ntS] = lctX, lctY
                            # 写回模板Stamp中心坐标
                            ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                            # 写回模板Stamp像素和、均值
                            ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                            # 写回模板Stamp中位数、众数
                            ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                            # 写回模板Stamp标准差、半高全宽
                            ctLfwhm[ntS] = lctLfwhm
                            # 写回模板Stamp大尺度半高全宽
                            ctNss[ntS] = lctNss
                            # 写回模板Stamp子Stamp总数
                            ctXss[ntS] = lctXss
                            # 写回模板Stamp子中心X坐标
                            ctYss[ntS] = lctYss
                            # 写回模板Stamp子中心Y坐标
                            ciX0[niS], ciY0[niS] = lciX0, lciY0
                            # 写回图像Stamp起始坐标
                            ciX[niS], ciY[niS] = lciX, lciY
                            # 写回图像Stamp中心坐标
                            ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                            # 写回图像Stamp像素和、均值
                            ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                            # 写回图像Stamp中位数、众数
                            ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                            # 写回图像Stamp标准差、半高全宽
                            ciLfwhm[niS] = lciLfwhm
                            # 写回图像Stamp大尺度半高全宽
                            ciNss[niS] = lciNss
                            # 写回图像Stamp子Stamp总数
                            ciXss[niS] = lciXss
                            # 写回图像Stamp子中心X坐标
                            ciYss[niS] = lciYss
                            # 写回图像Stamp子中心Y坐标
                            mRData1d[:] = lmRData
                            # 将内部修改后的遮罩数据同步回主遮罩数组

                    if findSSC:
                    # 如果开启自动搜索子Stamp中心模式
                        logger.debug("Automatically finding additional centers")
                        # 输出调试信息：自动搜索额外的子Stamp中心

                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                        # 再次调用buildStampsNumba，getCenters=1启用自动Psf峰值检测
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                                                          lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          # 调用buildStampsNumba构建Stamp（自动检测模式）
                                                          sYMax, 1, rXBMin,
                                                          # getCenters=1自动检测Psf峰值作为子Stamp中心、区域X起始
                                                          rYBMin, ctNss[ntS],
                                                          # 区域Y起始、当前模板Stamp已有子Stamp数
                                                          ctX0[ntS], ctY0[ntS],
                                                          # 当前模板Stamp起始坐标
                                                          ctX[ntS], ctY[ntS],
                                                          # 当前模板Stamp中心坐标
                                                          ctSumVal[ntS], ctMeanVal[ntS],
                                                          # 当前模板Stamp像素和与均值
                                                          ctMedian[ntS], ctMode[ntS],
                                                          # 当前模板Stamp中位数与众数
                                                          ctSd[ntS], ctFwhm[ntS], ctLfwhm[ntS],
                                                          # 当前模板Stamp标准差、半高全宽、大尺度半高全宽
                                                          ctXss[ntS].copy(), ctYss[ntS].copy(),
                                                          # 当前模板Stamp子中心坐标副本
                                                          ciNss[niS], ciX0[niS], ciY0[niS],
                                                          # 当前图像Stamp子Stamp数和起始坐标
                                                          ciX[niS], ciY[niS],
                                                          # 当前图像Stamp中心坐标
                                                          ciSumVal[niS], ciMeanVal[niS],
                                                          # 当前图像Stamp像素和与均值
                                                          ciMedian[niS], ciMode[niS],
                                                          # 当前图像Stamp中位数与众数
                                                          ciSd[niS], ciFwhm[niS], ciLfwhm[niS],
                                                          # 当前图像Stamp标准差、半高全宽、大尺度半高全宽
                                                          ciXss[niS].copy(), ciYss[niS].copy(),
                                                          # 当前图像Stamp子中心坐标副本
                                                          iRData1d, tRData1d, 0,
                                                          # 图像和模板1D数据、hardX=0无手动中心
                                                          0, localForceConvolve,
                                                          # hardY=0无手动中心、卷积方向标志
                                                          rPixX, rPixY,
                                                          # 区域X/Y像素数
                                                          tUKThresh, iUKThresh,
                                                          # 模板/图像Psf检测上限阈值
                                                          hwKSStamp, fwStamp,
                                                          # 子Stamp半宽、Stamp全宽
                                                          nKSStamps, kerFitThresh,
                                                          # 最大子Stamp数、核拟合阈值
                                                          mRData1d.copy(), statSig)
                                                          # 遮罩数据副本、Sigma-clip阈值
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                        # 写回模板Stamp起始坐标
                        ctX[ntS], ctY[ntS] = lctX, lctY
                        # 写回模板Stamp中心坐标
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        # 写回模板Stamp像素和与均值
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        # 写回模板Stamp中位数与众数
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                        # 写回模板Stamp标准差与半高全宽
                        ctLfwhm[ntS] = lctLfwhm
                        # 写回模板Stamp大尺度半高全宽
                        ctNss[ntS] = lctNss
                        # 写回模板Stamp子Stamp总数
                        ctXss[ntS] = lctXss
                        # 写回模板Stamp子中心X坐标
                        ctYss[ntS] = lctYss
                        # 写回模板Stamp子中心Y坐标
                        ciX0[niS], ciY0[niS] = lciX0, lciY0
                        # 写回图像Stamp起始坐标
                        ciX[niS], ciY[niS] = lciX, lciY
                        # 写回图像Stamp中心坐标
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        # 写回图像Stamp像素和与均值
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        # 写回图像Stamp中位数与众数
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                        # 写回图像Stamp标准差与半高全宽
                        ciLfwhm[niS] = lciLfwhm
                        # 写回图像Stamp大尺度半高全宽
                        ciNss[niS] = lciNss
                        # 写回图像Stamp子Stamp总数
                        ciXss[niS] = lciXss
                        # 写回图像Stamp子中心X坐标
                        ciYss[niS] = lciYss
                        # 写回图像Stamp子中心Y坐标
                        mRData1d[:] = lmRData
                        # 同步遮罩数据

                else:
                # 既无手动中心也非自动检测模式：对每个Stamp网格点调用buildStampsNumba
                    if useFullSS:
                    # 全子Stamp模式：每个像素都是子中心
                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal,
                        # 调用buildStampsNumba（全子Stamp模式，参数同前模式）
                         lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal,
                         lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          sYMax, 1, rXBMin,
                                                          # getCenters=1：全子模式也检测峰值
                                                          rYBMin, ctNss[ntS],
                                                          ctX0[ntS], ctY0[ntS],
                                                          ctX[ntS], ctY[ntS],
                                                          ctSumVal[ntS],
                                                          ctMeanVal[ntS],
                                                          ctMedian[ntS],
                                                          ctMode[ntS],
                                                          ctSd[ntS],
                                                          ctFwhm[ntS],
                                                          ctLfwhm[ntS],
                                                          ctXss[ntS].copy(),
                                                          ctYss[ntS].copy(),
                                                          ciNss[niS],
                                                          ciX0[niS],
                                                          ciY0[niS],
                                                          ciX[niS],
                                                          ciY[niS],
                                                          ciSumVal[niS],
                                                          ciMeanVal[niS],
                                                          ciMedian[niS],
                                                          ciMode[niS],
                                                          ciSd[niS],
                                                          ciFwhm[niS],
                                                          ciLfwhm[niS],
                                                          ciXss[niS].copy(),
                                                          ciYss[niS].copy(),
                                                          iRData1d, tRData1d, 0,
                                                          # 模板和图像1D数据，无手动中心
                                                          0, localForceConvolve,
                                                          # 卷积方向
                                                          rPixX, rPixY,
                                                          tUKThresh, iUKThresh,
                                                          hwKSStamp, fwStamp,
                                                          nKSStamps,
                                                          kerFitThresh,
                                                          mRData1d.copy(),
                                                          statSig)
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0; ctX[ntS], ctY[ntS] = lctX, lctY
                        # 写回模板Stamp起始/中心坐标
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        # 写回模板Stamp像素和与均值
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        # 写回中位数与众数
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm; ctLfwhm[ntS] = lctLfwhm
                        # 写回标准差、半高全宽、大尺度半高全宽
                        ctNss[ntS] = lctNss; ctXss[ntS] = lctXss; ctYss[ntS] = lctYss
                        # 写回子Stamp数和子中心坐标
                        ciX0[niS], ciY0[niS] = lciX0, lciY0; ciX[niS], ciY[niS] = lciX, lciY
                        # 写回图像Stamp起始/中心坐标
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        # 写回图像Stamp像素和与均值
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        # 写回中位数与众数
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm; ciLfwhm[niS] = lciLfwhm
                        # 写回标准差、半高全宽、大尺度半高全宽
                        ciNss[niS] = lciNss; ciXss[niS] = lciXss; ciYss[niS] = lciYss
                        # 写回子Stamp数和子中心坐标
                        mRData1d[:] = lmRData
                        # 同步遮罩数据

                    else:
                    # 标准模式：非全子Stamp，使用Stamp中心
                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, 
                        # 调用buildStampsNumba（标准模式，getCenters=1但无全子）
                         lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         # 接收模板Stamp的其他返回字段
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal,
                         # 接收图像Stamp的起始坐标/中心坐标/像素和/均值
                         lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         # 接收图像Stamp的其他返回字段
                         lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                         # 调用buildStampsNumba构建Stamp（标准模式）
                                                          sYMax, 1, rXBMin,
                                                          # Y结束坐标、getCenters=1（自动检测）、区域X起始
                                                          rYBMin, ctNss[ntS],
                                                          # 区域Y起始、当前模板Stamp已有子Stamp数
                                                          ctX0[ntS], ctY0[ntS],
                                                          # 当前模板Stamp起始X/Y坐标
                                                          ctX[ntS], ctY[ntS],
                                                          # 当前模板Stamp中心X/Y坐标
                                                          ctSumVal[ntS],
                                                          # 当前模板Stamp像素和
                                                          ctMeanVal[ntS],
                                                          # 当前模板Stamp像素均值
                                                          ctMedian[ntS],
                                                          # 当前模板Stamp像素中位数
                                                          ctMode[ntS],
                                                          # 当前模板Stamp像素众数（天空背景估计值）
                                                          ctSd[ntS],
                                                          # 当前模板Stamp像素标准差
                                                          ctFwhm[ntS],
                                                          # 当前模板Stamp半高全宽
                                                          ctLfwhm[ntS],
                                                          # 当前模板Stamp大尺度半高全宽
                                                          ctXss[ntS].copy(),
                                                          # 当前模板Stamp子中心X坐标数组副本
                                                          ctYss[ntS].copy(),
                                                          # 当前模板Stamp子中心Y坐标数组副本
                                                          ciNss[niS],
                                                          # 当前图像Stamp已有子Stamp数
                                                          ciX0[niS],
                                                          # 当前图像Stamp起始X坐标
                                                          ciY0[niS],
                                                          # 当前图像Stamp起始Y坐标
                                                          ciX[niS],
                                                          # 当前图像Stamp中心X坐标
                                                          ciY[niS],
                                                          # 当前图像Stamp中心Y坐标
                                                          ciSumVal[niS],
                                                          # 当前图像Stamp像素和
                                                          ciMeanVal[niS],
                                                          # 当前图像Stamp像素均值
                                                          ciMedian[niS],
                                                          # 当前图像Stamp像素中位数
                                                          ciMode[niS],
                                                          # 当前图像Stamp像素众数
                                                          ciSd[niS],
                                                          # 当前图像Stamp像素标准差
                                                          ciFwhm[niS],
                                                          # 当前图像Stamp半高全宽
                                                          ciLfwhm[niS],
                                                          # 当前图像Stamp大尺度半高全宽
                                                          ciXss[niS].copy(),
                                                          # 当前图像Stamp子中心X坐标数组副本
                                                          ciYss[niS].copy(),
                                                          # 当前图像Stamp子中心Y坐标数组副本
                                                          iRData1d, tRData1d, 0,
                                                          # 图像和模板1D数据、hardX=0无手动中心
                                                          0, localForceConvolve,
                                                          # hardY=0无手动中心、卷积方向标志
                                                          rPixX, rPixY,
                                                          # 区域X/Y像素数
                                                          tUKThresh, iUKThresh,
                                                          # 模板/图像Psf检测上限阈值
                                                          hwKSStamp, fwStamp,
                                                          # 子Stamp半宽、Stamp全宽
                                                          nKSStamps,
                                                          # 最大子Stamp数
                                                          kerFitThresh,
                                                          # 核拟合阈值
                                                          mRData1d.copy(),
                                                          # 遮罩数据副本
                                                          statSig)
                                                          # Sigma-clip阈值
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0; ctX[ntS], ctY[ntS] = lctX, lctY
                        # 写回模板Stamp起始/中心坐标
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        # 写回模板Stamp像素和与均值
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        # 写回中位数与众数
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm; ctLfwhm[ntS] = lctLfwhm
                        # 写回标准差、半高全宽、大尺度半高全宽
                        ctNss[ntS] = lctNss; ctXss[ntS] = lctXss; ctYss[ntS] = lctYss
                        # 写回子Stamp数和子中心坐标
                        ciX0[niS], ciY0[niS] = lciX0, lciY0; ciX[niS], ciY[niS] = lciX, lciY
                        # 写回图像Stamp起始/中心坐标
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        # 写回图像Stamp像素和与均值
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        # 写回中位数与众数
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm; ciLfwhm[niS] = lciLfwhm
                        # 写回标准差、半高全宽、大尺度半高全宽
                        ciNss[niS] = lciNss; ciXss[niS] = lciXss; ciYss[niS] = lciYss
                        # 写回子Stamp数和子中心坐标
                        mRData1d[:] = lmRData
                        # 同步遮罩数据

                if localForceConvolve != "i":
                # 如果构建了模板Stamp
                    logger.debug("    templ: %d substamps", ctSa['nss'][ntS])
                    # 输出当前模板Stamp检测到的子Stamp数
                    if ctSa['nss'][ntS] > 0:
                    # 如果模板Stamp有至少一个有效子Stamp
                        ntS += 1
                        # 模板Stamp有效计数器加1
                if localForceConvolve != "t":
                # 如果构建了图像Stamp
                    logger.debug("    image: %d substamps", ciSa['nss'][niS])
                    # 输出当前图像Stamp检测到的子Stamp数
                    if ciSa['nss'][niS] > 0:
                    # 如果图像Stamp有至少一个有效子Stamp
                        niS += 1
                        # 图像Stamp有效计数器加1

        iSFrac = niS / float(nStamps)
        # 计算有效图像Stamp占总Stamp数的比例
        tSFrac = ntS / float(nStamps)
        # 计算有效模板Stamp占总Stamp数的比例

        if localForceConvolve == "i":
        # 仅图像卷积：检查图像Stamp是否足够
            logger.info("%d stamps built (%.2f%%)", niS, iSFrac)
            # 输出图像Stamp构建结果
            if iSFrac < minFracGoodStamps:
            # 如果有效图像Stamp比例低于阈值
                flag = 1
                # 设置重试标志
        elif localForceConvolve == "t":
        # 仅模板卷积
            logger.info("%d stamps built (%.2f%%)", ntS, tSFrac)
            # 输出模板Stamp构建结果
            if tSFrac < minFracGoodStamps:
            # 比例不足
                flag = 1
                # 设置重试标志
        elif (iSFrac < minFracGoodStamps) or (tSFrac < minFracGoodStamps):
        # 双向卷积，任一方向不足
            logger.info("%d and %d stamps built (%.2f%%, %.2f%%)", ntS, niS, tSFrac, iSFrac)
            # 输出构建结果
            flag = 1
            # 设置重试标志
        else:
        # Stamp数量满足要求
            logger.info("%d and %d stamps built (%.2f%%, %.2f%%)", ntS, niS, tSFrac, iSFrac)
            # 输出最终结果
            break
            # 退出while循环

        if flag and (status <= 1) and (scaleFitThresh < 1.0):
        # 需要重试且阈值可缩放
            kerFitThresh *= scaleFitThresh
            # 按缩放因子降低核拟合阈值（放宽条件）
            logger.info("Too few stamps were fit, scaling down fitting threshold to %.2f", kerFitThresh)
            # 输出缩放后的新阈值

            if localForceConvolve != "i":
            # 重建模板Stamp数组（清空后重新构建）
                nVec = nCompKer + nBGVectors
                # 计算向量总数：核分量向量数+背景多项式向量数
                fwSq = fwKSStamp * fwKSStamp
                # 计算子Stamp的正方形像素数
                ct = {}
                # 创建空字典存储模板Stamp数据
                ct['nss'] = np.zeros(nStamps, dtype=np.int32)
                # 子Stamp数量数组
                ct['sscnt'] = np.zeros(nStamps, dtype=np.int32)
                # 当前子Stamp计数器数组
                ct['x0'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp起始X坐标数组
                ct['y0'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp起始Y坐标数组
                ct['x'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp中心X坐标数组
                ct['y'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp中心Y坐标数组
                ct['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                # 各Stamp的子中心X坐标二维数组
                ct['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                # 各Stamp的子中心Y坐标二维数组
                ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
                # 卷积向量三维数组 (nStamps × nVec × fwSq)
                ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
                # 局部拟合矩阵三维数组 (nStamps × nC × nC)
                ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
                # 局部标量积二维数组 (nStamps × nC)
                ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
                # 参考区域像素值二维数组 (nStamps × fwSq)
                ct['chi2'] = np.zeros(nStamps, dtype=np.float64)
                # 卡方拟合残差数组
                ct['norm'] = np.zeros(nStamps, dtype=np.float64)
                # 核归一化系数数组
                ct['diff'] = np.zeros(nStamps, dtype=np.float64)
                # 偏差值数组
                ct['sum_val'] = np.zeros(nStamps, dtype=np.float64)
                # 像素绝对值之和数组
                ct['mean_val'] = np.zeros(nStamps, dtype=np.float64)
                # 像素均值数组
                ct['median'] = np.zeros(nStamps, dtype=np.float64)
                # 像素中位数数组
                ct['mode'] = np.zeros(nStamps, dtype=np.float64)
                # 像素众数数组（天空背景估计）
                ct['sd'] = np.zeros(nStamps, dtype=np.float64)
                # 像素标准差数组
                ct['fwhm'] = np.zeros(nStamps, dtype=np.float64)
                # 半高全宽数组
                ct['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
                # 大尺度半高全宽数组
                ct['valid'] = np.zeros(nStamps, dtype=np.bool_)
                # 有效标志数组
                ct['nKSStamps'] = nKSStamps
                # 保存最大子Stamp数
                ct['fwKSStamp'] = fwKSStamp
                # 保存子Stamp窗口全宽
                ct['nCompKer'] = nCompKer
                # 保存核分量数
                ct['nBGVectors'] = nBGVectors
                # 保存背景向量数
                ct['nC'] = nC
                # 保存拟合矩阵列数
                ct['fwSq'] = fwSq
                # 保存子Stamp像素数
                ct['nVec'] = nVec
                # 保存向量总数
                ctSa = ct
                # 将字典赋值给ctSa变量
            if localForceConvolve != "t":
            # 重建图像Stamp数组
                nVec = nCompKer + nBGVectors
                # 计算向量总数
                fwSq = fwKSStamp * fwKSStamp
                # 计算子Stamp像素数
                ci = {}
                # 创建空字典
                ci['nss'] = np.zeros(nStamps, dtype=np.int32)
                # 子Stamp数量数组
                ci['sscnt'] = np.zeros(nStamps, dtype=np.int32)
                # 当前子Stamp计数器
                ci['x0'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp起始X坐标
                ci['y0'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp起始Y坐标
                ci['x'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp中心X坐标
                ci['y'] = np.zeros(nStamps, dtype=np.int32)
                # Stamp中心Y坐标
                ci['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                # 子中心X坐标二维数组
                ci['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                # 子中心Y坐标二维数组
                ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
                # 卷积向量三维数组
                ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
                # 局部拟合矩阵三维数组
                ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
                # 局部标量积二维数组
                ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
                # 参考区域像素值二维数组
                ci['chi2'] = np.zeros(nStamps, dtype=np.float64)
                # 卡方拟合残差
                ci['norm'] = np.zeros(nStamps, dtype=np.float64)
                # 核归一化系数
                ci['diff'] = np.zeros(nStamps, dtype=np.float64)
                # 偏差值
                ci['sum_val'] = np.zeros(nStamps, dtype=np.float64)
                # 像素绝对值之和
                ci['mean_val'] = np.zeros(nStamps, dtype=np.float64)
                # 像素均值
                ci['median'] = np.zeros(nStamps, dtype=np.float64)
                # 像素中位数
                ci['mode'] = np.zeros(nStamps, dtype=np.float64)
                # 像素众数
                ci['sd'] = np.zeros(nStamps, dtype=np.float64)
                # 像素标准差
                ci['fwhm'] = np.zeros(nStamps, dtype=np.float64)
                # 半高全宽
                ci['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
                # 大尺度半高全宽
                ci['valid'] = np.zeros(nStamps, dtype=np.bool_)
                # 有效标志
                ci['nKSStamps'] = nKSStamps
                # 保存最大子Stamp数
                ci['fwKSStamp'] = fwKSStamp
                # 保存子Stamp窗口全宽
                ci['nCompKer'] = nCompKer
                # 保存核分量数
                ci['nBGVectors'] = nBGVectors
                # 保存背景向量数
                ci['nC'] = nC
                # 保存拟合矩阵列数
                ci['fwSq'] = fwSq
                # 保存子Stamp像素数
                ci['nVec'] = nVec
                # 保存向量总数
                ciSa = ci
                # 赋值给ciSa

            mRData1d[:] = mRData1d & ~0xa00
            # 清除遮罩中的T_BAD(0x100)和I_BAD(0x400)位（重试前重置）

        status += 1
        # 状态计数器加1：每轮循环递增

    if (niS == 0) and (ntS == 0):
    # 如果模板和图像Stamp数都为0（完全失败）
        result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
        # 返回包含Stamp计数和空Stamp数据的字典
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
              # 核向量和滤波器设为None，状态-1表示失败
        return result
        # 提前返回失败结果
    if localForceConvolve == "i":
    # 如果仅图像卷积模式
        if niS == 0:
        # 图像Stamp数为0
            result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
            # 返回失败结果
                  'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
                  # 状态-1
            return result
            # 提前返回
    if localForceConvolve == "t":
    # 如果仅模板卷积模式
        if ntS == 0:
        # 模板Stamp数为0
            result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
            # 返回失败结果
                  'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
                  # 状态-1
            return result
            # 提前返回

    filter_x = np.zeros(fwKernel * nCompKer, dtype=np.float64)
    # 分配X方向滤波器数组（每个核分量×fwKernel个元素）
    filter_y = np.zeros(fwKernel * nCompKer, dtype=np.float64)
    # 分配Y方向滤波器数组
    kernel_vec = get_kernel_vec_numpy(ngauss, deg_fixe, usePCA, fwKernel, hwKernel,
    # 调用get_kernel_vec_numpy生成完整核基向量列表
                                      sigma_gauss, filter_x, filter_y, PCA)
                                      # 传入各高斯分量sigma、滤波器数组（原地写入）、PCA矩阵

    logger.debug("region_buildstamps_numpy done")
    # 输出调试日志：区域Stamp构建完成
    result = {'niS': niS, 'ntS': ntS,'ctStamps': ctSa,'ciStamps': ciSa,
    # 构建返回字典：图像/模板Stamp计数和Stamp数组
              'kernel_vec': kernel_vec,'filter_x': filter_x,'filter_y': filter_y,'status': 0}
              # 核基向量列表、滤波器数组、成功状态0
    return result
    # 返回构建结果


def region_fit_numpy(buildstamps_result: Dict, setup_result: Dict, ctx_info: Dict, params_info: Dict, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:
# 区域核拟合函数：对构建好的Stamp进行填充和迭代核拟合。返回fit结果字典
    logger.debug("region_fit_numpy start")
    # 输出调试日志

    nCompKer = ctx_info['nCompKer']
    # 取核分量数
    # nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    # 取拟合矩阵列数
    # nCompTotal = ctx_info['nCompTotal']
    fwKSStamp = ctx_info['fwKSStamp']
    # 取子Stamp窗口全宽
    fwKernel = ctx_info['fwKernel']
    # 取卷积核全宽

    hwKernel = params_info['hwKernel']
    # 取核半宽
    ngauss = params_info['ngauss']
    # 取高斯分量数
    deg_fixe = params_info['deg_fixe']
    # 取各高斯分量的多项式阶数列表
    kerOrder = params_info['kerOrder']
    # 取核多项式阶数
    bgOrder = params_info['bgOrder']
    # 取背景多项式阶数
    hwKSStamp = params_info['hwKSStamp']
    # 取子Stamp窗口半宽
    # verbose = params_info.get('verbose', 0)
    # usePCA = params_info.get('usePCA', 0)
    # PCA = params_info.get('PCA', None)
    statSig = params_info['statSig']
    # 取Sigma-clip阈值
    kerSigReject = params_info['kerSigReject']
    # 取核拟合拒绝阈值
    fillVal = params_info['fillVal']
    # 取无效像素填充值
    figMerit = params_info['figMerit']
    # 取品质指标类型(v=方差/s=偏度/h=峰度)

    tRData = setup_result['tRData']
    # 取模板区域数组
    iRData = setup_result['iRData']
    # 取科学图像区域数组
    oRData = setup_result['oRData']
    # 取方差区域数组
    mRData = setup_result['mRData']
    # 取综合遮罩数组
    rPixX = setup_result['rPixX']
    # 取区域X像素数
    rPixY = setup_result['rPixY']
    # 取区域Y像素数

    tRData1d = tRData.ravel()
    # 展平模板图像为1D
    iRData1d = iRData.ravel()
    # 展平科学图像为1D
    oRData1d = oRData.ravel()
    # 展平方差图像为1D
    mRData1d = mRData.ravel()
    # 展平遮罩为1D

    # ctStamps = buildstamps_result['ctStamps']
    ctSa = buildstamps_result['ctStamps']
    # 取模板Stamp字典
    # ciStamps = buildstamps_result['ciStamps']
    ciSa = buildstamps_result['ciStamps']
    # 取图像Stamp字典
    ntS = buildstamps_result['ntS']
    # 取模板有效Stamp数
    niS = buildstamps_result['niS']
    # 取图像有效Stamp数
    kernel_vec = buildstamps_result['kernel_vec']
    # 取核基向量列表
    filter_x = buildstamps_result['filter_x']
    # 取X方向滤波器数组
    filter_y = buildstamps_result['filter_y']
    # 取Y方向滤波器数组
    # 扁平数组字段（从 ctSa/ciSa dict 读取，替代原来 build_stamps_flatten_helper 展开的字段）
    if ctSa is not None:
    # 如果模板Stamp存在（非纯图像卷积模式）
        ctSscnt = ctSa['sscnt']
        # 取出子Stamp计数器别名
        ctNss = ctSa['nss']
        # 取出子Stamp数量别名
        ctVectors = ctSa['vectors']
        # 取出卷积向量别名
        ctMat = ctSa['mat']
        # 取出局部矩阵别名
        ctScprod = ctSa['scprod']
        # 取出局部标量积别名
        ctXss = ctSa['xss']
        # 取出子中心X坐标别名
        ctYss = ctSa['yss']
        # 取出子中心Y坐标别名
        ctKrefArea = ctSa['krefArea']
        # 取出参考区域别名
        ctSumVal = ctSa['sum_val']
        # 取出像素和别名
        ctNorm = ctSa['norm']
        # 取出归一化系数别名
        ctDiff = ctSa['diff']
        # 取出偏差值别名
        ctNKSStamps = ctSa['nKSStamps']
        # 取出最大子Stamp数
    else:
    # 模板Stamp不存在时的占位None值
        ctSscnt = None; ctNss = None; ctVectors = None; ctMat = None; ctScprod = None
        # 全部设为None
        ctXss = None; ctYss = None; ctKrefArea = None; ctSumVal = None
        # 坐标和统计量为None
        ctNorm = None; ctDiff = None
        # 归一化和偏差为None
        ctNKSStamps = params_info.get('nKSStamps')
        # 从参数取默认最大子Stamp数
    if ciSa is not None:
    # 如果图像Stamp存在（非纯模板卷积模式）
        ciSscnt = ciSa['sscnt']
        # 取出子Stamp计数器别名
        ciNss = ciSa['nss']
        # 取出子Stamp数量别名
        ciVectors = ciSa['vectors']
        # 取出卷积向量别名
        ciMat = ciSa['mat']
        # 取出局部矩阵别名
        ciScprod = ciSa['scprod']
        # 取出局部标量积别名
        ciXss = ciSa['xss']
        # 取出子中心X坐标别名
        ciYss = ciSa['yss']
        # 取出子中心Y坐标别名
        ciKrefArea = ciSa['krefArea']
        # 取出参考区域别名
        ciSumVal = ciSa['sum_val']
        # 取出像素和别名
        ciNorm = ciSa['norm']
        # 取出归一化系数别名
        ciDiff = ciSa['diff']
        # 取出偏差值别名
        ciNKSStamps = ciSa['nKSStamps']
        # 取出最大子Stamp数
    else:
    # 图像Stamp不存在时的占位None值
        ciSscnt = None; ciNss = None; ciVectors = None; ciMat = None; ciScprod = None
        # 全部设为None
        ciXss = None; ciYss = None; ciKrefArea = None; ciSumVal = None
        # 坐标和统计量为None
        ciNorm = None; ciDiff = None
        # 归一化和偏差为None
        ciNKSStamps = params_info.get('nKSStamps')
        # 从参数取默认值

    tMerit = 0.0
    # 模板品质指标初始化为0
    iMerit = 0.0
    # 图像品质指标初始化为0
    convTmpl = 0
    # 卷积模式标志：0=原方向，1=反向（模板→图像）

    logger.info("Filling Template sub-stamps")
    # 输出日志：开始填充模板子Stamp
    if localForceConvolve != "i":
    # 如果需要模板Stamp
        for k in range(ntS):
        # 遍历所有模板Stamp
            ctSa['sscnt'][k] = 0
            # 重置每个模板Stamp的子Stamp计数器为0（第一次填充）
        ct_si_list = [k for k in range(ntS) if ctSscnt[k] < ctNss[k]]
        # 批量收集有子Stamp的模板Stamp索引列表
        if ct_si_list:
        # 如果存在有效Stamps
            ct_out = fill_stamp_numba(ctXss, ctYss, ctSscnt, ctNss, ct_si_list,
            # 调用fill_stamp_numba对指定Stamps进行批量填充（当前sscnt对应的子中心）
                                       tRData1d, iRData1d, rPixX, rPixY, ngauss, deg_fixe,
                                       # 模板和图像1D数据、区域尺寸、高斯参数
                                       hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                                       # Stamp和核的尺寸参数、背景阶数、核分量数
                                       filter_x, filter_y, fillVal, mRData1d)
                                       # 核滤波器数组、无效填充值、遮罩数据
            ct_out_v, ct_out_k, ct_out_m, ct_out_s, ct_out_sum = ct_out
            # 解包返回值：向量、参考区域、矩阵、标量积、像素和
            nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
            # 计算背景多项式项数
            fwSq = fwKSStamp * fwKSStamp
            # 计算子Stamp正方形像素数
            nC = nCompKer + 1
            # 当前操作的矩阵列数
            for bi, si in enumerate(ct_si_list):
            # 遍历每个成功填充的Stamp
                ctVectors[si, :nCompKer + nbg, :fwSq] = ct_out_v[bi]
                # 将返回的卷积向量写入ctVectors对应位置
                ctKrefArea[si, :] = ct_out_k[bi]
                # 将参考区域数据写入ctKrefArea
                ctMat[si, :nC + 1, :nC + 1] = ct_out_m[bi]
                # 将局部拟合矩阵写入ctMat
                ctScprod[si, :nC + 1] = ct_out_s[bi]
                # 将局部标量积写入ctScprod
                ctSumVal[si] = ct_out_sum[bi]
                # 将像素和写入ctSumVal
        if localForceConvolve == "b":
        # 如果是双向卷积模式，需要做check_stamps评估
            logger.info("Trying to convolve the TEMPLATE to fit IMAGE")
            # 输出：尝试模板卷积拟合图像
            tMerit = check_stamps_numpy(ctScprod, ctMat, ctNorm, ctDiff, ctSscnt,
            # 调用check_stamps_numpy对模板Stamps做独立拟合和异常检测
                ctNss, ctXss, ctYss, ctVectors, ctKrefArea, ntS, iRData1d, oRData1d,
                # 子Stamp信息、参考区域、图像和方差数据
                nCompKer, kerOrder, bgOrder,
                # 核分量数、核阶数、背景阶数
                localForceConvolve, figMerit, kerSigReject, statSig,
                # 卷积方向、品质指标、拒绝阈值、Sigma阈值
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                # Stamp和图像尺寸、核尺寸
                kernel_vec, mRData1d, nKSStamps=ctNKSStamps)
                # 核基向量、遮罩数据、最大子Stamp数
            logger.info("    Result : merit = %.3f", tMerit)
            # 输出品质指标值
        else:
        # 非双向模式
            tMerit = 0.0; iMerit = 0.0
            # 品质指标置0

    logger.info("Filling Image sub-stamps")
    # 输出日志：开始填充图像子Stamp
    if localForceConvolve != "t":
    # 如果需要图像Stamp
        for k in range(niS):
        # 遍历所有图像Stamp
            ciSa['sscnt'][k] = 0
            # 重置每个图像Stamp的子Stamp计数器
        ci_si_list = [k for k in range(niS) if ciSscnt[k] < ciNss[k]]
        # 批量收集有子Stamp的图像Stamp索引列表
        if ci_si_list:
        # 如果存在有效Stamps
            ci_out = fill_stamp_numba(ciXss, ciYss, ciSscnt, ciNss, ci_si_list,
            # 调用fill_stamp_numba对图像Stamps进行批量填充
                                       iRData1d, tRData1d, rPixX, rPixY, ngauss, deg_fixe,
                                       # 注意：imConv和imRef交换了（图像作为被卷积对象，模板作为参考）
                                       hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                                       filter_x, filter_y, fillVal, mRData1d)
            ci_out_v, ci_out_k, ci_out_m, ci_out_s, ci_out_sum = ci_out
            # 解包返回结果
            nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
            # 背景项数
            fwSq = fwKSStamp * fwKSStamp
            # 子Stamp像素数
            nC = nCompKer + 1
            # 矩阵列数
            for bi, si in enumerate(ci_si_list):
            # 遍历每个成功填充的图像Stamp
                ciVectors[si, :nCompKer + nbg, :fwSq] = ci_out_v[bi]
                # 写回卷积向量
                ciKrefArea[si, :] = ci_out_k[bi]
                # 写回参考区域
                ciMat[si, :nC + 1, :nC + 1] = ci_out_m[bi]
                # 写回局部矩阵
                ciScprod[si, :nC + 1] = ci_out_s[bi]
                # 写回局部标量积
                ciSumVal[si] = ci_out_sum[bi]
                # 写回像素和
        if localForceConvolve == "b":
        # 双向模式检测
            logger.info("Trying to convolve the IMAGE to fit TEMPLATE")
            # 输出提示
            iMerit = check_stamps_numpy(ciScprod, ciMat, ciNorm, ciDiff, ciSscnt,
            # 对图像Stamps做独立拟合和异常检测
                ciNss, ciXss, ciYss, ciVectors, ciKrefArea, niS, tRData1d, oRData1d,
                nCompKer, kerOrder, bgOrder,
                localForceConvolve, figMerit, kerSigReject, statSig,
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                kernel_vec, mRData1d, nKSStamps=ciNKSStamps)
            logger.info("    Result : merit = %.3f", iMerit)
            # 输出品质指标
        else:
            iMerit = 0.0; tMerit = 0.0
            # 非双向：品质指标置0

    if (localForceConvolve == "t") or \
       ((tMerit < iMerit) and (localForceConvolve != "i")):
    # 如果强制模板卷积，或（模板品质<图像品质）且非强制图像模式
        convTmpl = 1
        # 设置为模板卷积模式（用模板去拟合图像）
    else:
    # 否则
        convTmpl = 0
        # 使用图像卷积模式（用图像去拟合模板）

    logger.debug("region_fit_numpy done, convTmpl=%s", convTmpl)
    # 输出最终卷积模式
    result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
    # 构建返回字典
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': 0}
              # 核向量和滤波器后续从buildstamps_result取，此处暂设None
    result['convTmpl'] = convTmpl
    # 附加卷积模式标志
    result['tMerit'] = tMerit
    # 附加模板品质指标
    result['iMerit'] = iMerit
    # 附加图像品质指标
    return result
    # 返回拟合结果字典


def region_convolve_diff_numpy(fit_result: Dict, setup_result: Dict, buildstamps_result: Dict,
# 区域卷积分差函数：对模板/图像执行空间卷积、计算差异图像和噪声组合。
                                 ctx_info: Dict, params_info: Dict, region_idx: int, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:
                                 # 接收上下文、参数、区域索引、卷积方向、日志记录器
    nCompKer = ctx_info['nCompKer']
    # 取核分量数
    fwKSStamp = ctx_info['fwKSStamp']
    # 取子Stamp窗口全宽
    fwKernel = ctx_info['fwKernel']
    # 取卷积核全宽
    kcStep = ctx_info['kcStep']
    # 取核缓存步长

    hwKernel = params_info['hwKernel']
    # 取核半宽
    ngauss = params_info['ngauss']
    # 取高斯分量数
    deg_fixe = params_info['deg_fixe']
    # 取各高斯分量的多项式阶数列表
    kerOrder = params_info['kerOrder']
    # 取核多项式阶数
    bgOrder = params_info['bgOrder']
    # 取背景多项式阶数
    hwKSStamp = params_info['hwKSStamp']
    # 取子Stamp窗口半宽
    # verbose = params_info.get('verbose', 0)
    # usePCA = params_info.get('usePCA', 0)
    # PCA = params_info.get('PCA', None)
    statSig = params_info['statSig']
    # 取Sigma-clip阈值
    kerSigReject = params_info['kerSigReject']
    # 取核拟合拒绝阈值
    kerFracMask = params_info['kerFracMask']
    # 取核分数遮罩阈值（低于此值的核像素被标记）
    fillVal = params_info['fillVal']
    # 取无效像素填充值
    fillValNoise = params_info['fillValNoise']
    # 取噪声填充值
    figMerit = params_info['figMerit']
    # 取品质指标类型
    tUThresh = params_info['tUThresh']
    # 取模板上限阈值
    tLThresh = params_info['tLThresh']
    # 取模板下限阈值
    tGain = params_info['tGain']
    # 取模板增益
    tRdnoise = params_info['tRdnoise']
    # 取模板读出噪声
    iUThresh = params_info['iUThresh']
    # 取科学图像上限阈值
    iLThresh = params_info['iLThresh']
    # 取科学图像下限阈值
    iGain = params_info['iGain']
    # 取科学图像增益
    iRdnoise = params_info['iRdnoise']
    # 取科学图像读出噪声
    convolveVariance = params_info.get('convolveVariance', 0)
    # 取方差卷积模式标志
    sameConv = params_info.get('sameConv', 0)
    # 取是否使用相同核的标志
    savexyflag = params_info.get('savexyflag', 0)
    # 取是否保存Stamp坐标的标志
    tnoise_2d = params_info.get('tNoiseFullData', None)
    # 取模板噪声全图数据
    inoise_2d = params_info.get('iNoiseFullData', None)
    # 取科学图像噪声全图数据

    tRData = setup_result['tRData'].copy()
    # 复制模板区域数据（独立副本）
    iRData = setup_result['iRData'].copy()
    # 复制科学图像区域数据
    mRData = setup_result['mRData'].copy()
    # 复制综合遮罩数据
    misRData = setup_result['misRData'].copy()
    # 复制科学图像遮罩
    mtsRData = setup_result['mtsRData'].copy()
    # 复制模板遮罩
    rXBMin = setup_result['rXBMin']
    # 取缓冲后区域X起始
    rYBMin = setup_result['rYBMin']
    # 取缓冲后区域Y起始
    rXBMax = setup_result['rXBMax']
    # 取缓冲后区域X结束
    rYBMax = setup_result['rYBMax']
    # 取缓冲后区域Y结束
    rPixX = setup_result['rPixX']
    # 取区域X像素数
    rPixY = setup_result['rPixY']
    # 取区域Y像素数
    rXMin = setup_result['rXMin']
    # 取原始区域X起始
    rYMin = setup_result['rYMin']
    # 取原始区域Y起始
    rXMax = setup_result['rXMax']
    # 取原始区域X结束
    rYMax = setup_result['rYMax']
    # 取原始区域Y结束

    convTmpl = fit_result['convTmpl']
    # 取卷积模式标志
    ctSa = fit_result.get('ctStamps')
    # 取模板Stamp字典
    ciSa = fit_result.get('ciStamps')
    # 取图像Stamp字典

    ntS = buildstamps_result['ntS']
    # 取模板有效Stamp数
    niS = buildstamps_result['niS']
    # 取图像有效Stamp数
    kernel_vec = buildstamps_result['kernel_vec']
    # 取核基向量列表
    filter_x = buildstamps_result['filter_x']
    # 取X方向滤波器数组
    filter_y = buildstamps_result['filter_y']
    # 取Y方向滤波器数组

    kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
    # 分配核空间系数的临时数组
    kernel = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    # 分配单个卷积核的临时数组（fwKernel×fwKernel）

    tRData1d = tRData.ravel()
    iRData1d = iRData.ravel()
    mRData1d = mRData.ravel()
    misRData1d = misRData.ravel()
    mtsRData1d = mtsRData.ravel()

    meansigSubstamps = 0.0
    # 子Stamp平均信噪比初始化为0
    scatterSubstamps = 0.0
    # 子Stamp信噪比弥散度初始化为0
    NskippedSubstamps = 0
    # 跳过的子Stamp计数初始化为0
    savexy_entries = []
    # 保存坐标条目列表（空）
    sumKernel = 0.0
    # 核总和初始化为0
    kerSol = None
    # 核解向量暂设为None
    nS = 0
    # 当前处理的Stamp总数初始化为0

    if convTmpl:
    # 如果是模板卷积模式（用模板去拟合图像）
        logger.info("Region %d:%d,%d:%d : Convolving TEMPLATE", rXMin, rXMax, rYMin, rYMax)
        # 输出日志：正在对模板进行卷积
        nS = ntS
        # 设置当前处理的Stamp数为模板Stamp数

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        # 调试日志：开始核拟合
        oRData1d = setup_result['oRData'].ravel().copy()
        # 取方差数据副本
        fit_result_k = fit_kernel_numpy(ctSa, iRData1d, tRData1d, oRData1d,
        # 调用fit_kernel_numpy进行迭代核拟合（模板→图像方向）
            nCompKer, kerOrder, bgOrder, nS,
            # 核分量数、核阶数、背景阶数、Stamp数
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            # Stamp和区域尺寸、品质指标
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            # 拒绝阈值、Sigma阈值、遮罩、高斯参数
            hwKernel, fwKernel, filter_x, filter_y, fillVal,
            # 核尺寸、滤波器、无效填充值
            logger=logger)
            # 日志记录器
        tKerSol = fit_result_k['kernelSol']
        # 取模板核解向量
        meansigSubstamps = fit_result_k['meansigSubstamps']
        # 取平均信噪比
        scatterSubstamps = fit_result_k['scatterSubstamps']
        # 取信噪比弥散度
        NskippedSubstamps = fit_result_k['NskippedSubstamps']
        # 取跳过子Stamp数
        ctSa = fit_result_k['stamps']
        # 取更新后的模板Stamp字典
        kerSol = tKerSol
        # 保存当前核解向量

        logger.debug("[region %d] convolve_diff: fitKernel done", region_idx)
        # 核拟合完成
        logger.debug("[region %d] convolve_diff: realloc+mask start", region_idx)
        # 开始重新分配和遮罩设置
        oRData1d = np.full(rPixX * rPixY, fillVal, dtype=np.float32)
        # 重新分配输出差异区域数组，全部填充为无效值

        tdata = tRData1d
        # 创建模板数据的局部别名
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tdata == fillVal).astype(np.int32)
        # 标记模板填充值为坏像素（INPUT_ISBAD | BAD_PIXVAL）
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (tdata >= tUThresh).astype(np.int32)
        # 标记模板饱和像素
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (tdata <= tLThresh).astype(np.int32)
        # 标记模板过低像素

        mRData1d[:] = 0
        # 清除综合遮罩（重新开始）

        logger.debug("[region %d] convolve_diff: realloc+mask done", region_idx)
        # 重新分配完成
        logger.debug("[region %d] convolve_diff: noise rebuild start", region_idx)
        # 开始重建噪声
        eRData1d = np.full(rPixX * rPixY, fillValNoise, dtype=np.float32)
        # 分配噪声方差数组，全部填充为噪声填充值
        if tnoise_2d is not None:
        # 如果提供了模板噪声方差全图
            eRData1d[:] = tnoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            # 提取对应区域的噪声数据
            eRData1d[:] = eRData1d * eRData1d
            # 噪声值平方得到方差
        else:
        # 未提供噪声全图
            eRData1d = make_noise_image4_numpy(tRData1d, 1.0 / tGain, tRdnoise / tGain)
            # 根据增益和读出噪声公式计算方差

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        # 噪声重建完成
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        # 开始空间卷积
        logger.info("Convolving...")  # Template
        # 输出提示：正在对模板做卷积

        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
        # 调用快速空间卷积函数（预计算块核后批量卷积）
            tRData1d, eRData1d, rPixX, rPixY, tKerSol, mtsRData1d,
            # 输入：模板图像、方差、区域尺寸、核解、模板遮罩
            kcStep, hwKernel, fwKernel,
            # 核缓存步长、核半宽、核全宽
            convolveVariance, kerFracMask,
            # 方差卷积模式、核分数遮罩阈值
            rPixX, rPixY, nCompKer, kerOrder, bgOrder, kernel_vec)
            # 区域尺寸归一化、核分量数、核阶数、背景阶数、核基向量
        oRData1d[:] = oRData1d_new.ravel()
        # 将卷积结果写入输出差异区域
        mRData1d[:] = mRData1d_new.ravel()
        # 将更新后的遮罩写入综合遮罩

        if vData is not None:
        # 如果返回了方差数据
            eRData1d = vData
            # 用新的方差数据替换

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        # 空间卷积完成
        logger.debug("[region %d] convolve_diff: background start", region_idx)
        # 开始背景计算
        # background_loop_jit(oRData1d, tKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)
        # 背景循环已合入spatial_convolve_jit_kernel内部，此处不再单独调用
        logger.debug("[region %d] convolve_diff: background done", region_idx)
        # 背景计算完成
        logger.debug("[region %d] convolve_diff: make_kernel start", region_idx)
        # 开始构造核并计算核总和
        sumKernel = make_kernel_numpy(rXMin, rYMin, tKerSol, rPixX, rPixY,
        # 在区域左上角构造空间可变核，计算核总和
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
                                      # 核分量数、核阶数、核全宽、核基向量、系数数组、核数组
        logger.info(" Sum Kernel at %d,%d: %f", rXMin, rYMin, sumKernel)
        # 输出左上角的核总和
        sumKernel = make_kernel_numpy(rXMax, rYMax, tKerSol, rPixX, rPixY,
        # 在区域右下角构造核
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Sum Kernel at %d,%d: %f", rXMax, rYMax, sumKernel)
        # 输出右下角的核总和
        sumKernel = make_kernel_numpy(rPixX // 2, rPixY // 2, tKerSol, rPixX, rPixY,
        # 在区域中心构造核，作为最终使用的核总和
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Using Kernel Sum = %f", sumKernel)
        # 输出最终采用的核总和

        logger.debug("[region %d] convolve_diff: make_kernel done", region_idx)
        # 核构造完成
        logger.debug("[region %d] convolve_diff: noise_combine start", region_idx)
        # 开始噪声组合
        tRData1d[:] = fillValNoise
        # 用噪声填充值初始化tRData（后续用作噪声合并）
        if inoise_2d is not None:
        # 如果提供了科学图像噪声全图
            tRData1d[:] = inoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            # 提取对应区域噪声数据
            tRData1d[:] = tRData1d * tRData1d
            # 平方得到方差
        else:
        # 未提供噪声全图
            tRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain)
            # 根据科学图像增益和读出噪声计算方差
            
        tRData1d = np.sqrt(tRData1d + eRData1d)
        # 总噪声 = sqrt(科学图像方差 + 模板方差)：两图像噪声的平方和开根

        idata = iRData1d
        # 创建科学图像数据的局部别名
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (idata == fillVal).astype(np.int32)
        # 标记科学图像填充值像素为输出+输入坏像素
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (idata >= iUThresh).astype(np.int32)
        # 标记科学图像饱和像素
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (idata <= iLThresh).astype(np.int32)
        # 标记科学图像过低像素
        mRData1d |= misRData1d
        # 合并科学图像输入遮罩到综合遮罩
        mRData1d |= FLAG_OUTPUT_ISBAD * ((misRData1d & FLAG_INPUT_ISBAD) > 0).astype(np.int32)
        # 对输入遮罩中已有INPUT_ISBAD的像素额外设置OUTPUT_ISBAD

        logger.debug("[region %d] convolve_diff: noise_combine done", region_idx)
        # 噪声组合完成
        if savexyflag:
        # 如果需要保存Stamp坐标信息
            for si in range(ntS):
            # 遍历所有模板Stamp
                for sc in range(ctSa['nss'][si]):
                # 遍历当前Stamp的所有子Stamp
                    entry = {'x': int(ctSa['xss'][si, sc]), 'y': int(ctSa['yss'][si, sc])}
                    # 记录子Stamp的坐标
                    if sc == ctSa['sscnt'][si]:
                    # 当前正在使用的子Stamp
                        entry['isUsed'] = 1
                        # 标记为使用中
                    elif sc < ctSa['sscnt'][si]:
                    # 已使用过但被跳过的子Stamp
                        entry['isUsed'] = -1
                        # 标记为已跳过
                    else:
                    # 尚未使用的子Stamp
                        entry['isUsed'] = 0
                        # 标记为未使用
                    savexy_entries.append(entry)
                    # 添加到保存列表

        if sameConv:
        # 如果启用相同核模式（模板卷积和图像卷积使用相同结果）
            localForceConvolve = "t"
            # 强制设为模板卷积模式
            ciSa = None
            # 清除图像Stamp数据

    else:
    # 图像卷积模式：用科学图像去拟合模板
        logger.info("Region %d:%d %d,%d : Convolving IMAGE", rXMin, rXMax, rYMin, rYMax)
        # 输出日志：正在对图像做卷积
        nS = niS
        # 设置当前处理的Stamp数为图像Stamp数

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        # 开始核拟合（图像→模板方向）
        oRData1d = setup_result['oRData'].ravel().copy()
        # 取方差数据副本
        fit_result_k = fit_kernel_numpy(ciSa, tRData1d, iRData1d, oRData1d,
        # 调用fit_kernel_numpy进行迭代核拟合（注意ciSa传入，imRef和imConv交换）
            nCompKer, kerOrder, bgOrder, nS,
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            hwKernel, fwKernel, filter_x, filter_y, fillVal,
            logger=logger)
        iKerSol = fit_result_k['kernelSol']
        # 取图像核解向量
        meansigSubstamps = fit_result_k['meansigSubstamps']
        # 取平均信噪比
        scatterSubstamps = fit_result_k['scatterSubstamps']
        # 取信噪比弥散度
        NskippedSubstamps = fit_result_k['NskippedSubstamps']
        # 取跳过子Stamp数
        ciSa = fit_result_k['stamps']
        # 取更新后的图像Stamp字典
        kerSol = iKerSol
        # 保存当前核解向量

        logger.debug("[region %d] convolve_diff: fitKernel done", region_idx)
        # 核拟合完成
        logger.debug("[region %d] convolve_diff: realloc+mask start", region_idx)
        # 开始重新分配和遮罩设置
        oRData1d = np.full(rPixX * rPixY, fillVal, dtype=np.float32)
        # 重新分配输出差异区域

        tdata = iRData1d
        # 创建科学图像数据的局部别名
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tdata == fillVal).astype(np.int32)
        # 标记科学图像填充值为坏像素
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (tdata >= iUThresh).astype(np.int32)
        # 标记科学图像饱和像素
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (tdata <= iLThresh).astype(np.int32)
        # 标记科学图像过低像素

        mRData1d[:] = 0
        # 清除综合遮罩

        logger.debug("[region %d] convolve_diff: realloc+mask done", region_idx)
        logger.debug("[region %d] convolve_diff: noise rebuild start", region_idx)
        eRData1d = np.full(rPixX * rPixY, fillValNoise, dtype=np.float32)
        if inoise_2d is not None:
            eRData1d[:] = inoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            eRData1d[:] = eRData1d * eRData1d
        else:
            eRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain)

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        logger.info("Convolving...")  # Image

        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
            iRData1d, eRData1d, rPixX, rPixY, iKerSol, misRData1d,
            kcStep, hwKernel, fwKernel,
            convolveVariance, kerFracMask,
            rPixX, rPixY, nCompKer, kerOrder, bgOrder, kernel_vec)
        oRData1d[:] = oRData1d_new.ravel()
        mRData1d[:] = mRData1d_new.ravel()
        if vData is not None:
            eRData1d = vData

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        logger.debug("[region %d] convolve_diff: background start", region_idx)

        # background_loop_jit(oRData1d, iKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)

        logger.debug("[region %d] convolve_diff: background done", region_idx)
        logger.debug("[region %d] convolve_diff: make_kernel start", region_idx)
        sumKernel = make_kernel_numpy(rXMin, rYMin, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Sum Kernel at %d,%d: %f", rXMin, rYMin, sumKernel)
        sumKernel = make_kernel_numpy(rXMax, rYMax, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Sum Kernel at %d,%d: %f", rXMax, rYMax, sumKernel)
        sumKernel = make_kernel_numpy(rPixX // 2, rPixY // 2, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Using Kernel Sum = %f", sumKernel)

        logger.debug("[region %d] convolve_diff: make_kernel done", region_idx)
        logger.debug("[region %d] convolve_diff: noise_combine start", region_idx)
        iRData1d[:] = fillValNoise
        if tnoise_2d is not None:
            iRData1d[:] = tnoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            iRData1d[:] = iRData1d * iRData1d
        else:
            iRData1d = make_noise_image4_numpy(tRData1d, 1.0 / tGain, tRdnoise / tGain)

        iRData1d = np.sqrt(iRData1d + eRData1d)
        tdata = tRData1d
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tdata == fillVal).astype(np.int32)
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (tdata >= tUThresh).astype(np.int32)
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (tdata <= tLThresh).astype(np.int32)
        mRData1d |= mtsRData1d
        mRData1d |= FLAG_OUTPUT_ISBAD * ((mtsRData1d & FLAG_INPUT_ISBAD) > 0).astype(np.int32)

        logger.debug("[region %d] convolve_diff: noise_combine done", region_idx)
        if savexyflag:
            for si in range(niS):
                # for sc in range(ciStamps[si]['nss']):
                for sc in range(ciSa['nss'][si]):
                    entry = {'x': int(ciSa['xss'][si, sc]), 'y': int(ciSa['yss'][si, sc])}
                    # if sc == ciStamps[si]['sscnt']:
                    if sc == ciSa['sscnt'][si]:
                        entry['isUsed'] = 1
                    # elif sc < ciStamps[si]['sscnt']:
                    elif sc < ciSa['sscnt'][si]:
                        entry['isUsed'] = -1
                    else:
                        entry['isUsed'] = 0
                    savexy_entries.append(entry)

        if sameConv:
        # 如果启用相同核模式
            localForceConvolve = "i"
            # 强制设为图像卷积模式
            ctSa = None
            # 清除模板Stamp数据

    noiseData1d = tRData1d if convTmpl else iRData1d
    # 选择噪声数据：模板卷积模式用模板噪声，图像卷积模式用图像噪声

    result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
    # 构建返回字典：Stamp计数和Stamp数据
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': 0}
              # 核向量和滤波器设为None（上层从buildstamps_result获取）
    result.update({'oRData': oRData1d.reshape(rPixY, rPixX),
    # 卷积后的差异图像（2D）
        'noiseData': noiseData1d.reshape(rPixY, rPixX), 'mRData': mRData1d.reshape(rPixY, rPixX),
        # 噪声数据和综合遮罩（2D）
        'sumKernel': sumKernel, 'meansigSubstamps': meansigSubstamps,
        # 核总和、平均子Stamp信噪比
        'scatterSubstamps': scatterSubstamps, 'NskippedSubstamps': NskippedSubstamps,
        # 信噪比弥散度、跳过子Stamp数
        'convTmpl': convTmpl, 'nS': nS, 'localForceConvolve': localForceConvolve,
        # 卷积模式、Stamp数、卷积方向
        'kerSol': kerSol, 'savexy_entries': savexy_entries,
        # 核解向量、坐标保存条目
        'savexyXmin': rXBMin + (0 if convTmpl else 1),
        # 保存坐标的最小X（FITS 1起始）
        'savexyYmin': rYBMin + (0 if convTmpl else 1),
        # 保存坐标的最小Y
        'ctStamps': ctSa, 'ciStamps': ciSa,
        # 模板和图像Stamp字典
        'tRData': tRData1d.reshape(rPixY, rPixX), 'iRData': iRData1d.reshape(rPixY, rPixX)})
        # 模板和图像区域数据（2D）
    return result
    # 返回卷积分差结果字典


def region_output_numpy(convolve_result: Dict, setup_result: Dict,
# 区域输出函数：将卷积结果转换为最终的diff/conv/noise/mask图像，统计噪声和品质指标。
                         diff_out: np.ndarray, noise_out: np.ndarray, conv_out: np.ndarray, mask_out: np.ndarray,
                         # 接收4个输出数组（全图，原地写入对应区域）
                         ctx_info: Dict, params_info: Dict, region_idx: int, stats_list: List,
                         # 上下文、参数、区域索引、统计列表
                         logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    logger.debug("[region %d] region_output_numpy start", region_idx)
    # 调试日志：开始区域输出

    fwKSStamp = ctx_info['fwKSStamp']
    # 取子Stamp窗口全宽

    hwKernel = params_info['hwKernel']
    # 取核半宽
    hwKSStamp = params_info['hwKSStamp']
    # 取子Stamp窗口半宽
    statSig = params_info['statSig']
    # 取Sigma-clip阈值
    fillVal = params_info['fillVal']
    # 取无效像素填充值
    fillValNoise = params_info['fillValNoise']
    # 取噪声填充值
    photNormalize = params_info['photNormalize']
    # 取光度归一化模式
    figMerit = params_info['figMerit']
    # 取品质指标类型
    rescaleOK = params_info.get('rescaleOK', 0)
    # 取是否允许重新缩放
    kfSpreadMask2 = params_info.get('kfSpreadMask2', -1.0)
    # 取第二扩散遮罩因子

    rPixX = setup_result['rPixX']
    # 取区域X像素数
    rPixY = setup_result['rPixY']
    # 取区域Y像素数
    xBufLo = setup_result['xBufLo']
    # 取左缓冲宽度
    yBufLo = setup_result['yBufLo']
    # 取上缓冲宽度
    fpixelOutX = setup_result['fpixelOutX']
    # 取输出起始X像素
    fpixelOutY = setup_result['fpixelOutY']
    # 取输出起始Y像素
    lpixelOutX = setup_result['lpixelOutX']
    # 取输出结束X像素
    lpixelOutY = setup_result['lpixelOutY']
    # 取输出结束Y像素

    convTmpl = convolve_result['convTmpl']
    # 取卷积模式标志
    nS = convolve_result['nS']
    # 取Stamp数
    sumKernel = convolve_result['sumKernel']
    # 取核总和
    meansigSubstamps = convolve_result['meansigSubstamps']
    # 取平均信噪比
    scatterSubstamps = convolve_result['scatterSubstamps']
    # 取信噪比弥散度

    oRData = convolve_result['oRData'].copy()
    # 复制差异图像数据
    noiseData = convolve_result['noiseData'].copy()
    # 复制噪声数据
    mRData = convolve_result['mRData'].copy()
    # 复制综合遮罩数据
    tRData = convolve_result['tRData']
    # 引用模板图像数据
    iRData = convolve_result['iRData']
    # 引用科学图像数据
    ctSa = convolve_result.get('ctStamps')
    # 取模板Stamp字典
    ciSa = convolve_result.get('ciStamps')
    # 取图像Stamp字典
    ctSscntOut = ctSa['sscnt'] if ctSa is not None else None
    # 取模板子Stamp计数器别名
    ctNssOut = ctSa['nss'] if ctSa is not None else None
    # 取模板子Stamp数别名
    ctXssOut = ctSa['xss'] if ctSa is not None else None
    # 取模板子中心X坐标别名
    ctYssOut = ctSa['yss'] if ctSa is not None else None
    # 取模板子中心Y坐标别名
    ciSscntOut = ciSa['sscnt'] if ciSa is not None else None
    # 取图像子Stamp计数器别名
    ciNssOut = ciSa['nss'] if ciSa is not None else None
    # 取图像子Stamp数别名
    ciXssOut = ciSa['xss'] if ciSa is not None else None
    # 取图像子中心X坐标别名
    ciYssOut = ciSa['yss'] if ciSa is not None else None
    # 取图像子中心Y坐标别名

    oRData1d = oRData.ravel()
    # 展平差异图像
    noiseData1d = noiseData.ravel()
    # 展平噪声数据
    mRData1d = mRData.ravel()
    # 展平遮罩数据
    tRData1d = tRData.ravel()
    # 展平模板图像
    iRData1d = iRData.ravel()
    # 展平科学图像

    mRData2d = mRData1d.reshape(rPixY, rPixX)
    # 将遮罩变回2D以便切片操作
    mRData2d[:, :hwKernel] |= FLAG_OUTPUT_ISBAD
    # 标记左边界列为输出坏像素
    mRData2d[:, rPixX - hwKernel:rPixX] |= FLAG_OUTPUT_ISBAD
    # 标记右边界列为输出坏像素
    mRData2d[:hwKernel, hwKernel:rPixX - hwKernel] |= FLAG_OUTPUT_ISBAD
    # 标记上边界行为输出坏像素
    mRData2d[rPixY - hwKernel:, hwKernel:rPixX - hwKernel] |= FLAG_OUTPUT_ISBAD
    # 标记下边界行为输出坏像素

    logger.info(" Creating and writing output images...")
    # 开始创建输出图像

    inv1 = 1.0 / sumKernel
    # 计算核总和的倒数（用于归一化）

    if conv_out is not None:
    # 如果需要输出卷积图像（convOut）
        inner_sy = slice(hwKernel, rPixY - hwKernel)
        # 计算Y方向内区域切片（排除核半宽边界）
        inner_sx = slice(hwKernel, rPixX - hwKernel)
        # 计算X方向内区域切片
        norm_ok = (photNormalize[0:1] != "u") and \
                  ((convTmpl and photNormalize[0:1] == "t") or
                   (not convTmpl and photNormalize[0:1] == "i"))
        # 如果不是非归一化模式：模板卷积+模板归一化 或 图像卷积+图像归一化
                   # 或图像卷积+图像归一化
        if norm_ok:
        # 如果需要归一化
            oRData2d = oRData1d.reshape(rPixY, rPixX)
            # 将差异图像转回2D
            oRData2d[inner_sy, inner_sx] *= inv1
            # 内区域乘以核总和的倒数（归一化到1）

        insert_subregion_flt_numpy(
        # 将区域插入到全图conv_out的对应位置
            oRData, conv_out, fpixelOutX, fpixelOutY,
            # 区域数据、输出数组、输出起始像素
            lpixelOutX, lpixelOutY, xBufLo, yBufLo)
            # 输出结束像素、缓冲宽度

        if norm_ok:
        # 如果做了归一化，需要恢复
            oRData2d[inner_sy, inner_sx] *= sumKernel
            # 乘以核总和恢复到归一化前的值

    meansigSubstampsF = 0.0
    # 最终平均信噪比初始化为0
    scatterSubstampsF = 0.0
    # 最终信噪比弥散度初始化为0
    inner_sy = slice(hwKernel, rPixY - hwKernel)
    # 重新计算内区域Y切片
    inner_sx = slice(hwKernel, rPixX - hwKernel)
    # 重新计算内区域X切片

    if convTmpl:
    # 模板卷积模式：diff = convolve(template) - image
        oRData2d = oRData1d.reshape(rPixY, rPixX)
        # 差异图像转2D
        iRData2d = iRData1d.reshape(rPixY, rPixX)
        # 科学图像转2D
        oRData2d[inner_sy, inner_sx] -= iRData2d[inner_sy, inner_sx]
        # diff = convolvedTemplate - scienceImage
        if (photNormalize[0:1] != "u") and (photNormalize[0:1] == "t"):
        # 如果需要模板归一化
            oRData2d[inner_sy, inner_sx] *= inv1
            # 归一化diff
            noiseData2d = noiseData1d.reshape(rPixY, rPixX)
            # 噪声数据转2D
            noiseData2d[inner_sy, inner_sx] *= inv1
            # 归一化噪声
        oRData2d[inner_sy, inner_sx] *= -1.0
        # 翻转符号使正值为源（模板卷积→图像=模板减图像→翻转后正=比模板亮）

        if figMerit[0:1] == "v":
        # 方差模式：计算diff图像上各Stamp的最终信噪比
            temp2 = np.zeros(nS, dtype=np.float32)
            # 临时数组存储各Stamp的信噪比
            kk = 0
            # 有效信噪比计数器
            for l in range(nS):
            # 遍历所有模板Stamp
                if ctSscntOut is not None and ctNssOut is not None and ctXssOut is not None and ctYssOut is not None:
                # 如果stamp扁平数组可用
                    if ctSscntOut[l] < ctNssOut[l]:
                    # 如果当前stamp有有效的子stamp
                        sig = get_final_stamp_sig_numpy(
                        # 计算最终stamp信噪比（diff²/噪声²的加权和）
                            ctXssOut, ctYssOut, ctSscntOut, l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        # 保存信噪比
                        kk += 1
                        # 计数器加1
                elif ctSa is not None:
                # 如果stamp字典可用
                    if ctSa['sscnt'][l] < ctSa['nss'][l]:
                        sig = get_final_stamp_sig_numpy(
                            ctSa['xss'], ctSa['yss'], ctSa['sscnt'], l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        kk += 1
            mean_f, stdev_f = sigma_clip_numpy(temp2[:kk], 10, statSig)[:2]
            # 对信噪比做sigma-clip去除离群值，返回均值和标准差
            meansigSubstampsF = mean_f
            # 保存最终平均信噪比
            scatterSubstampsF = stdev_f
            # 保存最终信噪比弥散度
            logger.info("   FINAL Mean sig: %6.3f stdev: %6.3f", meansigSubstampsF, scatterSubstampsF)
            # 输出最终品质指标

    else:
    # 图像卷积模式：diff = image - convolve(image)
        oRData2d = oRData1d.reshape(rPixY, rPixX)
        # 差异图像转2D
        tRData2d = tRData1d.reshape(rPixY, rPixX)
        # 模板图像转2D
        oRData2d[inner_sy, inner_sx] -= tRData2d[inner_sy, inner_sx]
        # diff = convolvedImage - template（与模板模式相反）
        if (photNormalize[0:1] != "u") and (photNormalize[0:1] == "i"):
        # 如果需要图像归一化
            oRData2d[inner_sy, inner_sx] *= inv1
            # 归一化diff
            noiseData2d = noiseData1d.reshape(rPixY, rPixX)
            # 噪声转2D
            noiseData2d[inner_sy, inner_sx] *= inv1
            # 归一化噪声

        if figMerit[0:1] == "v":
        # 方差模式：计算图像Stamp的最终信噪比（逻辑同上）
            temp2 = np.zeros(nS, dtype=np.float32)
            # 分配临时信噪比数组
            kk = 0
            # 有效信噪比计数器
            for l in range(nS):
            # 遍历所有图像Stamp
                if ciSscntOut is not None and ciNssOut is not None and ciXssOut is not None and ciYssOut is not None:
                # 如果图像Stamp扁平数组可用
                    if ciSscntOut[l] < ciNssOut[l]:
                    # 如果当前stamp有有效的子stamp
                        sig = get_final_stamp_sig_numpy(
                        # 计算最终信噪比
                            ciXssOut, ciYssOut, ciSscntOut, l, oRData1d, noiseData1d,
                            # 子中心坐标、当前计数器、差异和噪声数据
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                            # 窗口尺寸、区域宽、遮罩
                        temp2[kk] = sig
                        # 保存信噪比值
                        kk += 1
                        # 计数器加1
                elif ciSa is not None:
                # 如果图像Stamp字典可用
                    if ciSa['sscnt'][l] < ciSa['nss'][l]:
                    # 有效子stamp检查
                        sig = get_final_stamp_sig_numpy(
                        # 计算最终信噪比
                            ciSa['xss'], ciSa['yss'], ciSa['sscnt'], l, oRData1d, noiseData1d,
                            # 从字典取子中心坐标
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        # 保存信噪比
                        kk += 1
                        # 计数器加1
            mean_f, stdev_f = sigma_clip_numpy(temp2[:kk], 10, statSig)[:2]
            # Sigma-clip剔除离群值后求均值和标准差
            meansigSubstampsF = mean_f
            # 最终平均信噪比
            scatterSubstampsF = stdev_f
            # 最终信噪比弥散度
            logger.info("    FINAL Mean sig: %6.3f stdev: %6.3f", meansigSubstampsF, scatterSubstampsF)
            # 输出最终品质指标

    oRData_2d = oRData1d.reshape(rPixY, rPixX)
    # 最终差异图像转2D
    noiseData_2d = noiseData1d.reshape(rPixY, rPixX)
    # 最终噪声图像转2D
    mRData_2d = mRData1d.reshape(rPixY, rPixX)
    # 最终遮罩图像转2D

    logger.info(" Getting diffim stats for GOOD pixels : ")
    # 计算差异图像在GOOD像素上的统计量
    res_good = get_stamp_stats3_numpy(
        oRData_2d, 0, 0, rPixX, rPixY,
        0x0, 0xffff,          5, mRData_2d, statSig)
        # 输入差异图像、坐标原点、尺寸、无条件掩码、最大迭代5次、sigma阈值
    mean_val = res_good[1]
    # 均值
    sd_val = res_good[4]
    # 标准差
    logger.info("   Mean   : %.2f", res_good[1])
    logger.info("   Median : %.2f", res_good[2])
    logger.info("   Mode   : %.2f", res_good[3])
    logger.info("   Stdev  : %.2f", res_good[4])
    # 输出差异图像的均值/中位数/众数/标准差

    logger.info(" Getting noiseim stats for GOOD pixels : ")
    nres_good = get_stamp_stats3_numpy(
        noiseData_2d, 0, 0, rPixX, rPixY,
        0x0, 0xffff,          5, mRData_2d, statSig)
    nmean_val = nres_good[1]

    x2norm, nx2norm = get_noise_stats3_numpy(
        oRData1d, noiseData1d, 0x0, 0xffff, rPixX, rPixY, mRData1d)

    logger.info("   Mean   : %.2f", nres_good[1])
    logger.info("   Median : %.2f", nres_good[2])
    logger.info("   Mode   : %.2f", nres_good[3])
    logger.info("   Stdev  : %.2f", nres_good[4])
    logger.info(" Emperical / Expected Noise for GOOD pixels = %.2f", sd_val / nmean_val if nmean_val != 0 else 0.0)
    logger.info(" X2NORM = %.2f", x2norm)

    diffrat = sd_val / nmean_val if nmean_val != 0 else 0.0

    logger.info(" Getting diffim stats for OK pixels : ")
    res_ok = get_stamp_stats3_numpy(
        oRData_2d, 0, 0, rPixX, rPixY,
        0xff, FLAG_OUTPUT_ISBAD,          5, mRData_2d, statSig)
    meanm_val = res_ok[1]
    sdm_val = res_ok[4]
    logger.info("   Mean   : %.2f", res_ok[1])
    logger.info("   Median : %.2f", res_ok[2])
    logger.info("   Mode   : %.2f", res_ok[3])
    logger.info("   Stdev  : %.2f", res_ok[4])

    logger.info(" Getting noiseim stats for OK pixels : ")
    nres_ok = get_stamp_stats3_numpy(
        noiseData_2d, 0, 0, rPixX, rPixY,
        0xff, FLAG_OUTPUT_ISBAD,          5, mRData_2d, statSig)
    nmeanm_val = nres_ok[1]
    logger.info("   Mean   : %.2f", nres_ok[1])
    logger.info("   Median : %.2f", nres_ok[2])
    logger.info("   Mode   : %.2f", nres_ok[3])
    logger.info("   Stdev  : %.2f", nres_ok[4])

    logger.info(" Emperical / Expected Noise for OK pixels = %.2f", sdm_val / nmeanm_val if nmeanm_val != 0 else 0.0)

    # tm5 = time.time(); logger.debug("[out] get_stamp_stats3 done, %.3fs", tm5 - tm4)

    if rescaleOK:
        if diffrat != 0:
            diffrat = (sdm_val / nmeanm_val if nmeanm_val != 0 else 0.0) / diffrat
        if diffrat > 1:
            logger.info(" Scale OK pixel noise by = %.2f", diffrat)
            ok_mask = (mRData1d & 0xff).astype(np.bool_) & ~((mRData1d & FLAG_OUTPUT_ISBAD).astype(np.bool_))
            noiseData1d[ok_mask] *= diffrat
        else:
            logger.info(" Leave OK pixel noise as-is")

    if kfSpreadMask2 >= 0:
        oRData2d = oRData1d.reshape(rPixY, rPixX)
        noiseData2d = noiseData1d.reshape(rPixY, rPixX)
        bad_flag = (mRData1d.reshape(rPixY, rPixX)[inner_sy, inner_sx] & FLAG_OUTPUT_ISBAD).astype(np.bool_)
        oRData2d[inner_sy, inner_sx][bad_flag] = fillVal
        noiseData2d[inner_sy, inner_sx][bad_flag] = fillValNoise

    oRData_2d = oRData1d.reshape(rPixY, rPixX)
    noiseData_2d = noiseData1d.reshape(rPixY, rPixX)
    mRData_2d = mRData1d.reshape(rPixY, rPixX)

    # tm6 = time.time(); logger.debug("[out] insert_subregion start")

    insert_subregion_flt_numpy(
        oRData_2d, diff_out, fpixelOutX, fpixelOutY,
        lpixelOutX, lpixelOutY, xBufLo, yBufLo)

    if noise_out is not None:
        insert_subregion_flt_numpy(
            noiseData_2d, noise_out, fpixelOutX, fpixelOutY,
            lpixelOutX, lpixelOutY, xBufLo, yBufLo)

    if mask_out is not None:
        insert_subregion_int_numpy(
            mRData_2d, mask_out, fpixelOutX, fpixelOutY,
            lpixelOutX, lpixelOutY, xBufLo, yBufLo)

    # tm7 = time.time(); logger.debug("[out] insert_subregion done, %.3fs", tm7 - tm6)

    kerSol = convolve_result['kerSol']

    stats = {'convTmpl': convTmpl, 'sumKernel': sumKernel,
        'meansigSubstamps': meansigSubstamps, 'scatterSubstamps': scatterSubstamps,
        'meansigSubstampsF': meansigSubstampsF, 'scatterSubstampsF': scatterSubstampsF,
        'x2norm': x2norm, 'nx2norm': nx2norm,
        'mean': mean_val, 'sd': sd_val, 'nmean': nmean_val,
        'meanm': meanm_val, 'sdm': sdm_val, 'nmeanm': nmeanm_val,
        'diffrat': diffrat, 'kerSol': kerSol}

    if stats_list is not None and region_idx < len(stats_list):
        stats_list[region_idx] = stats

    logger.info("Region %i finished", region_idx)

    logger.debug("[region %d] region_output_numpy done", region_idx)
    return stats

def hotpants(inim: np.ndarray, tmplim: np.ndarray,
# hotpants主入口函数：接收科学图像和模板图像，执行图像差分。返回(diff, noise, conv, mask, stats)
    tni: Optional[np.ndarray] = None, ini: Optional[np.ndarray] = None,
    # 模板噪声图像(方差)、科学图像噪声图像(方差)
    tmi: Optional[np.ndarray] = None, imi: Optional[np.ndarray] = None,
    # 模板遮罩(整数)、科学图像遮罩(整数)
    tu: float = 25000., tuk: Optional[float] = None, tl: float = 0.,
    # 模板上限阈值、模板核上限阈值、模板下限阈值
    tg: float = 1., tr: float = 0., tp: float = 0.,
    # 模板增益(e-/ADU)、模板读出噪声(e-)、模板Pedestal
    iu: float = 25000., iuk: Optional[float] = None, il: float = 0.,
    # 科学图像上限阈值、核上限阈值、下限阈值
    ig: float = 1., ir: float = 0., ip: float = 0.,
    # 科学图像增益、读出噪声、Pedestal
    r: int = 10, ko: int = 2, bgo: int = 1,
    # 核半宽、核多项式阶数、背景多项式阶数
    ng: int = 3, ng_deg: Optional[List[int]] = None, ng_sig: Optional[List[float]] = None,
    # 高斯分量数、各分量多项式阶数列表(默认[6,4,2])、各分量sigma列表(默认[0.7,1.5,3.0])
    pca: Optional[np.ndarray] = None,
    # PCA基矩阵(None时使用解析高斯基)
    nrx: int = 1, nry: int = 1, rf: Optional[int] = None,
    # X方向区域数、Y方向区域数、核扇出半径
    nsx: int = 10, nsy: int = 10, ssf: Optional[List] = None,
    # X方向Stamp数、Y方向Stamp数、手动Stamp中心列表
    afssc: int = 1, nss: int = 3, rss: int = 15,
    # 核拟合缩放模式、每个Stamp最大子Stamp数、子Stamp窗口半宽
    ft: float = 20.0, sft: float = 0.5, nft: float = 0.1,
    # 核拟合阈值、缩放因子、最少有效Stamp比例
    ssig: float = 3.0, ks: float = 2.0, kfm: float = 0.99,
    # Sigma-clip阈值、核拒绝阈值、核分数遮罩阈值
    mins: float = 1.0, mous: float = 1.0,
    # 最小/最大输出缩放
    fi: float = 1e-30, fin: float = 0.,
    # 无效输出填充值、无效噪声填充值
    c: str = 'b', n: str = 't', fom: str = 'v',
    # 卷积方向(b双向/t模板/i图像)、归一化模式(t模板/i图像/u不归一化)、品质指标(v方差/s偏度/h峰度)
    sconv: int = 0, okn: int = 0, convvar: int = 0,
    # 相同卷积标志、仅OK像素噪声标志、方差卷积模式标志
    v: int = 1, kcs: int = 0,
    # 详细输出级别、核缓存步长(0时自动取fwKernel)
    uss: int = 0, savexy: int = 0,
    # 全子Stamp模式标志、保存子Stamp坐标标志
    dump_dir: Optional[str] = None,
    # dump输出目录
    logger: Optional[logging.Logger] = None,
    # 日志记录器
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    # 返回(diffOut, noiseOut, convOut, maskOut, stats_list)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG, stream=sys.stderr)
    # 配置日志格式：时间戳+消息
    if logger is None:
    # 如果未提供日志记录器
        logger = logging.getLogger('hotpants')
        # 获取默认日志记录器
    if ng_deg is None:
    # 如果未指定高斯多项式阶数
        ng_deg = [6, 4, 2]
        # 默认三个高斯分量：阶数6/4/2
    if ng_sig is None:
    # 如果未指定高斯sigma值
        ng_sig = [0.7, 1.5, 3.0]
        # 默认三个高斯分量：sigma=0.7/1.5/3.0像素

    tmpl_arr = np.ascontiguousarray(tmplim, dtype=np.float32)
    # 将模板转为连续内存的float32数组
    sci_arr = np.ascontiguousarray(inim, dtype=np.float32)
    # 将科学图像转为连续内存的float32数组
    tNx = tmpl_arr.shape[1]
    # 模板图像X像素数
    tNy = tmpl_arr.shape[0]
    # 模板图像Y像素数
    iNx = sci_arr.shape[1]
    # 科学图像X像素数
    iNy = sci_arr.shape[0]
    # 科学图像Y像素数

    tni_arr = None
    # 模板噪声数组初始化为None
    if tni is not None:
        tni_arr = np.ascontiguousarray(tni, dtype=np.float32)
        # 如果提供了模板噪声，转为连续float32数组

    ini_arr = None
    # 科学图像噪声数组初始化为None
    if ini is not None:
        ini_arr = np.ascontiguousarray(ini, dtype=np.float32)
        # 如果提供了科学图像噪声，转为连续float32

    tmi_arr = None
    # 模板遮罩数组初始化为None
    if tmi is not None:
        tmi_arr = np.ascontiguousarray(tmi, dtype=np.int32)
        # 如果提供了模板遮罩，转为连续int32

    imi_arr = None
    # 科学图像遮罩数组初始化为None
    if imi is not None:
        imi_arr = np.ascontiguousarray(imi, dtype=np.int32)
        # 如果提供了科学图像遮罩，转为连续int32

    tuk_val = float(tu) if tuk is None else float(tuk)
    # 模板核上限阈值：未指定时默认用普通上限阈值
    iuk_val = float(iu) if iuk is None else float(iuk)
    # 科学图像核上限阈值：未指定时默认用普通上限阈值

    sig_arr = np.array([1.0 / (2.0 * s * s) for s in ng_sig], dtype=np.float32)
    # 将sigma值转换为1/(2σ²)形式，保存为float32数组
    deg_arr = np.array(ng_deg, dtype=np.int32)
    # 多项式阶数列表转为int32数组

    use_pca = 0
    # PCA模式标志初始为0
    pca_arrs = []
    # PCA数组列表初始为空
    if pca is not None:
    # 如果提供了PCA基矩阵
        use_pca = 1
        # 启用PCA模式
        pca_ng = len(pca)
        # PCA核的数量
        pca_arrs = [np.ascontiguousarray(pca[pi], dtype=np.float32) for pi in range(pca_ng)]
        # 将PCA各层转为连续float32数组
        r = pca[0].shape[0] // 2
        # 从PCA第一层尺寸反推核半宽
        deg_arr = np.zeros(pca_ng, dtype=np.int32)
        # PCA模式下多项式阶数无效，全部置0
        sig_arr = np.full(pca_ng, -1.0, dtype=np.float32)
        # sigma无效，用-1标记
        ng = pca_ng
        # 高斯分量数改为PCA核数

    xMin = 0; yMin = 0
    # 全局最小X/Y坐标
    xMax = min(tNx, iNx) - 1
    # 全局最大X（模板和图像中较小尺寸减1，0索引）
    yMax = min(tNy, iNy) - 1
    # 全局最大Y

    if rf is not None:
    # 如果手动指定了各区域的范围列表
        nR = len(rf)
        # 区域数等于范围列表长度
        rxmins = np.array([reg[0] for reg in rf], dtype=np.int32)
        # 提取各区域X起始
        rxmaxs = np.array([reg[1] for reg in rf], dtype=np.int32)
        # 提取各区域X结束
        rymins = np.array([reg[2] for reg in rf], dtype=np.int32)
        # 提取各区域Y起始
        rymaxs = np.array([reg[3] for reg in rf], dtype=np.int32)
        # 提取各区域Y结束
    else:
    # 自动等分图像为nrx×nry个区域
        nR = nrx * nry
        # 区域总数
        rxmins_l, rxmaxs_l, rymins_l, rymaxs_l = [], [], [], []
        # 临时列表存储各区域坐标
        for j in range(nry):
        # 遍历Y方向区域索引
            for i in range(nrx):
            # 遍历X方向区域索引
                rxmins_l.append(xMin + i * xMax // nrx)
                # 等分X起始
                rymins_l.append(yMin + j * yMax // nry)
                # 等分Y起始
                rxmaxs_l.append(min((i + 1) * xMax // nrx, xMax))
                # 等分X结束（不超过全局最大X）
                rymaxs_l.append(min((j + 1) * yMax // nry, yMax))
                # 等分Y结束
        rxmins = np.array(rxmins_l, dtype=np.int32); rxmaxs = np.array(rxmaxs_l, dtype=np.int32)
        # 转为numpy数组
        rymins = np.array(rymins_l, dtype=np.int32); rymaxs = np.array(rymaxs_l, dtype=np.int32)
        # 转为numpy数组

    xcmp_arr = None; ycmp_arr = None; ncmp = 0
    # 手动Stamp中心坐标和数量默认为None/0
    if ssf is not None:
    # 如果提供了手动Stamp中心列表
        xcmp_arr = np.array([p[0] - 1 for p in ssf], dtype=np.float32)
        # X坐标减1转为0索引
        ycmp_arr = np.array([p[1] - 1 for p in ssf], dtype=np.float32)
        # Y坐标减1
        ncmp = len(ssf)
        # 手动中心数量

    oNx = max(tNx, iNx)
    # 输出图像X尺寸（取模板和图像中较大者）
    oNy = max(tNy, iNy)
    # 输出图像Y尺寸
    diff_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    # 分配差异输出图像，初始填充无效值
    noise_out = np.full((oNy, oNx), float(fin), dtype=np.float32)
    # 分配噪声输出图像
    conv_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    # 分配卷积输出图像
    mask_out = np.zeros((oNy, oNx), dtype=np.int32)
    # 分配遮罩输出图像，初始全0

    if dump_dir is not None:
    # 如果指定了dump目录，将所有输入参数和图像写入二进制文件
        os.makedirs(dump_dir, exist_ok=True)
        # 创建dump目录
        path = os.path.join(dump_dir, "py_input.bin")
        # dump文件路径
        with open(path, "wb") as f:
        # 以二进制写模式打开
            def de(nm, db):
            # dump_entry辅助函数：写入名称+数据长度+原始字节
                nb = nm.encode('ascii')
                f.write(struct.pack('i', len(nb))); f.write(nb)
                # 写入名称长度和名称
                dl = len(db) if db else 0
                f.write(struct.pack('l', dl))
                # 写入数据长度
                if dl > 0: f.write(db)
                # 写入数据
            def di(nm, val): de(nm, struct.pack('i', int(val)))
            # dump int类型
            def dl(nm, val): de(nm, struct.pack('l', int(val)))
            # dump long类型
            def df(nm, val): de(nm, struct.pack('f', float(val)))
            # dump float类型
            def ds(nm, s):
            # dump字符串类型（空终止）
                if s is None: de(nm, None)
                else: de(nm, s.encode('ascii') + b'\x00')
            dl("tNx",tNx); dl("tNy",tNy); dl("iNx",iNx); dl("iNy",iNy)
            # dump图像尺寸
            dl("oNx",oNx); dl("oNy",oNy)
            # dump输出尺寸
            di("nR",nR); di("hwKernel",r); di("ngauss",ng)
            # dump区域数和核参数
            di("kerOrder",ko); di("bgOrder",bgo)
            # dump多项式阶数
            di("nStampX",nsx); di("nStampY",nsy)
            # dump Stamp网格尺寸
            di("nKSStamps",nss); di("hwKSStamp",rss)
            # dump子Stamp参数
            di("useFullSS",uss); di("findSSC",afssc)
            # dump全子Stamp和自动搜索标志
            df("kerFitThresh",ft); df("scaleFitThresh",sft)
            # dump拟合阈值
            df("minFracGoodStamps",nft)
            # dump最小有效Stamp比例
            df("statSig",ssig); df("kerSigReject",ks); df("kerFracMask",kfm)
            # dump Sigma-clip和核拒绝参数
            df("tUThresh",tu); df("tLThresh",tl)
            # dump模板阈值
            df("tGain",tg); df("tRdnoise",tr); df("tPedestal",tp)
            # dump模板增益/噪声/Pedestal
            df("iUThresh",iu); df("iLThresh",il)
            # dump科学图像阈值
            df("iGain",ig); df("iRdnoise",ir); df("iPedestal",ip)
            # dump科学图像增益/噪声/Pedestal
            df("tUKThresh",tuk_val); df("iUKThresh",iuk_val)
            # dump核上限阈值
            df("kfSpreadMask1",mins); df("kfSpreadMask2",mous)
            # dump扩散遮罩参数
            df("fillVal",fi); df("fillValNoise",fin)
            # dump填充值
            di("sameConv",sconv); di("rescaleOK",okn)
            # dump相同卷积和重缩放标志
            di("convolveVariance",convvar)
            # dump方差卷积模式
            di("usePCA",use_pca); di("Ncmp",ncmp)
            # dump PCA和手动中心标志
            di("verbose",v); di("kcStep",kcs); di("savexyflag",savexy)
            # dump详细级别、核步长、坐标保存标志
            ds("forceConvolve",c); ds("photNormalize",n); ds("figMerit",fom)
            # dump字符串参数
            de("deg_fixe", bytes(np.ascontiguousarray(deg_arr)))
            # dump多项式阶数数组
            de("sigma_gauss", bytes(np.ascontiguousarray(sig_arr)))
            # dump sigma数组
            de("tFullData", bytes(np.ascontiguousarray(tmpl_arr)))
            # dump完整模板图像数据
            de("iFullData", bytes(np.ascontiguousarray(sci_arr)))
            # dump完整科学图像数据
            de("tNoiseFullData", bytes(np.ascontiguousarray(tni_arr)) if tni_arr is not None else None)
            # dump模板噪声数据（可选）
            de("iNoiseFullData", bytes(np.ascontiguousarray(ini_arr)) if ini_arr is not None else None)
            # dump科学图像噪声数据（可选）
            de("tMaskFullData", bytes(np.ascontiguousarray(tmi_arr)) if tmi_arr is not None else None)
            # dump模板遮罩数据（可选）
            de("iMaskFullData", bytes(np.ascontiguousarray(imi_arr)) if imi_arr is not None else None)
            # dump科学图像遮罩数据（可选）
            de("rXMins", bytes(np.ascontiguousarray(rxmins)))
            # dump各区域X起始坐标
            de("rXMaxs", bytes(np.ascontiguousarray(rxmaxs)))
            # dump各区域X结束坐标
            de("rYMins", bytes(np.ascontiguousarray(rymins)))
            # dump各区域Y起始坐标
            de("rYMaxs", bytes(np.ascontiguousarray(rymaxs)))
            # dump各区域Y结束坐标
            de("xcmp", bytes(np.ascontiguousarray(xcmp_arr)) if ssf is not None else None)
            # dump手动中心X坐标（可选）
            de("ycmp", bytes(np.ascontiguousarray(ycmp_arr)) if ssf is not None else None)
            # dump手动中心Y坐标（可选）
            de("diffOut", bytes(np.ascontiguousarray(diff_out)))
            # dump输出差异图像（初始空）
            de("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            # dump输出噪声图像
            de("convOut", bytes(np.ascontiguousarray(conv_out)))
            # dump输出卷积图像
            de("maskOut", bytes(np.ascontiguousarray(mask_out)))
            # dump输出遮罩图像

    ctx_info = compute_ctx_info(
    # 计算上下文信息：核尺寸、Stamp布局、kcStep等
        int(r), int(ng), deg_arr,
        # 核半宽、高斯分量数、阶数数组
        int(ko), int(bgo),
        # 核多项式阶数、背景多项式阶数
        int(nsx), int(nsy), int(rss),
        # X/Y方向Stamp数、子Stamp半宽
        int(uss), float(ft), int(kcs),
        # 全子Stamp标志、拟合阈值、核缓存步长
        tNx, tNy, iNx, iNy, nR)
        # 模板和科学图像尺寸、区域总数

    params_info = {'hwKernel': int(r), 'ngauss': int(ng),'deg_fixe': deg_arr, 'sigma_gauss': sig_arr,'kerOrder': int(ko), 'bgOrder': int(bgo),
    # 构建参数信息字典：核和背景参数
        'hwKSStamp': int(rss), 'nKSStamps': int(nss),'scaleFitThresh': float(sft), 'minFracGoodStamps': float(nft),
        # Stamp和阈值参数
        'tUKThresh': float(tuk_val), 'iUKThresh': float(iuk_val),'tUThresh': float(tu), 'tLThresh': float(tl),
        # 模板/图像阈值
        'tGain': float(tg), 'tRdnoise': float(tr),'iUThresh': float(iu), 'iLThresh': float(il),'iGain': float(ig), 'iRdnoise': float(ir),
        # 增益和噪声参数
        'verbose': int(v), 'usePCA': int(use_pca),
        # 详细级别和PCA
        'PCA': pca_arrs if use_pca else None,
        # PCA数据或None
        'xcmp': np.asarray(xcmp_arr) if ssf is not None else None,
        # 手动中心X坐标
        'ycmp': np.asarray(ycmp_arr) if ssf is not None else None,
        # 手动中心Y坐标
        'Ncmp': ncmp,'statSig': float(ssig), 'kerSigReject': float(ks), 'kerFracMask': float(kfm),
        # 手动中心数、统计和核拒绝参数
        'fillVal': float(fi), 'fillValNoise': float(fin),'figMerit': fom, 'photNormalize': n,
        # 填充值、品质指标、光度归一化模式
        'convolveVariance': int(convvar), 'sameConv': int(sconv),'savexyflag': int(savexy), 'rescaleOK': int(okn),
        # 卷积模式标志
        'kfSpreadMask2': float(mous), 'findSSC': int(afssc),'tNoiseFullData': tni_arr,'iNoiseFullData': ini_arr}
        # 扩散遮罩、自动搜索、噪声全图

    localFC_py = c
    stats_list_py = [None] * nR

    localFC_py = c
    # 卷积方向标志（可在循环中被conv_result修改）
    stats_list_py = [None] * nR
    # 创建统计列表，每个区域一个slot

    for ri in range(nR):
        t0 = tm.time()
        # 遍历每个区域，计时开始
        # 区域计时开始
        logger.debug("  [%d] setup start", ri)
        py = region_setup_numpy(
        # 区域初始化：提取子图像、创建遮罩、计算噪声
            tmpl_arr, sci_arr,
            # 模板和科学图像全图
            tni_arr, ini_arr, tmi_arr, imi_arr,
            # 噪声方差全图和遮罩全图
            ri, rxmins, rxmaxs, rymins, rymaxs, nR,
            # 区域索引、各区域坐标数组、区域总数
            r, ctx_info['fwStamp'], ctx_info['sBorder'],
            # 核半宽、Stamp全宽、边界宽度
            ctx_info['xMin'], ctx_info['yMin'], ctx_info['xMax'], ctx_info['yMax'],
            # 全局图像坐标范围
            fi, fin,
            # 无效像素填充值和噪声填充值
            tp, ip,
            # 模板和科学图像Pedestal
            tg, tr, ig, ir,
            # 模板和科学图像的增益和读出噪声
            tu, tl, iu, il,
            # 模板和科学图像的上下限阈值
            mins, logger=logger)
            # 扩散遮罩因子、日志记录器
        logger.info("[%d] setup: %.3fs", ri, tm.time()-t0)
        # 输出耗时

        t1 = tm.time()
        bs_result = region_buildstamps_numpy(py, ctx_info, params_info, localFC_py, logger=logger)
        # 区域Stamp构建：在区域内构建模板和图像的Stamp数组
        logger.info("[%d] buildstamps: %.3fs", ri, tm.time()-t1)
        # 输出耗时
        if bs_result['status'] != 0:
        # 如果Stamp构建失败（状态非0）
            continue
            # 跳过当前区域

        t1 = tm.time()
        fit_result = region_fit_numpy(bs_result, py, ctx_info, params_info, localFC_py, logger=logger)
        # 区域核拟合：填充Stamp并执行迭代核拟合
        logger.info("[%d] fit: %.3fs", ri, tm.time()-t1)
        # 输出耗时

        t1 = tm.time()
        conv_result = region_convolve_diff_numpy(
        # 区域卷积分差：执行空间卷积、计算差异图像
            fit_result, py, bs_result, ctx_info, params_info, ri, localFC_py, logger=logger)
        logger.info("[%d] convolve_diff: %.3fs", ri, tm.time()-t1)
        # 输出耗时
        localFC_py = conv_result.get('localForceConvolve', localFC_py)
        # 更新卷积方向标志

        t1 = tm.time()
        stats_entry = region_output_numpy(
        # 区域输出：将差异/噪声/卷积/遮罩插入全图，计算统计量
            conv_result, py,
            diff_out, noise_out, conv_out, mask_out,
            # 全图输出数组（原地写入）
            ctx_info, params_info, ri, None, logger=logger)
        logger.info("[%d] output: %.3fs", ri, tm.time()-t1)
        # 输出耗时
        stats_list_py[ri] = stats_entry
        # 保存当前区域统计

    if dump_dir is not None:
    # 如果指定了dump目录，将最终输出写入文件
        opath = os.path.join(dump_dir, "py_output.bin")
        # 输出dump文件路径
        with open(opath, "wb") as fo:
        # 以二进制写模式打开
            def de2(nm, db):
            # dump输出辅助函数
                nb = nm.encode('ascii')
                fo.write(struct.pack('i', len(nb))); fo.write(nb)
                # 写入名称长度和名称
                dl2 = len(db) if db else 0
                fo.write(struct.pack('l', dl2))
                # 写入数据长度
                if dl2 > 0: fo.write(db)
                # 写入数据
            de2("diffOut", bytes(np.ascontiguousarray(diff_out)))
            # dump差异输出图像
            de2("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            # dump噪声输出图像
            de2("convOut", bytes(np.ascontiguousarray(conv_out)))
            # dump卷积输出图像
            de2("maskOut", bytes(np.ascontiguousarray(mask_out)))
            # dump遮罩输出图像

    stats_list = []
    # 构建最终统计列表
    for si in range(nR):
    # 遍历每个区域
        if stats_list_py[si] is not None:
        # 如果该区域有有效统计
            entry = stats_list_py[si]
            stats_list.append({'conv_tmpl': entry['convTmpl'],
            # 卷积模式标志
                'sum_kernel': entry['sumKernel'],
                # 核总和
                'mean_sig': entry['meansigSubstamps'], 'scatter_sig': entry['scatterSubstamps'],
                # 拟合阶段信噪比统计
                'final_mean_sig': entry['meansigSubstampsF'], 'final_scatter_sig': entry['scatterSubstampsF'],
                # 最终差异图像上的信噪比统计
                'x2norm': entry['x2norm'], 'nx2norm': entry['nx2norm'],
                # 卡方归一化因子
                'diff_mean': entry['mean'], 'diff_sd': entry['sd'], 'noise_mean': entry['nmean'],
                # GOOD像素上的差异统计
                'diff_mean_ok': entry['meanm'], 'diff_sd_ok': entry['sdm'], 'noise_mean_ok': entry['nmeanm'],
                # OK像素上的差异统计
                'diffrat': entry['diffrat']})
                # OK/GOOD噪声比率
        else:
        # 区域失败，填充默认值
            stats_list.append({'conv_tmpl': 0, 'sum_kernel': 0.0,
                'mean_sig': 0.0, 'scatter_sig': 0.0,
                'final_mean_sig': 0.0, 'final_scatter_sig': 0.0,
                'x2norm': 0.0, 'nx2norm': 0,
                'diff_mean': 0.0, 'diff_sd': 0.0, 'noise_mean': 0.0,
                'diff_mean_ok': 0.0, 'diff_sd_ok': 0.0, 'noise_mean_ok': 0.0,
                'diffrat': 0.0})

    return diff_out, noise_out, conv_out, mask_out, stats_list
    # 返回差异、噪声、卷积、遮罩四种输出图像和统计列表


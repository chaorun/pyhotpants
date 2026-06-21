import numpy as np
import numba
import math
import logging
# logger = logging.getLogger('hotpants')  # disabled: use parameter

from .functions import (
    ZEROVAL, MAXVAL,
    FLAG_BAD_PIXVAL, FLAG_SAT_PIXEL, FLAG_NEG_PIXEL, FLAG_SPREAD,
    FLAG_MASKED, FLAG_BORDER, FLAG_SUBREGION, FLAG_INVALID,
    BUILD_STAMP_FLAT_CACHE, LAST_SIGMA_CLIP, LAST_N, LAST_MEDIAN,
    Ran1,
    sigma_clip_numpy, get_noise_stats3_numpy,
    insert_subregion_flt_numpy, insert_subregion_int_numpy,
    cut_stamp_numpy, get_stamp_stats3_numpy, bin_quartile_numpy,
    get_stamp_stats3_fast_numpy, cut_sstamp_numpy,
    check_psf_center_numba, check_psf_center_numpy,
    quick_sort_impl, quick_sort_recurse,
    psfCentersJit, buildStampsNumba,
    lubksb_numpy, ludcmp_numpy,
    make_noise_image4_numpy,
)

from .alard import (
    kernel_vector_pca_numpy, kernel_vector_numpy, get_kernel_vec_numpy,
    xy_conv_stamp_numpy, xy_conv_stamp_fast_numba_kernel,
    xy_conv_stamp_fast_numba, xy_conv_stamp_pca_numpy,
    build_matrix0_jit, build_matrix_jit, build_matrix0_numpy, build_matrix_numpy,
    build_scprod0_jit, build_scprod_jit, build_scprod0_numpy, build_scprod_numpy,
    fill_stamp_numba_kernel, fill_stamp_numba, fill_stamp_numpy,
    make_kernel_numpy, make_model_jit, make_model_numpy,
    get_background_numpy, background_loop_jit,
    get_final_stamp_sig_numpy, get_stamp_sig_batch_jit,
    get_stamp_sig_jit, get_stamp_sig_numpy,
    spatial_convolve_numpy_fast, spatial_convolve_numpy,
    spatial_convolve_jit_kernel, buildAllKernels, variance_convolve_jit,
    mask_check_loop_jit, spatial_convolve_fast_numpy,
    check_stamps_numpy, check_again_numpy, fit_kernel_numpy,
)
def compute_ctx_info(hwKernel, ngauss, deg_fixe, sigma_gauss,
                     kerOrder, bgOrder,
                     nStampX_in, nStampY_in, nKSStamps, hwKSStamp,
                     useFullSS, kerFitThresh, kcStep_in,
                     tNx, tNy, iNx, iNy, nR):
    import numpy as np
    nCompKer = 0
    for i in range(ngauss):
        nCompKer += ((deg_fixe[i] + 1) * (deg_fixe[i] + 2)) // 2

    nComp      = ((kerOrder + 1) * (kerOrder + 2)) // 2
    nC         = nCompKer + 2
    nCompBG    = (nCompKer - 1) * nComp + 1
    nBGVectors = ((bgOrder + 1) * (bgOrder + 2)) // 2
    nCompTotal = nCompKer * nComp + nBGVectors

    fwKernel   = hwKernel * 2 + 1
    nStampX    = nStampX_in
    nStampY    = nStampY_in

    if useFullSS:
        fwKSStamp = fwKernel
        fwStamp   = fwKernel
        nStampX   = int(min(tNx, iNx) / nR / fwStamp)
        nStampY   = int(min(tNy, iNy) / nR / fwStamp)
    else:
        fwKSStamp = hwKSStamp * 2 + 1
        fwStamp = min(min(tNx, iNx) // int(np.sqrt(nR)) // nStampX,
                        min(tNy, iNy) // int(np.sqrt(nR)) // nStampY)
        fwStamp -= fwKernel
        if fwStamp % 2 == 0:
            fwStamp -= 1

        if fwStamp < fwKSStamp:
            fwStamp  = fwKSStamp + fwKernel
            if fwStamp % 2 == 0:
                fwStamp -= 1
            nStampX = min(tNx, iNx) // int(np.sqrt(nR)) // fwStamp
            nStampY = min(tNy, iNy) // int(np.sqrt(nR)) // fwStamp

    kcStep  = kcStep_in if kcStep_in else fwKernel
    nStamps = nStampX * nStampY
    sBorder = hwKSStamp + hwKernel

    xMin = 0
    yMin = 0
    xMax = min(tNx, iNx) - 1
    yMax = min(tNy, iNy) - 1
    fitThresh = kerFitThresh

    return {
        'nCompKer': nCompKer, 'nComp': nComp, 'nC': nC,
        'nCompBG': nCompBG, 'nBGVectors': nBGVectors, 'nCompTotal': nCompTotal,
        'fwKernel': fwKernel, 'fwStamp': fwStamp, 'fwKSStamp': fwKSStamp,
        'sBorder': sBorder, 'nStamps': nStamps,
        'nStampX': nStampX, 'nStampY': nStampY,
        'xMin': xMin, 'yMin': yMin, 'xMax': xMax, 'yMax': yMax,
        'fitThresh': fitThresh, 'kcStep': kcStep,
    }

def region_setup_numpy(tmpl_2d, sci_2d, tnoise_2d, inoise_2d, tmask_2d, imask_2d,
                        ri, rxmins_np, rxmaxs_np, rymins_np, rymaxs_np, nR,
                        hwKernel, fwStamp, sBorder,
                        xMin, yMin, xMax, yMax,
                        fillVal, fillValNoise,
                        tPedestal, iPedestal,
                        tGain, tRdnoise, iGain, iRdnoise,
                        tUThresh, tLThresh, iUThresh, iLThresh,
                        kfSpreadMask1, logger=None):
    if logger is None:
        logger = logging.getLogger('hotpants')
    logger.debug("  region_setup: ri=%d rXMin=%d rXMax=%d rYMin=%d rYMax=%d",
                 ri, rxmins_np[ri], rxmaxs_np[ri], rymins_np[ri], rymaxs_np[ri])
    rXMin = int(rxmins_np[ri])
    rXMax = int(rxmaxs_np[ri])
    rYMin = int(rymins_np[ri])
    rYMax = int(rymaxs_np[ri])

    fwStamp = int(fwStamp)
    hwKernel = int(hwKernel)
    sBorder = int(sBorder)
    xMin = int(xMin); yMin = int(yMin); xMax = int(xMax); yMax = int(yMax)

    if nR > 1:
        rXBMin = max(xMin, rXMin - fwStamp // 2)
        rYBMin = max(yMin, rYMin - fwStamp // 2)
        rXBMax = min(xMax, rXMax + fwStamp // 2)
        rYBMax = min(yMax, rYMax + fwStamp // 2)
    else:
        rXBMin = max(xMin, rXMin - hwKernel)
        rYBMin = max(yMin, rYMin - hwKernel)
        rXBMax = min(xMax, rXMax + hwKernel)
        rYBMax = min(yMax, rYMax + hwKernel)

    rPixX = rXBMax - rXBMin + 1
    rPixY = rYBMax - rYBMin + 1

    fillVal_f = np.float32(fillVal)
    fillValNoise_f = np.float32(fillValNoise)

    tRData_py = np.full((rPixY, rPixX), fillVal_f, dtype=np.float32)
    iRData_py = np.full((rPixY, rPixX), fillVal_f, dtype=np.float32)
    oRData_py = np.full((rPixY, rPixX), fillValNoise_f, dtype=np.float32)
    eRData_py = np.full((rPixY, rPixX), fillValNoise_f, dtype=np.float32)
    mRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)
    misRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)
    mtsRData_py = np.zeros((rPixY, rPixX), dtype=np.int32)

    tRData_py[:, :] = tmpl_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
    iRData_py[:, :] = sci_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]

    tPedestal_f = np.float32(tPedestal)
    iPedestal_f = np.float32(iPedestal)
    if tPedestal_f != 0. or iPedestal_f != 0.:
        tRData_py -= tPedestal_f
        iRData_py -= iPedestal_f

    if inoise_2d is not None:
        oRData_py[:, :] = inoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        oRData_py *= oRData_py
    else:
        invGain_f = np.float32(1.0 / float(iGain))
        quad_f = np.float32(float(iRdnoise) / float(iGain))
        qquad = np.float64(quad_f) * np.float64(quad_f)
        oRData_py = (np.abs(iRData_py.astype(np.float64)) * np.float64(invGain_f) + qquad).astype(np.float32)

    if tnoise_2d is not None:
        eRData_py[:, :] = tnoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        eRData_py *= eRData_py
    else:
        invGain_t = np.float32(1.0 / float(tGain))
        quad_t = np.float32(float(tRdnoise) / float(tGain))
        qquad_t = np.float64(quad_t) * np.float64(quad_t)
        eRData_py = (np.abs(tRData_py.astype(np.float64)) * np.float64(invGain_t) + qquad_t).astype(np.float32)

    oRData_py += eRData_py

    if imask_2d is not None:
        misRData_py[:, :] = imask_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        misRData_py |= np.int32(0x20) * (misRData_py > 0).astype(np.int32)
        mRData_py |= misRData_py

    if tmask_2d is not None:
        mtsRData_py[:, :] = tmask_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        mtsRData_py |= np.int32(0x20) * (mtsRData_py > 0).astype(np.int32)
        mRData_py |= mtsRData_py

    tUThresh_f = np.float32(tUThresh)
    tLThresh_f = np.float32(tLThresh)
    iUThresh_f = np.float32(iUThresh)
    iLThresh_f = np.float32(iLThresh)
    mRData_py |= np.int32(0x80 | 0x01) * ((tRData_py == fillVal_f) | (iRData_py == fillVal_f)).astype(np.int32)
    mRData_py |= np.int32(0x80 | 0x02) * ((tRData_py >= tUThresh_f) | (iRData_py >= iUThresh_f)).astype(np.int32)
    mRData_py |= np.int32(0x80 | 0x04) * ((tRData_py <= tLThresh_f) | (iRData_py <= iLThresh_f)).astype(np.int32)

    width = int(hwKernel * float(kfSpreadMask1))
    if width > 0:
        w2 = width // 2
        bad = (mRData_py & 0x80) != 0
        spread = np.zeros((rPixY, rPixX), dtype=np.bool_)
        for dy in range(-w2, w2 + 1):
            for dx in range(-w2, w2 + 1):
                shifted = np.zeros((rPixY, rPixX), dtype=np.bool_)
                sy1, sy2 = max(0, -dy), min(rPixY, rPixY - dy)
                dy1_s, dy2_s = max(0, dy), min(rPixY, rPixY + dy)
                sx1, sx2 = max(0, -dx), min(rPixX, rPixX - dx)
                dx1_s, dx2_s = max(0, dx), min(rPixX, rPixX + dx)
                shifted[dy1_s:dy2_s, dx1_s:dx2_s] = bad[sy1:sy2, sx1:sx2]
                spread |= shifted
        mRData_py[spread & ~bad] |= np.int32(0x40)

    if sBorder > 0:
        mRData_py[:, :sBorder] |= np.int32(0x100 | 0x400)
        mRData_py[:, rPixX-sBorder:] |= np.int32(0x100 | 0x400)
        mRData_py[:sBorder, sBorder:rPixX-sBorder] |= np.int32(0x100 | 0x400)
        mRData_py[rPixY-sBorder:, sBorder:rPixX-sBorder] |= np.int32(0x100 | 0x400)

    xBufLo = rXMin - rXBMin
    xBufHi = rXBMax - rXMax
    yBufLo = rYMin - rYBMin
    yBufHi = rYBMax - rYMax
    fpixelOutX = rXBMin + xBufLo + 1
    fpixelOutY = rYBMin + yBufLo + 1
    lpixelOutX = fpixelOutX + (rPixX - xBufHi - xBufLo - 1)
    lpixelOutY = fpixelOutY + (rPixY - yBufHi - yBufLo - 1)

    logger.debug("  region_setup: done rPixX=%d rPixY=%d fpixelOut=(%d,%d) lpixelOut=(%d,%d)",
                 rPixX, rPixY, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY)
    return {
        'tRData': tRData_py, 'iRData': iRData_py,
        'oRData': oRData_py, 'eRData': eRData_py,
        'mRData': mRData_py, 'misRData': misRData_py, 'mtsRData': mtsRData_py,
        'rXMin': rXMin, 'rYMin': rYMin, 'rXMax': rXMax, 'rYMax': rYMax,
        'rXBMin': rXBMin, 'rYBMin': rYBMin, 'rXBMax': rXBMax, 'rYBMax': rYBMax,
        'xBufLo': xBufLo, 'xBufHi': xBufHi, 'yBufLo': yBufLo, 'yBufHi': yBufHi,
        'fpixelOutX': fpixelOutX, 'fpixelOutY': fpixelOutY,
        'lpixelOutX': lpixelOutX, 'lpixelOutY': lpixelOutY,
        'rPixX': rPixX, 'rPixY': rPixY,
    }

ZEROVAL = 1e-10
MAXVAL = 1e10
FLAG_BAD_PIXVAL = 0x01
FLAG_SAT_PIXEL = 0x02
FLAG_LOW_PIXEL = 0x04
FLAG_ISNAN = 0x08
FLAG_BAD_CONV = 0x10
FLAG_INPUT_MASK = 0x20
FLAG_OK_CONV = 0x40
FLAG_INPUT_ISBAD = 0x80
FLAG_T_BAD = 0x100
FLAG_T_SKIP = 0x200
FLAG_I_BAD = 0x400
FLAG_I_SKIP = 0x800
FLAG_OUTPUT_ISBAD = 0x8000


# class StampsArray:
#     def __init__(self, nS, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC):
#         fwSq = fwKSStamp * fwKSStamp
#         nVec = nCompKer + nBGVectors
#
#         self.nS = nS
#         self.nKSStamps = nKSStamps
#         self.fwKSStamp = fwKSStamp
#         self.nCompKer = nCompKer
#         self.nBGVectors = nBGVectors
#         self.nC = nC
#         self.fwSq = fwSq
#         self.nVec = nVec
#
#         self.sscnt   = np.zeros(nS, dtype=np.int32)
#         self.nss     = np.zeros(nS, dtype=np.int32)
#         self.x0      = np.zeros(nS, dtype=np.int32)
#         self.y0      = np.zeros(nS, dtype=np.int32)
#         self.x       = np.zeros(nS, dtype=np.int32)
#         self.y       = np.zeros(nS, dtype=np.int32)
#
#         self.xss     = np.zeros((nS, nKSStamps), dtype=np.int32)
#         self.yss     = np.zeros((nS, nKSStamps), dtype=np.int32)
#
#         self.vectors  = np.zeros((nS, nVec, fwSq), dtype=np.float64)
#         self.mat      = np.zeros((nS, nC, nC), dtype=np.float64)
#         self.scprod   = np.zeros((nS, nC), dtype=np.float64)
#         self.krefArea = np.zeros((nS, fwSq), dtype=np.float64)
#
#         self.chi2    = np.zeros(nS, dtype=np.float64)
#         self.norm    = np.zeros(nS, dtype=np.float64)
#         self.diff    = np.zeros(nS, dtype=np.float64)
#         self.sum_val = np.zeros(nS, dtype=np.float64)
#         self.mean_val = np.zeros(nS, dtype=np.float64)
#         self.median  = np.zeros(nS, dtype=np.float64)
#         self.mode    = np.zeros(nS, dtype=np.float64)
#         self.sd      = np.zeros(nS, dtype=np.float64)
#         self.fwhm    = np.zeros(nS, dtype=np.float64)
#         self.lfwhm   = np.zeros(nS, dtype=np.float64)
#
#         self.valid   = np.zeros(nS, dtype=np.bool_)
#         self.ntS     = 0
#
#     @staticmethod
#     def subset(src, mask):
#         nSub = mask.sum()
#         if nSub == 0:
#             return None
#         sa = StampsArray(nSub, src.nKSStamps, src.fwKSStamp, src.nCompKer, src.nBGVectors, src.nC)
#         sa.sscnt[:]     = src.sscnt[mask]
#         sa.nss[:]       = src.nss[mask]
#         sa.x0[:]        = src.x0[mask]
#         sa.y0[:]        = src.y0[mask]
#         sa.x[:]         = src.x[mask]
#         sa.y[:]         = src.y[mask]
#         sa.xss[:]       = src.xss[mask]
#         sa.yss[:]       = src.yss[mask]
#         sa.vectors[:]   = src.vectors[mask]
#         sa.mat[:]       = src.mat[mask]
#         sa.scprod[:]    = src.scprod[mask]
#         sa.krefArea[:]  = src.krefArea[mask]
#         sa.chi2[:]      = src.chi2[mask]
#         sa.norm[:]      = src.norm[mask]
#         sa.diff[:]      = src.diff[mask]
#         sa.sum_val[:]   = src.sum_val[mask]
#         sa.mean_val[:]  = src.mean_val[mask]
#         sa.median[:]    = src.median[mask]
#         sa.mode[:]      = src.mode[mask]
#         sa.sd[:]        = src.sd[mask]
#         sa.fwhm[:]      = src.fwhm[mask]
#         sa.lfwhm[:]     = src.lfwhm[mask]
#         sa.valid[:]     = src.valid[mask]
#         sa.ntS          = nSub
#         return sa
#
#     def deepCopy(self):
#         """返回完整深拷贝"""
#         cp = StampsArray(self.nS, self.nKSStamps, self.fwKSStamp, self.nCompKer, self.nBGVectors, self.nC)
#         cp.sscnt[:] = self.sscnt
#         cp.nss[:] = self.nss
#         cp.x0[:] = self.x0
#         cp.y0[:] = self.y0
#         cp.x[:] = self.x
#         cp.y[:] = self.y
#         cp.xss[:] = self.xss
#         cp.yss[:] = self.yss
#         cp.vectors[:] = self.vectors
#         cp.mat[:] = self.mat
#         cp.scprod[:] = self.scprod
#         cp.krefArea[:] = self.krefArea
#         cp.chi2[:] = self.chi2
#         cp.norm[:] = self.norm
#         cp.diff[:] = self.diff
#         cp.sum_val[:] = self.sum_val
#         cp.mean_val[:] = self.mean_val
#         cp.median[:] = self.median
#         cp.mode[:] = self.mode
#         cp.sd[:] = self.sd
#         cp.fwhm[:] = self.fwhm
#         cp.lfwhm[:] = self.lfwhm
#         cp.valid[:] = self.valid
#         cp.ntS = self.ntS
#         return cp


def sigma_clip_numpy(data, maxiter=10, stat_sig=3.0):
    """返回 (mean, stdev, return_code) — 矢量化版本"""
    count = len(data)
    if count == 0:
        return (0.0, MAXVAL, 1)

    # 截断为float32再转float64，模拟原始 float32 中间精度行为
    arr = np.asarray(data, dtype=np.float32).astype(np.float64)
    mask = np.zeros(count, dtype=bool)
    cnt = 0
    ncnt = count
    iternum = 0
    mean_val = 0.0
    stdev_val = 0.0

    while (ncnt != cnt) and (iternum < maxiter):
        cnt = ncnt

        good = arr[~mask]
        ncnt = len(good)

        if ncnt > 0:
            mean_val = float(good.mean())
        else:
            return (0.0, MAXVAL, 2)

        if ncnt > 1:
            stdev_val = float(good.std(ddof=1))
        else:
            return (mean_val, MAXVAL, 3)

        istdev = 1.0 / stdev_val

        deviations = np.abs(good - mean_val) * istdev
        new_outliers = deviations > stat_sig

        good_indices = np.where(~mask)[0]
        mask[good_indices[new_outliers]] = True
        ncnt = count - int(np.sum(mask))

        iternum += 1

    return (mean_val, stdev_val, 0)

# def sigma_clip_numpy_old(data, maxiter=10, stat_sig=3.0):
#     """返回 (mean, stdev, return_code)"""
#     count = len(data)
#     if count == 0:
#         return (0.0, MAXVAL, 1)
#
#     smask = [False] * count
#     cnt = 0
#     ncnt = count
#     iternum = 0
#     mean_val = 0.0
#     stdev_val = 0.0
#
#     while (ncnt != cnt) and (iternum < maxiter):
#         cnt = ncnt
#         mean_val = 0.0
#         stdev_val = 0.0
#         for i in range(count):
#             if not smask[i]:
#                 d = np.float32(data[i])
#                 mean_val += float(d)
#                 stdev_val += float(d * d)
#
#         if ncnt > 0:
#             mean_val /= ncnt
#         else:
#             return (0.0, MAXVAL, 2)
#
#         if ncnt > 1:
#             stdev_val = stdev_val - ncnt * mean_val * mean_val
#             stdev_val = math.sqrt(stdev_val / float(ncnt - 1))
#         else:
#             return (mean_val, MAXVAL, 3)
#
#         ncnt = 0
#         istdev = 1.0 / stdev_val
#         for i in range(count):
#             if not smask[i]:
#                 if (abs(float(data[i]) - mean_val) * istdev) > stat_sig:
#                     smask[i] = True
#                 else:
#                     ncnt += 1
#         iternum += 1
#
#     return (mean_val, stdev_val, 0)


@numba.jit(nopython=True)
def get_noise_stats3_numpy(data, noise, umask, smask, rPixX, rPixY, mRData):
    """返回 (nnorm, nncount)"""
    nsum = 0.0
    n = 0
    total = rPixX * rPixY
    ZVAL = 1e-10
    data_flat = data.ravel()
    noise_flat = noise.ravel()
    mRData_flat = mRData.ravel()

    for i in range(total - 1, -1, -1):
        ddat = data_flat[i]
        mdat = mRData_flat[i]

        if ((umask > 0) and (not (mdat & umask))) or \
           ((smask > 0) and (mdat & smask)) or \
           (abs(ddat) <= ZVAL):
            continue

        ndat = 1.0 / noise_flat[i]
        n += 1
        prod = (ddat * ddat) * ndat * ndat
        nsum += prod

    if n > 1:
        return (nsum / n, n)
    else:
        return (MAXVAL, n)


def insert_subregion_flt_numpy(sub_2d, full_2d, fpixelX, fpixelY, lpixelX, lpixelY, xBufLo, yBufLo):
    """in-place 修改 full_2d"""
    pixX = lpixelX - fpixelX + 1
    pixY = lpixelY - fpixelY + 1
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]


def insert_subregion_int_numpy(sub_2d, full_2d, fpixelX, fpixelY, lpixelX, lpixelY, xBufLo, yBufLo):
    """in-place 修改 full_2d (int 类型)"""
    pixX = lpixelX - fpixelX + 1
    pixY = lpixelY - fpixelY + 1
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]


def cut_stamp_numpy(data_1d, dxLen, xMin, yMin, xMax, yMax):
    """返回 (refArea_1d, x0, y0, cx, cy)"""
    sxLen = xMax - xMin + 1
    data_2d = data_1d.reshape(-1, dxLen)
    refArea = data_2d[yMin:yMax + 1, xMin:xMax + 1].ravel().copy()
    x0 = xMin
    y0 = yMin
    cx = xMin + (xMax - xMin) // 2
    cy = yMin + (yMax - yMin) // 2
    return (refArea, x0, y0, cx, cy)


class Ran1:
    M1 = 259200;  IA1 = 7141;  IC1 = 54773;  RM1 = 1.0 / 259200
    M2 = 134456;  IA2 = 8121;  IC2 = 28411;  RM2 = 1.0 / 134456
    M3 = 243000;  IA3 = 4561;  IC3 = 51349

    def __init__(self, idum):
        self.ix1 = 0
        self.ix2 = 0
        self.ix3 = 0
        self.r = [0.0] * 98
        self.iff = 0
        self.idum = int(idum)

    def __call__(self):
        if self.idum < 0 or self.iff == 0:
            self.iff = 1
            self.ix1 = (self.IC1 - self.idum) % self.M1
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            self.ix2 = self.ix1 % self.M2
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            self.ix3 = self.ix1 % self.M3
            for j in range(1, 98):
                self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
                self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
                self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
            self.idum = 1

        self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
        self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
        self.ix3 = (self.IA3 * self.ix3 + self.IC3) % self.M3
        j = 1 + (97 * self.ix3) // self.M3
        temp = self.r[j]
        self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
        return temp


# def get_stamp_stats3_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
#                             umask, smask, maxiter, rPixX, mRData_2d, statSig):
#     nstat = 100
#     ufstat = 0.9
#     mfstat = 0.5
# 
#     npts = nPixX * nPixY
#     if npts < nstat:
#         return {'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
#                 'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 4}
# 
#     rng = Ran1(-666)
#     tries = 0
# 
#     goodcnt = 0
#     work = [0.0] * nstat
#     i = 0
#     while (i < nstat) and (goodcnt < npts):
#         xr = int(math.floor(rng() * nPixX))
#         yr = int(math.floor(rng() * nPixY))
# 
#         rdat = float(data_2d[yr, xr])
#         mdat = int(mRData_2d[yr + y0Reg, xr + x0Reg])
# 
#         if ((umask > 0) and not (mdat & umask)) or \
#            ((smask > 0) and (mdat & smask)) or \
#            (abs(rdat) <= ZEROVAL):
#             pass
#         else:
#             work[i] = rdat
#             i += 1
#         goodcnt += 1
# 
#     work[:i] = sorted(work[:i])
#     npts = i
# 
#     binsize = (work[int(ufstat * npts)] - work[int(mfstat * npts)]) / float(nstat)
#     bin1 = work[int(mfstat * npts)] - 128.0 * binsize
# 
#     goodcnt = 0
#     sdat = []
#     for j in range(nPixY):
#         for ci in range(nPixX):
#             rdat = float(data_2d[j, ci])
#             mdat = int(mRData_2d[j + y0Reg, ci + x0Reg])
# 
#             if ((umask > 0) and not (mdat & umask)) or \
#                ((smask > 0) and (mdat & smask)) or \
#                (abs(rdat) <= ZEROVAL):
#                 continue
# 
#             if rdat * 0.0 != 0.0:
#                 mRData_2d[j + y0Reg, ci + x0Reg] = int(mRData_2d[j + y0Reg, ci + x0Reg]) | (FLAG_INPUT_ISBAD | FLAG_ISNAN)
#                 continue
# 
#             sdat.append(rdat)
#             goodcnt += 1
# 
#     sdat_arr = np.array(sdat, dtype=np.float32)
#     mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat_arr, maxiter, statSig)
#     if sc_rc != 0:
#         return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
#                 'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 5}
# 
#     isd = 1.0 / sd_val
# 
#     repeat = 1
#     ssum = 0.0
#     lower_val = 0.0
#     upper_val = 0.0
#     mode_val = 0.0
#     while repeat:
#         if tries >= 5:
#             return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
#                     'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 1}
# 
#         bins = [0] * 256
# 
#         ssum = 0.0
#         sumx = 0.0
#         sumxx = 0.0
#         goodcnt = 0
# 
#         for j in range(nPixY):
#             for ci in range(nPixX):
#                 rdat = float(data_2d[j, ci])
#                 mdat = int(mRData_2d[j + y0Reg, ci + x0Reg])
# 
#                 if ((umask > 0) and not (mdat & umask)) or \
#                    ((smask > 0) and (mdat & smask)) or \
#                    (abs(rdat) <= ZEROVAL):
#                     continue
# 
#                 if rdat * 0.0 != 0.0:
#                     mRData_2d[j + y0Reg, ci + x0Reg] = int(mRData_2d[j + y0Reg, ci + x0Reg]) | (FLAG_INPUT_ISBAD | FLAG_ISNAN)
#                     continue
# 
#                 if (abs(rdat - mean_val) * isd) > statSig:
#                     continue
# 
#                 index = int(math.floor((rdat - bin1) / binsize)) + 1
#                 if index < 0:
#                     index = 0
#                 if index > 255:
#                     index = 255
# 
#                 bins[index] += 1
#                 ssum += abs(rdat)
#                 goodcnt += 1
# 
#         if goodcnt == 0:
#             mode_val = work[int(mfstat * npts)]
#             median_val = mode_val
#             return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
#                     'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 2}
# 
#         if binsize == 0.0:
#             mode_val = work[int(mfstat * npts)]
#             median_val = mode_val
#             return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
#                     'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 3}
# 
#         sumx = 0.0
#         maxdens = 0.0
#         imax = 0
#         ilower = 1
#         iupper = 1
#         while iupper < 255:
#             while (sumx < goodcnt / 10.0) and (iupper < 255):
#                 sumx += bins[iupper]
#                 iupper += 1
# 
#             if (iupper - ilower) > 0 and sumx / (iupper - ilower) > maxdens:
#                 maxdens = sumx / (iupper - ilower)
#                 imax = ilower
# 
#             sumx -= bins[ilower]
#             ilower += 1
# 
#         if imax < 0 or imax > 255:
#             imax = 0
# 
#         sumxx = 0.0
#         sumx = 0.0
#         ci = imax
#         while (sumx < goodcnt / 10.0) and (ci < 255):
#             sumx += bins[ci]
#             sumxx += ci * bins[ci]
#             ci += 1
# 
#         mode_bin = sumxx / sumx + 0.5
#         mode_val = bin1 + binsize * (mode_bin - 1.0)
# 
#         imax_floor = int(math.floor(mode_bin))
#         sumx = 0.0
#         for ci in range(imax_floor):
#             sumx += bins[ci]
#         sumx += bins[imax_floor] * (mode_bin - imax_floor)
#         sumx /= goodcnt
#         moden = sumx
# 
#         lower = goodcnt * 0.25
#         upper = goodcnt * 0.75
#         sumx = 0.0
#         ci = 0
#         while sumx < lower:
#             sumx += bins[ci]
#             ci += 1
#         lower_val = ci - (sumx - lower) / bins[ci - 1]
# 
#         while sumx < upper:
#             sumx += bins[ci]
#             ci += 1
#         upper_val = ci - (sumx - upper) / bins[ci - 1]
# 
#         if (lower_val < 1.0) or (upper_val > 255.0):
#             bin1 -= 128.0 * binsize
#             binsize *= 2.0
#             tries += 1
#             repeat = 1
#         elif (upper_val - lower_val) < 40.0:
#             binsize /= 3.0
#             bin1 = mode_val - 128.0 * binsize
#             tries += 1
#             repeat = 1
#         else:
#             repeat = 0
# 
#     sum_val = ssum
# 
#     fwhm_val = binsize * (upper_val - lower_val) / 1.35
# 
#     sumx = 0.0
#     ci = 0
#     while sumx < goodcnt / 2.0:
#         sumx += bins[ci]
#         ci += 1
#     median_val = ci - (sumx - goodcnt / 2.0) / bins[ci - 1]
# 
#     lfwhm_val = binsize * (median_val - lower_val) * 2.0 / 1.35
# 
#     median_val = bin1 + binsize * (median_val - 1.0)
# 
#     return {'sum': sum_val, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
#             'sd': sd_val, 'fwhm': fwhm_val, 'lfwhm': lfwhm_val, 'return_code': 0}


def get_stamp_stats3_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                            umask, smask, maxiter, rPixX, mRData_2d, statSig):
    return get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                                        umask, smask, maxiter, rPixX, mRData_2d, statSig)


def bin_quartile_numpy(counts, target):
    cumsum = np.cumsum(counts).astype(np.float64)
    ci = int(np.searchsorted(cumsum, target))
    if ci >= len(cumsum):
        ci = len(cumsum) - 1
    if counts[ci] > 0:
        val = (ci + 1) - (cumsum[ci] - target) / counts[ci]
    else:
        val = float(ci + 1)
    return val


def get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                                 umask, smask, maxiter, rPixX, mRData_2d, statSig):
    # import time
    # t0 = time.time()
    nstat = 100
    ufstat = 0.9
    mfstat = 0.5

    npts = nPixX * nPixY
    if npts < nstat:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4, None)

    np.random.seed(666)
    flat_indices = np.random.randint(0, npts, size=nstat * 20)
    xr = flat_indices % nPixX
    yr = flat_indices // nPixX

    rdat_sample = data_2d[yr, xr].astype(np.float64)
    mdat_sample = mRData_2d[yr + y0Reg, xr + x0Reg]

    skip_sample = np.zeros(len(flat_indices), dtype=bool)
    if umask > 0:
        skip_sample |= (mdat_sample & umask) == 0
    if smask > 0:
        skip_sample |= (mdat_sample & smask) != 0
    skip_sample |= np.abs(rdat_sample) <= ZEROVAL

    good_samples = rdat_sample[~skip_sample]
    if len(good_samples) < nstat:
        good_samples = rdat_sample[~skip_sample][:len(good_samples)]
    else:
        good_samples = good_samples[:nstat]

    work = np.sort(good_samples)
    nfound = len(work)
    if nfound > 0:
        binsize = (work[int(ufstat * nfound)] - work[int(mfstat * nfound)]) / float(nstat)
        bin1 = work[int(mfstat * nfound)] - 128.0 * binsize
    else:
        binsize = 0.0
        bin1 = 0.0

    mRData_region = mRData_2d[y0Reg:y0Reg + nPixY, x0Reg:x0Reg + nPixX].ravel()

    data_flat = data_2d.ravel().astype(np.float64)
    ntotal = len(data_flat)

    skip_all = np.zeros(ntotal, dtype=bool)
    if umask > 0:
        skip_all |= (mRData_region & umask) == 0
    if smask > 0:
        skip_all |= (mRData_region & smask) != 0
    skip_all |= np.abs(data_flat) <= ZEROVAL
    # t_skip = time.time()

    nan_mask = np.isnan(data_flat)
    skip_all |= nan_mask
    # t_nan = time.time()

    # 收集 NaN 像素标记信息，不再原地修改 mRData_2d
    nan_updates = None
    if nan_mask.any():
        nan_y, nan_x = np.where(nan_mask.reshape(nPixY, nPixX))
        nan_updates = (nan_y, nan_x)
    # t_where = time.time()

    sdat = np.asarray(data_flat[~skip_all], dtype=np.float32)
    # t1 = time.time(); logger.debug("[stats3] skip=%.3fs nan=%.3fs where=%.3fs sdat=%.3fs total_setup=%.3fs",
    #     t_skip - t0, t_nan - t_skip, t_where - t_nan, t1 - t_where, t1 - t0)
    if len(sdat) == 0:
        mode_val = 0.0
        if nfound > 0:
            mode_val = work[int(mfstat * nfound)]
        return (0.0, 0.0, mode_val, mode_val, MAXVAL, 0.0, 0.0, 5, nan_updates)

    # t2 = time.time()
    mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat, maxiter, statSig)
    # t3 = time.time(); logger.debug("[stats3] sigma_clip %.3fs", t3 - t2)
    if sc_rc != 0:
        return (0.0, mean_val, 0.0, 0.0, sd_val, 0.0, 0.0, 5, nan_updates)

    isd = 1.0 / sd_val
    clip_mask = (np.abs(sdat.astype(np.float64) - mean_val) * isd) > statSig
    sdat_clipped = sdat[~clip_mask]

    ssum_val = float(np.sum(np.abs(sdat_clipped.astype(np.float64))))

    tries = 0
    current_binsize = binsize
    current_bin1 = bin1
    lower_val = 0.0
    upper_val = 0.0
    mode_val = 0.0
    goodcnt_h = 0
    bins = np.zeros(256, dtype=np.int64)

    while True:
        if tries >= 5:
            return (0.0, mean_val, 0.0, 0.0, sd_val, 0.0, 0.0, 1, nan_updates)

        if len(sdat_clipped) == 0:
            mode_val = 0.0
            if nfound > 0:
                mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return (0.0, mean_val, median_val, mode_val, sd_val, 0.0, 0.0, 2, nan_updates)

        if current_binsize == 0.0:
            mode_val = 0.0
            if nfound > 0:
                mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return (0.0, mean_val, median_val, mode_val, sd_val, 0.0, 0.0, 3, nan_updates)

        indices = ((sdat_clipped.astype(np.float64) - current_bin1) / current_binsize).astype(np.int32) + 1
        indices = np.clip(indices, 0, 255)
        bins.fill(0)
        np.add.at(bins, indices, 1)
        goodcnt_h = len(sdat_clipped)

        target10 = goodcnt_h / 10.0

        sumx = 0.0
        maxdens = 0.0
        imax_h = 0
        ilower = 1
        iupper = 1
        while iupper < 255:
            while (sumx < target10) and (iupper < 255):
                sumx += bins[iupper]
                iupper += 1
            if (iupper - ilower) > 0 and sumx / (iupper - ilower) > maxdens:
                maxdens = sumx / (iupper - ilower)
                imax_h = ilower
            sumx -= bins[ilower]
            ilower += 1

        if imax_h < 0 or imax_h > 255:
            imax_h = 0

        sumxx = 0.0
        sumx = 0.0
        ci = imax_h
        while (sumx < target10) and (ci < 255):
            sumx += bins[ci]
            sumxx += ci * bins[ci]
            ci += 1

        mode_bin = sumxx / sumx + 0.5
        mode_val = current_bin1 + current_binsize * (mode_bin - 1.0)

        lower_target = goodcnt_h * 0.25
        upper_target = goodcnt_h * 0.75

        lower_val = bin_quartile_numpy(bins, lower_target)
        upper_val = bin_quartile_numpy(bins, upper_target)

        if (lower_val < 1.0) or (upper_val > 255.0):
            current_bin1 -= 128.0 * current_binsize
            current_binsize *= 2.0
            tries += 1
        elif (upper_val - lower_val) < 40.0:
            current_binsize /= 3.0
            current_bin1 = mode_val - 128.0 * current_binsize
            tries += 1
        else:
            break

    # t4 = time.time(); logger.debug("[stats3] binning loop %.3fs", t4 - t3)
    fwhm_val = current_binsize * (upper_val - lower_val) / 1.35

    median_target = goodcnt_h / 2.0
    median_val = bin_quartile_numpy(bins, median_target)
    lfwhm_val = current_binsize * (median_val - lower_val) * 2.0 / 1.35
    median_val = current_bin1 + current_binsize * (median_val - 1.0)

    return (ssum_val, mean_val, median_val, mode_val, sd_val, fwhm_val, lfwhm_val, 0, nan_updates)


# cut_sstamp_numpy: 从参考图像切出 kernel stamp 区域，返回局部结果
def cut_sstamp_numpy(saXss, saYss, saX0, saY0, saNss, saSscnt, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose=0):
    # def cut_sstamp_numpy(sa, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose=0):
    # 局部分配 stamp 区域数组
    fwSqStamp = fwKSStamp * fwKSStamp
    outKrefArea = np.full(fwSqStamp, fillVal, dtype=np.float64)

    nss = saNss[si]
    sscnt = saSscnt[si]
    xStamp = int(saXss[si, sscnt]) - saX0[si]
    yStamp = int(saYss[si, sscnt]) - saY0[si]

    if sscnt >= nss:
        return 1  # 失败返回

    sumVal = 0.0
    for j in range(yStamp - hwKSStamp, yStamp + hwKSStamp + 1):
        y = j - (yStamp - hwKSStamp)
        dy = j + saY0[si]
        for i in range(xStamp - hwKSStamp, xStamp + hwKSStamp + 1):
            x = i - (xStamp - hwKSStamp)
            k = i + saX0[si] + rPixX * dy
            dpt = float(iData[k])
            # 写入局部数组，不再原地修改
            outKrefArea[x + y * fwKSStamp] = dpt
            if not (int(mRData[k]) & FLAG_INPUT_ISBAD):
                sumVal += abs(dpt)

    # 返回局部结果，由调用方写回 sa 数组
    return outKrefArea, sumVal


@numba.jit(nopython=True)
def check_psf_center_numba(iData, imax, jmax, xLen, yLen, sx0, sy0,
                             hiThresh, sky, invdsky,
                             xbuffer, ybuffer, bbit, bbit1,
                             rPixX, hwKSStamp, mRData, kerFitThresh):
    kerFitThresh = np.float64(kerFitThresh)
    brk = 0
    dmax2 = 0.0
    for l in range(jmax - hwKSStamp, jmax + hwKSStamp + 1):
        if l < ybuffer or l >= yLen - ybuffer:
            continue
        yr2 = l + sy0
        for k in range(imax - hwKSStamp, imax + hwKSStamp + 1):
            if k < xbuffer or k >= xLen - xbuffer:
                continue
            xr2 = k + sx0
            nr2 = xr2 + rPixX * yr2
            if mRData[nr2] & bbit:
                brk = 1
                dmax2 = 0.0
                break
            dpt2 = iData[nr2]
            if dpt2 >= hiThresh:
                mRData[nr2] = mRData[nr2] | bbit1
                brk = 1
                dmax2 = 0.0
                break
            if ((dpt2 - sky) * invdsky) > kerFitThresh:
                dmax2 += dpt2
        if brk == 1:
            break
    return dmax2


def check_psf_center_numpy(iData, imax, jmax, xLen, yLen, sx0, sy0,
                            hiThresh, sky, invdsky,
                            xbuffer, ybuffer, bbit, bbit1,
                            rPixX, hwKSStamp, kerFitThresh):
    # 不再接收 mRData 入参，改为收集需标记的像素索引
    sky = float(np.float32(sky))
    invdsky = float(np.float32(invdsky))
    kerFitThresh = float(np.float32(kerFitThresh))
    brk = 0
    dmax2 = 0.0
    maskUpdates = []  # 收集需要标记 bbit1 的像素索引

    for l in range(jmax - hwKSStamp, jmax + hwKSStamp + 1):
        if l < ybuffer or l >= yLen - ybuffer:
            continue

        yr2 = l + sy0

        for k in range(imax - hwKSStamp, imax + hwKSStamp + 1):
            if k < xbuffer or k >= xLen - xbuffer:
                continue

            xr2 = k + sx0
            nr2 = xr2 + rPixX * yr2

            # 不再读取/写入 mRData，改为收集索引交调用方处理
            dpt2 = float(iData[nr2])

            if dpt2 >= hiThresh:
                maskUpdates.append(nr2)  # 收集高亮像素索引
                brk = 1
                dmax2 = 0.0
                break

            if ((dpt2 - sky) * invdsky) > kerFitThresh:
                dmax2 += dpt2

        if brk == 1:
            break

    return dmax2, maskUpdates  # 返回标记列表，调用方负责应用


def quick_sort_impl(listArr, n):
    index = list(range(n))
    if n > 1:
        quick_sort_recurse(listArr, index, 0, n - 1)
    return index


def quick_sort_recurse(listArr, index, leftEnd, rightEnd):
    chosen = listArr[index[(leftEnd + rightEnd) // 2]]
    i = leftEnd - 1
    j = rightEnd + 1

    while True:
        i += 1
        while listArr[index[i]] < chosen:
            i += 1
        j -= 1
        while listArr[index[j]] > chosen:
            j -= 1
        if i < j:
            index[i], index[j] = index[j], index[i]
        elif i == j:
            i += 1
            break
        else:
            break

    if leftEnd < j:
        quick_sort_recurse(listArr, index, leftEnd, j)
    if i < rightEnd:
        quick_sort_recurse(listArr, index, i, rightEnd)


def get_psf_centers_numpy(sa, si, iData, xLen, yLen, hiThresh, bbit1, bbit2,
                           nKSStamps, hwKSStamp, rPixX, mRData, kerFitThresh,
                           verbose=0):
    # def get_psf_centers_numpy(stamp, iData, xLen, yLen, hiThresh, bbit1, bbit2,
    #                            nKSStamps, hwKSStamp, rPixX, mRData, kerFitThresh,
    #                            verbose=0):
    kerFitThresh = float(np.float32(kerFitThresh))
    dfrac = 0.9

    # if stamp['nss'] >= nKSStamps:
    if sa.nss[si] >= nKSStamps:
        return 0

    bbit = bbit1 | bbit2 | 0xbf
    # sky = stamp['mode']
    sky = sa.mode[si]
    # invdsky = 1.0 / stamp['fwhm']
    invdsky = 1.0 / sa.fwhm[si]

    # sx0 = stamp['x0']
    sx0 = sa.x0[si]
    # sy0 = stamp['y0']
    sy0 = sa.y0[si]

    xbuffer = 0
    ybuffer = 0

    # floorVal = sky + kerFitThresh * stamp['fwhm']
    floorVal = sky + kerFitThresh * sa.fwhm[si]

    allocSize = max(1, (xLen * yLen) // hwKSStamp)
    # xloc = [0] * allocSize
    xloc = np.zeros(allocSize, dtype=np.int32)
    # yloc = [0] * allocSize
    yloc = np.zeros(allocSize, dtype=np.int32)
    # peaks = [0.0] * allocSize
    peaks = np.zeros(allocSize, dtype=np.float64)

    brk = 0
    pcnt = 0
    fcnt = 2 * nKSStamps

    while pcnt < fcnt:
        loPsf = sky + (hiThresh - sky) * dfrac
        loPsf = max(loPsf, floorVal)

        for j in range(ybuffer, yLen - ybuffer):
            yr = j + sy0

            for i in range(xbuffer, xLen - xbuffer):
                xr = i + sx0
                nr = xr + rPixX * yr

                if int(mRData[nr]) & bbit:
                    continue

                dpt = float(iData[nr])

                if dpt >= hiThresh:
                    mRData[nr] = int(mRData[nr]) | bbit1
                    continue

                if ((dpt - sky) * invdsky) < kerFitThresh:
                    continue

                if dpt > loPsf:
                    dmax = dpt
                    imaxVal = i
                    jmaxVal = j

                    for l in range(j - hwKSStamp, j + hwKSStamp + 1):
                        yr2 = l + sy0

                        if l < ybuffer or l >= yLen - ybuffer:
                            continue

                        for k in range(i - hwKSStamp, i + hwKSStamp + 1):
                            xr2 = k + sx0
                            nr2 = xr2 + rPixX * yr2

                            if k < xbuffer or k >= xLen - xbuffer:
                                continue

                            if int(mRData[nr2]) & bbit:
                                continue

                            dpt2 = float(iData[nr2])

                            if dpt2 >= hiThresh:
                                mRData[nr2] = int(mRData[nr2]) | bbit1
                                continue

                            if ((dpt2 - sky) * invdsky) < kerFitThresh:
                                continue

                            if dpt2 > dmax:
                                dmax = dpt2
                                imaxVal = k
                                jmaxVal = l

                    dmax2 = check_psf_center_numpy(
                        iData, imaxVal, jmaxVal, xLen, yLen,
                        sx0, sy0, hiThresh, sky, invdsky,
                        xbuffer, ybuffer, bbit, bbit1,
                        rPixX, hwKSStamp, mRData, kerFitThresh)

                    if dmax2 == 0.0:
                        continue

                    xloc[pcnt] = imaxVal
                    yloc[pcnt] = jmaxVal
                    peaks[pcnt] = dmax2
                    pcnt += 1

                    for l in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
                        yr2 = l + sy0

                        for k in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
                            xr2 = k + sx0
                            nr2 = xr2 + rPixX * yr2

                            if (k > 0) and (k < xLen) and (l > 0) and (l < yLen):
                                mRData[nr2] = int(mRData[nr2]) | bbit2

                    if pcnt >= fcnt:
                        brk = 2

                if brk == 2:
                    break
            if brk == 2:
                break

        if loPsf == floorVal:
            break
        dfrac -= 0.2

    if pcnt == 0:
        return 1
    else:
        # qs = quick_sort_impl(peaks, pcnt)
        qs = np.argsort(peaks[:pcnt])
        # nssOrig = stamp['nss']
        nssOrig = sa.nss[si]
        idx = nssOrig
        jj = 0
        while jj < pcnt and idx < nKSStamps:
            # stamp['xss'][idx] = xloc[qs[pcnt - jj - 1]] + sx0
            sa.xss[si, idx] = xloc[qs[pcnt - jj - 1]] + sx0
            # stamp['yss'][idx] = yloc[qs[pcnt - jj - 1]] + sy0
            sa.yss[si, idx] = yloc[qs[pcnt - jj - 1]] + sy0
            # stamp['nss'] += 1
            sa.nss[si] += 1
            idx += 1
            jj += 1
        return 0


# def build_stamps_numpy(sXMin, sXMax, sYMin, sYMax, niS, ntS,
#                         getCenters, rXBMin, rYBMin, ciSa, ctSa,
#                         iRData1d, tRData1d, hardX, hardY,
#                         verbose, forceConvolve, rPixX, rPixY,
#                         tUKThresh, iUKThresh, hwKSStamp,
#                         fwStamp, nKSStamps, kerFitThresh,
#                         mRData1d, statSig):
#     # def build_stamps_numpy(sXMin, sXMax, sYMin, sYMax, niS, ntS,
#     #                         getCenters, rXBMin, rYBMin, ciStamps, ctStamps,
#     #                         iRData1d, tRData1d, hardX, hardY,
#     #                         verbose, forceConvolve, rPixX, rPixY,
#     #                         tUKThresh, iUKThresh, hwKSStamp,
#     #                         fwStamp, nKSStamps, kerFitThresh,
#     #                         mRData1d, statSig):
#     sPixX = sXMax - sXMin + 1
#     sPixY = sYMax - sYMin + 1
#
#     bbitt1 = FLAG_T_BAD
#     bbitt2 = FLAG_T_SKIP
#     bbiti1 = FLAG_I_BAD
#     bbiti2 = FLAG_I_SKIP
#
#     mRData2d = mRData1d.reshape(rPixY, rPixX)
#
#     if forceConvolve != "i":
#         # if ctStamps[ntS]['nss'] == 0:
#         if ctSa.nss[ntS] == 0:
#             refArea, x0, y0, cx, cy = cut_stamp_numpy(
#                 tRData1d, rPixX,
#                 sXMin - rXBMin, sYMin - rYBMin,
#                 sXMax - rXBMin, sYMax - rYBMin)
#             # ctStamps[ntS]['x0'] = x0
#             ctSa.x0[ntS] = x0
#             # ctStamps[ntS]['y0'] = y0
#             ctSa.y0[ntS] = y0
#             # ctStamps[ntS]['x'] = cx
#             ctSa.x[ntS] = cx
#             # ctStamps[ntS]['y'] = cy
#             ctSa.y[ntS] = cy
#
#             refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
#             result = get_stamp_stats3_numpy(
#                 # refArea2d, ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
#                 refArea2d, ctSa.x0[ntS], ctSa.y0[ntS],
#                 sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)
#
#             if result['return_code'] == 0:
#                 # ctStamps[ntS]['sum'] = result['sum']
#                 ctSa.sum_val[ntS] = result['sum']
#                 # ctStamps[ntS]['mean'] = result['mean']
#                 ctSa.mean_val[ntS] = result['mean']
#                 # ctStamps[ntS]['median'] = result['median']
#                 ctSa.median[ntS] = result['median']
#                 # ctStamps[ntS]['mode'] = result['mode']
#                 ctSa.mode[ntS] = result['mode']
#                 # ctStamps[ntS]['sd'] = result['sd']
#                 ctSa.sd[ntS] = result['sd']
#                 # ctStamps[ntS]['fwhm'] = result['fwhm']
#                 ctSa.fwhm[ntS] = result['fwhm']
#                 # ctStamps[ntS]['lfwhm'] = result['lfwhm']
#                 ctSa.lfwhm[ntS] = result['lfwhm']
#
#     if forceConvolve != "t":
#         # if ciStamps[niS]['nss'] == 0:
#         if ciSa.nss[niS] == 0:
#             refArea, x0, y0, cx, cy = cut_stamp_numpy(
#                 iRData1d, rPixX,
#                 sXMin - rXBMin, sYMin - rYBMin,
#                 sXMax - rXBMin, sYMax - rYBMin)
#             # ciStamps[niS]['x0'] = x0
#             ciSa.x0[niS] = x0
#             # ciStamps[niS]['y0'] = y0
#             ciSa.y0[niS] = y0
#             # ciStamps[niS]['x'] = cx
#             ciSa.x[niS] = cx
#             # ciStamps[niS]['y'] = cy
#             ciSa.y[niS] = cy
#
#             refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
#             result = get_stamp_stats3_numpy(
#                 # refArea2d, ciStamps[niS]['x0'], ciStamps[niS]['y0'],
#                 refArea2d, ciSa.x0[niS], ciSa.y0[niS],
#                 sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)
#
#             if result['return_code'] == 0:
#                 # ciStamps[niS]['sum'] = result['sum']
#                 ciSa.sum_val[niS] = result['sum']
#                 # ciStamps[niS]['mean'] = result['mean']
#                 ciSa.mean_val[niS] = result['mean']
#                 # ciStamps[niS]['median'] = result['median']
#                 ciSa.median[niS] = result['median']
#                 # ciStamps[niS]['mode'] = result['mode']
#                 ciSa.mode[niS] = result['mode']
#                 # ciStamps[niS]['sd'] = result['sd']
#                 ciSa.sd[niS] = result['sd']
#                 # ciStamps[niS]['fwhm'] = result['fwhm']
#                 ciSa.fwhm[niS] = result['fwhm']
#                 # ciStamps[niS]['lfwhm'] = result['lfwhm']
#                 ciSa.lfwhm[niS] = result['lfwhm']
#
#     if forceConvolve != "i":
#         # nss = ctStamps[ntS]['nss']
#         nss = ctSa.nss[ntS]
#         # if getCenters:
#         #     # get_psf_centers_numpy(
#         #     #     ctStamps[ntS], tRData1d, sPixX, sPixY,
#         #     #     tUKThresh, bbitt1, bbitt2, nKSStamps,
#         #     #     hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
#         #     get_psf_centers_numpy(
#         #         ctSa, ntS, tRData1d, sPixX, sPixY,
#         #         tUKThresh, bbitt1, bbitt2, nKSStamps,
#         #         hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
#         # else:
#         #     [注释] 原 else 分支逻辑由下方内联代码替代
#         if getCenters:
#             # [内联] get_psf_centers_numpy(ctSa, ntS, tRData1d, ...) 内联开始
#             kerFitThresh_t = float(np.float32(kerFitThresh))
#             dfrac = 0.9
#             if ctSa.nss[ntS] < nKSStamps:
#                 bbit = bbitt1 | bbitt2 | 0xbf
#                 sky = ctSa.mode[ntS]
#                 invdsky = 1.0 / ctSa.fwhm[ntS]
#                 sx0 = ctSa.x0[ntS]
#                 sy0 = ctSa.y0[ntS]
#                 xbuffer = 0
#                 ybuffer = 0
#                 floorVal = sky + kerFitThresh_t * ctSa.fwhm[ntS]
#                 allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
#                 xloc = np.zeros(allocSize, dtype=np.int32)
#                 yloc = np.zeros(allocSize, dtype=np.int32)
#                 peaks = np.zeros(allocSize, dtype=np.float64)
#                 brk = 0
#                 pcnt = 0
#                 fcnt = 2 * nKSStamps
#                 while pcnt < fcnt:
#                     loPsf = sky + (tUKThresh - sky) * dfrac
#                     loPsf = max(loPsf, floorVal)
#                     for j in range(ybuffer, sPixY - ybuffer):
#                         yr = j + sy0
#                         for i in range(xbuffer, sPixX - xbuffer):
#                             xr = i + sx0
#                             nr = xr + rPixX * yr
#                             if int(mRData1d[nr]) & bbit:
#                                 continue
#                             dpt = float(tRData1d[nr])
#                             if dpt >= tUKThresh:
#                                 mRData1d[nr] = int(mRData1d[nr]) | bbitt1
#                                 continue
#                             if ((dpt - sky) * invdsky) < kerFitThresh_t:
#                                 continue
#                             if dpt > loPsf:
#                                 dmax = dpt
#                                 imaxVal = i
#                                 jmaxVal = j
#                                 for l in range(j - hwKSStamp, j + hwKSStamp + 1):
#                                     yr2 = l + sy0
#                                     if l < ybuffer or l >= sPixY - ybuffer:
#                                         continue
#                                     for k in range(i - hwKSStamp, i + hwKSStamp + 1):
#                                         xr2 = k + sx0
#                                         nr2 = xr2 + rPixX * yr2
#                                         if k < xbuffer or k >= sPixX - xbuffer:
#                                             continue
#                                         if int(mRData1d[nr2]) & bbit:
#                                             continue
#                                         dpt2 = float(tRData1d[nr2])
#                                         if dpt2 >= tUKThresh:
#                                             mRData1d[nr2] = int(mRData1d[nr2]) | bbitt1
#                                             continue
#                                         if ((dpt2 - sky) * invdsky) < kerFitThresh_t:
#                                             continue
#                                         if dpt2 > dmax:
#                                             dmax = dpt2
#                                             imaxVal = k
#                                             jmaxVal = l
#                                 dmax2 = check_psf_center_numba(
#                                     tRData1d, imaxVal, jmaxVal, sPixX, sPixY,
#                                     sx0, sy0, tUKThresh, sky, invdsky,
#                                     xbuffer, ybuffer, bbit, bbitt1,
#                                     rPixX, hwKSStamp, mRData1d, kerFitThresh_t)
#                                 if dmax2 == 0.0:
#                                     continue
#                                 xloc[pcnt] = imaxVal
#                                 yloc[pcnt] = jmaxVal
#                                 peaks[pcnt] = dmax2
#                                 pcnt += 1
#                                 for l in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
#                                     yr2 = l + sy0
#                                     for k in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
#                                         xr2 = k + sx0
#                                         nr2 = xr2 + rPixX * yr2
#                                         if (k > 0) and (k < sPixX) and (l > 0) and (l < sPixY):
#                                             mRData1d[nr2] = int(mRData1d[nr2]) | bbitt2
#                                 if pcnt >= fcnt:
#                                     brk = 2
#                             if brk == 2:
#                                 break
#                         if brk == 2:
#                             break
#                     if loPsf == floorVal:
#                         break
#                     dfrac -= 0.2
#                 if pcnt > 0:
#                     qs = np.argsort(peaks[:pcnt])
#                     nssOrig = ctSa.nss[ntS]
#                     idx = nssOrig
#                     jj = 0
#                     while jj < pcnt and idx < nKSStamps:
#                         ctSa.xss[ntS, idx] = xloc[qs[pcnt - jj - 1]] + sx0
#                         ctSa.yss[ntS, idx] = yloc[qs[pcnt - jj - 1]] + sy0
#                         ctSa.nss[ntS] += 1
#                         idx += 1
#                         jj += 1
#             # [内联] get_psf_centers_numpy 内联结束
#         else:
#             if nss < nKSStamps:
#                 if hardX:
#                     xmax = int(hardX)
#                 else:
#                     xmax = sXMin + fwStamp // 2
#
#                 if hardY:
#                     ymax = int(hardY)
#                 else:
#                     ymax = sYMin + fwStamp // 2
#
#                 check = check_psf_center_numba(
#                     tRData1d,
#                     # xmax - ctStamps[ntS]['x0'],
#                     xmax - ctSa.x0[ntS],
#                     # ymax - ctStamps[ntS]['y0'],
#                     ymax - ctSa.y0[ntS],
#                     sPixX, sPixY,
#                     # ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
#                     ctSa.x0[ntS], ctSa.y0[ntS],
#                     tUKThresh,
#                     # ctStamps[ntS]['mode'],
#                     ctSa.mode[ntS],
#                     # 1.0 / ctStamps[ntS]['fwhm'],
#                     1.0 / ctSa.fwhm[ntS],
#                     0, 0,
#                     bbitt1 | bbitt2 | 0xbf, bbitt1,
#                     rPixX, hwKSStamp, mRData1d, kerFitThresh)
#
#                 if check != 0.0:
#                     for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
#                         for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
#                             nr2 = l + rPixX * k
#                             if nr2 >= 0 and nr2 < rPixX * rPixY:
#                                 mRData1d[nr2] = int(mRData1d[nr2]) | bbitt2
#
#                     # ctStamps[ntS]['xss'][nss] = xmax
#                     ctSa.xss[ntS, nss] = xmax
#                     # ctStamps[ntS]['yss'][nss] = ymax
#                     ctSa.yss[ntS, nss] = ymax
#                     # ctStamps[ntS]['nss'] += 1
#                     ctSa.nss[ntS] += 1
#
#     if forceConvolve != "t":
#         # nss = ciStamps[niS]['nss']
#         nss = ciSa.nss[niS]
#         # if getCenters:
#         #     # get_psf_centers_numpy(
#         #     #     ciStamps[niS], iRData1d, sPixX, sPixY,
#         #     #     iUKThresh, bbiti1, bbiti2, nKSStamps,
#         #     #     hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
#         #     get_psf_centers_numpy(
#         #         ciSa, niS, iRData1d, sPixX, sPixY,
#         #         iUKThresh, bbiti1, bbiti2, nKSStamps,
#         #         hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
#         # else:
#         #     [注释] 原 else 分支逻辑由下方内联代码替代
#         if getCenters:
#             # [内联] get_psf_centers_numpy(ciSa, niS, iRData1d, ...) 内联开始
#             kerFitThresh_i = float(np.float32(kerFitThresh))
#             dfrac = 0.9
#             if ciSa.nss[niS] < nKSStamps:
#                 bbit = bbiti1 | bbiti2 | 0xbf
#                 sky = ciSa.mode[niS]
#                 invdsky = 1.0 / ciSa.fwhm[niS]
#                 sx0 = ciSa.x0[niS]
#                 sy0 = ciSa.y0[niS]
#                 xbuffer = 0
#                 ybuffer = 0
#                 floorVal = sky + kerFitThresh_i * ciSa.fwhm[niS]
#                 allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
#                 xloc = np.zeros(allocSize, dtype=np.int32)
#                 yloc = np.zeros(allocSize, dtype=np.int32)
#                 peaks = np.zeros(allocSize, dtype=np.float64)
#                 brk = 0
#                 pcnt = 0
#                 fcnt = 2 * nKSStamps
#                 while pcnt < fcnt:
#                     loPsf = sky + (iUKThresh - sky) * dfrac
#                     loPsf = max(loPsf, floorVal)
#                     for j in range(ybuffer, sPixY - ybuffer):
#                         yr = j + sy0
#                         for i in range(xbuffer, sPixX - xbuffer):
#                             xr = i + sx0
#                             nr = xr + rPixX * yr
#                             if int(mRData1d[nr]) & bbit:
#                                 continue
#                             dpt = float(iRData1d[nr])
#                             if dpt >= iUKThresh:
#                                 mRData1d[nr] = int(mRData1d[nr]) | bbiti1
#                                 continue
#                             if ((dpt - sky) * invdsky) < kerFitThresh_i:
#                                 continue
#                             if dpt > loPsf:
#                                 dmax = dpt
#                                 imaxVal = i
#                                 jmaxVal = j
#                                 for l in range(j - hwKSStamp, j + hwKSStamp + 1):
#                                     yr2 = l + sy0
#                                     if l < ybuffer or l >= sPixY - ybuffer:
#                                         continue
#                                     for k in range(i - hwKSStamp, i + hwKSStamp + 1):
#                                         xr2 = k + sx0
#                                         nr2 = xr2 + rPixX * yr2
#                                         if k < xbuffer or k >= sPixX - xbuffer:
#                                             continue
#                                         if int(mRData1d[nr2]) & bbit:
#                                             continue
#                                         dpt2 = float(iRData1d[nr2])
#                                         if dpt2 >= iUKThresh:
#                                             mRData1d[nr2] = int(mRData1d[nr2]) | bbiti1
#                                             continue
#                                         if ((dpt2 - sky) * invdsky) < kerFitThresh_i:
#                                             continue
#                                         if dpt2 > dmax:
#                                             dmax = dpt2
#                                             imaxVal = k
#                                             jmaxVal = l
#                                 dmax2 = check_psf_center_numba(
#                                     iRData1d, imaxVal, jmaxVal, sPixX, sPixY,
#                                     sx0, sy0, iUKThresh, sky, invdsky,
#                                     xbuffer, ybuffer, bbit, bbiti1,
#                                     rPixX, hwKSStamp, mRData1d, kerFitThresh_i)
#                                 if dmax2 == 0.0:
#                                     continue
#                                 xloc[pcnt] = imaxVal
#                                 yloc[pcnt] = jmaxVal
#                                 peaks[pcnt] = dmax2
#                                 pcnt += 1
#                                 for l in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
#                                     yr2 = l + sy0
#                                     for k in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
#                                         xr2 = k + sx0
#                                         nr2 = xr2 + rPixX * yr2
#                                         if (k > 0) and (k < sPixX) and (l > 0) and (l < sPixY):
#                                             mRData1d[nr2] = int(mRData1d[nr2]) | bbiti2
#                                 if pcnt >= fcnt:
#                                     brk = 2
#                             if brk == 2:
#                                 break
#                         if brk == 2:
#                             break
#                     if loPsf == floorVal:
#                         break
#                     dfrac -= 0.2
#                 if pcnt > 0:
#                     qs = np.argsort(peaks[:pcnt])
#                     nssOrig = ciSa.nss[niS]
#                     idx = nssOrig
#                     jj = 0
#                     while jj < pcnt and idx < nKSStamps:
#                         ciSa.xss[niS, idx] = xloc[qs[pcnt - jj - 1]] + sx0
#                         ciSa.yss[niS, idx] = yloc[qs[pcnt - jj - 1]] + sy0
#                         ciSa.nss[niS] += 1
#                         idx += 1
#                         jj += 1
#             # [内联] get_psf_centers_numpy 内联结束
#         else:
#             if nss < nKSStamps:
#                 if hardX:
#                     xmax = int(hardX)
#                 else:
#                     xmax = sXMin + fwStamp // 2
#
#                 if hardY:
#                     ymax = int(hardY)
#                 else:
#                     ymax = sYMin + fwStamp // 2
#
#                 check = check_psf_center_numba(
#                     iRData1d,
#                     # xmax - ciStamps[niS]['x0'],
#                     xmax - ciSa.x0[niS],
#                     # ymax - ciStamps[niS]['y0'],
#                     ymax - ciSa.y0[niS],
#                     sPixX, sPixY,
#                     # ciStamps[niS]['x0'], ciStamps[niS]['y0'],
#                     ciSa.x0[niS], ciSa.y0[niS],
#                     iUKThresh,
#                     # ciStamps[niS]['mode'],
#                     ciSa.mode[niS],
#                     # 1.0 / ciStamps[niS]['fwhm'],
#                     1.0 / ciSa.fwhm[niS],
#                     0, 0,
#                     bbiti1 | bbiti2 | 0xbf, bbiti1,
#                     rPixX, hwKSStamp, mRData1d, kerFitThresh)
#
#                 if check != 0.0:
#                     for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
#                         for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
#                             nr2 = l + rPixX * k
#                             if nr2 >= 0 and nr2 < rPixX * rPixY:
#                                 mRData1d[nr2] = int(mRData1d[nr2]) | bbiti2
#
#                     # ciStamps[niS]['xss'][nss] = xmax
#                     # ciStamps[niS]['yss'][nss] = ymax
#                     # ciStamps[niS]['nss'] += 1
#                     ciSa.xss[niS, nss] = xmax
#                     ciSa.yss[niS, nss] = ymax
#                     ciSa.nss[niS] += 1


@numba.jit(nopython=True)
def psfCentersJit(iData1d, mRData1d, sPixX, sPixY, hwKSStamp, rPixX,
                   hiThresh, kerFitThreshVal, sky_val, invdsky_val,
                   sx0_val, sy0_val, nKSStamps, bbit, bbit1, bbit2,
                   xloc_out, yloc_out, peaks_out):
    dfrac = 0.9
    floorVal = sky_val + kerFitThreshVal / invdsky_val
    xbuffer = 0
    ybuffer = 0
    pcnt = 0
    fcnt = 2 * nKSStamps
    brk = 0
    while pcnt < fcnt:
        loPsf = sky_val + (hiThresh - sky_val) * dfrac
        loPsf = max(loPsf, floorVal)
        for j in range(ybuffer, sPixY - ybuffer):
            yr = j + sy0_val
            for i in range(xbuffer, sPixX - xbuffer):
                xr = i + sx0_val
                nr = xr + rPixX * yr
                if mRData1d[nr] & bbit:
                    continue
                dpt = iData1d[nr]
                if dpt >= hiThresh:
                    mRData1d[nr] = mRData1d[nr] | bbit1
                    continue
                if ((dpt - sky_val) * invdsky_val) < kerFitThreshVal:
                    continue
                if dpt > loPsf:
                    dmax = dpt
                    imaxVal = i
                    jmaxVal = j
                    for l in range(j - hwKSStamp, j + hwKSStamp + 1):
                        yr2 = l + sy0_val
                        if l < ybuffer or l >= sPixY - ybuffer:
                            continue
                        for k in range(i - hwKSStamp, i + hwKSStamp + 1):
                            xr2 = k + sx0_val
                            nr2 = xr2 + rPixX * yr2
                            if k < xbuffer or k >= sPixX - xbuffer:
                                continue
                            if mRData1d[nr2] & bbit:
                                continue
                            dpt2 = iData1d[nr2]
                            if dpt2 >= hiThresh:
                                mRData1d[nr2] = mRData1d[nr2] | bbit1
                                continue
                            if ((dpt2 - sky_val) * invdsky_val) < kerFitThreshVal:
                                continue
                            if dpt2 > dmax:
                                dmax = dpt2
                                imaxVal = k
                                jmaxVal = l
                    dmax2 = check_psf_center_numba(
                        iData1d, imaxVal, jmaxVal, sPixX, sPixY,
                        sx0_val, sy0_val, hiThresh, sky_val, invdsky_val,
                        xbuffer, ybuffer, bbit, bbit1,
                        rPixX, hwKSStamp, mRData1d, kerFitThreshVal)
                    if dmax2 == 0.0:
                        continue
                    xloc_out[pcnt] = imaxVal
                    yloc_out[pcnt] = jmaxVal
                    peaks_out[pcnt] = dmax2
                    pcnt += 1
                    for l in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
                        yr2 = l + sy0_val
                        for k in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
                            xr2 = k + sx0_val
                            nr2 = xr2 + rPixX * yr2
                            if (k > 0) and (k < sPixX) and (l > 0) and (l < sPixY):
                                mRData1d[nr2] = mRData1d[nr2] | bbit2
                    if pcnt >= fcnt:
                        brk = 2
                if brk == 2:
                    break
            if brk == 2:
                break
        if loPsf == floorVal:
            break
        dfrac -= 0.2
    return pcnt


def buildStampsNumba(sXMin, sXMax, sYMin, sYMax, niS, ntS,
                      getCenters, rXBMin, rYBMin,
                      lctNss_in, lctX0_in, lctY0_in, lctX_in, lctY_in, lctSumVal_in, lctMeanVal_in, lctMedian_in, lctMode_in, lctSd_in, lctFwhm_in, lctLfwhm_in, ctXss_in, ctYss_in, ctSscnt_in,
                      lciNss_in, lciX0_in, lciY0_in, lciX_in, lciY_in, lciSumVal_in, lciMeanVal_in, lciMedian_in, lciMode_in, lciSd_in, lciFwhm_in, lciLfwhm_in, ciXss_in, ciYss_in, ciSscnt_in,
                      iRData1d, tRData1d, hardX, hardY,
                      verbose, forceConvolve, rPixX, rPixY,
                      tUKThresh, iUKThresh, hwKSStamp,
                      fwStamp, nKSStamps, kerFitThresh,
                      mRData1d_in, statSig):
    # return build_stamps_numpy(sXMin, sXMax, sYMin, sYMax, niS, ntS,
    #                             getCenters, rXBMin, rYBMin, ciSa, ctSa,
    #                             iRData1d, tRData1d, hardX, hardY,
    #                             verbose, forceConvolve, rPixX, rPixY,
    #                             tUKThresh, iUKThresh, hwKSStamp,
    #                             fwStamp, nKSStamps, kerFitThresh,
    #                             mRData1d, statSig)
    sPixX = sXMax - sXMin + 1
    sPixY = sYMax - sYMin + 1

    # 读入本地变量
    lctX0, lctY0 = lctX0_in, lctY0_in
    lctX, lctY = lctX_in, lctY_in
    lctSumVal, lctMeanVal = lctSumVal_in, lctMeanVal_in
    lctMedian, lctMode = lctMedian_in, lctMode_in
    lctSd, lctFwhm = lctSd_in, lctFwhm_in
    lctLfwhm = lctLfwhm_in
    lctNss = lctNss_in
    lctXss = ctXss_in.copy()
    lctYss = ctYss_in.copy()
    lciX0, lciY0 = lciX0_in, lciY0_in
    lciX, lciY = lciX_in, lciY_in
    lciSumVal, lciMeanVal = lciSumVal_in, lciMeanVal_in
    lciMedian, lciMode = lciMedian_in, lciMode_in
    lciSd, lciFwhm = lciSd_in, lciFwhm_in
    lciLfwhm = lciLfwhm_in
    lciNss = lciNss_in
    lciXss = ciXss_in.copy()
    lciYss = ciYss_in.copy()
    lmRData = mRData1d_in.copy()

    bbitt1 = FLAG_T_BAD
    bbitt2 = FLAG_T_SKIP
    bbiti1 = FLAG_I_BAD
    bbiti2 = FLAG_I_SKIP

    mRData2d = lmRData.reshape(rPixY, rPixX)

    if forceConvolve != "i":
        # if ctStamps[ntS]['nss'] == 0:
        if lctNss == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                tRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            # ctStamps[ntS]['x0'] = x0
            lctX0 = x0
            # ctStamps[ntS]['y0'] = y0
            lctY0 = y0
            # ctStamps[ntS]['x'] = cx
            lctX = cx
            # ctStamps[ntS]['y'] = cy
            lctY = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                # refArea2d, ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
                refArea2d, lctX0, lctY0,
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result[7] == 0:
                # ctStamps[ntS]['sum'] = result['sum']
                lctSumVal = result[0]
                # ctStamps[ntS]['mean'] = result['mean']
                lctMeanVal = result[1]
                # ctStamps[ntS]['median'] = result['median']
                lctMedian = result[2]
                # ctStamps[ntS]['mode'] = result['mode']
                lctMode = result[3]
                # ctStamps[ntS]['sd'] = result['sd']
                lctSd = result[4]
                # ctStamps[ntS]['fwhm'] = result['fwhm']
                lctFwhm = result[5]
                # ctStamps[ntS]['lfwhm'] = result['lfwhm']
                lctLfwhm = result[6]

    if forceConvolve != "t":
        # if ciStamps[niS]['nss'] == 0:
        if lciNss == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                iRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            # ciStamps[niS]['x0'] = x0
            lciX0 = x0
            # ciStamps[niS]['y0'] = y0
            lciY0 = y0
            # ciStamps[niS]['x'] = cx
            lciX = cx
            # ciStamps[niS]['y'] = cy
            lciY = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                # refArea2d, ciStamps[niS]['x0'], ciStamps[niS]['y0'],
                refArea2d, lciX0, lciY0,
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result[7] == 0:
                # ciStamps[niS]['sum'] = result['sum']
                lciSumVal = result[0]
                # ciStamps[niS]['mean'] = result['mean']
                lciMeanVal = result[1]
                # ciStamps[niS]['median'] = result['median']
                lciMedian = result[2]
                # ciStamps[niS]['mode'] = result['mode']
                lciMode = result[3]
                # ciStamps[niS]['sd'] = result['sd']
                lciSd = result[4]
                # ciStamps[niS]['fwhm'] = result['fwhm']
                lciFwhm = result[5]
                # ciStamps[niS]['lfwhm'] = result['lfwhm']
                lciLfwhm = result[6]

    if forceConvolve != "i":
        # nss = ctStamps[ntS]['nss']
        nss = lctNss
        # if getCenters:
        #     # get_psf_centers_numpy(
        #     #     ctStamps[ntS], tRData1d, sPixX, sPixY,
        #     #     tUKThresh, bbitt1, bbitt2, nKSStamps,
        #     #     hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
        #     get_psf_centers_numpy(
        #         ctSa, ntS, tRData1d, sPixX, sPixY,
        #         tUKThresh, bbitt1, bbitt2, nKSStamps,
        #         hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
        # else:
        #     [注释] 原 else 分支逻辑由下方内联代码替代
        if getCenters:
            # [内联] get_psf_centers_numpy(ctSa, ntS, tRData1d, ...) 内联开始
            kerFitThresh_t = float(np.float32(kerFitThresh))
            if lctNss < nKSStamps:
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                pcnt = psfCentersJit(
                    tRData1d, lmRData, sPixX, sPixY, hwKSStamp, rPixX,
                    tUKThresh, kerFitThresh_t, lctMode, 1.0 / lctFwhm,
                    lctX0, lctY0, nKSStamps,
                    bbitt1 | bbitt2 | 0xbf, bbitt1, bbitt2,
                    xloc, yloc, peaks)
                if pcnt > 0:
                    qs = np.argsort(peaks[:pcnt])
                    nssOrig = lctNss
                    idx = nssOrig
                    jj = 0
                    while jj < pcnt and idx < nKSStamps:
                        lctXss[idx] = xloc[qs[pcnt - jj - 1]] + lctX0
                        lctYss[idx] = yloc[qs[pcnt - jj - 1]] + lctY0
                        lctNss += 1
                        idx += 1
                        jj += 1
            # [内联] get_psf_centers_numpy 内联结束
        else:
            if nss < nKSStamps:
                if hardX:
                    xmax = int(hardX)
                else:
                    xmax = sXMin + fwStamp // 2

                if hardY:
                    ymax = int(hardY)
                else:
                    ymax = sYMin + fwStamp // 2

                check = check_psf_center_numba(
                    tRData1d,
                    # xmax - ctStamps[ntS]['x0'],
                    xmax - lctX0,
                    # ymax - ctStamps[ntS]['y0'],
                    ymax - lctY0,
                    sPixX, sPixY,
                    # ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
                    lctX0, lctY0,
                    tUKThresh,
                    # ctStamps[ntS]['mode'],
                    lctMode,
                    # 1.0 / ctStamps[ntS]['fwhm'],
                    1.0 / lctFwhm,
                    0, 0,
                    bbitt1 | bbitt2 | 0xbf, bbitt1,
                    rPixX, hwKSStamp, lmRData, kerFitThresh)

                if check != 0.0:
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                            nr2 = l + rPixX * k
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                lmRData[nr2] = int(lmRData[nr2]) | bbitt2

                    # ctStamps[ntS]['xss'][nss] = xmax
                    lctXss[nss] = xmax
                    # ctStamps[ntS]['yss'][nss] = ymax
                    lctYss[nss] = ymax
                    # ctStamps[ntS]['nss'] += 1
                    lctNss += 1

    if forceConvolve != "t":
        # nss = ciStamps[niS]['nss']
        nss = lciNss
        # if getCenters:
        #     # get_psf_centers_numpy(
        #     #     ciStamps[niS], iRData1d, sPixX, sPixY,
        #     #     iUKThresh, bbiti1, bbiti2, nKSStamps,
        #     #     hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
        #     get_psf_centers_numpy(
        #         ciSa, niS, iRData1d, sPixX, sPixY,
        #         iUKThresh, bbiti1, bbiti2, nKSStamps,
        #         hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
        # else:
        #     [注释] 原 else 分支逻辑由下方内联代码替代
        if getCenters:
            # [内联] get_psf_centers_numpy(ciSa, niS, iRData1d, ...) 内联开始
            kerFitThresh_i = float(np.float32(kerFitThresh))
            if lciNss < nKSStamps:
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                pcnt = psfCentersJit(
                    iRData1d, lmRData, sPixX, sPixY, hwKSStamp, rPixX,
                    iUKThresh, kerFitThresh_i, lciMode, 1.0 / lciFwhm,
                    lciX0, lciY0, nKSStamps,
                    bbiti1 | bbiti2 | 0xbf, bbiti1, bbiti2,
                    xloc, yloc, peaks)
                if pcnt > 0:
                    qs = np.argsort(peaks[:pcnt])
                    nssOrig = lciNss
                    idx = nssOrig
                    jj = 0
                    while jj < pcnt and idx < nKSStamps:
                        lciXss[idx] = xloc[qs[pcnt - jj - 1]] + lciX0
                        lciYss[idx] = yloc[qs[pcnt - jj - 1]] + lciY0
                        lciNss += 1
                        idx += 1
                        jj += 1
            # [内联] get_psf_centers_numpy 内联结束
        else:
            if nss < nKSStamps:
                if hardX:
                    xmax = int(hardX)
                else:
                    xmax = sXMin + fwStamp // 2

                if hardY:
                    ymax = int(hardY)
                else:
                    ymax = sYMin + fwStamp // 2

                check = check_psf_center_numba(
                    iRData1d,
                    # xmax - ciStamps[niS]['x0'],
                    xmax - lciX0,
                    # ymax - ciStamps[niS]['y0'],
                    ymax - lciY0,
                    sPixX, sPixY,
                    # ciStamps[niS]['x0'], ciStamps[niS]['y0'],
                    lciX0, lciY0,
                    iUKThresh,
                    # ciStamps[niS]['mode'],
                    lciMode,
                    # 1.0 / ciStamps[niS]['fwhm'],
                    1.0 / lciFwhm,
                    0, 0,
                    bbiti1 | bbiti2 | 0xbf, bbiti1,
                    rPixX, hwKSStamp, lmRData, kerFitThresh)

                if check != 0.0:
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                            nr2 = l + rPixX * k
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                lmRData[nr2] = int(lmRData[nr2]) | bbiti2

                    # ciStamps[niS]['xss'][nss] = xmax
                    # ciStamps[niS]['yss'][nss] = ymax
                    # ciStamps[niS]['nss'] += 1
                    lciXss[nss] = xmax
                    lciYss[nss] = ymax
                    lciNss += 1

    return (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
            lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
            lmRData)


def get_background_numpy(xi, yi, kernelSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY):
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
    return background


@numba.jit(nopython=True)
def background_loop_jit(oRData1d, kernelSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel):
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
def get_final_stamp_sig_numpy(saXss, saYss, saSscnt, si, imDiff, imNoise, fwKSStamp, hwKSStamp, rPixX, mRData):
    # def get_final_stamp_sig_numpy(sa, si, imDiff, imNoise, fwKSStamp, hwKSStamp, rPixX, mRData):
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


def make_kernel_numpy(xi, yi, kernelSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel,
                      kernel_vec, kernel_coeffs, kernel):
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


def lubksb_numpy(a, n, indx, b):
    ii = 0
    for i in range(1, n + 1):
        ip = int(indx[i])
        sum_val = b[ip]
        b[ip] = b[i]
        if ii:
            for j in range(ii, i):
                sum_val -= a[i, j] * b[j]
        elif sum_val != 0.0:
            ii = i
        b[i] = sum_val
    for i in range(n, 0, -1):
        sum_val = b[i]
        for j in range(i + 1, n + 1):
            sum_val -= a[i, j] * b[j]
        b[i] = sum_val / a[i, i]


def kernel_vector_pca_numpy(n, deg_x, deg_y, ig, fwKernel, PCA, kernel_vec):
    vector = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    for i in range(fwKernel):
        for j in range(fwKernel):
            vector[i + fwKernel * j] = float(PCA[n][i + fwKernel * j])
    ren = 0
    if n > 0:
        kernel0 = kernel_vec[0]
        for i in range(fwKernel * fwKernel):
            vector[i] -= kernel0[i]
        ren = 1
    return vector, ren


def kernel_vector_numpy(n, deg_x, deg_y, ig, usePCA, fwKernel, hwKernel,
                        sigma_gauss, filter_x, filter_y, kernel_vec, PCA):
    if usePCA:
        vec, ren = kernel_vector_pca_numpy(n, deg_x, deg_y, ig, fwKernel, PCA, kernel_vec)
        return vec, ren

    vector = np.zeros(fwKernel * fwKernel, dtype=np.float64)
    dx = (deg_x // 2) * 2 - deg_x
    dy = (deg_y // 2) * 2 - deg_y
    sum_x = 0.0
    sum_y = 0.0
    ren = 0

    for ix in range(fwKernel):
        x = float(ix - hwKernel)
        k = ix + n * fwKernel
        qe = math.exp(-x * x * float(sigma_gauss[ig]))
        filter_x[k] = qe * math.pow(x, deg_x)
        filter_y[k] = qe * math.pow(x, deg_y)
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


def get_kernel_vec_numpy(ngauss, deg_fixe, usePCA, fwKernel, hwKernel,
                         sigma_gauss, filter_x, filter_y, PCA):
    kernel_vec = []
    nvec = 0
    for ig in range(ngauss):
        for idegx in range(int(deg_fixe[ig]) + 1):
            for idegy in range(int(deg_fixe[ig]) - idegx + 1):
                vec, ren = kernel_vector_numpy(nvec, idegx, idegy, ig, usePCA,
                                               fwKernel, hwKernel, sigma_gauss,
                                               filter_x, filter_y, kernel_vec, PCA)
                kernel_vec.append(vec)
                nvec += 1
    return kernel_vec


def ludcmp_numpy(a, n, indx):
    TINY = 1.0e-20
    d = 1.0
    vv = np.zeros(n + 1, dtype=np.float64)

    for i in range(1, n + 1):
        big = 0.0
        for j in range(1, n + 1):
            temp2 = abs(a[i, j])
            if temp2 > big:
                big = temp2
        if big == 0.0:
            return 1, d
        vv[i] = 1.0 / big

    imax = 0
    for j in range(1, n + 1):
        for i in range(1, j):
            sum_val = a[i, j]
            for k in range(1, i):
                sum_val -= a[i, k] * a[k, j]
            a[i, j] = sum_val

        big = 0.0
        for i in range(j, n + 1):
            sum_val = a[i, j]
            for k in range(1, j):
                sum_val -= a[i, k] * a[k, j]
            a[i, j] = sum_val
            dum = vv[i] * abs(sum_val)
            if dum >= big:
                big = dum
                imax = i

        if j != imax:
            for k in range(1, n + 1):
                dum = a[imax, k]
                a[imax, k] = a[j, k]
                a[j, k] = dum
            d = -d
            vv[imax] = vv[j]

        indx[j] = imax
        if a[j, j] == 0.0:
            a[j, j] = TINY
        if j != n:
            dum = 1.0 / a[j, j]
            for i in range(j + 1, n + 1):
                a[i, j] *= dum

    return 0, d


def xy_conv_stamp_numpy(stamp, image, n, ren, usePCA, fwKSStamp, fwKernel,
                         hwKSStamp, hwKernel, rPixX, filter_x, filter_y, temp, PCA):
    if usePCA:
        xy_conv_stamp_pca_numpy(stamp, image, n, ren, fwKSStamp, hwKSStamp,
                                hwKernel, fwKernel, rPixX, PCA)
        return

    xi = int(stamp['xss'][stamp['sscnt']])
    yi = int(stamp['yss'][stamp['sscnt']])

    sub_width = fwKSStamp + fwKernel - 1

    for i in range(xi - hwKSStamp - hwKernel, xi + hwKSStamp + hwKernel + 1):
        for j in range(yi - hwKSStamp, yi + hwKSStamp + 1):
            xij = i - xi + sub_width // 2 + sub_width * (j - yi + hwKSStamp)
            temp[xij] = 0.0
            for yc in range(-hwKernel, hwKernel + 1):
                temp[xij] += float(image[i + rPixX * (j + yc)]) * filter_y[hwKernel - yc + n * fwKernel]

    for j in range(-hwKSStamp, hwKSStamp + 1):
        for i in range(-hwKSStamp, hwKSStamp + 1):
            xij = i + hwKSStamp + fwKSStamp * (j + hwKSStamp)
            stamp['vectors'][n][xij] = 0.0
            for xc in range(-hwKernel, hwKernel + 1):
                stamp['vectors'][n][xij] += temp[i + xc + sub_width // 2 + sub_width * (j + hwKSStamp)] * filter_x[hwKernel - xc + n * fwKernel]

    if ren:
        for i in range(fwKSStamp * fwKSStamp):
            stamp['vectors'][n][i] -= stamp['vectors'][0][i]


@numba.jit(nopython=True)
def xy_conv_stamp_fast_numba_kernel(image, filterX, filterY, xi, yi,
                            fwKSStamp, fwKernel, hwKSStamp, hwKernel,
                            rPixX, nvecTotal, renFlags,
                            out_vectors):
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel

    kernels = np.zeros((nvecTotal, fwSqKernel), dtype=np.float64)
    for n in range(nvecTotal):
        for jc in range(fwKernel):
            for ic in range(fwKernel):
                fy_idx = fwKernel - 1 - jc
                fx_idx = fwKernel - 1 - ic
                kernels[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]

    for n in range(nvecTotal):
        for fsi in range(fwSqStamp):
            out_vectors[n, fsi] = 0.0

    for fsi in range(fwSqStamp):
        fsi_x = fsi % fwKSStamp
        fsi_y = fsi // fwKSStamp
        img_base_x = xi - hwKSStamp + fsi_x - hwKernel
        img_base_y = yi - hwKSStamp + fsi_y - hwKernel
        for fki in range(fwSqKernel):
            kx = fki % fwKernel
            ky = fki // fwKernel
            img_x = img_base_x + kx
            img_y = img_base_y + ky
            img_val = image[img_x + rPixX * img_y]
            for n in range(nvecTotal):
                out_vectors[n, fsi] += kernels[n, fki] * img_val

    for n in range(nvecTotal):
        if renFlags[n]:
            for fsi in range(fwSqStamp):
                out_vectors[n, fsi] -= out_vectors[0, fsi]


def xy_conv_stamp_fast_numba(saVectors, saXss, saYss, saSscnt, si, image, nvecTotal, renFlags, usePCA, fwKSStamp, fwKernel,
                              hwKSStamp, hwKernel, rPixX, filterX, filterY, temp, PCA):
    # def xy_conv_stamp_fast_numba(sa, si, image, nvecTotal, renFlags, usePCA, fwKSStamp, fwKernel,
    #                               hwKSStamp, hwKernel, rPixX, filterX, filterY, temp, PCA):
    if usePCA:
        for n in range(nvecTotal):
            xy_conv_stamp_pca_numpy(saVectors, saXss, saYss, saSscnt, si, image, n, renFlags[n], fwKSStamp, hwKSStamp,
                                    hwKernel, fwKernel, rPixX, PCA)
        return

    xi = int(saXss[si, saSscnt[si]])
    yi = int(saYss[si, saSscnt[si]])
    img_flat = np.asarray(image, dtype=np.float64).ravel()
    fx = np.asarray(filterX, dtype=np.float64)
    fy = np.asarray(filterY, dtype=np.float64)
    rflags = np.array(renFlags, dtype=np.int32)
    out_vec = np.zeros((nvecTotal, fwKSStamp * fwKSStamp), dtype=np.float64)

    xy_conv_stamp_fast_numba_kernel(img_flat, fx, fy, xi, yi,
                            fwKSStamp, fwKernel, hwKSStamp, hwKernel,
                            rPixX, nvecTotal, rflags, out_vec)

    for n in range(nvecTotal):
        saVectors[si, n, :fwKSStamp * fwKSStamp] = out_vec[n]


# def xy_conv_stamp_fast_numpy_blas(sa, si, image, nvecTotal, renFlags, usePCA, fwKSStamp, fwKernel,
#                                    hwKSStamp, hwKernel, rPixX, filterX, filterY, temp, PCA):
#     if usePCA:
#         for n in range(nvecTotal):
#             xy_conv_stamp_pca_numpy(sa, si, image, n, renFlags[n], fwKSStamp, hwKSStamp,
#                                     hwKernel, fwKernel, rPixX, PCA)
#         return
# 
#     xi = int(sa.xss[si, sa.sscnt[si]])
#     yi = int(sa.yss[si, sa.sscnt[si]])
# 
#     iCenters = np.arange(xi - hwKSStamp, xi + hwKSStamp + 1)
#     jCenters = np.arange(yi - hwKSStamp, yi + hwKSStamp + 1)
#     xcOffsets = np.arange(-hwKernel, hwKernel + 1)
#     ycOffsets = np.arange(-hwKernel, hwKernel + 1)
# 
#     xPart = iCenters[None, :, None, None] + xcOffsets[None, None, None, :]
#     yPart = jCenters[:, None, None, None] + ycOffsets[None, None, :, None]
#     flatIdx = (xPart + rPixX * yPart).ravel()
# 
#     imageArr = np.asarray(image, dtype=np.float32)
#     patches = imageArr[flatIdx].reshape(fwKSStamp * fwKSStamp, fwKernel * fwKernel)
#     patches64 = patches.astype(np.float64)
# 
#     fxArr = np.asarray(filterX, dtype=np.float64)
#     fyArr = np.asarray(filterY, dtype=np.float64)
#     fxAll = fxArr[:nvecTotal * fwKernel].reshape(nvecTotal, fwKernel)[:, ::-1]
#     fyAll = fyArr[:nvecTotal * fwKernel].reshape(nvecTotal, fwKernel)[:, ::-1]
# 
#     kernels = fyAll[:, :, None] * fxAll[:, None, :]
#     kernels = kernels.reshape(nvecTotal, fwKernel * fwKernel)
# 
#     result = kernels @ patches64.T
# 
#     for n in range(nvecTotal):
#         if renFlags[n]:
#             result[n] -= result[0]
#         sa.vectors[si, n, :fwKSStamp * fwKSStamp] = result[n]


# def xy_conv_stamp_fast_numpy_deprecated_1781965833(sa, si, image, nvecTotal, renFlags, usePCA, fwKSStamp, fwKernel,
#                                   hwKSStamp, hwKernel, rPixX, filterX, filterY, temp, PCA):
#     if usePCA:
#         return xy_conv_stamp_fast_numba(sa, si, image, nvecTotal, renFlags, usePCA,
#                                          fwKSStamp, fwKernel, hwKSStamp, hwKernel,
#                                          rPixX, filterX, filterY, temp, PCA)
# 
#     xi = int(sa.xss[si, sa.sscnt[si]])
#     yi = int(sa.yss[si, sa.sscnt[si]])
#     img_flat = np.asarray(image, dtype=np.float64).ravel()
#     fx = np.asarray(filterX, dtype=np.float64)
#     fy = np.asarray(filterY, dtype=np.float64)
#     rflags = np.array(renFlags, dtype=np.int32)
#     out_vec = np.zeros((nvecTotal, fwKSStamp * fwKSStamp), dtype=np.float64)
# 
#     xy_conv_stamp_fast_numba_kernel(img_flat, fx, fy, xi, yi,
#                             fwKSStamp, fwKernel, hwKSStamp, hwKernel,
#                             rPixX, nvecTotal, rflags, out_vec)
# 
#     for n in range(nvecTotal):
#         sa.vectors[si, n, :fwKSStamp * fwKSStamp] = out_vec[n]


def xy_conv_stamp_pca_numpy(sa, si, image, n, ren, fwKSStamp, hwKSStamp,
                             hwKernel, fwKernel, rPixX, PCA):
    # def xy_conv_stamp_pca_numpy(stamp, image, n, ren, fwKSStamp, hwKSStamp,
    #                              hwKernel, fwKernel, rPixX, PCA):
    # xi = int(stamp['xss'][stamp['sscnt']])
    xi = int(sa.xss[si, sa.sscnt[si]])
    # yi = int(stamp['yss'][stamp['sscnt']])
    yi = int(sa.yss[si, sa.sscnt[si]])

    for j in range(yi - hwKSStamp, yi + hwKSStamp + 1):
        for i in range(xi - hwKSStamp, xi + hwKSStamp + 1):
            xij = i - (xi - hwKSStamp) + fwKSStamp * (j - (yi - hwKSStamp))
            # stamp['vectors'][n][xij] = 0.0
            sa.vectors[si, n, xij] = 0.0
            for yc in range(-hwKernel, hwKernel + 1):
                for xc in range(-hwKernel, hwKernel + 1):
                    val_img = np.float32(image[(i + xc) + rPixX * (j + yc)])
                    val_pca = np.float32(PCA[n][(xc + hwKernel) + fwKernel * (yc + hwKernel)])
                    # stamp['vectors'][n][xij] += float(val_img * val_pca)
                    sa.vectors[si, n, xij] += float(val_img * val_pca)

    if ren:
        for i in range(fwKSStamp * fwKSStamp):
            # stamp['vectors'][n][i] -= stamp['vectors'][0][i]
            sa.vectors[si, n, i] -= sa.vectors[si, 0, i]


# def build_matrix0_numpy(stamp, nCompKer, kerOrder, bgOrder, fwKSStamp):
#     ncomp1 = nCompKer
#     pixStamp = fwKSStamp * fwKSStamp
#     vec = stamp['vectors']
# 
#     for i in range(ncomp1):
#         for j in range(i + 1):
#             q = 0.0
#             for k in range(pixStamp):
#                 q += vec[i][k] * vec[j][k]
#             stamp['mat'][i + 1][j + 1] = q
# 
#     ivecbg = 0
#     for i1 in range(ncomp1):
#         ivecbg = ncomp1
#         p0 = 0.0
#         for k in range(pixStamp):
#             p0 += vec[i1][k] * vec[ivecbg][k]
#         stamp['mat'][ncomp1 + 1][i1 + 1] = p0
# 
#     q = 0.0
#     for k in range(pixStamp):
#         q += vec[ivecbg][k] * vec[ncomp1][k]
#     stamp['mat'][ncomp1 + 1][ncomp1 + 1] = q


@numba.jit(nopython=True)
def build_matrix0_jit(vectors, mat, nCompKer, kerOrder, bgOrder, fwKSStamp):
    ncomp1 = nCompKer
    pixStamp = fwKSStamp * fwKSStamp

    for i in range(ncomp1):
        for j in range(i + 1):
            q = 0.0
            for k in range(pixStamp):
                q += vectors[i, k] * vectors[j, k]
            mat[i + 1, j + 1] = q

    ivecbg = ncomp1
    for i1 in range(ncomp1):
        p0 = 0.0
        for k in range(pixStamp):
            p0 += vectors[i1, k] * vectors[ivecbg, k]
        mat[ncomp1 + 1, i1 + 1] = p0

    q = 0.0
    for k in range(pixStamp):
        q += vectors[ivecbg, k] * vectors[ncomp1, k]
    mat[ncomp1 + 1, ncomp1 + 1] = q


@numba.jit(nopython=True)
def build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
                     wxy, matrix,
                     nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec, mat_size, pixStamp):
    rPixX2 = np.float64(0.5 * rPixX)
    rPixY2 = np.float64(0.5 * rPixY)

    for istamp in range(nS):
        if valid_mask[istamp] == 0:
            continue

        xstamp = all_x[istamp]
        ystamp = all_y[istamp]
        fx = np.float64((np.float64(xstamp) - rPixX2) / rPixX2)
        fy = np.float64((np.float64(ystamp) - rPixY2) / rPixY2)

        kk = 0
        a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[istamp, kk] = a1 * a2
                kk += 1
                a2 *= fy
            a1 *= fx

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
                p0 = 0.0
                for kk in range(pixStamp):
                    p0 += all_vectors[istamp, i1, kk] * all_vectors[istamp, ivecbg, kk]

                for i2 in range(ncomp2):
                    jj = (i1 - 1) * ncomp2 + i2 + 1
                    matrix[ii + 1, jj + 1] += p0 * wxy[istamp, i2]

            p0 = 0.0
            for kk in range(pixStamp):
                p0 += all_vectors[istamp, 0, kk] * all_vectors[istamp, ivecbg, kk]
            matrix[ii + 1, 1] += p0

            for jbg in range(ibg + 1):
                q = 0.0
                for kk in range(pixStamp):
                    q += all_vectors[istamp, ivecbg, kk] * all_vectors[istamp, ncomp1 + jbg + 1, kk]
                matrix[ii + 1, ncomp + jbg + 2] += q


def build_matrix0_numpy(saVectors, saMat, si, nCompKer, kerOrder, bgOrder, fwKSStamp):
    # def build_matrix0_numpy(sa, si, nCompKer, kerOrder, bgOrder, fwKSStamp):
    #     build_matrix0_jit(sa.vectors[si], sa.mat[si], nCompKer, kerOrder, bgOrder, fwKSStamp)
    build_matrix0_jit(saVectors[si], saMat[si], nCompKer, kerOrder, bgOrder, fwKSStamp)


# def build_scprod0_numpy(stamp, image, nCompKer, kerOrder, bgOrder,
#                          fwKSStamp, hwKSStamp, rPixX):
#     ncomp1 = nCompKer
#     vec = stamp['vectors']
#     xi = int(stamp['xss'][stamp['sscnt']])
#     yi = int(stamp['yss'][stamp['sscnt']])
# 
#     for i1 in range(ncomp1):
#         p0 = 0.0
#         for xc in range(-hwKSStamp, hwKSStamp + 1):
#             for yc in range(-hwKSStamp, hwKSStamp + 1):
#                 k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
#                 p0 += vec[i1][k] * float(image[xc + xi + rPixX * (yc + yi)])
#         stamp['scprod'][i1 + 1] = p0
# 
#     q = 0.0
#     for xc in range(-hwKSStamp, hwKSStamp + 1):
#         for yc in range(-hwKSStamp, hwKSStamp + 1):
#             k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
#             q += vec[ncomp1][k] * float(image[xc + xi + rPixX * (yc + yi)])
#     stamp['scprod'][ncomp1 + 1] = q


@numba.jit(nopython=True)
def build_scprod0_jit(vectors, scprod, image, nCompKer, xi, yi, fwKSStamp, hwKSStamp, rPixX):
    ncomp1 = nCompKer

    for i1 in range(ncomp1):
        p0 = 0.0
        for xc in range(-hwKSStamp, hwKSStamp + 1):
            for yc in range(-hwKSStamp, hwKSStamp + 1):
                k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                p0 += vectors[i1, k] * image[xc + xi + rPixX * (yc + yi)]
        scprod[i1 + 1] = p0

    q = 0.0
    for xc in range(-hwKSStamp, hwKSStamp + 1):
        for yc in range(-hwKSStamp, hwKSStamp + 1):
            k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
            q += vectors[ncomp1, k] * image[xc + xi + rPixX * (yc + yi)]
    scprod[ncomp1 + 1] = q


@numba.jit(nopython=True)
def build_scprod_jit(all_vectors, all_scprod, valid_mask, all_x, all_y,
                     wxy, image_flat, kernelSol,
                     nS, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec, pixStamp):
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

        for ibg in range(nbg_vec):
            q = 0.0
            for xc in range(-hwKSStamp, hwKSStamp + 1):
                for yc in range(-hwKSStamp, hwKSStamp + 1):
                    k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                    q += all_vectors[istamp, ncomp1 + ibg + 1, k] * image_flat[xc + xi + rPixX * (yc + yi)]
            kernelSol[ncomp + ibg + 2] += q


def build_scprod0_numpy(saVectors, saScprod, saXss, saYss, saSscnt, si, image, nCompKer, kerOrder, bgOrder,
                         fwKSStamp, hwKSStamp, rPixX):
    # def build_scprod0_numpy(sa, si, image, nCompKer, kerOrder, bgOrder,
    #                          fwKSStamp, hwKSStamp, rPixX):
    xi = int(saXss[si, saSscnt[si]])
    yi = int(saYss[si, saSscnt[si]])
    image_arr = np.asarray(image, dtype=np.float64)
    build_scprod0_jit(saVectors[si], saScprod[si], image_arr,
                      nCompKer, xi, yi, fwKSStamp, hwKSStamp, rPixX)


# def build_matrix_numpy(sa, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, verbose, wxy):
def build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, verbose, nC=None, nKSStamps=None):
    # def build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, verbose, wxy):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    # valid_mask = np.zeros(nS, dtype=np.int32)
    # all_x = np.zeros(nS, dtype=np.int64)
    # all_y = np.zeros(nS, dtype=np.int64)
    # n_valid = 0
    # for i in range(nS):
    #     if stamps_dicts[i]['sscnt'] < stamps_dicts[i]['nss']:
    #         valid_mask[i] = 1
    #         sscnt = stamps_dicts[i]['sscnt']
    #         all_x[i] = int(stamps_dicts[i]['xss'][sscnt])
    #         all_y[i] = int(stamps_dicts[i]['yss'][sscnt])
    #         n_valid += 1
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
    all_mat = saMat[:, :nC_valid, :nC_valid].copy()
    nvec_total = nCompKer + nbg_vec
    # all_vectors = np.zeros((nS, nvec_total, pixStamp), dtype=np.float64)
    all_vectors = saVectors[:, :nvec_total, :].copy()
    # for i in range(nS):
    #     if valid_mask[i]:
    #         all_mat[i] = stamps_dicts[i]['mat'][:nC_valid, :nC_valid]
    #         all_vectors[i] = np.asarray(stamps_dicts[i]['vectors'])[:nvec_total]

    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)

    build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
                     wxy, matrix,
                     nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec, mat_size, pixStamp)

    for i in range(mat_size):
        for j in range(i + 1):
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]

    return matrix, wxy

# def build_matrix_numpy_old(stamps_dicts, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, verbose, wxy):
#     ncomp1 = nCompKer - 1
#     ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
#     ncomp = ncomp1 * ncomp2
#     nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
# 
#     pixStamp = fwKSStamp * fwKSStamp
#     rPixX2 = float(np.float32(0.5 * rPixX))
#     rPixY2 = float(np.float32(0.5 * rPixY))
# 
#     mat_size = ncomp1 * ncomp2 + nbg_vec + 1
# 
#     matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
# 
#     for i in range(nS):
#         for j in range(ncomp2):
#             wxy[i, j] = 0.0
# 
#     istamp = 0
#     while istamp < nS:
#         while stamps_dicts[istamp]['sscnt'] >= stamps_dicts[istamp]['nss']:
#             istamp += 1
#             if istamp >= nS:
#                 break
#         if istamp >= nS:
#             break
# 
#         vec = stamps_dicts[istamp]['vectors']
#         sscnt = stamps_dicts[istamp]['sscnt']
#         xstamp = int(stamps_dicts[istamp]['xss'][sscnt])
#         ystamp = int(stamps_dicts[istamp]['yss'][sscnt])
#         fx = float(np.float32(np.float32(xstamp) - np.float32(rPixX2)) / np.float32(rPixX2))
#         fy = float(np.float32(np.float32(ystamp) - np.float32(rPixY2)) / np.float32(rPixY2))
# 
#         k = 0
#         a1 = 1.0
#         for ideg1 in range(kerOrder + 1):
#             a2 = 1.0
#             for ideg2 in range(kerOrder - ideg1 + 1):
#                 wxy[istamp, k] = a1 * a2
#                 k += 1
#                 a2 *= fy
#             a1 *= fx
# 
#         matrix0 = stamps_dicts[istamp]['mat']
#         for i in range(ncomp):
#             i1 = i // ncomp2
#             i2 = i - i1 * ncomp2
# 
#             for j in range(i + 1):
#                 j1 = j // ncomp2
#                 j2 = j - j1 * ncomp2
# 
#                 matrix[i + 2, j + 2] += wxy[istamp, i2] * wxy[istamp, j2] * matrix0[i1 + 2, j1 + 2]
# 
#         matrix[1, 1] += matrix0[1, 1]
#         for i in range(ncomp):
#             i1 = i // ncomp2
#             i2 = i - i1 * ncomp2
#             matrix[i + 2, 1] += wxy[istamp, i2] * matrix0[i1 + 2, 1]
# 
#         for ibg in range(nbg_vec):
#             i = ncomp + ibg + 1
#             ivecbg = ncomp1 + ibg + 1
#             for i1 in range(1, ncomp1 + 1):
#                 p0 = 0.0
#                 for k in range(pixStamp):
#                     p0 += vec[i1, k] * vec[ivecbg, k]
# 
#                 for i2 in range(ncomp2):
#                     jj = (i1 - 1) * ncomp2 + i2 + 1
#                     matrix[i + 1, jj + 1] += p0 * wxy[istamp, i2]
# 
#             p0 = 0.0
#             for k in range(pixStamp):
#                 p0 += vec[0, k] * vec[ivecbg, k]
#             matrix[i + 1, 1] += p0
# 
#             for jbg in range(ibg + 1):
#                 q = 0.0
#                 for k in range(pixStamp):
#                     q += vec[ivecbg, k] * vec[ncomp1 + jbg + 1, k]
#                 matrix[i + 1, ncomp + jbg + 2] += q
# 
#         istamp += 1
# 
#     for i in range(mat_size):
#         for j in range(i + 1):
#             matrix[j + 1, i + 1] = matrix[i + 1, j + 1]
# 
#     return matrix


# def build_scprod_numpy(sa, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy):
def build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=None):
    # def build_scprod_numpy(stamps_dicts, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp

    # valid_mask = np.zeros(nS, dtype=np.int32)
    # all_x = np.zeros(nS, dtype=np.int64)
    # all_y = np.zeros(nS, dtype=np.int64)
    # n_valid = 0
    # for i in range(nS):
    #     if stamps_dicts[i]['sscnt'] < stamps_dicts[i]['nss']:
    #         valid_mask[i] = 1
    #         sscnt = stamps_dicts[i]['sscnt']
    #         all_x[i] = int(stamps_dicts[i]['xss'][sscnt])
    #         all_y[i] = int(stamps_dicts[i]['yss'][sscnt])
    #         n_valid += 1
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
    all_scprod = saScprod[:, :nCompKer + 1].copy()
    nvec_total = nCompKer + nbg_vec
    # all_vectors = np.zeros((nS, nvec_total, pixStamp), dtype=np.float64)
    all_vectors = saVectors[:, :nvec_total, :].copy()
    # for i in range(nS):
    #     if valid_mask[i]:
    #         all_scprod[i] = stamps_dicts[i]['scprod'][:nCompKer + 1]
    #         all_vectors[i] = np.asarray(stamps_dicts[i]['vectors'])[:nvec_total]

    image_arr = np.asarray(image, dtype=np.float64)
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)

    build_scprod_jit(all_vectors, all_scprod, valid_mask, all_x, all_y,
                     wxy, image_arr, kernelSol,
                     nS, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec, pixStamp)

    return kernelSol

# def build_scprod_numpy_old(stamps_dicts, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy):
#     ncomp1 = nCompKer - 1
#     ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
#     ncomp = ncomp1 * ncomp2
#     nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
# 
#     kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
# 
#     istamp = 0
#     while istamp < nS:
#         while stamps_dicts[istamp]['sscnt'] >= stamps_dicts[istamp]['nss']:
#             istamp += 1
#             if istamp >= nS:
#                 break
#         if istamp >= nS:
#             break
# 
#         vec = stamps_dicts[istamp]['vectors']
#         sscnt = stamps_dicts[istamp]['sscnt']
#         xi = int(stamps_dicts[istamp]['xss'][sscnt])
#         yi = int(stamps_dicts[istamp]['yss'][sscnt])
# 
#         p0 = stamps_dicts[istamp]['scprod'][1]
#         kernelSol[1] += p0
# 
#         for i1 in range(1, ncomp1 + 1):
#             p0 = stamps_dicts[istamp]['scprod'][i1 + 1]
#             for i2 in range(ncomp2):
#                 ii = (i1 - 1) * ncomp2 + i2 + 1
#                 kernelSol[ii + 1] += p0 * wxy[istamp, i2]
# 
#         for ibg in range(nbg_vec):
#             q = 0.0
#             for xc in range(-hwKSStamp, hwKSStamp + 1):
#                 for yc in range(-hwKSStamp, hwKSStamp + 1):
#                     k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
#                     q += vec[ncomp1 + ibg + 1, k] * float(image[xc + xi + rPixX * (yc + yi)])
#             kernelSol[ncomp + ibg + 2] += q
# 
#         istamp += 1
# 
#     return kernelSol


@numba.jit(nopython=True)
def make_model_jit(vectors, kernelSol, csModel, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp,
                   xi, yi):
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)

    fwSq = fwKSStamp * fwKSStamp

    coeff = kernelSol[1]
    for i in range(fwSq):
        csModel[i] = coeff * vectors[0, i]

    k = 2
    for i1 in range(1, nCompKer):
        coeff = 0.0
        ax = 1.0
        for ix in range(kerOrder + 1):
            ay = 1.0
            for iy in range(kerOrder - ix + 1):
                coeff += kernelSol[k] * ax * ay
                k += 1
                ay *= yf
            ax *= xf

        for i in range(fwSq):
            csModel[i] += coeff * vectors[i1, i]


def make_model_numpy(sa, si, kernelSol, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp):
    # def make_model_numpy(stamp, kernelSol, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp):
    # xi = int(stamp['xss'][stamp['sscnt']])
    xi = int(sa.xss[si, sa.sscnt[si]])
    # yi = int(stamp['yss'][stamp['sscnt']])
    yi = int(sa.yss[si, sa.sscnt[si]])

    fwSq = fwKSStamp * fwKSStamp
    csModel = np.zeros(fwSq, dtype=np.float64)
    # vectors = np.asarray(stamp['vectors'], dtype=np.float64)
    vectors = np.asarray(sa.vectors[si], dtype=np.float64)
    kSol = np.asarray(kernelSol, dtype=np.float64)
    make_model_jit(vectors, kSol, csModel, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp,
                   xi, yi)
    return csModel.astype(np.float32)

# # 旧版本 make_model_numpy — 已替换为 jit 版本
# def make_model_numpy(stamp, kernelSol, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp):
#     xi = int(stamp['xss'][stamp['sscnt']])
#     yi = int(stamp['yss'][stamp['sscnt']])
# 
#     xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
#     yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
# 
#     fwSq = fwKSStamp * fwKSStamp
#     csModel = np.zeros(fwSq, dtype=np.float32)
# 
#     vector = stamp['vectors'][0]
#     coeff = kernelSol[1]
#     for i in range(fwSq):
#         csModel[i] += np.float32(coeff * vector[i])
# 
#     k = 2
#     for i1 in range(1, nCompKer):
#         vector = stamp['vectors'][i1]
#         coeff = 0.0
#         ax = 1.0
#         for ix in range(kerOrder + 1):
#             ay = 1.0
#             for iy in range(kerOrder - ix + 1):
#                 coeff += kernelSol[k] * ax * ay
#                 k += 1
#                 ay *= yf
#             ax *= xf
# 
#         for i in range(fwSq):
#             csModel[i] += np.float32(coeff * vector[i])
# 
#     return csModel


@numba.jit(nopython=True)
def fill_stamp_numba_kernel_local(
    image, imRef, filterX, filterY,
    xi, yi, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
    rPixX, rPixY, nCompKer, kerOrder, bgOrder,
    nvec, renFlags_arr, fillVal, mRData1d, verbose,
    out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val):

    LOCAL_FLAG_INPUT_ISBAD = 0x80
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
# 
    # ========== Step 1: inlined xy_conv_stamp_fast_numba_kernel ==========
    kernels = np.zeros((nvec, fwSqKernel), dtype=np.float64)
    for n in range(nvec):
        for jc in range(fwKernel):
            for ic in range(fwKernel):
                fy_idx = fwKernel - 1 - jc
                fx_idx = fwKernel - 1 - ic
                kernels[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]

    for n in range(nvec):
        for fsi in range(fwSqStamp):
            out_vectors[n, fsi] = 0.0

    for fsi in range(fwSqStamp):
        fsi_x = fsi % fwKSStamp
        fsi_y = fsi // fwKSStamp
        img_base_x = xi - hwKSStamp + fsi_x - hwKernel
        img_base_y = yi - hwKSStamp + fsi_y - hwKernel
        for fki in range(fwSqKernel):
            kx = fki % fwKernel
            ky = fki // fwKernel
            img_x = img_base_x + kx
            img_y = img_base_y + ky
            img_val = image[img_x + rPixX * img_y]
            for n in range(nvec):
                out_vectors[n, fsi] += kernels[n, fki] * img_val

    for n in range(nvec):
        if renFlags_arr[n]:
            for fsi in range(fwSqStamp):
                out_vectors[n, fsi] -= out_vectors[0, fsi]
# 
    # ========== Step 2: inlined cut_sstamp_numpy ==========
    for fsi in range(fwSqStamp):
        out_krefArea[fsi] = fillVal

    sumVal = 0.0
    for y_offset in range(fwKSStamp):
        img_y = yi - hwKSStamp + y_offset
        for x_offset in range(fwKSStamp):
            img_x = xi - hwKSStamp + x_offset
            k = img_x + rPixX * img_y
            dpt = imRef[k]
            out_krefArea[x_offset + y_offset * fwKSStamp] = dpt
            if (mRData1d[k] & LOCAL_FLAG_INPUT_ISBAD) == 0:
                sumVal += abs(dpt)

    out_sum_val[0] = sumVal

    # ========== Step 3: background vectors ==========
    rPixX2 = np.float64(0.5 * rPixX)
    rPixY2 = np.float64(0.5 * rPixY)
    for y_offset in range(fwKSStamp):
        j = yi - hwKSStamp + y_offset
        yf = (j - rPixY2) / rPixY2
        for x_offset in range(fwKSStamp):
            i = xi - hwKSStamp + x_offset
            xf = (i - rPixX2) / rPixX2
            ipix = x_offset + y_offset * fwKSStamp
            ax = 1.0
            nv = nvec
            for idegx in range(bgOrder + 1):
                ay = 1.0
                for idegy in range(bgOrder - idegx + 1):
                    out_vectors[nv, ipix] = ax * ay
                    ay *= yf
                    nv += 1
                ax *= xf

    # ========== Step 4: inlined build_matrix0_jit ==========
    ncomp1 = nCompKer
    pixStamp = fwSqStamp

    for i in range(ncomp1):
        for j in range(i + 1):
            q = 0.0
            for k in range(pixStamp):
                q += out_vectors[i, k] * out_vectors[j, k]
            out_mat[i + 1, j + 1] = q

    ivecbg = ncomp1
    for i1 in range(ncomp1):
        p0 = 0.0
        for k in range(pixStamp):
            p0 += out_vectors[i1, k] * out_vectors[ivecbg, k]
        out_mat[ncomp1 + 1, i1 + 1] = p0

    q = 0.0
    for k in range(pixStamp):
        q += out_vectors[ivecbg, k] * out_vectors[ncomp1, k]
    out_mat[ncomp1 + 1, ncomp1 + 1] = q

    # ========== Step 5: inlined build_scprod0_jit ==========
    for i1 in range(ncomp1):
        p0 = 0.0
        for xc in range(-hwKSStamp, hwKSStamp + 1):
            for yc in range(-hwKSStamp, hwKSStamp + 1):
                k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                p0 += out_vectors[i1, k] * out_krefArea[k]
        out_scprod[i1 + 1] = p0

    q = 0.0
    for xc in range(-hwKSStamp, hwKSStamp + 1):
        for yc in range(-hwKSStamp, hwKSStamp + 1):
            k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
            q += out_vectors[ncomp1, k] * out_krefArea[k]
    out_scprod[ncomp1 + 1] = q

    return 0


def fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, saX0, saY0, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
                     usePCA, filter_x, filter_y, PCA, fillVal, mRData):
    # def fill_stamp_numpy(stamp_dict, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
    #                      hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
    #                      usePCA, filter_x, filter_y, PCA, fillVal, mRData):
    # rPixX2 = float(np.float32(0.5 * rPixX))
    # rPixY2 = float(np.float32(0.5 * rPixY))
    #
    # # if stamp_dict['sscnt'] >= stamp_dict['nss']:
    # if saSscnt[si] >= saNss[si]:
    #     return 1
    #
    # sub_width = fwKSStamp + fwKernel - 1
    # temp = np.zeros(sub_width * fwKSStamp, dtype=np.float32)
    #
    # nvec = 0
    # renFlags = []
    # for ig in range(ngauss):
    #     for idegx in range(int(deg_fixe[ig]) + 1):
    #         for idegy in range(int(deg_fixe[ig]) - idegx + 1):
    #             ren = 0
    #             dx = (idegx // 2) * 2 - idegx
    #             dy = (idegy // 2) * 2 - idegy
    #             if dx == 0 and dy == 0 and nvec > 0:
    #                 ren = 1
    #             # xy_conv_stamp_numpy(stamp_dict, imConv, nvec, ren, usePCA,
    #             #                     fwKSStamp, fwKernel, hwKSStamp, hwKernel,
    #             #                     rPixX, filter_x, filter_y, temp, PCA)
    #             renFlags.append(ren)
    #             nvec += 1
    # # xy_conv_stamp_fast_numpy(stamp_dict, imConv, nvec, renFlags, usePCA,
    # #                           fwKSStamp, fwKernel, hwKSStamp, hwKernel,
    # #                           rPixX, filter_x, filter_y, temp, PCA)
    # xy_conv_stamp_fast_numba(sa, si, imConv, nvec, renFlags, usePCA,
    #                           fwKSStamp, fwKernel, hwKSStamp, hwKernel,
    #                           rPixX, filter_x, filter_y, temp, PCA)

    # # if cut_sstamp_numpy(stamp_dict, imRef, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose):
    # if cut_sstamp_numpy(sa, si, imRef, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose):
    #     return 1
    #
    # # xi = int(stamp_dict['xss'][stamp_dict['sscnt']])
    # xi = int(saXss[si, saSscnt[si]])
    # # yi = int(stamp_dict['yss'][stamp_dict['sscnt']])
    # yi = int(saYss[si, saSscnt[si]])
    # di = xi - hwKSStamp
    # dj = yi - hwKSStamp
    # for i in range(xi - hwKSStamp, xi + hwKSStamp + 1):
    #     xf = (i - rPixX2) / rPixX2
    #     for j in range(yi - hwKSStamp, yi + hwKSStamp + 1):
    #         yf = (j - rPixY2) / rPixY2
    #         ax = 1.0
    #         nv = nvec
    #         for idegx in range(bgOrder + 1):
    #             ay = 1.0
    #             for idegy in range(bgOrder - idegx + 1):
    #                 # stamp_dict['vectors'][nv][i - di + fwKSStamp * (j - dj)] = ax * ay
    #                 saVectors[si, nv, i - di + fwKSStamp * (j - dj)] = ax * ay
    #                 ay *= yf
    #                 nv += 1
    #             ax *= xf
    #
    # # build_matrix0_numpy(stamp_dict, nCompKer, kerOrder, bgOrder, fwKSStamp)
    # build_matrix0_numpy(sa, si, nCompKer, kerOrder, bgOrder, fwKSStamp)
    # # build_scprod0_numpy(stamp_dict, imRef, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX)
    # build_scprod0_numpy(sa, si, imRef, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX)

    # # ===== comparison: fill_stamp_numba vs fill_stamp_numpy =====
    # if verbose:
    #     nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
    #     fwSqStamp = fwKSStamp * fwKSStamp
    #     nC = nCompKer + nbg
    #
    #     cmp_vectors = np.zeros((nvec + nbg, fwSqStamp), dtype=np.float64)
    #     cmp_krefArea = np.zeros(fwSqStamp, dtype=np.float64)
    #     cmp_mat = np.zeros((nC, nC), dtype=np.float64)
    #     cmp_scprod = np.zeros(nC, dtype=np.float64)
    #     cmp_sum_val = np.zeros(1, dtype=np.float64)
    #
    #     xi_val = int(saXss[si, saSscnt[si]])
    #     yi_val = int(saYss[si, saSscnt[si]])
    #     img_flat = np.asarray(imConv, dtype=np.float64).ravel()
    #     imRef_flat = np.asarray(imRef, dtype=np.float64).ravel()
    #     fx = np.asarray(filter_x, dtype=np.float64)
    #     fy = np.asarray(filter_y, dtype=np.float64)
    #     rflags = np.array(renFlags, dtype=np.int32)
    #     mRData1d = mRData.ravel()
    #
    #     fill_stamp_numba_kernel(
    #         img_flat, imRef_flat, fx, fy,
    #         xi_val, yi_val, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
    #         rPixX, rPixY, nCompKer, kerOrder, bgOrder,
    #         nvec, rflags, fillVal, mRData1d, verbose,
    #         cmp_vectors, cmp_krefArea, cmp_mat, cmp_scprod, cmp_sum_val)
    #
    #     diff_vectors = np.max(np.abs(cmp_vectors - saVectors[si, :nvec + nbg, :fwSqStamp]))
    #     diff_krefArea = np.max(np.abs(cmp_krefArea - saKrefArea[si, :fwSqStamp]))
    #     diff_mat = np.max(np.abs(cmp_mat[:nCompKer + 2, :nCompKer + 2] - saMat[si, :nCompKer + 2, :nCompKer + 2]))
    #     diff_scprod = np.max(np.abs(cmp_scprod[:nCompKer + 2] - saScprod[si, :nCompKer + 2]))
    #
    #     import sys
    #     sys.stderr.write("fill_stamp compare si=%d: vectors_diff=%.6e krefArea_diff=%.6e mat_diff=%.6e scprod_diff=%.6e\n" % (
    #         si, diff_vectors, diff_krefArea, diff_mat, diff_scprod))
    #     sys.stderr.flush()
    #
    # return 0

    # ===== new: delegate to fill_stamp_numba =====
    result = fill_stamp_numba(saXss, saYss, saSscnt, saNss, saX0, saY0, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
                              hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
                              usePCA, filter_x, filter_y, PCA, fillVal, mRData)
    if isinstance(result, int):
        return result
    out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val = result
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
    fwSqStamp = fwKSStamp * fwKSStamp
    nC = nCompKer + 1
    saVectors[si, :nCompKer + nbg, :fwSqStamp] = out_vectors
    saKrefArea[si, :] = out_krefArea
    saMat[si, :nC + 1, :nC + 1] = out_mat
    saScprod[si, :nC + 1] = out_scprod
    saSumVal[si] = out_sum_val
    return 0


# def fill_stamp_numpy_deprecated_1781967154(sa, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
#                      hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
#                      usePCA, filter_x, filter_y, PCA, fillVal, mRData):
#     # def fill_stamp_numpy(stamp_dict, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
#     #                      hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
#     #                      usePCA, filter_x, filter_y, PCA, fillVal, mRData):
#     rPixX2 = float(np.float32(0.5 * rPixX))
#     rPixY2 = float(np.float32(0.5 * rPixY))
# 
#     # if stamp_dict['sscnt'] >= stamp_dict['nss']:
#     if sa.sscnt[si] >= sa.nss[si]:
#         return 1
# 
#     sub_width = fwKSStamp + fwKernel - 1
#     temp = np.zeros(sub_width * fwKSStamp, dtype=np.float32)
# 
#     nvec = 0
#     renFlags = []
#     for ig in range(ngauss):
#         for idegx in range(int(deg_fixe[ig]) + 1):
#             for idegy in range(int(deg_fixe[ig]) - idegx + 1):
#                 ren = 0
#                 dx = (idegx // 2) * 2 - idegx
#                 dy = (idegy // 2) * 2 - idegy
#                 if dx == 0 and dy == 0 and nvec > 0:
#                     ren = 1
#                 renFlags.append(ren)
#                 nvec += 1
#     xy_conv_stamp_fast_numba(sa.vectors, sa.xss, sa.yss, sa.sscnt, si, imConv, nvec, renFlags, usePCA,
#                               fwKSStamp, fwKernel, hwKSStamp, hwKernel,
#                               rPixX, filter_x, filter_y, temp, PCA)
# 
#     if cut_sstamp_numpy(sa.krefArea, sa.xss, sa.yss, sa.x0, sa.y0, sa.nss, sa.sscnt, sa.sum_val, si, imRef, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose):
#         return 1
# 
#     xi = int(sa.xss[si, sa.sscnt[si]])
#     yi = int(sa.yss[si, sa.sscnt[si]])
#     di = xi - hwKSStamp
#     dj = yi - hwKSStamp
#     for i in range(xi - hwKSStamp, xi + hwKSStamp + 1):
#         xf = (i - rPixX2) / rPixX2
#         for j in range(yi - hwKSStamp, yi + hwKSStamp + 1):
#             yf = (j - rPixY2) / rPixY2
#             ax = 1.0
#             nv = nvec
#             for idegx in range(bgOrder + 1):
#                 ay = 1.0
#                 for idegy in range(bgOrder - idegx + 1):
#                     sa.vectors[si, nv, i - di + fwKSStamp * (j - dj)] = ax * ay
#                     ay *= yf
#                     nv += 1
#                 ax *= xf
# 
#     build_matrix0_numpy(sa.vectors, sa.mat, si, nCompKer, kerOrder, bgOrder, fwKSStamp)
#     build_scprod0_numpy(sa.vectors, sa.scprod, sa.xss, sa.yss, sa.sscnt, si, imRef, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX)
# 
#     return 0


def fill_stamp_numba(saXss, saYss, saSscnt, saNss, saX0, saY0, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
                     usePCA, filter_x, filter_y, PCA, fillVal, mRData,
                     img_flat=None, imRef_flat=None, kernels_all=None):
    # def fill_stamp_numba(sa, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
    #                      hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
    #                      usePCA, filter_x, filter_y, PCA, fillVal, mRData):
    if saSscnt[si] >= saNss[si]:
        return 1

    # if usePCA:
    #     return 1

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

    xi = int(saXss[si, saSscnt[si]])
    yi = int(saYss[si, saSscnt[si]])

    img_flat = np.asarray(imConv, dtype=np.float64).ravel()
    imRef_flat = np.asarray(imRef, dtype=np.float64).ravel()
    fx = np.asarray(filter_x, dtype=np.float64)
    fy = np.asarray(filter_y, dtype=np.float64)
    rflags = np.array(renFlags, dtype=np.int32)
    mRData1d = mRData.ravel()

    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
    fwSqStamp = fwKSStamp * fwKSStamp
    nC = nCompKer + 1

    out_vectors = np.zeros((nvec + nbg, fwSqStamp), dtype=np.float64)
    out_krefArea = np.zeros(fwSqStamp, dtype=np.float64)
    out_mat = np.zeros((nC + 1, nC + 1), dtype=np.float64)
    out_scprod = np.zeros(nC + 1, dtype=np.float64)
    out_sum_val = np.zeros(1, dtype=np.float64)

    logging.getLogger('hotpants').debug("fill_stamp_numba si=%d xi=%d yi=%d", si, xi, yi)

    fill_stamp_numba_kernel_local(
        img_flat, imRef_flat, fx, fy,
        xi, yi, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
        rPixX, rPixY, nCompKer, kerOrder, bgOrder,
        nvec, rflags, fillVal, mRData1d, verbose,
        out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val)

    return out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val[0]


@numba.jit(nopython=True)
def get_stamp_sig_batch_jit(
    sa_vectors, sa_krefArea, sa_sscnt, sa_nss, sa_xss, sa_yss,
    kernelSol, imNoise, mRData1d,
    fwKSStamp, hwKSStamp, rPixX, rPixY,
    nCompKer, kerOrder, bgOrder, nS,
    figMerit_is_v, statSig,
    out_sig1, out_sig2, out_sig3,
    batched_bg=None, batched_coeffs=None):

    LOCAL_ZEROVAL = 1e-10
    LOCAL_MAXVAL = 1e10
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    LOCAL_FLAG_ISNAN = 0x08
    fwSq = fwKSStamp * fwKSStamp

    for si in range(nS):
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
                if math.isnan(tdat) or math.isnan(idat):
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


@numba.jit(nopython=True)
def get_stamp_sig_jit(vectors, kernelSol, imNoise, mRData1d, im,
                       fwKSStamp, hwKSStamp, rPixX, rPixY,
                       nCompKer, kerOrder, bgOrder, xi, yi,
                       figMerit_is_v, statSig, temp):
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

            if math.isnan(tdat) or math.isnan(idat):
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


def get_stamp_sig_numpy(sa, si, kernelSol, imNoise, fwKSStamp, hwKSStamp,
                        rPixX, rPixY, mRData, figMerit, statSig, nCompKer, kerOrder, bgOrder):
    # def get_stamp_sig_numpy(stamp_dict, kernelSol, imNoise, fwKSStamp, hwKSStamp,
    #                         rPixX, rPixY, mRData, figMerit, statSig, nCompKer, kerOrder, bgOrder):
    # sscnt = stamp_dict['sscnt']
    sscnt = sa.sscnt[si]
    # xRegion = int(stamp_dict['xss'][sscnt])
    xRegion = int(sa.xss[si, sscnt])
    # yRegion = int(stamp_dict['yss'][sscnt])
    yRegion = int(sa.yss[si, sscnt])

    # im = stamp_dict['krefArea']
    im = sa.krefArea[si]
    # vectors = np.asarray(stamp_dict['vectors'], dtype=np.float64)
    vectors = np.asarray(sa.vectors[si], dtype=np.float64)
    kSol = np.asarray(kernelSol, dtype=np.float64)
    imNoiseArr = np.asarray(imNoise, dtype=np.float64)
    mRDataArr = np.asarray(mRData, dtype=np.int64)
    imArr = np.asarray(im, dtype=np.float64)

    figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
    temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)

    sig1, sig2, sig3, nsig = get_stamp_sig_jit(
        vectors, kSol, imNoiseArr, mRDataArr, imArr,
        fwKSStamp, hwKSStamp, rPixX, rPixY,
        nCompKer, kerOrder, bgOrder, xRegion, yRegion,
        figMerit_is_v, statSig, temp)

    if figMerit[0:1] != "v":
        temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
        mRData_2d = mRData.reshape(-1, rPixX)
        result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                        fwKSStamp, fwKSStamp,
                                        0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
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

    return (sig1, sig2, sig3)

# # 旧版本 get_stamp_sig_numpy — 已替换为 jit 版本
# def get_stamp_sig_numpy(stamp_dict, kernelSol, imNoise, fwKSStamp, hwKSStamp,
#                         rPixX, rPixY, mRData, figMerit, statSig, nCompKer, kerOrder, bgOrder):
#     sscnt = stamp_dict['sscnt']
#     xRegion = int(stamp_dict['xss'][sscnt])
#     yRegion = int(stamp_dict['yss'][sscnt])
# 
#     im = stamp_dict['krefArea']
#     bg = get_background_numpy(xRegion, yRegion, kernelSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY)
#     csModel = make_model_numpy(stamp_dict, kernelSol, rPixX, rPixY, nCompKer, kerOrder, fwKSStamp)
# 
#     nsig = 0
#     sig1 = 0.0
#     sig2 = -1.0
#     sig3 = -1.0
# 
#     temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float32)
# 
#     for j in range(fwKSStamp):
#         yRegion2 = yRegion - hwKSStamp + j
#         for i in range(fwKSStamp):
#             xRegion2 = xRegion - hwKSStamp + i
#             idx = i + j * fwKSStamp
# 
#             tdat = float(csModel[idx])
#             idat = float(im[idx])
#             ndat = float(imNoise[xRegion2 + rPixX * yRegion2])
# 
#             diff = tdat - idat + bg
# 
#             if (int(mRData[xRegion2 + rPixX * yRegion2]) & FLAG_INPUT_ISBAD) or (abs(idat) <= ZEROVAL):
#                 continue
#             else:
#                 temp[idx] = diff
# 
#             if (tdat * 0.0 != 0.0) or (idat * 0.0 != 0.0):
#                 mRData[xRegion2 + rPixX * yRegion2] = int(mRData[xRegion2 + rPixX * yRegion2]) | (FLAG_INPUT_ISBAD | FLAG_ISNAN)
#                 continue
# 
#             nsig += 1
#             sig1 += diff * diff / ndat
# 
#     if nsig > 0:
#         sig1 /= nsig
#         if sig1 >= MAXVAL:
#             sig1 = -1.0
#     else:
#         sig1 = -1.0
# 
#     if figMerit[0:1] != "v":
#         temp_2d = temp.reshape(fwKSStamp, fwKSStamp)
#         mRData_2d = mRData.reshape(-1, rPixX)
#         result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
#                                         fwKSStamp, fwKSStamp,
#                                         0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
#         if result['return_code'] != 0:
#             sig2 = -1.0
#             sig3 = -1.0
#         else:
#             sig2 = result['sd']
#             sig3 = result['fwhm']
#             if sig2 < 0 or sig2 >= MAXVAL:
#                 sig2 = -1.0
#             elif sig3 < 0 or sig3 >= MAXVAL:
#                 sig3 = -1.0
# 
#     return (sig1, sig2, sig3)


def spatial_convolve_numpy_fast(image, variance, xSize, ySize, kernelSol, cRdata, cMask, kcStep,
                                hwKernel, fwKernel, kernel, kernel_coeffs,
                                convolveVariance, kerFracMask, mRData,
                                rPixX, rPixY, nCompKer, kerOrder, kernel_vec):
    from scipy.signal import fftconvolve
    from scipy.ndimage import maximum_filter
    import sys
    import time

    t0 = time.time()

    dovar = variance is not None
    vData = None
    if dovar:
        vData = np.zeros(xSize * ySize, dtype=np.float32)

    fwSq = fwKernel * fwKernel
    image_2d = np.asarray(image, dtype=np.float64).reshape(ySize, xSize)

    basis_list = []
    conv_maps = []
    for idx in range(nCompKer):
        basis = np.asarray(kernel_vec[idx][:fwSq], dtype=np.float64).reshape(fwKernel, fwKernel)
        basis_list.append(basis)
        conv_maps.append(fftconvolve(image_2d, basis, mode='same'))

    t1 = time.time()
    sys.stderr.write("  FFT convolve (%d basis): %.2f s\n" % (nCompKer, t1 - t0))

    halfX = 0.5 * rPixX
    halfY = 0.5 * rPixY
    xf1d = (np.arange(xSize, dtype=np.float64) + hwKernel - halfX) / halfX
    yf1d = (np.arange(ySize, dtype=np.float64) + hwKernel - halfY) / halfY

    xfPow = [np.ones(xSize, dtype=np.float64)]
    for p in range(1, kerOrder + 1):
        xfPow.append(xfPow[-1] * xf1d)

    yfPow = [np.ones(ySize, dtype=np.float64)]
    for p in range(1, kerOrder + 1):
        yfPow.append(yfPow[-1] * yf1d)

    output = float(kernelSol[1]) * conv_maps[0]
    coeffFields = [np.full((ySize, xSize), float(kernelSol[1]), dtype=np.float64)]

    poly_basis = []
    for ix in range(kerOrder + 1):
        for iy in range(kerOrder - ix + 1):
            poly_basis.append(np.outer(yfPow[iy], xfPow[ix]))
    num_poly_terms = len(poly_basis)

    k = 2
    for i1 in range(1, nCompKer):
        cf = np.zeros((ySize, xSize), dtype=np.float64)
        for p in range(num_poly_terms):
            if k + p >= len(kernelSol):
                break
            cf += float(kernelSol[k + p]) * poly_basis[p]
        k += num_poly_terms
        coeffFields.append(cf)
        output += cf * conv_maps[i1]

    t2 = time.time()
    sys.stderr.write("  Poly weighting: %.2f s\n" % (t2 - t1))

    sy = slice(hwKernel, ySize - hwKernel)
    sx = slice(hwKernel, xSize - hwKernel)
    cRdata2d = np.asarray(cRdata).reshape(ySize, xSize)
    cRdata2d[sy, sx] = output[sy, sx]

    if dovar:
        vData2d = vData.reshape(ySize, xSize)
        var2d = np.asarray(variance, dtype=np.float64).reshape(ySize, xSize)

        if convolveVariance:
            varConvMaps = [fftconvolve(var2d, basis_list[idx], mode='same')
                           for idx in range(nCompKer)]
            varOutput = np.zeros((ySize, xSize), dtype=np.float64)
            for i1 in range(nCompKer):
                varOutput += coeffFields[i1] * varConvMaps[i1]
        else:
            varCross = {}
            for i in range(nCompKer):
                for j in range(i, nCompKer):
                    prod = basis_list[i] * basis_list[j]
                    varCross[(i, j)] = fftconvolve(var2d, prod, mode='same')

            varOutput = np.zeros((ySize, xSize), dtype=np.float64)
            for i in range(nCompKer):
                for j in range(i, nCompKer):
                    factor = 2.0 if i != j else 1.0
                    varOutput += factor * coeffFields[i] * coeffFields[j] * varCross[(i, j)]

        vData2d[sy, sx] = varOutput[sy, sx]

    t3 = time.time()
    sys.stderr.write("  Variance: %.2f s\n" % (t3 - t2))

    cMaskView = np.asarray(cMask).reshape(ySize, xSize)
    mRDataView = np.asarray(mRData).reshape(ySize, xSize)

    innerCMask = cMaskView[sy, sx]
    mRDataView[sy, sx] |= innerCMask
    badSelf = (innerCMask.astype(np.int32) & FLAG_INPUT_ISBAD) > 0
    mRDataView[sy, sx] |= (FLAG_OUTPUT_ISBAD * badSelf.astype(np.int32))

    mbitField = maximum_filter(cMaskView.astype(np.int32), size=fwKernel)
    maskPixY, maskPixX = np.where(mbitField[sy, sx] > 0)
    maskPixY += hwKernel
    maskPixX += hwKernel
    nMaskPix = len(maskPixY)

    sys.stderr.write("  Mask pixels to check: %d\n" % nMaskPix)

    for pidx in range(nMaskPix):
        j = int(maskPixY[pidx])
        i = int(maskPixX[pidx])
        # make_kernel_numpy(i + hwKernel, j + hwKernel, kernelSol, rPixX, rPixY,
        #                   nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        blockI = (i - hwKernel) // kcStep
        i0 = blockI * kcStep + hwKernel
        blockJ = (j - hwKernel) // kcStep
        j0 = blockJ * kcStep + hwKernel
        make_kernel_numpy(i0 + hwKernel, j0 + hwKernel, kernelSol, rPixX, rPixY,
                          nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        aks = 0.0
        uks = 0.0
        for jc in range(j - hwKernel, j + hwKernel + 1):
            jk = j - jc + hwKernel
            for ic in range(i - hwKernel, i + hwKernel + 1):
                ik = i - ic + hwKernel
                nc = ic + xSize * jc
                kk = abs(float(kernel[ik + jk * fwKernel]))
                aks += kk
                if not (int(cMask[nc]) & FLAG_INPUT_ISBAD):
                    uks += kk
        ni = i + xSize * j
        if aks > 0.0 and (uks / aks) < kerFracMask:
            mRData[ni] = int(mRData[ni]) | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
        else:
            mRData[ni] = int(mRData[ni]) | FLAG_OK_CONV

    t4 = time.time()
    sys.stderr.write("  Mask handling: %.2f s\n" % (t4 - t3))
    sys.stderr.write("  Total spatial_convolve_fast: %.2f s\n" % (t4 - t0))

    return vData


def spatial_convolve_numpy(image, variance, xSize, ySize, kernelSol, cRdata, cMask, kcStep,
                           hwKernel, fwKernel, kernel, kernel_coeffs,
                           convolveVariance, kerFracMask, mRData,
                           rPixX, rPixY, nCompKer, kerOrder, kernel_vec):
    dovar = variance is not None
    vData = None
    if dovar:
        vData = np.zeros(xSize * ySize, dtype=np.float32)

    nstepsX = int(math.ceil(float(xSize) / float(kcStep)))
    nstepsY = int(math.ceil(float(ySize) / float(kcStep)))

    for j1 in range(nstepsY):
        j0 = j1 * kcStep + hwKernel

        for i1 in range(nstepsX):
            i0 = i1 * kcStep + hwKernel

            make_kernel_numpy(i0 + hwKernel, j0 + hwKernel, kernelSol, rPixX, rPixY,
                              nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)

            for j2 in range(kcStep):
                j = j0 + j2
                if j >= ySize - hwKernel:
                    break

                for i2 in range(kcStep):
                    i = i0 + i2
                    if i >= xSize - hwKernel:
                        break

                    ni = i + xSize * j
                    qv = 0.0
                    q = 0.0
                    aks = 0.0
                    uks = 0.0
                    mbit = 0x0

                    for jc in range(j - hwKernel, j + hwKernel + 1):
                        jk = j - jc + hwKernel

                        for ic in range(i - hwKernel, i + hwKernel + 1):
                            ik = i - ic + hwKernel
                            nc = ic + xSize * jc
                            kk = float(kernel[ik + jk * fwKernel])

                            q += float(image[nc]) * kk
                            if dovar:
                                if convolveVariance:
                                    qv += float(variance[nc]) * kk
                                else:
                                    qv += float(variance[nc]) * kk * kk

                            mbit |= int(cMask[nc])
                            aks += abs(kk)
                            if not (int(cMask[nc]) & FLAG_INPUT_ISBAD):
                                uks += abs(kk)

                    cRdata[ni] = q
                    if dovar:
                        vData[ni] = qv

                    mRData[ni] = int(mRData[ni]) | int(cMask[ni])
                    mRData[ni] = int(mRData[ni]) | (FLAG_OUTPUT_ISBAD * int((int(cMask[ni]) & FLAG_INPUT_ISBAD) > 0))

                    if mbit:
                        if aks > 0.0 and (uks / aks) < kerFracMask:
                            mRData[ni] = int(mRData[ni]) | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
                        else:
                            mRData[ni] = int(mRData[ni]) | FLAG_OK_CONV

    logger.debug("    sc_fast: mask loop done")
    return vData
    # return spatial_convolve_numpy_fast(image, variance, xSize, ySize, kernelSol, cRdata, cMask, kcStep,
    #                                    hwKernel, fwKernel, kernel, kernel_coeffs,
    #                                    convolveVariance, kerFracMask, mRData,
    #                                    rPixX, rPixY, nCompKer, kerOrder, kernel_vec)


# @numba.jit(nopython=True, parallel=True)
# def spatial_convolve_jit_kernel(
#     image, variance, cMask,
#     cRdata, vData, mRData,
#     kernelSol,
#     xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
#     kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
#     kernel_vec_2d,
#     allKernels=None,
#     nstepsX_in=None):
# 
#     fwSq = fwKernel * fwKernel
#     FLAG_INPUT_ISBAD = np.int32(0x80)
#     FLAG_OUTPUT_ISBAD = np.int32(0x8000)
#     FLAG_BAD_CONV = np.int32(0x10)
#     FLAG_OK_CONV = np.int32(0x40)
#     halfX = 0.5 * rPixX
#     halfY = 0.5 * rPixY
# 
#     # kernel = np.zeros(fwSq, dtype=np.float64)  # moved inside prange
#     # kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)  # moved inside prange
# 
#     nsteps_x = int(np.ceil(xSize / kcStep))
#     nsteps_y = int(np.ceil(ySize / kcStep))
# 
#     for j1 in numba.prange(nsteps_y):
#         j0 = j1 * kcStep + hwKernel
#         for i1 in range(nsteps_x):
#             i0 = i1 * kcStep + hwKernel
# 
#             # ---- make_kernel(i0+hwKernel, j0+hwKernel) ----
#             kernel = np.zeros(fwSq, dtype=np.float64)
#             kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
#             xi = i0 + hwKernel
#             yi = j0 + hwKernel
#             xf = (xi - halfX) / halfX
#             yf = (yi - halfY) / halfY
# 
#             k = 2
#             for i1k in range(1, nCompKer):
#                 coeff = 0.0
#                 ax = 1.0
#                 for ix in range(kerOrder + 1):
#                     ay = 1.0
#                     for iy in range(kerOrder - ix + 1):
#                         coeff += kernelSol[k] * ax * ay
#                         k += 1
#                         ay *= yf
#                     ax *= xf
#                 kernel_coeffs[i1k] = coeff
#             kernel_coeffs[0] = kernelSol[1]
# 
#             # for ii in range(fwSq):
#             #     kernel[ii] = 0.0  # already zeros from np.zeros
#             for ii in range(fwSq):
#                 for c in range(nCompKer):
#                     kernel[ii] += kernel_coeffs[c] * kernel_vec_2d[c, ii]
#             # ---- end make_kernel ----
# 
#             for j2 in range(kcStep):
#                 j = j0 + j2
#                 if j >= ySize - hwKernel:
#                     break
#                 for i2 in range(kcStep):
#                     i = i0 + i2
#                     if i >= xSize - hwKernel:
#                         break
# 
#                     ni = i + xSize * j
#                     q = 0.0
#                     qv = 0.0
#                     aks = 0.0
#                     uks = 0.0
#                     mbit = np.int32(0)
# 
#                     for jc in range(j - hwKernel, j + hwKernel + 1):
#                         jk = j - jc + hwKernel
#                         for ic in range(i - hwKernel, i + hwKernel + 1):
#                             ik = i - ic + hwKernel
#                             nc = ic + xSize * jc
#                             kk = kernel[ik + jk * fwKernel]
# 
#                             q += image[nc] * kk
#                             if dovar:
#                                 if convolveVariance:
#                                     qv += variance[nc] * kk * kk
#                                 else:
#                                     qv += variance[nc] * kk * kk
#                             mbit |= cMask[nc]
#                             aks += abs(kk)
#                             if not (cMask[nc] & FLAG_INPUT_ISBAD):
#                                 uks += abs(kk)
# 
#                     cRdata[ni] = q
#                     if dovar:
#                         vData[ni] = qv
# 
#                     mRData[ni] = mRData[ni] | cMask[ni]
#                     mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD * np.int32((cMask[ni] & FLAG_INPUT_ISBAD) > 0))
# 
#                     if mbit:
#                         if aks > 0.0 and (uks / aks) < kerFracMask:
#                             mRData[ni] = mRData[ni] | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
#                         else:
#                             mRData[ni] = mRData[ni] | FLAG_OK_CONV


@numba.jit(nopython=True)
def variance_convolve_jit(var_in, var_out, xSize, ySize, hwKernel, fwKernel,
                           kernelSol, kernel_vec_2d, nCompKer, kerOrder,
                           kcStep, rPixX, rPixY, convolveVariance):
    fwSq = fwKernel * fwKernel
    halfX = np.float64(0.5 * rPixX)
    halfY = np.float64(0.5 * rPixY)

    prev_i0 = -1
    prev_j0 = -1
    kernel = np.zeros(fwSq, dtype=np.float64)

    for j in range(hwKernel, ySize - hwKernel):
        for i in range(hwKernel, xSize - hwKernel):
            blockI = (i - hwKernel) // kcStep
            i0 = blockI * kcStep + hwKernel
            blockJ = (j - hwKernel) // kcStep
            j0 = blockJ * kcStep + hwKernel

            if i0 != prev_i0 or j0 != prev_j0:
                xi = np.float64(i0 + hwKernel)
                yi = np.float64(j0 + hwKernel)
                xf = (xi - halfX) / halfX
                yf = (yi - halfY) / halfY

                for jj in range(fwSq):
                    kernel[jj] = kernelSol[1] * kernel_vec_2d[0, jj]

                ksol = 2
                for ig in range(1, nCompKer):
                    coeff = np.float64(0.0)
                    ax = np.float64(1.0)
                    for ix in range(kerOrder + 1):
                        ay = np.float64(1.0)
                        for iy in range(kerOrder - ix + 1):
                            coeff += kernelSol[ksol] * ax * ay
                            ksol += 1
                            ay *= yf
                        ax *= xf
                    for jj in range(fwSq):
                        kernel[jj] += coeff * kernel_vec_2d[ig, jj]

                prev_i0 = i0
                prev_j0 = j0

            var_sum = np.float64(0.0)
            if convolveVariance:
                for jc in range(fwKernel):
                    cy = j - hwKernel + jc
                    for ic in range(fwKernel):
                        cx = i - hwKernel + ic
                        k_idx = (fwKernel - 1 - ic) + (fwKernel - 1 - jc) * fwKernel
                        kk = kernel[k_idx]
                        var_sum += kk * kk * var_in[cy, cx]
            else:
                for jc in range(fwKernel):
                    cy = j - hwKernel + jc
                    for ic in range(fwKernel):
                        cx = i - hwKernel + ic
                        k_idx = (fwKernel - 1 - ic) + (fwKernel - 1 - jc) * fwKernel
                        kk = kernel[k_idx]
                        # C 代码实际效果等价于 kk*kk（方差传播），非 abs(kk)
                        var_sum += kk * kk * var_in[cy, cx]

            var_out[j, i] = var_sum


@numba.jit(nopython=True)
def mask_check_loop_jit(maskPixY, maskPixX, cMask2d, mRData1d,
                         kernelSol, kernel_vec_2d,
                         nCompKer, kerOrder, fwKernel, hwKernel, kcStep,
                         rPixX, rPixY, xSize, ySize, kerFracMask):
    fwSq = fwKernel * fwKernel
    halfX = np.float64(0.5 * rPixX)
    halfY = np.float64(0.5 * rPixY)
    FLAG_INPUT_ISBAD = np.int32(0x80)
    FLAG_OUTPUT_ISBAD = np.int32(0x8000)
    FLAG_BAD_CONV = np.int32(0x10)
    FLAG_OK_CONV = np.int32(0x40)
    nTerms = (kerOrder + 1) * (kerOrder + 2) // 2

    for pidx in range(len(maskPixY)):
        j = maskPixY[pidx]
        i = maskPixX[pidx]

        blockI = (i - hwKernel) // kcStep
        i0 = blockI * kcStep + hwKernel
        blockJ = (j - hwKernel) // kcStep
        j0 = blockJ * kcStep + hwKernel

        xi = np.float64(i0 + hwKernel)
        yi = np.float64(j0 + hwKernel)
        xf = (xi - halfX) / halfX
        yf = (yi - halfY) / halfY

        kernel = np.zeros(fwSq, dtype=np.float64)

        coeff0 = kernelSol[1]
        for jj in range(fwSq):
            kernel[jj] = coeff0 * kernel_vec_2d[0, jj]

        ksol = 2
        for ig in range(1, nCompKer):
            coeff = np.float64(0.0)
            ax = np.float64(1.0)
            for ix in range(kerOrder + 1):
                ay = np.float64(1.0)
                for iy in range(kerOrder - ix + 1):
                    coeff += kernelSol[ksol] * ax * ay
                    ksol += 1
                    ay *= yf
                ax *= xf
            for jj in range(fwSq):
                kernel[jj] += coeff * kernel_vec_2d[ig, jj]

        aks = np.float64(0.0)
        uks = np.float64(0.0)
        for jc in range(fwKernel):
            cy = j - hwKernel + jc
            for ic in range(fwKernel):
                cx = i - hwKernel + ic
                kk = abs(kernel[(fwKernel - 1 - ic) + (fwKernel - 1 - jc) * fwKernel])
                aks += kk
                if not (cMask2d[cy, cx] & FLAG_INPUT_ISBAD):
                    uks += kk

        ni = i + xSize * j
        if aks > 0.0 and (uks / aks) < kerFracMask:
            mRData1d[ni] = mRData1d[ni] | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
        else:
            mRData1d[ni] = mRData1d[ni] | FLAG_OK_CONV


# def spatial_convolve_fast_numpy(image, variance, xSize, ySize, kernelSol, cRdata, cMask, kcStep,
#                                 hwKernel, fwKernel, kernel, kernel_coeffs,
#                                 convolveVariance, kerFracMask, mRData,
#                                 rPixX, rPixY, nCompKer, kerOrder, kernel_vec):
#     from scipy.signal import fftconvolve
#     from scipy.ndimage import maximum_filter
# 
#     dovar = variance is not None
#     vData = None
#     if dovar:
#         vData = np.zeros(xSize * ySize, dtype=np.float32)
# 
#     fwSq = fwKernel * fwKernel
#     kernel_vec_2d = np.array([kernel_vec[idx][:fwSq] for idx in range(nCompKer)], dtype=np.float64)
#     image2d = np.asarray(image, dtype=np.float64).reshape(ySize, xSize)
# 
#     basisList = []
#     convMaps = []
#     logger.debug("    sc_fast: FFT convolve start (%d bases)", nCompKer)
#     for idx in range(nCompKer):
#         basis = np.asarray(kernel_vec[idx][:fwSq], dtype=np.float64).reshape(fwKernel, fwKernel)
#         basisList.append(basis)
#         convMaps.append(fftconvolve(image2d, basis, mode='same'))
#     logger.debug("    sc_fast: FFT convolve done")
# 
#     halfX = 0.5 * rPixX
#     halfY = 0.5 * rPixY
#     blockIdxX = np.clip((np.arange(xSize) - hwKernel) // kcStep, 0, None)
#     i0Arr = blockIdxX * kcStep + hwKernel
#     xf1d = (i0Arr.astype(np.float64) + hwKernel - halfX) / halfX
#     blockIdxY = np.clip((np.arange(ySize) - hwKernel) // kcStep, 0, None)
#     j0Arr = blockIdxY * kcStep + hwKernel
#     yf1d = (j0Arr.astype(np.float64) + hwKernel - halfY) / halfY
# 
#     xfPow = [np.ones(xSize, dtype=np.float64)]
#     for p in range(1, kerOrder + 1):
#         xfPow.append(xfPow[-1] * xf1d)
#     yfPow = [np.ones(ySize, dtype=np.float64)]
#     for p in range(1, kerOrder + 1):
#         yfPow.append(yfPow[-1] * yf1d)
# 
#     output = float(kernelSol[1]) * convMaps[0]
#     coeffFields = [np.full((ySize, xSize), float(kernelSol[1]), dtype=np.float64)]
# 
#     k = 2
#     for i1 in range(1, nCompKer):
#         cf = np.zeros((ySize, xSize), dtype=np.float64)
#         for ix in range(kerOrder + 1):
#             for iy in range(kerOrder - ix + 1):
#                 cf += float(kernelSol[k]) * np.outer(yfPow[iy], xfPow[ix])
#                 k += 1
#         coeffFields.append(cf)
#         output += cf * convMaps[i1]
#     logger.debug("    sc_fast: coeff fields + weighted sum done")
# 
#     sy = slice(hwKernel, ySize - hwKernel)
#     sx = slice(hwKernel, xSize - hwKernel)
#     cRdata2d = np.asarray(cRdata).reshape(ySize, xSize)
#     cRdata2d[sy, sx] = output[sy, sx]
# 
#     if dovar:
#         # 旧版本：用 fftconvolve 做 variance 卷积，nCompKer²/2 次调用，耗时长
#         # vData2d = vData.reshape(ySize, xSize)
#         # var2d = np.asarray(variance, dtype=np.float64).reshape(ySize, xSize)
#         # if convolveVariance:
#         #     varConvMaps = [fftconvolve(var2d, b, mode='same') for b in basisList]
#         #     varOutput = np.zeros((ySize, xSize), dtype=np.float64)
#         #     for i1 in range(nCompKer):
#         #         varOutput += coeffFields[i1] * varConvMaps[i1]
#         # else:
#         #     varOutput = np.zeros((ySize, xSize), dtype=np.float64)
#         #     for i1 in range(nCompKer):
#         #         for j1 in range(i1, nCompKer):
#         #             prod = basisList[i1] * basisList[j1]
#         #             cross = fftconvolve(var2d, prod, mode='same')
#         #             factor = 2.0 if i1 != j1 else 1.0
#         #             varOutput += factor * coeffFields[i1] * coeffFields[j1] * cross
#         # vData2d[sy, sx] = varOutput[sy, sx]
#         vData2d = vData.reshape(ySize, xSize)
#         var2d = np.asarray(variance, dtype=np.float64).reshape(ySize, xSize)
#         varOutput = np.zeros((ySize, xSize), dtype=np.float64)
#         variance_convolve_jit(var2d, varOutput, xSize, ySize, hwKernel, fwKernel,
#                                np.asarray(kernelSol, dtype=np.float64), kernel_vec_2d,
#                                nCompKer, kerOrder, kcStep, rPixX, rPixY, convolveVariance)
#         vData2d[sy, sx] = varOutput[sy, sx]
# 
#     cMaskView = np.asarray(cMask).reshape(ySize, xSize)
#     mRDataView = np.asarray(mRData).reshape(ySize, xSize)
# 
#     innerCMask = cMaskView[sy, sx]
#     mRDataView[sy, sx] |= innerCMask
#     badSelf = (innerCMask.astype(np.int32) & FLAG_INPUT_ISBAD) > 0
#     mRDataView[sy, sx] |= (FLAG_OUTPUT_ISBAD * badSelf.astype(np.int32))
# 
#     # mbitField = maximum_filter(cMaskView.astype(np.int32), size=fwKernel)
#     # maskPixY, maskPixX = np.where(mbitField[sy, sx] > 0)
#     badFlag2d = ((cMaskView.astype(np.int32) & FLAG_INPUT_ISBAD) > 0).astype(np.float64)
#     badCountField = fftconvolve(badFlag2d, np.ones((fwKernel, fwKernel)), mode='same')
#     badCountField = np.round(badCountField).astype(int)
#     maskPixY, maskPixX = np.where(badCountField[sy, sx] > 0)
#     maskPixY += hwKernel
#     maskPixX += hwKernel
#     logger.debug("    sc_fast: bad_count done, %d pixels to check", len(maskPixY))
# 
#     # for pidx in range(len(maskPixY)):
#     #     j = int(maskPixY[pidx])
#     #     i = int(maskPixX[pidx])
#     #     # make_kernel_numpy(i + hwKernel, j + hwKernel, kernelSol, rPixX, rPixY,
#     #     #                   nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
#     #     blockI = (i - hwKernel) // kcStep
#     #     i0 = blockI * kcStep + hwKernel
#     #     blockJ = (j - hwKernel) // kcStep
#     #     j0 = blockJ * kcStep + hwKernel
#     #     make_kernel_numpy(i0 + hwKernel, j0 + hwKernel, kernelSol, rPixX, rPixY,
#     #                       nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
#     #     # aks = 0.0
#     #     # uks = 0.0
#     #     # for jc in range(j - hwKernel, j + hwKernel + 1):
#     #     #     jk = j - jc + hwKernel
#     #     #     for ic in range(i - hwKernel, i + hwKernel + 1):
#     #     #         ik = i - ic + hwKernel
#     #     #         nc = ic + xSize * jc
#     #     #         kk = abs(float(kernel[ik + jk * fwKernel]))
#     #     #         aks += kk
#     #     #         if not (int(cMask[nc]) & FLAG_INPUT_ISBAD):
#     #     #             uks += kk
#     #     # 矢量化内层循环
#     #     kern2d = np.abs(kernel[:fwSq].reshape(fwKernel, fwKernel))[::-1, ::-1]
#     #     cmask_patch = cMaskView[j-hwKernel:j+hwKernel+1, i-hwKernel:i+hwKernel+1]
#     #     bad_patch = (cmask_patch.astype(np.int32) & FLAG_INPUT_ISBAD) > 0
#     #     aks = float(kern2d.sum())
#     #     uks = float(kern2d[~bad_patch].sum())
#     #     ni = i + xSize * j
#     #     if aks > 0.0 and (uks / aks) < kerFracMask:
#     #         mRData[ni] = int(mRData[ni]) | (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
#     #     else:
#     #         mRData[ni] = int(mRData[ni]) | FLAG_OK_CONV
#     # jit version: inline make_kernel + mask check
#     # fwSq = fwKernel * fwKernel  # 已提前创建
#     # kernel_vec_2d = np.array([kernel_vec[idx][:fwSq] for idx in range(nCompKer)], dtype=np.float64)  # 已提前创建
#     mask_check_loop_jit(maskPixY, maskPixX, cMaskView, np.asarray(mRData),
#                          np.asarray(kernelSol), kernel_vec_2d,
#                          nCompKer, kerOrder, fwKernel, hwKernel, kcStep,
#                          rPixX, rPixY, xSize, ySize, kerFracMask)
# 
#     logger.debug("    sc_fast: mask loop done (fast)|(jit)")
#     return vData

def spatial_convolve_fast_numpy(image, variance, xSize, ySize, kernelSol, cMask, kcStep,
                                hwKernel, fwKernel, kernel, kernel_coeffs,
                                convolveVariance, kerFracMask,
                                rPixX, rPixY, nCompKer, kerOrder, kernel_vec,
                                logger=None):
    # import time
    # t0 = time.time()
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
    # t1 = time.time(); logger.debug("  sc_fast: prep %.3fs", t1 - t0)

    allKernels, nstepsX, nstepsY = buildAllKernels(
        kernelSol.astype(np.float64), kernel_vec_2d,
        nCompKer, kerOrder, fwKernel, hwKernel, kcStep,
        rPixX, rPixY, xSize, ySize)

    spatial_convolve_jit_kernel(
        image1d, var1d, cMask1d,
        cRdata64, vData64, mRData64,
        kernelSol.astype(np.float64),
        xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
        kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
        kernel_vec_2d,
        allKernels=allKernels,
        nstepsX_in=nstepsX)
    # t2 = time.time(); logger.debug("  sc_fast: jit_kernel %.3fs", t2 - t1)

    if dovar:
        vData = vData64.astype(np.float32)

    cRdata_out = cRdata64.astype(np.float32)
    mRData_out = mRData64
    # t3 = time.time(); logger.debug("  sc_fast: post %.3fs total %.3fs", t3 - t2, t3 - t0)
    return vData, cRdata_out, mRData_out


# def spatial_convolve_fast_numpy_deprecated_1781965833(image, variance, xSize, ySize, kernelSol, cMask, kcStep,
#                                      hwKernel, fwKernel, kernel, kernel_coeffs,
#                                      convolveVariance, kerFracMask,
#                                      rPixX, rPixY, nCompKer, kerOrder, kernel_vec):
#     fwSq = fwKernel * fwKernel
#     dovar = variance is not None
# 
#     # 输入: 转float64做计算
#     image1d = image.astype(np.float64).ravel()
#     cMask1d = cMask.astype(np.int32).ravel()
# 
#     # 输出: float64临时数组
#     cRdata64 = np.zeros(xSize * ySize, dtype=np.float64)
#     mRData64 = np.zeros(xSize * ySize, dtype=np.int32)
# 
#     if dovar:
#         var1d = variance.astype(np.float64).ravel()
#     else:
#         var1d = np.zeros(1, dtype=np.float64)
# 
#     vData = None
#     vData64 = np.zeros(xSize * ySize, dtype=np.float64)
# 
#     kernel_vec_2d = np.array([np.asarray(kernel_vec[idx][:fwSq], dtype=np.float64)
#                               for idx in range(nCompKer)])
# 
#     spatial_convolve_jit_kernel(
#         image1d, var1d, cMask1d,
#         cRdata64, vData64, mRData64,
#         kernelSol.astype(np.float64),
#         xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
#         kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
#         kernel_vec_2d)
# 
#     if dovar:
#         vData = vData64.astype(np.float32)
# 
#     cRdata_out = cRdata64.astype(np.float32)
#     mRData_out = mRData64.astype(np.int32)
#     return vData, cRdata_out, mRData_out




# def check_stamps_numpy(sa, nS, imRef, imNoise, nCompKer, kerOrder, bgOrder,
#                        nCompTotal, verbose, forceConvolve, figMerit,
#                        kerSigReject, statSig, fwKSStamp, hwKSStamp,
#                        rPixX, rPixY, fwKernel, kernel_vec, mRData):
def check_stamps_numpy(saScprod, saMat, saNorm, saDiff, saSscnt, saNss, saXss, saYss,
                       saVectors, saKrefArea, nS, imRef, imNoise, nCompKer, kerOrder, bgOrder,
                       nCompTotal, verbose, forceConvolve, figMerit,
                       kerSigReject, statSig, fwKSStamp, hwKSStamp,
                       rPixX, rPixY, fwKernel, kernel_vec, mRData,
                       nKSStamps=None, nC=None):
    # def check_stamps_numpy(stamps_dicts, nS, imRef, imNoise, nCompKer, kerOrder, bgOrder,
    #                        nCompTotal, verbose, forceConvolve, figMerit,
    #                        kerSigReject, statSig, fwKSStamp, hwKSStamp,
    #                        rPixX, rPixY, fwKernel, kernel_vec, mRData):
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

        # d = 0.0
        # ludcmp_numpy(check_mat, nComps, indx)
        # lubksb_numpy(check_mat, nComps, indx, check_vec)
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

        # matrix = build_matrix_numpy(testStamps, ntestStamps, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        # matrix = build_matrix_numpy(testSa, ntestStamps, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        matrix, _ = build_matrix_numpy(testMat, testVectors, testSscnt, testNss, testXss, testYss,
                                        ntestStamps, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, rPixX, rPixY, verbose, nC=nC, nKSStamps=nKSStamps)

        # testKerSol = build_scprod_numpy(testStamps, ntestStamps, imRef, nCompKer, kerOrder,
        #                                 bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy)
        # testKerSol = build_scprod_numpy(testSa, ntestStamps, imRef, nCompKer, kerOrder,
        #                                 bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy)
        testKerSol = build_scprod_numpy(testScprod, testVectors, testSscnt, testNss, testXss, testYss,
                                        ntestStamps, imRef, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)

        # indx2 = np.zeros(mat_size + 1, dtype=np.int32)
        # ludcmp_numpy(matrix, mat_size, indx2)
        # lubksb_numpy(matrix, mat_size, indx2, testKerSol)
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
            # sig1, sig2, sig3 = get_stamp_sig_numpy(
            #     testStamps[i], testKerSol, imNoise, fwKSStamp, hwKSStamp,
            #     rPixX, rPixY, mRData, figMerit, statSig, nCompKer, kerOrder, bgOrder)
            # sig1, sig2, sig3 = get_stamp_sig_numpy(
            #     testSa, i, testKerSol, imNoise, fwKSStamp, hwKSStamp,
            #     rPixX, rPixY, mRData, figMerit, statSig, nCompKer, kerOrder, bgOrder)
            # 内联 get_stamp_sig_numpy，使用扁平数组
            sscnt = testSscnt[i]
            xRegion = int(testXss[i, sscnt])
            yRegion = int(testYss[i, sscnt])
            im = testKrefArea[i]
            vectors = np.asarray(testVectors[i], dtype=np.float64)
            kSol = np.asarray(testKerSol, dtype=np.float64)
            imNoiseArr = np.asarray(imNoise, dtype=np.float64)
            mRDataArr = np.asarray(mRData, dtype=np.int64)
            imArr = np.asarray(im, dtype=np.float64)

            figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
            temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)

            sig1, sig2, sig3, nsig = get_stamp_sig_jit(
                vectors, kSol, imNoiseArr, mRDataArr, imArr,
                fwKSStamp, hwKSStamp, rPixX, rPixY,
                nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                figMerit_is_v, statSig, temp)

            if figMerit[0:1] != "v":
                temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                mRData_2d = mRData.reshape(-1, rPixX)
                result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                                fwKSStamp, fwKSStamp,
                                                0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
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


# def check_again_numpy(sa, kernelSol, imConv, imRef, imNoise,
#                       nS, verbose, figMerit, kerSigReject, statSig,
#                       fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
#                       nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
#                       hwKernel, fwKernel, usePCA, filter_x, filter_y,
#                       PCA, fillVal):
def check_again_numpy(saSscnt, saNss, saChi2, saXss, saYss, saVectors, saMat, saScprod, saKrefArea, saSumVal, kernelSol,
                      imConv, imRef, imNoise,
                      nS, verbose, figMerit, kerSigReject, statSig,
                      fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
                      nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
                      hwKernel, fwKernel, usePCA, filter_x, filter_y,
                      PCA, fillVal):
    # def check_again_numpy(stamps, kernelSol, imConv, imRef, imNoise,
    #                       nS, verbose, figMerit, kerSigReject, statSig,
    #                       fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
    #                       nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
    #                       hwKernel, fwKernel, usePCA, filter_x, filter_y,
    #                       PCA, fillVal):
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

        get_stamp_sig_batch_jit(
            np.asarray(saVectors, dtype=np.float64), np.asarray(saKrefArea, dtype=np.float64),
            saSscnt, saNss, saXss, saYss,
            np.asarray(kernelSol, dtype=np.float64),
            np.asarray(imNoise, dtype=np.float64),
            np.asarray(mRData, dtype=np.int32).ravel(),
            fwKSStamp, hwKSStamp, rPixX, rPixY,
            nCompKer, kerOrder, bgOrder, nS,
            1, statSig,
            batch_sig1, batch_sig2, batch_sig3,
            batched_bg=batched_bg, batched_coeffs=batched_coeffs4)

    for istamp in range(nS):
        if sscnt_local[istamp] < saNss[istamp]:
            if batch_sig1 is not None:
                sig1 = batch_sig1[istamp]
                sig2 = batch_sig2[istamp]
                sig3 = batch_sig3[istamp]
            else:
                # sig1, sig2, sig3 = get_stamp_sig_numpy(
                #     sa, istamp, kernelSol, imNoise,
                #     fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
                #     figMerit, statSig, nCompKer, kerOrder, bgOrder)
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

                figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
                temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64)

                sig1, sig2, sig3, nsig = get_stamp_sig_jit(
                    vectors, kSol, imNoiseArr, mRDataArr, imArr,
                    fwKSStamp, hwKSStamp, rPixX, rPixY,
                    nCompKer, kerOrder, bgOrder, xRegion, yRegion,
                    figMerit_is_v, statSig, temp)

                if figMerit[0:1] != "v":
                    temp_2d = temp.reshape(fwKSStamp, fwKSStamp).astype(np.float32)
                    mRData_2d = mRData.reshape(-1, rPixX)
                    result = get_stamp_stats3_numpy(temp_2d, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                                    fwKSStamp, fwKSStamp,
                                                    0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
                    if result['return_code'] != 0:
                        sig2 = -1.0
                        sig3 = -1.0
                    else:
                        sig2 = result['sd']
                        sig3 = result['fwhm']
                        if sig2 < 0 or sig2 >= MAXVAL:
                            sig2 = -1.0
                        elif sig3 < 0 or sig3 >= MAXVAL:
                            sig3 = -1.0

            if (figMerit[0:1] == "v" and sig1 == -1) or \
               (figMerit[0:1] == "s" and sig2 == -1) or \
               (figMerit[0:1] == "h" and sig3 == -1):
                sscnt_local[istamp] += 1
                # fill_stamp_numpy(sa, istamp, imConv, imRef, rPixX, rPixY, verbose,
                #                  ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                #                  bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                #                  PCA, fillVal, mRData)
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

    scnt = 0
    for istamp in range(nS):
        # if stamps[istamp]['sscnt'] < stamps[istamp]['nss']:
        if sscnt_local[istamp] < saNss[istamp]:
            # if (stamps[istamp]['chi2'] - mean) > kerSigReject * stdev:
            if (chi2_local[istamp] - mean) > kerSigReject * stdev:
                # stamps[istamp]['sscnt'] += 1
                sscnt_local[istamp] += 1
                # rc = fill_stamp_numpy(stamps[istamp], imConv, imRef, rPixX, rPixY, verbose,
                #                       ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                #                       bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                #                       PCA, fillVal, mRData)
                # rc = fill_stamp_numpy(sa, istamp, imConv, imRef, rPixX, rPixY, verbose,
                #                       ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                #                       bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                #                       PCA, fillVal, mRData)
                # scnt += (1 if rc == 0 else 0)
                refill_indices.append(istamp)
                check = 1
            else:
                scnt += 1

    return (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
            refill_indices, sscnt_local, chi2_local)


# def fit_kernel_numpy(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
#                      verbose, nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
#                      kerSigReject, statSig, mRData, ngauss, deg_fixe,
#                      hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal):
def fit_kernel_numpy(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
                     verbose, nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
                      kerSigReject, statSig, mRData, ngauss, deg_fixe,
                      hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal,
                      logger=None):
    # def fit_kernel_numpy(stamps_dicts, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
    #                      verbose, nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
    #                      kerSigReject, statSig, mRData, ngauss, deg_fixe,
    #                      hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal):
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
    saX0       = sa['x0']
    saY0       = sa['y0']
    saChi2     = sa['chi2']
    nC         = sa['nC']
    nKSStamps  = sa['nKSStamps']

    # import time
    iter_count = 0
    # t_start = time.time()
    logger.debug("  fitKernel: iteration %d start", iter_count)
    # matrix = build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder,
    #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
    # matrix = build_matrix_numpy(sa, nS, nCompKer, kerOrder, bgOrder,
    #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
    # tm = time.time()
    matrix, wxy = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                    nS, nCompKer, kerOrder, bgOrder,
                                    fwKSStamp, rPixX, rPixY, verbose, nC=nC, nKSStamps=nKSStamps)
    # kernelSol = build_scprod_numpy(stamps_dicts, nS, imRef, nCompKer, kerOrder, bgOrder,
    #                                fwKSStamp, hwKSStamp, rPixX, wxy)
    # kernelSol = build_scprod_numpy(sa, nS, imRef, nCompKer, kerOrder, bgOrder,
    #                                fwKSStamp, hwKSStamp, rPixX, wxy)
    kernelSol = build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                   nS, imRef, nCompKer, kerOrder, bgOrder,
                                   fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)
    logger.debug("  fitKernel: build_matrix0+scprod0 done")
    # tm_bm = time.time(); logger.debug("  fitKernel: build+scprod %.3fs", tm_bm - tm)

    # indx = np.zeros(mat_size + 1, dtype=np.int32)
    # ludcmp_numpy(matrix, mat_size, indx)
    # lubksb_numpy(matrix, mat_size, indx, kernelSol)
    kernelSol[1:mat_size+1] = np.linalg.solve(
        matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
    logger.debug("  fitKernel: solve done")
    # tm_slv = time.time(); logger.debug("  fitKernel: solve %.3fs", tm_slv - tm_bm)

    # check, meansigSubstamps, scatterSubstamps, nskippedSubstamps = check_again_numpy(
    #     stamps_dicts, kernelSol, imConv, imRef, imNoise,
    #     nS, verbose, figMerit, kerSigReject, statSig,
    #     fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
    #     nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
    #     hwKernel, fwKernel, usePCA, filter_x, filter_y,
    #     PCA, fillVal)
    # check, meansigSubstamps, scatterSubstamps, nskippedSubstamps = check_again_numpy(
    #     sa, kernelSol, imConv, imRef, imNoise,
    #     nS, verbose, figMerit, kerSigReject, statSig,
    #     fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
    #     nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
    #     hwKernel, fwKernel, usePCA, filter_x, filter_y,
    #     PCA, fillVal)
    (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
     refill_indices, sscnt_update, chi2_update) = check_again_numpy(
        saSscnt, saNss, saChi2, saXss, saYss, saVectors, saMat, saScprod, saKrefArea, saSumVal, kernelSol,
        imConv, imRef, imNoise,
        nS, verbose, figMerit, kerSigReject, statSig,
        fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
        nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
        hwKernel, fwKernel, usePCA, filter_x, filter_y,
        PCA, fillVal)
    saSscnt[:nS] = sscnt_update
    saChi2[:nS] = chi2_update

    for idx in refill_indices:
        fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, saX0, saY0, idx, imConv, imRef, rPixX, rPixY, verbose,
                         ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                         bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                         PCA, fillVal, mRData)
    logger.debug("  fitKernel: check_again done, check=%s", check)
    # tm_ca = time.time(); logger.debug("  fitKernel: check_again %.3fs", tm_ca - tm_slv)

    while check:
        iter_count += 1
        logger.debug("  fitKernel: iteration %d start", iter_count)
        # matrix = build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        # matrix = build_matrix_numpy(sa, nS, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        # tm = time.time()
        matrix, wxy = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                        nS, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, rPixX, rPixY, verbose, nC=nC, nKSStamps=nKSStamps)
        # kernelSol = build_scprod_numpy(stamps_dicts, nS, imRef, nCompKer, kerOrder, bgOrder,
        #                                fwKSStamp, hwKSStamp, rPixX, wxy)
        # kernelSol = build_scprod_numpy(sa, nS, imRef, nCompKer, kerOrder, bgOrder,
        #                                fwKSStamp, hwKSStamp, rPixX, wxy)
        kernelSol = build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                       nS, imRef, nCompKer, kerOrder, bgOrder,
                                       fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)
        logger.debug("  fitKernel: build_matrix+scprod done")
        # tm_bm = time.time(); logger.debug("  fitKernel: build+scprod %.3fs", tm_bm - tm)

        # indx = np.zeros(mat_size + 1, dtype=np.int32)
        # ludcmp_numpy(matrix, mat_size, indx)
        # lubksb_numpy(matrix, mat_size, indx, kernelSol)
        kernelSol[1:mat_size+1] = np.linalg.solve(
            matrix[1:mat_size+1, 1:mat_size+1], kernelSol[1:mat_size+1])
        logger.debug("  fitKernel: solve done")
        # tm_slv = time.time(); logger.debug("  fitKernel: solve %.3fs", tm_slv - tm_bm)

        # check, meansigSubstamps, scatterSubstamps, nskippedSubstamps = check_again_numpy(
        #     stamps_dicts, kernelSol, imConv, imRef, imNoise,
        #     nS, verbose, figMerit, kerSigReject, statSig,
        #     fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
        #     nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
        #     hwKernel, fwKernel, usePCA, filter_x, filter_y,
        #     PCA, fillVal)
        # check, meansigSubstamps, scatterSubstamps, nskippedSubstamps = check_again_numpy(
        #     sa, kernelSol, imConv, imRef, imNoise,
        #     nS, verbose, figMerit, kerSigReject, statSig,
        #     fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
        #     nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
        #     hwKernel, fwKernel, usePCA, filter_x, filter_y,
        #     PCA, fillVal)
        (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
         refill_indices, sscnt_update, chi2_update) = check_again_numpy(
            saSscnt, saNss, saChi2, saXss, saYss, saVectors, saMat, saScprod, saKrefArea, saSumVal, kernelSol,
            imConv, imRef, imNoise,
            nS, verbose, figMerit, kerSigReject, statSig,
            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
            nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
            hwKernel, fwKernel, usePCA, filter_x, filter_y,
            PCA, fillVal)
        saSscnt[:nS] = sscnt_update
        saChi2[:nS] = chi2_update

        for idx in refill_indices:
            fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, saX0, saY0, idx, imConv, imRef, rPixX, rPixY, verbose,
                             ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                             PCA, fillVal, mRData)
        logger.debug("  fitKernel: check_again done, check=%s", check)
        # tm_ca = time.time(); logger.debug("  fitKernel: check_again %.3fs", tm_ca - tm_slv)

    return {
        'kernelSol': kernelSol,
        'meansigSubstamps': meansigSubstamps,
        'scatterSubstamps': scatterSubstamps,
        'NskippedSubstamps': nskippedSubstamps,
        # 'stamps': stamps_dicts,
        'stamps': sa,
    }


# def allocate_stamp_dict(nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC):
#     fwSq = fwKSStamp * fwKSStamp
#     nVec = nCompKer + nBGVectors
#     return {
#         'x0': 0, 'y0': 0,
#         'x': 0, 'y': 0,
#         'nx': 0, 'ny': 0,
#         'nss': 0, 'sscnt': 0,
#         'xss': np.zeros(nKSStamps, dtype=np.int32),
#         'yss': np.zeros(nKSStamps, dtype=np.int32),
#         'krefArea': np.zeros(fwSq, dtype=np.float64),
#         'scprod': np.zeros(nC, dtype=np.float64),
#         'vectors': np.zeros((nVec, fwSq), dtype=np.float64),
#         'mat': np.zeros((nC, nC), dtype=np.float64),
#         'chi2': 0.0, 'norm': 0.0, 'diff': 0.0,
#         'sum': 0.0, 'mean': 0.0, 'median': 0.0,
#         'mode': 0.0, 'sd': 0.0,
#         'fwhm': 0.0, 'lfwhm': 0.0,
#     }


def make_noise_image4_numpy(data1d, invGain, quad, rPixX, rPixY):
    qquad = float(quad) * float(quad)
    nData = np.abs(data1d.astype(np.float64)) * float(invGain) + qquad
    return nData.astype(np.float32)


# def build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, kernel_vec, filter_x, filter_y, status):
#     """将 StampsArray 对象的字段展开为扁平 numpy 数组，与原有对象一起放入 dict"""
#     result = {
#         'niS': niS, 'ntS': ntS,
#         'ctStamps': ctSa,
#         'ciStamps': ciSa,
#         'kernel_vec': kernel_vec,
#         'filter_x': filter_x,
#         'filter_y': filter_y,
#         'status': status,
#     }
#     if ctSa is not None:
#         result['ctSscnt'] = ctSa.sscnt
#         result['ctNss'] = ctSa.nss
#         result['ctX0'] = ctSa.x0
#         result['ctY0'] = ctSa.y0
#         result['ctX'] = ctSa.x
#         result['ctY'] = ctSa.y
#         result['ctXss'] = ctSa.xss
#         result['ctYss'] = ctSa.yss
#         result['ctVectors'] = ctSa.vectors
#         result['ctMat'] = ctSa.mat
#         result['ctScprod'] = ctSa.scprod
#         result['ctKrefArea'] = ctSa.krefArea
#         result['ctChi2'] = ctSa.chi2
#         result['ctNorm'] = ctSa.norm
#         result['ctDiff'] = ctSa.diff
#         result['ctSumVal'] = ctSa.sum_val
#         result['ctMeanVal'] = ctSa.mean_val
#         result['ctMedian'] = ctSa.median
#         result['ctMode'] = ctSa.mode
#         result['ctSd'] = ctSa.sd
#         result['ctFwhm'] = ctSa.fwhm
#         result['ctLfwhm'] = ctSa.lfwhm
#         result['ctValid'] = ctSa.valid
#         result['ctNKSStamps'] = ctSa.nKSStamps
#         result['ctNC'] = ctSa.nC
#         result['ctNVec'] = ctSa.nVec
#         result['ctFwSq'] = ctSa.fwSq
#     if ciSa is not None:
#         result['ciSscnt'] = ciSa.sscnt
#         result['ciNss'] = ciSa.nss
#         result['ciX0'] = ciSa.x0
#         result['ciY0'] = ciSa.y0
#         result['ciX'] = ciSa.x
#         result['ciY'] = ciSa.y
#         result['ciXss'] = ciSa.xss
#         result['ciYss'] = ciSa.yss
#         result['ciVectors'] = ciSa.vectors
#         result['ciMat'] = ciSa.mat
#         result['ciScprod'] = ciSa.scprod
#         result['ciKrefArea'] = ciSa.krefArea
#         result['ciChi2'] = ciSa.chi2
#         result['ciNorm'] = ciSa.norm
#         result['ciDiff'] = ciSa.diff
#         result['ciSumVal'] = ciSa.sum_val
#         result['ciMeanVal'] = ciSa.mean_val
#         result['ciMedian'] = ciSa.median
#         result['ciMode'] = ciSa.mode
#         result['ciSd'] = ciSa.sd
#         result['ciFwhm'] = ciSa.fwhm
#         result['ciLfwhm'] = ciSa.lfwhm
#         result['ciValid'] = ciSa.valid
#         result['ciNKSStamps'] = ciSa.nKSStamps
#         result['ciNC'] = ciSa.nC
#         result['ciNVec'] = ciSa.nVec
#         result['ciFwSq'] = ciSa.fwSq
#     return result

def region_buildstamps_numpy(setup_result, ctx_info, params_info, localForceConvolve, logger=None):
    import sys
    logger.debug("region_buildstamps_numpy start")

    nCompKer = ctx_info['nCompKer']
    nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    fwStamp = ctx_info['fwStamp']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']
    nStamps = ctx_info['nStamps']
    nStampX = ctx_info['nStampX']
    nStampY = ctx_info['nStampY']
    fitThresh = ctx_info['fitThresh']

    hwKernel = params_info['hwKernel']
    ngauss = params_info['ngauss']
    deg_fixe = params_info['deg_fixe']
    sigma_gauss = params_info['sigma_gauss']
    hwKSStamp = params_info['hwKSStamp']
    nKSStamps = params_info['nKSStamps']
    scaleFitThresh = params_info['scaleFitThresh']
    minFracGoodStamps = params_info['minFracGoodStamps']
    tUKThresh = params_info['tUKThresh']
    iUKThresh = params_info['iUKThresh']
    verbose = params_info.get('verbose', 0)
    usePCA = params_info.get('usePCA', 0)
    PCA = params_info.get('PCA', None)
    xcmp = params_info.get('xcmp', None)
    ycmp = params_info.get('ycmp', None)
    Ncmp = params_info.get('Ncmp', 0)
    statSig = params_info['statSig']
    findSSC = params_info.get('findSSC', 0)

    tRData = setup_result['tRData']
    iRData = setup_result['iRData']
    mRData = setup_result['mRData']
    rXBMin = setup_result['rXBMin']
    rYBMin = setup_result['rYBMin']
    rXBMax = setup_result['rXBMax']
    rYBMax = setup_result['rYBMax']
    rPixX = setup_result['rPixX']
    rPixY = setup_result['rPixY']

    tRData1d = tRData.ravel()
    iRData1d = iRData.ravel()
    mRData1d = mRData.ravel()

    useFullSS = 0

    status = 1
    flag = 1
    kerFitThresh = fitThresh
    # ctStamps = None
    # ciStamps = None
    ctSa = None
    ciSa = None

    while (status <= 2) and flag:
        flag = 0
        niS = 0
        ntS = 0

        if localForceConvolve != "i":
            # ctStamps = [allocate_stamp_dict(nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
            #             for _ in range(nStamps)]
            # ctSa = StampsArray(nStamps, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
            ct = {}  # template stamps flat arrays dict
            nVec = nCompKer + nBGVectors
            fwSq = fwKSStamp * fwKSStamp
            ct['nss'] = np.zeros(nStamps, dtype=np.int32)
            ct['sscnt'] = np.zeros(nStamps, dtype=np.int32)
            ct['x0'] = np.zeros(nStamps, dtype=np.int32)
            ct['y0'] = np.zeros(nStamps, dtype=np.int32)
            ct['x'] = np.zeros(nStamps, dtype=np.int32)
            ct['y'] = np.zeros(nStamps, dtype=np.int32)
            ct['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            ct['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
            ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
            ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
            ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
            ct['chi2'] = np.zeros(nStamps, dtype=np.float64)
            ct['norm'] = np.zeros(nStamps, dtype=np.float64)
            ct['diff'] = np.zeros(nStamps, dtype=np.float64)
            ct['sum_val'] = np.zeros(nStamps, dtype=np.float64)
            ct['mean_val'] = np.zeros(nStamps, dtype=np.float64)
            ct['median'] = np.zeros(nStamps, dtype=np.float64)
            ct['mode'] = np.zeros(nStamps, dtype=np.float64)
            ct['sd'] = np.zeros(nStamps, dtype=np.float64)
            ct['fwhm'] = np.zeros(nStamps, dtype=np.float64)
            ct['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
            ct['valid'] = np.zeros(nStamps, dtype=np.bool_)
            ct['nKSStamps'] = nKSStamps
            ct['fwKSStamp'] = fwKSStamp
            ct['nCompKer'] = nCompKer
            ct['nBGVectors'] = nBGVectors
            ct['nC'] = nC
            ct['fwSq'] = fwSq
            ct['nVec'] = nVec
            ctSa = ct
        if localForceConvolve != "t":
            # ciStamps = [allocate_stamp_dict(nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
            #             for _ in range(nStamps)]
            # ciSa = StampsArray(nStamps, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
            ci = {}  # image stamps flat arrays dict
            nVec = nCompKer + nBGVectors
            fwSq = fwKSStamp * fwKSStamp
            ci['nss'] = np.zeros(nStamps, dtype=np.int32)
            ci['sscnt'] = np.zeros(nStamps, dtype=np.int32)
            ci['x0'] = np.zeros(nStamps, dtype=np.int32)
            ci['y0'] = np.zeros(nStamps, dtype=np.int32)
            ci['x'] = np.zeros(nStamps, dtype=np.int32)
            ci['y'] = np.zeros(nStamps, dtype=np.int32)
            ci['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            ci['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
            ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
            ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
            ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
            ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
            ci['chi2'] = np.zeros(nStamps, dtype=np.float64)
            ci['norm'] = np.zeros(nStamps, dtype=np.float64)
            ci['diff'] = np.zeros(nStamps, dtype=np.float64)
            ci['sum_val'] = np.zeros(nStamps, dtype=np.float64)
            ci['mean_val'] = np.zeros(nStamps, dtype=np.float64)
            ci['median'] = np.zeros(nStamps, dtype=np.float64)
            ci['mode'] = np.zeros(nStamps, dtype=np.float64)
            ci['sd'] = np.zeros(nStamps, dtype=np.float64)
            ci['fwhm'] = np.zeros(nStamps, dtype=np.float64)
            ci['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
            ci['valid'] = np.zeros(nStamps, dtype=np.bool_)
            ci['nKSStamps'] = nKSStamps
            ci['fwKSStamp'] = fwKSStamp
            ci['nCompKer'] = nCompKer
            ci['nBGVectors'] = nBGVectors
            ci['nC'] = nC
            ci['fwSq'] = fwSq
            ci['nVec'] = nVec
            ciSa = ci

        # None-safe 辅助变量：当 forceConvolve=="t" 时 ciSa 为 None，forceConvolve=="i" 时 ctSa 为 None
        if ctSa is not None:
            ctNss = ctSa['nss']; ctX0 = ctSa['x0']; ctY0 = ctSa['y0']; ctX = ctSa['x']; ctY = ctSa['y']
            ctSumVal = ctSa['sum_val']; ctMeanVal = ctSa['mean_val']; ctMedian = ctSa['median']; ctMode = ctSa['mode']
            ctSd = ctSa['sd']; ctFwhm = ctSa['fwhm']; ctLfwhm = ctSa['lfwhm']; ctXss = ctSa['xss']; ctYss = ctSa['yss']; ctSscnt = ctSa['sscnt']
        else:
            ctNss = np.zeros(1, dtype=np.int32); ctX0 = np.zeros(1, dtype=np.int32); ctY0 = np.zeros(1, dtype=np.int32)
            ctX = np.zeros(1, dtype=np.int32); ctY = np.zeros(1, dtype=np.int32)
            ctSumVal = np.zeros(1, dtype=np.float64); ctMeanVal = np.zeros(1, dtype=np.float64); ctMedian = np.zeros(1, dtype=np.float64)
            ctMode = np.zeros(1, dtype=np.float64); ctSd = np.zeros(1, dtype=np.float64); ctFwhm = np.zeros(1, dtype=np.float64)
            ctLfwhm = np.zeros(1, dtype=np.float64); ctXss = np.zeros((1, 1), dtype=np.int32); ctYss = np.zeros((1, 1), dtype=np.int32)
            ctSscnt = np.zeros(1, dtype=np.int32)
        if ciSa is not None:
            ciNss = ciSa['nss']; ciX0 = ciSa['x0']; ciY0 = ciSa['y0']; ciX = ciSa['x']; ciY = ciSa['y']
            ciSumVal = ciSa['sum_val']; ciMeanVal = ciSa['mean_val']; ciMedian = ciSa['median']; ciMode = ciSa['mode']
            ciSd = ciSa['sd']; ciFwhm = ciSa['fwhm']; ciLfwhm = ciSa['lfwhm']; ciXss = ciSa['xss']; ciYss = ciSa['yss']; ciSscnt = ciSa['sscnt']
        else:
            ciNss = np.zeros(1, dtype=np.int32); ciX0 = np.zeros(1, dtype=np.int32); ciY0 = np.zeros(1, dtype=np.int32)
            ciX = np.zeros(1, dtype=np.int32); ciY = np.zeros(1, dtype=np.int32)
            ciSumVal = np.zeros(1, dtype=np.float64); ciMeanVal = np.zeros(1, dtype=np.float64); ciMedian = np.zeros(1, dtype=np.float64)
            ciMode = np.zeros(1, dtype=np.float64); ciSd = np.zeros(1, dtype=np.float64); ciFwhm = np.zeros(1, dtype=np.float64)
            ciLfwhm = np.zeros(1, dtype=np.float64); ciXss = np.zeros((1, 1), dtype=np.int32); ciYss = np.zeros((1, 1), dtype=np.int32)
            ciSscnt = np.zeros(1, dtype=np.int32)

        for l in range(nStampY):
            for k in range(nStampX):
                sys.stderr.write("Build stamp  : t %4d i %4d (grid coord %2d %2d)\n" % (ntS, niS, k, l))

                sXMin = rXBMin + k * rPixX // nStampX
                sYMin = rYBMin + l * rPixY // nStampY
                sXMax = min(sXMin + fwStamp - 1, rXBMax)
                sYMax = min(sYMin + fwStamp - 1, rYBMax)

                if localForceConvolve != "i":
                    # ctStamps[ntS]['sscnt'] = 0
                    ctSa['sscnt'][ntS] = 0
                    # ctStamps[ntS]['nss'] = 0
                    ctSa['nss'][ntS] = 0
                    # ctStamps[ntS]['chi2'] = 0.0
                    ctSa['chi2'][ntS] = 0.0
                if localForceConvolve != "t":
                    # ciStamps[niS]['sscnt'] = 0
                    ciSa['sscnt'][niS] = 0
                    # ciStamps[niS]['nss'] = 0
                    ciSa['nss'][niS] = 0
                    # ciStamps[niS]['chi2'] = 0.0
                    ciSa['chi2'][niS] = 0.0

                if xcmp is not None and Ncmp > 0:
                    if verbose >= 2:
                        sys.stderr.write("Adding centers manually\n")
                    for m in range(Ncmp):
                        if (xcmp[m] > sXMin + hwKernel + 1) and (xcmp[m] < sXMax - hwKernel - 1) and \
                           (ycmp[m] > sYMin + hwKernel + 1) and (ycmp[m] < sYMax - hwKernel - 1):
                            # # 对比：深拷贝两份输入
                            # ctCopy = ctSa.deepCopy() if ctSa is not None else None
                            # ciCopy = ciSa.deepCopy() if ciSa is not None else None
                            # iDataCopy = iRData1d.copy()
                            # tDataCopy = tRData1d.copy()
                            # mDataCopy = mRData1d.copy()
                            # # 旧版
                            # build_stamps_numpy(
                            #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 0,
                            #     rXBMin, rYBMin, ciSa, ctSa,
                            #     iRData1d, tRData1d,
                            #     xcmp[m] - rXBMin, ycmp[m] - rYBMin,
                            #     verbose, localForceConvolve, rPixX, rPixY,
                            #     tUKThresh, iUKThresh, hwKSStamp,
                            #     fwStamp, nKSStamps, kerFitThresh,
                            #     mRData1d, statSig)
                            # bsNssOldT = int(ctSa.nss[ntS]) if ctSa is not None else -1
                            # bsNssOldI = int(ciSa.nss[niS]) if ciSa is not None else -1
                            # bsXssOldT = ctSa.xss[ntS].copy() if ctSa is not None else None
                            # bsYssOldT = ctSa.yss[ntS].copy() if ctSa is not None else None
                            # bsXssOldI = ciSa.xss[niS].copy() if ciSa is not None else None
                            # bsYssOldI = ciSa.yss[niS].copy() if ciSa is not None else None
                            # # 新版（用副本）
                            # buildStampsNumba(
                            #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 0,
                            #     rXBMin, rYBMin, ciCopy, ctCopy,
                            #     iDataCopy, tDataCopy,
                            #     xcmp[m] - rXBMin, ycmp[m] - rYBMin,
                            #     verbose, localForceConvolve, rPixX, rPixY,
                            #     tUKThresh, iUKThresh, hwKSStamp,
                            #     fwStamp, nKSStamps, kerFitThresh,
                            #     mDataCopy, statSig)
                            (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                             lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                             lmRData) = buildStampsNumba(
                                sXMin, sXMax, sYMin, sYMax, niS, ntS, 0,
                                rXBMin, rYBMin,
                                # ctSa 扁平
                                ctNss[ntS], ctX0[ntS], ctY0[ntS], ctX[ntS], ctY[ntS],
                                ctSumVal[ntS], ctMeanVal[ntS], ctMedian[ntS], ctMode[ntS], ctSd[ntS], ctFwhm[ntS], ctLfwhm[ntS],
                                ctXss[ntS].copy(), ctYss[ntS].copy(), ctSscnt[ntS],
                                # ciSa 扁平
                                ciNss[niS], ciX0[niS], ciY0[niS], ciX[niS], ciY[niS],
                                ciSumVal[niS], ciMeanVal[niS], ciMedian[niS], ciMode[niS], ciSd[niS], ciFwhm[niS], ciLfwhm[niS],
                                ciXss[niS].copy(), ciYss[niS].copy(), ciSscnt[niS],
                                iRData1d, tRData1d,
                                xcmp[m] - rXBMin, ycmp[m] - rYBMin,
                                verbose, localForceConvolve, rPixX, rPixY,
                                tUKThresh, iUKThresh, hwKSStamp,
                                fwStamp, nKSStamps, kerFitThresh,
                                mRData1d.copy(), statSig)
                            # 写回
                            ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                            ctX[ntS], ctY[ntS] = lctX, lctY
                            ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                            ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                            ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                            ctLfwhm[ntS] = lctLfwhm
                            ctNss[ntS] = lctNss
                            ctXss[ntS] = lctXss
                            ctYss[ntS] = lctYss
                            ciX0[niS], ciY0[niS] = lciX0, lciY0
                            ciX[niS], ciY[niS] = lciX, lciY
                            ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                            ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                            ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                            ciLfwhm[niS] = lciLfwhm
                            ciNss[niS] = lciNss
                            ciXss[niS] = lciXss
                            ciYss[niS] = lciYss
                            mRData1d[:] = lmRData
                            # bsNssNewT = int(ctCopy.nss[ntS]) if ctCopy is not None else -1
                            # bsNssNewI = int(ciCopy.nss[niS]) if ciCopy is not None else -1
                            # bsXssNewT = ctCopy.xss[ntS] if ctCopy is not None else None
                            # bsYssNewT = ctCopy.yss[ntS] if ctCopy is not None else None
                            # bsXssNewI = ciCopy.xss[niS] if ciCopy is not None else None
                            # bsYssNewI = ciCopy.yss[niS] if ciCopy is not None else None
                            # # 对比
                            # bsNssDiff = abs(bsNssOldT - bsNssNewT) + abs(bsNssOldI - bsNssNewI)
                            # bsXssMax = 0
                            # bsYssMax = 0
                            # if bsXssOldT is not None and bsXssNewT is not None:
                            #     bsXssMax = int(np.max(np.abs(bsXssOldT.astype(np.int64) - bsXssNewT.astype(np.int64))))
                            #     bsYssMax = int(np.max(np.abs(bsYssOldT.astype(np.int64) - bsYssNewT.astype(np.int64))))
                            # if bsXssOldI is not None and bsXssNewI is not None:
                            #     bsXssMax = max(bsXssMax, int(np.max(np.abs(bsXssOldI.astype(np.int64) - bsXssNewI.astype(np.int64)))))
                            #     bsYssMax = max(bsYssMax, int(np.max(np.abs(bsYssOldI.astype(np.int64) - bsYssNewI.astype(np.int64)))))
                            # sys.stderr.write('[BS_CMP] nss:T=%d=%d I=%d=%d' % (
                            #     bsNssOldT, bsNssNewT, bsNssOldI, bsNssNewI))
                            # bsXssOd = 0; bsYssOd = 0; bsXssNd = 0; bsYssNd = 0
                            # if bsXssOldT is not None and bsXssNewT is not None:
                            #     bsXssOd = int(np.max(bsXssOldT)); bsXssNd = int(np.max(bsXssNewT))
                            #     bsYssOd = int(np.max(bsYssOldT)); bsYssNd = int(np.max(bsYssNewT))
                            # sys.stderr.write(' xss:T=%d=%d I=%d=%d' % (bsXssOd, bsXssNd,
                            #     int(np.max(bsXssOldI) if bsXssOldI is not None else -1) if bsXssOldI is not None else -1,
                            #     int(np.max(bsXssNewI) if bsXssNewI is not None else -1) if bsXssNewI is not None else -1))
                            # if ctCopy is not None and ctSa is not None and ctSa.nss[ntS] > 0:
                            #     sys.stderr.write(' vec:Tmax=%.3e=%.3e mat:Tmax=%.3e=%.3e' % (
                            #         float(np.max(np.abs(ctSa.vectors[ntS].astype(np.float64)))),
                            #         float(np.max(np.abs(ctCopy.vectors[ntS].astype(np.float64)))),
                            #         float(np.max(np.abs(ctSa.mat[ntS].astype(np.float64)))),
                            #         float(np.max(np.abs(ctCopy.mat[ntS].astype(np.float64))))))
                            # sys.stderr.write('\n')
                    if findSSC:
                        if verbose >= 2:
                            sys.stderr.write("Automatically finding additional centers\n")
                        # ctCopy = ctSa.deepCopy() if ctSa is not None else None
                        # ciCopy = ciSa.deepCopy() if ciSa is not None else None
                        # iDataCopy = iRData1d.copy()
                        # tDataCopy = tRData1d.copy()
                        # mDataCopy = mRData1d.copy()
                        # build_stamps_numpy(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                        #     rXBMin, rYBMin, ciSa, ctSa,
                        #     iRData1d, tRData1d, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mRData1d, statSig)
                        # bsNssOldT = int(ctSa.nss[ntS]) if ctSa is not None else -1
                        # bsNssOldI = int(ciSa.nss[niS]) if ciSa is not None else -1
                        # bsXssOldT = ctSa.xss[ntS].copy() if ctSa is not None else None
                        # bsYssOldT = ctSa.yss[ntS].copy() if ctSa is not None else None
                        # bsXssOldI = ciSa.xss[niS].copy() if ciSa is not None else None
                        # bsYssOldI = ciSa.yss[niS].copy() if ciSa is not None else None
                        # buildStampsNumba(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                        #     rXBMin, rYBMin, ciCopy, ctCopy,
                        #     iDataCopy, tDataCopy, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mDataCopy, statSig)
                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(
                            sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                            rXBMin, rYBMin,
                            ctNss[ntS], ctX0[ntS], ctY0[ntS], ctX[ntS], ctY[ntS],
                            ctSumVal[ntS], ctMeanVal[ntS], ctMedian[ntS], ctMode[ntS], ctSd[ntS], ctFwhm[ntS], ctLfwhm[ntS],
                            ctXss[ntS].copy(), ctYss[ntS].copy(), ctSscnt[ntS],
                            ciNss[niS], ciX0[niS], ciY0[niS], ciX[niS], ciY[niS],
                            ciSumVal[niS], ciMeanVal[niS], ciMedian[niS], ciMode[niS], ciSd[niS], ciFwhm[niS], ciLfwhm[niS],
                            ciXss[niS].copy(), ciYss[niS].copy(), ciSscnt[niS],
                            iRData1d, tRData1d, 0, 0,
                            verbose, localForceConvolve, rPixX, rPixY,
                            tUKThresh, iUKThresh, hwKSStamp,
                            fwStamp, nKSStamps, kerFitThresh,
                            mRData1d.copy(), statSig)
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                        ctX[ntS], ctY[ntS] = lctX, lctY
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                        ctLfwhm[ntS] = lctLfwhm
                        ctNss[ntS] = lctNss
                        ctXss[ntS] = lctXss
                        ctYss[ntS] = lctYss
                        ciX0[niS], ciY0[niS] = lciX0, lciY0
                        ciX[niS], ciY[niS] = lciX, lciY
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                        ciLfwhm[niS] = lciLfwhm
                        ciNss[niS] = lciNss
                        ciXss[niS] = lciXss
                        ciYss[niS] = lciYss
                        mRData1d[:] = lmRData
                        # bsNssNewT = int(ctCopy.nss[ntS]) if ctCopy is not None else -1
                        # bsNssNewI = int(ciCopy.nss[niS]) if ciCopy is not None else -1
                        # bsXssNewT = ctCopy.xss[ntS] if ctCopy is not None else None
                        # bsYssNewT = ctCopy.yss[ntS] if ctCopy is not None else None
                        # bsXssNewI = ciCopy.xss[niS] if ciCopy is not None else None
                        # bsYssNewI = ciCopy.yss[niS] if ciCopy is not None else None
                        # bsNssDiff = abs(bsNssOldT - bsNssNewT) + abs(bsNssOldI - bsNssNewI)
                        # bsXssMax = 0
                        # bsYssMax = 0
                        # if bsXssOldT is not None and bsXssNewT is not None:
                        #     bsXssMax = int(np.max(np.abs(bsXssOldT.astype(np.int64) - bsXssNewT.astype(np.int64))))
                        #     bsYssMax = int(np.max(np.abs(bsYssOldT.astype(np.int64) - bsYssNewT.astype(np.int64))))
                        # if bsXssOldI is not None and bsXssNewI is not None:
                        #     bsXssMax = max(bsXssMax, int(np.max(np.abs(bsXssOldI.astype(np.int64) - bsXssNewI.astype(np.int64)))))
                        #     bsYssMax = max(bsYssMax, int(np.max(np.abs(bsYssOldI.astype(np.int64) - bsYssNewI.astype(np.int64)))))
                        # sys.stderr.write('[BS_CMP] nt=%d=%d' % (bsNssOldT, bsNssNewT))
                        # if bsXssOldT is not None and bsXssNewT is not None:
                        #     sys.stderr.write(' xt=%d=%d yt=%d=%d' % (int(np.max(bsXssOldT)), int(np.max(bsXssNewT)), int(np.max(bsYssOldT)), int(np.max(bsYssNewT))))
                        # if ctSa is not None and ctCopy is not None:
                        #     sys.stderr.write(' vt=%.3e=%.3e mt=%.3e=%.3e' % (float(np.max(np.abs(ctSa.vectors[ntS]))), float(np.max(np.abs(ctCopy.vectors[ntS]))), float(np.max(np.abs(ctSa.mat[ntS]))), float(np.max(np.abs(ctCopy.mat[ntS])))))
                        # if bsXssOldI is not None and bsXssNewI is not None:
                        #     sys.stderr.write(' ni=%d=%d xi=%d=%d' % (bsNssOldI, bsNssNewI, int(np.max(bsXssOldI)), int(np.max(bsXssNewI))))
                        # sys.stderr.write(' dif:n=%d x=%d y=%d' % (bsNssDiff, bsXssMax, bsYssMax))
                        # sys.stderr.write('\n')
                else:
                    if useFullSS:
                        # ctCopy = ctSa.deepCopy() if ctSa is not None else None
                        # ciCopy = ciSa.deepCopy() if ciSa is not None else None
                        # iDataCopy = iRData1d.copy()
                        # tDataCopy = tRData1d.copy()
                        # mDataCopy = mRData1d.copy()
                        # build_stamps_numpy(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 0,
                        #     rXBMin, rYBMin, ciSa, ctSa,
                        #     iRData1d, tRData1d, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mRData1d, statSig)
                        # bsNssOldT = int(ctSa.nss[ntS]) if ctSa is not None else -1
                        # bsNssOldI = int(ciSa.nss[niS]) if ciSa is not None else -1
                        # bsXssOldT = ctSa.xss[ntS].copy() if ctSa is not None else None
                        # bsYssOldT = ctSa.yss[ntS].copy() if ctSa is not None else None
                        # bsXssOldI = ciSa.xss[niS].copy() if ciSa is not None else None
                        # bsYssOldI = ciSa.yss[niS].copy() if ciSa is not None else None
                        # buildStampsNumba(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 0,
                        #     rXBMin, rYBMin, ciCopy, ctCopy,
                        #     iDataCopy, tDataCopy, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mDataCopy, statSig)
                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(
                            sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                            rXBMin, rYBMin,
                            # ctSa 扁平
                            ctNss[ntS], ctX0[ntS], ctY0[ntS], ctX[ntS], ctY[ntS],
                            ctSumVal[ntS], ctMeanVal[ntS], ctMedian[ntS], ctMode[ntS], ctSd[ntS], ctFwhm[ntS], ctLfwhm[ntS],
                            ctXss[ntS].copy(), ctYss[ntS].copy(), ctSscnt[ntS],
                            # ciSa 扁平
                            ciNss[niS], ciX0[niS], ciY0[niS], ciX[niS], ciY[niS],
                            ciSumVal[niS], ciMeanVal[niS], ciMedian[niS], ciMode[niS], ciSd[niS], ciFwhm[niS], ciLfwhm[niS],
                            ciXss[niS].copy(), ciYss[niS].copy(), ciSscnt[niS],
                            iRData1d, tRData1d, 0, 0,
                            verbose, localForceConvolve, rPixX, rPixY,
                            tUKThresh, iUKThresh, hwKSStamp,
                            fwStamp, nKSStamps, kerFitThresh,
                            mRData1d.copy(), statSig)
                        # 写回
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                        ctX[ntS], ctY[ntS] = lctX, lctY
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                        ctLfwhm[ntS] = lctLfwhm
                        ctNss[ntS] = lctNss
                        ctXss[ntS] = lctXss
                        ctYss[ntS] = lctYss
                        ciX0[niS], ciY0[niS] = lciX0, lciY0
                        ciX[niS], ciY[niS] = lciX, lciY
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                        ciLfwhm[niS] = lciLfwhm
                        ciNss[niS] = lciNss
                        ciXss[niS] = lciXss
                        ciYss[niS] = lciYss
                        mRData1d[:] = lmRData
                        # bsNssNewT = int(ctCopy.nss[ntS]) if ctCopy is not None else -1
                        # bsNssNewI = int(ciCopy.nss[niS]) if ciCopy is not None else -1
                        # bsXssNewT = ctCopy.xss[ntS] if ctCopy is not None else None
                        # bsYssNewT = ctCopy.yss[ntS] if ctCopy is not None else None
                        # bsXssNewI = ciCopy.xss[niS] if ciCopy is not None else None
                        # bsYssNewI = ciCopy.yss[niS] if ciCopy is not None else None
                        # bsNssDiff = abs(bsNssOldT - bsNssNewT) + abs(bsNssOldI - bsNssNewI)
                        # bsXssMax = 0
                        # bsYssMax = 0
                        # if bsXssOldT is not None and bsXssNewT is not None:
                        #     bsXssMax = int(np.max(np.abs(bsXssOldT.astype(np.int64) - ctCopy.xss[ntS].astype(np.int64))))
                        #     bsYssMax = int(np.max(np.abs(bsYssOldT.astype(np.int64) - ctCopy.yss[ntS].astype(np.int64))))
                        # if bsXssOldI is not None and bsXssNewI is not None:
                        #     bsXssMax = max(bsXssMax, int(np.max(np.abs(bsXssOldI.astype(np.int64) - ciCopy.xss[niS].astype(np.int64)))))
                        #     bsYssMax = max(bsYssMax, int(np.max(np.abs(bsYssOldI.astype(np.int64) - ciCopy.yss[niS].astype(np.int64)))))
                        # sys.stderr.write('[BS_CMP] nt=%d=%d' % (bsNssOldT, bsNssNewT))
                        # if bsXssOldT is not None and bsXssNewT is not None:
                        #     sys.stderr.write(' xt=%d=%d yt=%d=%d' % (int(np.max(bsXssOldT)), int(np.max(bsXssNewT)), int(np.max(bsYssOldT)), int(np.max(bsYssNewT))))
                        # if ctSa is not None and ctCopy is not None:
                        #     sys.stderr.write(' vt=%.3e=%.3e mt=%.3e=%.3e' % (float(np.max(np.abs(ctSa.vectors[ntS]))), float(np.max(np.abs(ctCopy.vectors[ntS]))), float(np.max(np.abs(ctSa.mat[ntS]))), float(np.max(np.abs(ctCopy.mat[ntS])))))
                        # if bsXssOldI is not None and bsXssNewI is not None:
                        #     sys.stderr.write(' ni=%d=%d xi=%d=%d' % (bsNssOldI, bsNssNewI, int(np.max(bsXssOldI)), int(np.max(bsXssNewI))))
                        # sys.stderr.write(' dif:n=%d x=%d y=%d' % (bsNssDiff, bsXssMax, bsYssMax))
                        # sys.stderr.write('\n')
                    else:
                        # ctCopy = ctSa.deepCopy() if ctSa is not None else None
                        # ciCopy = ciSa.deepCopy() if ciSa is not None else None
                        # iDataCopy = iRData1d.copy()
                        # tDataCopy = tRData1d.copy()
                        # mDataCopy = mRData1d.copy()
                        # build_stamps_numpy(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                        #     rXBMin, rYBMin, ciSa, ctSa,
                        #     iRData1d, tRData1d, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mRData1d, statSig)
                        # bsNssOldT = int(ctSa.nss[ntS]) if ctSa is not None else -1
                        # bsNssOldI = int(ciSa.nss[niS]) if ciSa is not None else -1
                        # bsXssOldT = ctSa.xss[ntS].copy() if ctSa is not None else None
                        # bsYssOldT = ctSa.yss[ntS].copy() if ctSa is not None else None
                        # bsXssOldI = ciSa.xss[niS].copy() if ciSa is not None else None
                        # bsYssOldI = ciSa.yss[niS].copy() if ciSa is not None else None
                        # buildStampsNumba(
                        #     sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                        #     rXBMin, rYBMin, ciCopy, ctCopy,
                        #     iDataCopy, tDataCopy, 0, 0,
                        #     verbose, localForceConvolve, rPixX, rPixY,
                        #     tUKThresh, iUKThresh, hwKSStamp,
                        #     fwStamp, nKSStamps, kerFitThresh,
                        #     mDataCopy, statSig)
                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(
                            sXMin, sXMax, sYMin, sYMax, niS, ntS, 1,
                            rXBMin, rYBMin,
                            ctNss[ntS], ctX0[ntS], ctY0[ntS], ctX[ntS], ctY[ntS],
                            ctSumVal[ntS], ctMeanVal[ntS], ctMedian[ntS], ctMode[ntS], ctSd[ntS], ctFwhm[ntS], ctLfwhm[ntS],
                            ctXss[ntS].copy(), ctYss[ntS].copy(), ctSscnt[ntS],
                            ciNss[niS], ciX0[niS], ciY0[niS], ciX[niS], ciY[niS],
                            ciSumVal[niS], ciMeanVal[niS], ciMedian[niS], ciMode[niS], ciSd[niS], ciFwhm[niS], ciLfwhm[niS],
                            ciXss[niS].copy(), ciYss[niS].copy(), ciSscnt[niS],
                            iRData1d, tRData1d, 0, 0,
                            verbose, localForceConvolve, rPixX, rPixY,
                            tUKThresh, iUKThresh, hwKSStamp,
                            fwStamp, nKSStamps, kerFitThresh,
                            mRData1d.copy(), statSig)
                        ctX0[ntS], ctY0[ntS] = lctX0, lctY0
                        ctX[ntS], ctY[ntS] = lctX, lctY
                        ctSumVal[ntS], ctMeanVal[ntS] = lctSumVal, lctMeanVal
                        ctMedian[ntS], ctMode[ntS] = lctMedian, lctMode
                        ctSd[ntS], ctFwhm[ntS] = lctSd, lctFwhm
                        ctLfwhm[ntS] = lctLfwhm
                        ctNss[ntS] = lctNss
                        ctXss[ntS] = lctXss
                        ctYss[ntS] = lctYss
                        ciX0[niS], ciY0[niS] = lciX0, lciY0
                        ciX[niS], ciY[niS] = lciX, lciY
                        ciSumVal[niS], ciMeanVal[niS] = lciSumVal, lciMeanVal
                        ciMedian[niS], ciMode[niS] = lciMedian, lciMode
                        ciSd[niS], ciFwhm[niS] = lciSd, lciFwhm
                        ciLfwhm[niS] = lciLfwhm
                        ciNss[niS] = lciNss
                        ciXss[niS] = lciXss
                        ciYss[niS] = lciYss
                        mRData1d[:] = lmRData
                        # bsNssNewT = int(ctCopy.nss[ntS]) if ctCopy is not None else -1
                        # bsNssNewI = int(ciCopy.nss[niS]) if ciCopy is not None else -1
                        # bsXssNewT = ctCopy.xss[ntS] if ctCopy is not None else None
                        # bsYssNewT = ctCopy.yss[ntS] if ctCopy is not None else None
                        # bsXssNewI = ciCopy.xss[niS] if ciCopy is not None else None
                        # bsYssNewI = ciCopy.yss[niS] if ciCopy is not None else None
                        # bsNssDiff = abs(bsNssOldT - bsNssNewT) + abs(bsNssOldI - bsNssNewI)
                        # bsXssMax = 0
                        # bsYssMax = 0
                        # if bsXssOldT is not None and bsNssNewT >= 0:
                        #     bsXssMax = int(np.max(np.abs(bsXssOldT.astype(np.int64) - ctCopy.xss[ntS].astype(np.int64))))
                        #     bsYssMax = int(np.max(np.abs(bsYssOldT.astype(np.int64) - ctCopy.yss[ntS].astype(np.int64))))
                        # if bsXssOldI is not None and bsNssNewI >= 0:
                        #     bsXssMax = max(bsXssMax, int(np.max(np.abs(bsXssOldI.astype(np.int64) - ciCopy.xss[niS].astype(np.int64)))))
                        #     bsYssMax = max(bsYssMax, int(np.max(np.abs(bsYssOldI.astype(np.int64) - ciCopy.yss[niS].astype(np.int64)))))
                        # sys.stderr.write('[BS_CMP] nt=%d=%d' % (bsNssOldT, bsNssNewT))
                        # if bsXssOldT is not None and bsXssNewT is not None:
                        #     sys.stderr.write(' xt=%d=%d yt=%d=%d' % (int(np.max(bsXssOldT)), int(np.max(bsXssNewT)), int(np.max(bsYssOldT)), int(np.max(bsYssNewT))))
                        # if ctSa is not None and ctCopy is not None:
                        #     sys.stderr.write(' vt=%.3e=%.3e mt=%.3e=%.3e' % (float(np.max(np.abs(ctSa.vectors[ntS]))), float(np.max(np.abs(ctCopy.vectors[ntS]))), float(np.max(np.abs(ctSa.mat[ntS]))), float(np.max(np.abs(ctCopy.mat[ntS])))))
                        # if bsXssOldI is not None and bsXssNewI is not None:
                        #     sys.stderr.write(' ni=%d=%d xi=%d=%d' % (bsNssOldI, bsNssNewI, int(np.max(bsXssOldI)), int(np.max(bsXssNewI))))
                        # sys.stderr.write(' dif:n=%d x=%d y=%d' % (bsNssDiff, bsXssMax, bsYssMax))
                        # sys.stderr.write('\n')

                if localForceConvolve != "i":
                    if verbose >= 2:
                        # sys.stderr.write("    templ: %d substamps\n" % ctStamps[ntS]['nss'])
                        sys.stderr.write("    templ: %d substamps\n" % ctSa['nss'][ntS])
                    # if ctStamps[ntS]['nss'] > 0:
                    if ctSa['nss'][ntS] > 0:
                        ntS += 1
                if localForceConvolve != "t":
                    if verbose >= 2:
                        # sys.stderr.write("    image: %d substamps\n" % ciStamps[niS]['nss'])
                        sys.stderr.write("    image: %d substamps\n" % ciSa['nss'][niS])
                    # if ciStamps[niS]['nss'] > 0:
                    if ciSa['nss'][niS] > 0:
                        niS += 1

        iSFrac = niS / float(nStamps)
        tSFrac = ntS / float(nStamps)

        if localForceConvolve == "i":
            sys.stderr.write("%d stamps built (%.2f%%)\n\n" % (niS, iSFrac))
            if iSFrac < minFracGoodStamps:
                flag = 1
        elif localForceConvolve == "t":
            sys.stderr.write("%d stamps built (%.2f%%)\n\n" % (ntS, tSFrac))
            if tSFrac < minFracGoodStamps:
                flag = 1
        elif (iSFrac < minFracGoodStamps) or (tSFrac < minFracGoodStamps):
            sys.stderr.write("%d and %d stamps built (%.2f%%, %.2f%%)\n\n" % (ntS, niS, tSFrac, iSFrac))
            flag = 1
        else:
            sys.stderr.write("%d and %d stamps built (%.2f%%, %.2f%%)\n\n" % (ntS, niS, tSFrac, iSFrac))
            break

        if flag and (status <= 1) and (scaleFitThresh < 1.0):
            kerFitThresh *= scaleFitThresh
            sys.stderr.write("Too few stamps were fit, scaling down fitting threshold to %.2f\n" % kerFitThresh)

            if localForceConvolve != "i":
                # ctStamps = [allocate_stamp_dict(nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
                #             for _ in range(nStamps)]
                # ctSa = StampsArray(nStamps, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
                nVec = nCompKer + nBGVectors
                fwSq = fwKSStamp * fwKSStamp
                ct = {}
                ct['nss'] = np.zeros(nStamps, dtype=np.int32)
                ct['sscnt'] = np.zeros(nStamps, dtype=np.int32)
                ct['x0'] = np.zeros(nStamps, dtype=np.int32)
                ct['y0'] = np.zeros(nStamps, dtype=np.int32)
                ct['x'] = np.zeros(nStamps, dtype=np.int32)
                ct['y'] = np.zeros(nStamps, dtype=np.int32)
                ct['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                ct['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
                ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
                ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
                ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
                ct['chi2'] = np.zeros(nStamps, dtype=np.float64)
                ct['norm'] = np.zeros(nStamps, dtype=np.float64)
                ct['diff'] = np.zeros(nStamps, dtype=np.float64)
                ct['sum_val'] = np.zeros(nStamps, dtype=np.float64)
                ct['mean_val'] = np.zeros(nStamps, dtype=np.float64)
                ct['median'] = np.zeros(nStamps, dtype=np.float64)
                ct['mode'] = np.zeros(nStamps, dtype=np.float64)
                ct['sd'] = np.zeros(nStamps, dtype=np.float64)
                ct['fwhm'] = np.zeros(nStamps, dtype=np.float64)
                ct['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
                ct['valid'] = np.zeros(nStamps, dtype=np.bool_)
                ct['nKSStamps'] = nKSStamps
                ct['fwKSStamp'] = fwKSStamp
                ct['nCompKer'] = nCompKer
                ct['nBGVectors'] = nBGVectors
                ct['nC'] = nC
                ct['fwSq'] = fwSq
                ct['nVec'] = nVec
                ctSa = ct
            if localForceConvolve != "t":
                # ciStamps = [allocate_stamp_dict(nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
                #             for _ in range(nStamps)]
                # ciSa = StampsArray(nStamps, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC)
                nVec = nCompKer + nBGVectors
                fwSq = fwKSStamp * fwKSStamp
                ci = {}
                ci['nss'] = np.zeros(nStamps, dtype=np.int32)
                ci['sscnt'] = np.zeros(nStamps, dtype=np.int32)
                ci['x0'] = np.zeros(nStamps, dtype=np.int32)
                ci['y0'] = np.zeros(nStamps, dtype=np.int32)
                ci['x'] = np.zeros(nStamps, dtype=np.int32)
                ci['y'] = np.zeros(nStamps, dtype=np.int32)
                ci['xss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                ci['yss'] = np.zeros((nStamps, nKSStamps), dtype=np.int32)
                ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float64)
                ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float64)
                ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float64)
                ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float64)
                ci['chi2'] = np.zeros(nStamps, dtype=np.float64)
                ci['norm'] = np.zeros(nStamps, dtype=np.float64)
                ci['diff'] = np.zeros(nStamps, dtype=np.float64)
                ci['sum_val'] = np.zeros(nStamps, dtype=np.float64)
                ci['mean_val'] = np.zeros(nStamps, dtype=np.float64)
                ci['median'] = np.zeros(nStamps, dtype=np.float64)
                ci['mode'] = np.zeros(nStamps, dtype=np.float64)
                ci['sd'] = np.zeros(nStamps, dtype=np.float64)
                ci['fwhm'] = np.zeros(nStamps, dtype=np.float64)
                ci['lfwhm'] = np.zeros(nStamps, dtype=np.float64)
                ci['valid'] = np.zeros(nStamps, dtype=np.bool_)
                ci['nKSStamps'] = nKSStamps
                ci['fwKSStamp'] = fwKSStamp
                ci['nCompKer'] = nCompKer
                ci['nBGVectors'] = nBGVectors
                ci['nC'] = nC
                ci['fwSq'] = fwSq
                ci['nVec'] = nVec
                ciSa = ci

            mRData1d[:] = mRData1d & ~0xa00

        status += 1

    if (niS == 0) and (ntS == 0):
        # return {'niS': 0, 'ntS': 0, 'ctStamps': ctStamps, 'ciStamps': ciStamps,
        #         'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
        # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, -1)
        result = {
            'niS': niS, 'ntS': ntS,
            'ctStamps': ctSa,
            'ciStamps': ciSa,
            'kernel_vec': None,
            'filter_x': None,
            'filter_y': None,
            'status': -1,
        }
        return result
    if localForceConvolve == "i":
        if niS == 0:
            # return {'niS': 0, 'ntS': ntS, 'ctStamps': ctStamps, 'ciStamps': ciStamps,
            #         'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
            # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, -1)
            result = {
                'niS': niS, 'ntS': ntS,
                'ctStamps': ctSa,
                'ciStamps': ciSa,
                'kernel_vec': None,
                'filter_x': None,
                'filter_y': None,
                'status': -1,
            }
            return result
    if localForceConvolve == "t":
        if ntS == 0:
            # return {'niS': niS, 'ntS': 0, 'ctStamps': ctStamps, 'ciStamps': ciStamps,
            #         'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
            # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, -1)
            result = {
                'niS': niS, 'ntS': ntS,
                'ctStamps': ctSa,
                'ciStamps': ciSa,
                'kernel_vec': None,
                'filter_x': None,
                'filter_y': None,
                'status': -1,
            }
            return result

    filter_x = np.zeros(fwKernel * nCompKer, dtype=np.float64)
    filter_y = np.zeros(fwKernel * nCompKer, dtype=np.float64)
    kernel_vec = get_kernel_vec_numpy(ngauss, deg_fixe, usePCA, fwKernel, hwKernel,
                                      sigma_gauss, filter_x, filter_y, PCA)

    logger.debug("region_buildstamps_numpy done")
    # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, kernel_vec, filter_x, filter_y, 0)
    result = {
        'niS': niS, 'ntS': ntS,
        'ctStamps': ctSa,
        'ciStamps': ciSa,
        'kernel_vec': kernel_vec,
        'filter_x': filter_x,
        'filter_y': filter_y,
        'status': 0,
    }
    return result
    # return {
    #     'niS': niS, 'ntS': ntS,
    #     # 'ctStamps': ctStamps,
    #     # 'ciStamps': ciStamps,
    #     'ctStamps': ctSa,
    #     'ciStamps': ciSa,
    #     'kernel_vec': kernel_vec,
    #     'filter_x': filter_x,
    #     'filter_y': filter_y,
    #     'status': 0,
    # }

def region_fit_numpy(buildstamps_result, setup_result, ctx_info, params_info, localForceConvolve, logger=None):
    import sys
    logger.debug("region_fit_numpy start")

    nCompKer = ctx_info['nCompKer']
    nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    nCompTotal = ctx_info['nCompTotal']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']

    hwKernel = params_info['hwKernel']
    ngauss = params_info['ngauss']
    deg_fixe = params_info['deg_fixe']
    kerOrder = params_info['kerOrder']
    bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    verbose = params_info.get('verbose', 0)
    usePCA = params_info.get('usePCA', 0)
    PCA = params_info.get('PCA', None)
    statSig = params_info['statSig']
    kerSigReject = params_info['kerSigReject']
    fillVal = params_info['fillVal']
    figMerit = params_info['figMerit']

    tRData = setup_result['tRData']
    iRData = setup_result['iRData']
    oRData = setup_result['oRData']
    mRData = setup_result['mRData']
    rPixX = setup_result['rPixX']
    rPixY = setup_result['rPixY']

    tRData1d = tRData.ravel()
    iRData1d = iRData.ravel()
    oRData1d = oRData.ravel()
    mRData1d = mRData.ravel()

    # ctStamps = buildstamps_result['ctStamps']
    ctSa = buildstamps_result['ctStamps']
    # ciStamps = buildstamps_result['ciStamps']
    ciSa = buildstamps_result['ciStamps']
    ntS = buildstamps_result['ntS']
    niS = buildstamps_result['niS']
    kernel_vec = buildstamps_result['kernel_vec']
    filter_x = buildstamps_result['filter_x']
    filter_y = buildstamps_result['filter_y']
    # 扁平数组字段（从 ctSa/ciSa dict 读取，替代原来 build_stamps_flatten_helper 展开的字段）
    ctSscnt = ctSa['sscnt'] if ctSa is not None else None
    ctNss = ctSa['nss'] if ctSa is not None else None
    ctVectors = ctSa['vectors'] if ctSa is not None else None
    ctMat = ctSa['mat'] if ctSa is not None else None
    ctScprod = ctSa['scprod'] if ctSa is not None else None
    ctXss = ctSa['xss'] if ctSa is not None else None
    ctYss = ctSa['yss'] if ctSa is not None else None
    ctKrefArea = ctSa['krefArea'] if ctSa is not None else None
    ctSumVal = ctSa['sum_val'] if ctSa is not None else None
    ctX0 = ctSa['x0'] if ctSa is not None else None
    ctY0 = ctSa['y0'] if ctSa is not None else None
    ctNorm = ctSa['norm'] if ctSa is not None else None
    ctDiff = ctSa['diff'] if ctSa is not None else None
    ctNKSStamps = ctSa['nKSStamps'] if ctSa is not None else params_info.get('nKSStamps')
    ctNC = ctSa['nC'] if ctSa is not None else ctx_info.get('nC')
    ciSscnt = ciSa['sscnt'] if ciSa is not None else None
    ciNss = ciSa['nss'] if ciSa is not None else None
    ciVectors = ciSa['vectors'] if ciSa is not None else None
    ciMat = ciSa['mat'] if ciSa is not None else None
    ciScprod = ciSa['scprod'] if ciSa is not None else None
    ciXss = ciSa['xss'] if ciSa is not None else None
    ciYss = ciSa['yss'] if ciSa is not None else None
    ciKrefArea = ciSa['krefArea'] if ciSa is not None else None
    ciSumVal = ciSa['sum_val'] if ciSa is not None else None
    ciX0 = ciSa['x0'] if ciSa is not None else None
    ciY0 = ciSa['y0'] if ciSa is not None else None
    ciNorm = ciSa['norm'] if ciSa is not None else None
    ciDiff = ciSa['diff'] if ciSa is not None else None
    ciNKSStamps = ciSa['nKSStamps'] if ciSa is not None else params_info.get('nKSStamps')
    ciNC = ciSa['nC'] if ciSa is not None else ctx_info.get('nC')

    tMerit = 0.0
    iMerit = 0.0
    convTmpl = 0

    sys.stderr.write("Filling Template sub-stamps\n")
    if localForceConvolve != "i":
        for k in range(ntS):
            # ctStamps[k]['sscnt'] = 0
            ctSa['sscnt'][k] = 0
            # fill_stamp_numpy(ctStamps[k], tRData1d, iRData1d,
            #                  rPixX, rPixY, verbose, ngauss, deg_fixe,
            #                  hwKSStamp, fwKSStamp, hwKernel, fwKernel,
            #                  bgOrder, nCompKer, kerOrder, usePCA,
            #                  filter_x, filter_y, PCA, fillVal, mRData1d)
            fill_stamp_numpy(ctVectors, ctMat, ctScprod, ctXss, ctYss, ctSscnt, ctNss, ctKrefArea, ctSumVal, ctX0, ctY0, k, tRData1d, iRData1d,
                             rPixX, rPixY, verbose, ngauss, deg_fixe,
                             hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, kerOrder, usePCA,
                             filter_x, filter_y, PCA, fillVal, mRData1d)
        if localForceConvolve == "b":
            sys.stderr.write("\n\nTrying to convolve the TEMPLATE to fit IMAGE\n")
            # tMerit = check_stamps_numpy(
            #     ctStamps, ntS, iRData1d, oRData1d,
            #     nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
            #     localForceConvolve, figMerit, kerSigReject, statSig,
            #     fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
            #     kernel_vec, mRData1d)
            # tMerit = check_stamps_numpy(
            #     ctSa, ntS, iRData1d, oRData1d,
            #     nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
            #     localForceConvolve, figMerit, kerSigReject, statSig,
            #     fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
            #     kernel_vec, mRData1d)
            tMerit = check_stamps_numpy(
                ctScprod, ctMat, ctNorm, ctDiff, ctSscnt, ctNss, ctXss, ctYss,
                ctVectors, ctKrefArea, ntS, iRData1d, oRData1d,
                nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
                localForceConvolve, figMerit, kerSigReject, statSig,
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                kernel_vec, mRData1d, nKSStamps=ctNKSStamps, nC=ctNC)
            sys.stderr.write("    Result : merit = %.3f\n" % tMerit)
        else:
            tMerit = 0.0
            iMerit = 0.0

    sys.stderr.write("Filling Image sub-stamps\n")
    if localForceConvolve != "t":
        for k in range(niS):
            # ciStamps[k]['sscnt'] = 0
            ciSa['sscnt'][k] = 0
            # fill_stamp_numpy(ciStamps[k], iRData1d, tRData1d,
            #                  rPixX, rPixY, verbose, ngauss, deg_fixe,
            #                  hwKSStamp, fwKSStamp, hwKernel, fwKernel,
            #                  bgOrder, nCompKer, kerOrder, usePCA,
            #                  filter_x, filter_y, PCA, fillVal, mRData1d)
            fill_stamp_numpy(ciVectors, ciMat, ciScprod, ciXss, ciYss, ciSscnt, ciNss, ciKrefArea, ciSumVal, ciX0, ciY0, k, iRData1d, tRData1d,
                             rPixX, rPixY, verbose, ngauss, deg_fixe,
                             hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, kerOrder, usePCA,
                             filter_x, filter_y, PCA, fillVal, mRData1d)
        if localForceConvolve == "b":
            sys.stderr.write("\n\nTrying to convolve the IMAGE to fit TEMPLATE \n")
            # iMerit = check_stamps_numpy(
            #     ciStamps, niS, tRData1d, oRData1d,
            #     nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
            #     localForceConvolve, figMerit, kerSigReject, statSig,
            #     fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
            #     kernel_vec, mRData1d)
            # iMerit = check_stamps_numpy(
            #     ciSa, niS, tRData1d, oRData1d,
            #     nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
            #     localForceConvolve, figMerit, kerSigReject, statSig,
            #     fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
            #     kernel_vec, mRData1d)
            iMerit = check_stamps_numpy(
                ciScprod, ciMat, ciNorm, ciDiff, ciSscnt, ciNss, ciXss, ciYss,
                ciVectors, ciKrefArea, niS, tRData1d, oRData1d,
                nCompKer, kerOrder, bgOrder, nCompTotal, verbose,
                localForceConvolve, figMerit, kerSigReject, statSig,
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                kernel_vec, mRData1d, nKSStamps=ciNKSStamps, nC=ciNC)
            sys.stderr.write("    Result : merit = %.3f\n" % iMerit)
        else:
            iMerit = 0.0
            tMerit = 0.0

    if (localForceConvolve == "t") or \
       ((tMerit < iMerit) and (localForceConvolve != "i")):
        convTmpl = 1
    else:
        convTmpl = 0

    logger.debug("region_fit_numpy done, convTmpl=%s", convTmpl)
    # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, 0)
    result = {
        'niS': niS, 'ntS': ntS,
        'ctStamps': ctSa,
        'ciStamps': ciSa,
        'kernel_vec': None,
        'filter_x': None,
        'filter_y': None,
        'status': 0,
    }
    result['convTmpl'] = convTmpl
    result['tMerit'] = tMerit
    result['iMerit'] = iMerit
    return result
    # return {
    #     'convTmpl': convTmpl,
    #     'tMerit': tMerit,
    #     'iMerit': iMerit,
    #     # 'ctStamps': ctStamps,
    #     # 'ciStamps': ciStamps,
    #     'ctStamps': ctSa,
    #     'ciStamps': ciSa,
    # }

def region_convolve_diff_numpy(fit_result, setup_result, buildstamps_result,
                                ctx_info, params_info, region_idx, localForceConvolve, logger=None):
    import sys

    nCompKer = ctx_info['nCompKer']
    nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    nStamps = ctx_info['nStamps']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']
    kcStep = ctx_info['kcStep']

    hwKernel = params_info['hwKernel']
    ngauss = params_info['ngauss']
    deg_fixe = params_info['deg_fixe']
    kerOrder = params_info['kerOrder']
    bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    verbose = params_info.get('verbose', 0)
    usePCA = params_info.get('usePCA', 0)
    PCA = params_info.get('PCA', None)
    statSig = params_info['statSig']
    kerSigReject = params_info['kerSigReject']
    kerFracMask = params_info['kerFracMask']
    fillVal = params_info['fillVal']
    fillValNoise = params_info['fillValNoise']
    figMerit = params_info['figMerit']
    tUThresh = params_info['tUThresh']
    tLThresh = params_info['tLThresh']
    tGain = params_info['tGain']
    tRdnoise = params_info['tRdnoise']
    iUThresh = params_info['iUThresh']
    iLThresh = params_info['iLThresh']
    iGain = params_info['iGain']
    iRdnoise = params_info['iRdnoise']
    convolveVariance = params_info.get('convolveVariance', 0)
    sameConv = params_info.get('sameConv', 0)
    savexyflag = params_info.get('savexyflag', 0)
    tnoise_2d = params_info.get('tNoiseFullData', None)
    inoise_2d = params_info.get('iNoiseFullData', None)

    tRData = setup_result['tRData'].copy()
    iRData = setup_result['iRData'].copy()
    mRData = setup_result['mRData'].copy()
    misRData = setup_result['misRData'].copy()
    mtsRData = setup_result['mtsRData'].copy()
    rXBMin = setup_result['rXBMin']
    rYBMin = setup_result['rYBMin']
    rXBMax = setup_result['rXBMax']
    rYBMax = setup_result['rYBMax']
    rPixX = setup_result['rPixX']
    rPixY = setup_result['rPixY']
    rXMin = setup_result['rXMin']
    rYMin = setup_result['rYMin']
    rXMax = setup_result['rXMax']
    rYMax = setup_result['rYMax']

    convTmpl = fit_result['convTmpl']
    # ctStamps = fit_result['ctStamps']
    ctSa = fit_result.get('ctStamps')
    # ciStamps = fit_result['ciStamps']
    ciSa = fit_result.get('ciStamps')
    # 扁平数组字段（从 ctSa/ciSa dict 读取）
    ctSscnt = ctSa['sscnt'] if ctSa is not None else None
    ctNss = ctSa['nss'] if ctSa is not None else None
    ctXss = ctSa['xss'] if ctSa is not None else None
    ctYss = ctSa['yss'] if ctSa is not None else None
    ciSscnt = ciSa['sscnt'] if ciSa is not None else None
    ciNss = ciSa['nss'] if ciSa is not None else None
    ciXss = ciSa['xss'] if ciSa is not None else None
    ciYss = ciSa['yss'] if ciSa is not None else None

    ntS = buildstamps_result['ntS']
    niS = buildstamps_result['niS']
    kernel_vec = buildstamps_result['kernel_vec']
    filter_x = buildstamps_result['filter_x']
    filter_y = buildstamps_result['filter_y']

    kernel_coeffs = np.zeros(nCompKer, dtype=np.float64)
    kernel = np.zeros(fwKernel * fwKernel, dtype=np.float64)

    tRData1d = tRData.ravel()
    iRData1d = iRData.ravel()
    mRData1d = mRData.ravel()
    misRData1d = misRData.ravel()
    mtsRData1d = mtsRData.ravel()

    meansigSubstamps = 0.0
    scatterSubstamps = 0.0
    NskippedSubstamps = 0
    savexy_entries = []
    sumKernel = 0.0
    kerSol = None
    nS = 0

    if convTmpl:
        sys.stderr.write("\n\n Region %d:%d,%d:%d : Convolving TEMPLATE\n" % (rXMin, rXMax, rYMin, rYMax))
        nS = ntS

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        oRData1d = setup_result['oRData'].ravel().copy()
        # fit_result_k = fit_kernel_numpy(
        #     ctStamps, iRData1d, tRData1d, oRData1d,
        #     nCompKer, kerOrder, bgOrder, verbose, nS,
        #     fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
        #     kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
        #     hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal)
        fit_result_k = fit_kernel_numpy(
            ctSa, iRData1d, tRData1d, oRData1d,
            nCompKer, kerOrder, bgOrder, verbose, nS,
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal, logger=logger)
        tKerSol = fit_result_k['kernelSol']
        meansigSubstamps = fit_result_k['meansigSubstamps']
        scatterSubstamps = fit_result_k['scatterSubstamps']
        NskippedSubstamps = fit_result_k['NskippedSubstamps']
        # ctStamps = fit_result_k['stamps']
        ctSa = fit_result_k['stamps']
        kerSol = tKerSol

        logger.debug("[region %d] convolve_diff: fitKernel done", region_idx)
        logger.debug("[region %d] convolve_diff: realloc+mask start", region_idx)
        oRData1d = np.full(rPixX * rPixY, fillVal, dtype=np.float32)

        # for idx in range(rPixX * rPixY):
        #     mtsRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * int(tRData1d[idx] == fillVal)
        #     mtsRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * int(tRData1d[idx] >= tUThresh)
        #     mtsRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * int(tRData1d[idx] <= tLThresh)
        tdata = tRData1d
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tdata == fillVal).astype(np.int32)
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (tdata >= tUThresh).astype(np.int32)
        mtsRData1d |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (tdata <= tLThresh).astype(np.int32)

        mRData1d[:] = 0

        logger.debug("[region %d] convolve_diff: realloc+mask done", region_idx)
        logger.debug("[region %d] convolve_diff: noise rebuild start", region_idx)
        eRData1d = np.full(rPixX * rPixY, fillValNoise, dtype=np.float32)
        if tnoise_2d is not None:
            eRData1d[:] = tnoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            eRData1d[:] = eRData1d * eRData1d
        else:
            eRData1d = make_noise_image4_numpy(tRData1d, 1.0 / tGain, tRdnoise / tGain, rPixX, rPixY)

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        sys.stderr.write("\n Convolving...\n")
        # vData = spatial_convolve_numpy(
        #     tRData1d, eRData1d, rPixX, rPixY, tKerSol, oRData1d, mtsRData1d,
        #     kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
        #     convolveVariance, kerFracMask, mRData1d,
        #     rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
#         vData = spatial_convolve_fast_numpy(
#             tRData1d, eRData1d, rPixX, rPixY, tKerSol, oRData1d, mtsRData1d,
#             kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
#             convolveVariance, kerFracMask, mRData1d,
#             rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
#         # 对比新老版本：用相同输入调用 spatial_convolve_fast_numpy_deprecated_1781965833
#         _image_new = tRData1d.copy()
#         _var_new = eRData1d.copy() if eRData1d is not None else None
#         _ksol_new = tKerSol.copy()
#         _cm_new = mtsRData1d.copy()
#         _kern_new = np.zeros(fwKernel * fwKernel, dtype=np.float64)
#         _kcoeff_new = np.zeros(nCompKer, dtype=np.float64)
#         _kvec_new = [kv.copy() for kv in kernel_vec]
#         _vData_new, _cr_new, _mr_new = spatial_convolve_fast_numpy_deprecated_1781965833(
#             _image_new, _var_new, rPixX, rPixY, _ksol_new, _cm_new,
#             kcStep, hwKernel, fwKernel, _kern_new, _kcoeff_new,
#             convolveVariance, kerFracMask,
#             rPixX, rPixY, nCompKer, kerOrder, _kvec_new)
#         # 逐字节比对输入
#         _input_ok = True
#         if not np.array_equal(_image_new.ravel(), tRData1d.ravel()):
#             sys.stderr.write('  INPUT DIFF: image\n'); _input_ok = False
#         if eRData1d is not None and not np.array_equal(_var_new.ravel(), eRData1d.ravel()):
#             sys.stderr.write('  INPUT DIFF: variance\n'); _input_ok = False
#         if not np.array_equal(_ksol_new, tKerSol):
#             sys.stderr.write('  INPUT DIFF: kernelSol\n'); _input_ok = False
#         if not np.array_equal(_cm_new.ravel(), mtsRData1d.ravel()):
#             sys.stderr.write('  INPUT DIFF: cMask\n'); _input_ok = False
#         # 对比输出
#         _cr_diff = np.abs(oRData1d.astype(np.float64) - _cr_new.astype(np.float64).ravel())
#         _mr_diff = np.abs(mRData1d.astype(np.int32).ravel() - _mr_new.astype(np.int32).ravel())
#         sys.stderr.write('  [SC_CMP] cr max=%.6e mean=%.6e mr max=%d different=%d\n' % (
#             float(_cr_diff.max()), float(_cr_diff.mean()),
#             int(_mr_diff.max()), int((_mr_diff > 0).sum())))
#         if not _input_ok:
#             sys.stderr.write('  ** INPUT MISMATCH DETECTED **\n')
        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
            tRData1d, eRData1d, rPixX, rPixY, tKerSol, mtsRData1d,
            kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
            convolveVariance, kerFracMask,
            rPixX, rPixY, nCompKer, kerOrder, kernel_vec, logger=logger)
        oRData1d[:] = oRData1d_new.ravel()
        mRData1d[:] = mRData1d_new.ravel()
        # vData = spatial_convolve_fast_numpy(  # 回滚：1K 测试精度不合格
        #     tRData1d, eRData1d, rPixX, rPixY, tKerSol, oRData1d, mtsRData1d,
        #     kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
        #     convolveVariance, kerFracMask, mRData1d,
        #     rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
        if vData is not None:
            eRData1d = vData

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        logger.debug("[region %d] convolve_diff: background start", region_idx)
        # for l in range(hwKernel, rPixY - hwKernel):
        #     for k in range(hwKernel, rPixX - hwKernel):
        #         oRData1d[k + rPixX * l] += get_background_numpy(
        #             k, l, tKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY)
        background_loop_jit(oRData1d, tKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)

        logger.debug("[region %d] convolve_diff: background done", region_idx)
        logger.debug("[region %d] convolve_diff: make_kernel start", region_idx)
        sumKernel = make_kernel_numpy(rXMin, rYMin, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Sum Kernel at %d,%d: %f\n" % (rXMin, rYMin, sumKernel))
        sumKernel = make_kernel_numpy(rXMax, rYMax, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Sum Kernel at %d,%d: %f\n" % (rXMax, rYMax, sumKernel))
        sumKernel = make_kernel_numpy(rPixX // 2, rPixY // 2, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Using Kernel Sum = %f\n\n" % sumKernel)

        logger.debug("[region %d] convolve_diff: make_kernel done", region_idx)
        logger.debug("[region %d] convolve_diff: noise_combine start", region_idx)
        tRData1d[:] = fillValNoise
        if inoise_2d is not None:
            tRData1d[:] = inoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            tRData1d[:] = tRData1d * tRData1d
        else:
            tRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain, rPixX, rPixY)

        # for idx in range(rPixX * rPixY):
        #     tRData1d[idx] = math.sqrt(float(tRData1d[idx]) + float(eRData1d[idx]))
        tRData1d = np.sqrt(tRData1d + eRData1d)

        # for idx in range(rPixX * rPixY):
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * int(iRData1d[idx] == fillVal)
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * int(iRData1d[idx] >= iUThresh)
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * int(iRData1d[idx] <= iLThresh)
        #     mRData1d[idx] |= misRData1d[idx]
        #     mRData1d[idx] |= FLAG_OUTPUT_ISBAD * int((misRData1d[idx] & FLAG_INPUT_ISBAD) > 0)
        idata = iRData1d
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (idata == fillVal).astype(np.int32)
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (idata >= iUThresh).astype(np.int32)
        mRData1d |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (idata <= iLThresh).astype(np.int32)
        mRData1d |= misRData1d
        mRData1d |= FLAG_OUTPUT_ISBAD * ((misRData1d & FLAG_INPUT_ISBAD) > 0).astype(np.int32)

        logger.debug("[region %d] convolve_diff: noise_combine done", region_idx)
        if savexyflag:
            for si in range(ntS):
                # for sc in range(ctStamps[si]['nss']):
                for sc in range(ctSa['nss'][si]):
                    entry = {
                        # 'x': int(ctStamps[si]['xss'][sc]),
                        'x': int(ctSa['xss'][si, sc]),
                        # 'y': int(ctStamps[si]['yss'][sc]),
                        'y': int(ctSa['yss'][si, sc]),
                    }
                    # if sc == ctStamps[si]['sscnt']:
                    if sc == ctSa['sscnt'][si]:
                        entry['isUsed'] = 1
                    # elif sc < ctStamps[si]['sscnt']:
                    elif sc < ctSa['sscnt'][si]:
                        entry['isUsed'] = -1
                    else:
                        entry['isUsed'] = 0
                    savexy_entries.append(entry)

        if sameConv:
            localForceConvolve = "t"
            # ciStamps = None
            ciSa = None

    else:
        sys.stderr.write("\n\n Region %d,%d %d,%d : Convolving IMAGE\n" % (rXMin, rXMax, rYMin, rYMax))
        nS = niS

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        oRData1d = setup_result['oRData'].ravel().copy()
        # fit_result_k = fit_kernel_numpy(
        #     ciStamps, tRData1d, iRData1d, oRData1d,
        #     nCompKer, kerOrder, bgOrder, verbose, nS,
        #     fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
        #     kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
        #     hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal)
        fit_result_k = fit_kernel_numpy(
            ciSa, tRData1d, iRData1d, oRData1d,
            nCompKer, kerOrder, bgOrder, verbose, nS,
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal, logger=logger)
        iKerSol = fit_result_k['kernelSol']
        meansigSubstamps = fit_result_k['meansigSubstamps']
        scatterSubstamps = fit_result_k['scatterSubstamps']
        NskippedSubstamps = fit_result_k['NskippedSubstamps']
        # ciStamps = fit_result_k['stamps']
        ciSa = fit_result_k['stamps']
        kerSol = iKerSol

        logger.debug("[region %d] convolve_diff: fitKernel done", region_idx)
        logger.debug("[region %d] convolve_diff: realloc+mask start", region_idx)
        oRData1d = np.full(rPixX * rPixY, fillVal, dtype=np.float32)

        # for idx in range(rPixX * rPixY):
        #     misRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * int(iRData1d[idx] == fillVal)
        #     misRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * int(iRData1d[idx] >= iUThresh)
        #     misRData1d[idx] |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * int(iRData1d[idx] <= iLThresh)
        tdata = iRData1d
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tdata == fillVal).astype(np.int32)
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * (tdata >= iUThresh).astype(np.int32)
        misRData1d |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * (tdata <= iLThresh).astype(np.int32)

        mRData1d[:] = 0

        logger.debug("[region %d] convolve_diff: realloc+mask done", region_idx)
        logger.debug("[region %d] convolve_diff: noise rebuild start", region_idx)
        eRData1d = np.full(rPixX * rPixY, fillValNoise, dtype=np.float32)
        if inoise_2d is not None:
            eRData1d[:] = inoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            eRData1d[:] = eRData1d * eRData1d
        else:
            eRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain, rPixX, rPixY)

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        sys.stderr.write("\n Convolving...\n")
        # vData = spatial_convolve_numpy(
        #     iRData1d, eRData1d, rPixX, rPixY, iKerSol, oRData1d, misRData1d,
        #     kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
        #     convolveVariance, kerFracMask, mRData1d,
        #     rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
#         vData = spatial_convolve_fast_numpy(
#             iRData1d, eRData1d, rPixX, rPixY, iKerSol, oRData1d, misRData1d,
#             kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
#             convolveVariance, kerFracMask, mRData1d,
#             rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
#         # vData = spatial_convolve_fast_numpy(  # 回滚：1K 测试精度不合格
#         #     iRData1d, eRData1d, rPixX, rPixY, iKerSol, oRData1d, misRData1d,
#         #     kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
#         #     convolveVariance, kerFracMask, mRData1d,
#         #     rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
            iRData1d, eRData1d, rPixX, rPixY, iKerSol, misRData1d,
            kcStep, hwKernel, fwKernel, kernel, kernel_coeffs,
            convolveVariance, kerFracMask,
            rPixX, rPixY, nCompKer, kerOrder, kernel_vec, logger=logger)
        oRData1d[:] = oRData1d_new.ravel()
        mRData1d[:] = mRData1d_new.ravel()
        if vData is not None:
            eRData1d = vData

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        logger.debug("[region %d] convolve_diff: background start", region_idx)
        # for l in range(hwKernel, rPixY - hwKernel):
        #     for k in range(hwKernel, rPixX - hwKernel):
        #         oRData1d[k + rPixX * l] += get_background_numpy(
        #             k, l, iKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY)
        background_loop_jit(oRData1d, iKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)

        logger.debug("[region %d] convolve_diff: background done", region_idx)
        logger.debug("[region %d] convolve_diff: make_kernel start", region_idx)
        sumKernel = make_kernel_numpy(rXMin, rYMin, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Sum Kernel at %d,%d: %f\n" % (rXMin, rYMin, sumKernel))
        sumKernel = make_kernel_numpy(rXMax, rYMax, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Sum Kernel at %d,%d: %f\n" % (rXMax, rYMax, sumKernel))
        sumKernel = make_kernel_numpy(rPixX // 2, rPixY // 2, iKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        sys.stderr.write(" Using Kernel Sum = %f\n\n" % sumKernel)

        logger.debug("[region %d] convolve_diff: make_kernel done", region_idx)
        logger.debug("[region %d] convolve_diff: noise_combine start", region_idx)
        iRData1d[:] = fillValNoise
        if tnoise_2d is not None:
            iRData1d[:] = tnoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            iRData1d[:] = iRData1d * iRData1d
        else:
            iRData1d = make_noise_image4_numpy(tRData1d, 1.0 / tGain, tRdnoise / tGain, rPixX, rPixY)

        # for idx in range(rPixX * rPixY):
        #     iRData1d[idx] = math.sqrt(float(iRData1d[idx]) + float(eRData1d[idx]))
        iRData1d = np.sqrt(iRData1d + eRData1d)

        # for idx in range(rPixX * rPixY):
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * int(tRData1d[idx] == fillVal)
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL) * int(tRData1d[idx] >= tUThresh)
        #     mRData1d[idx] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL) * int(tRData1d[idx] <= tLThresh)
        #     mRData1d[idx] |= mtsRData1d[idx]
        #     mRData1d[idx] |= FLAG_OUTPUT_ISBAD * int((mtsRData1d[idx] & FLAG_INPUT_ISBAD) > 0)
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
                    entry = {
                        # 'x': int(ciStamps[si]['xss'][sc]),
                        'x': int(ciSa['xss'][si, sc]),
                        # 'y': int(ciStamps[si]['yss'][sc]),
                        'y': int(ciSa['yss'][si, sc]),
                    }
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
            localForceConvolve = "i"
            # ctStamps = None
            ctSa = None

    noiseData1d = tRData1d if convTmpl else iRData1d

    # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, 0)
    result = {
        'niS': niS, 'ntS': ntS,
        'ctStamps': ctSa,
        'ciStamps': ciSa,
        'kernel_vec': None,
        'filter_x': None,
        'filter_y': None,
        'status': 0,
    }
    result.update({
        'oRData': oRData1d.reshape(rPixY, rPixX),
        'noiseData': noiseData1d.reshape(rPixY, rPixX),
        'mRData': mRData1d.reshape(rPixY, rPixX),
        'sumKernel': sumKernel,
        'meansigSubstamps': meansigSubstamps,
        'scatterSubstamps': scatterSubstamps,
        'NskippedSubstamps': NskippedSubstamps,
        'convTmpl': convTmpl,
        'nS': nS,
        'localForceConvolve': localForceConvolve,
        'kerSol': kerSol,
        'savexy_entries': savexy_entries,
        'savexyXmin': rXBMin + (0 if convTmpl else 1),
        'savexyYmin': rYBMin + (0 if convTmpl else 1),
        # 'ctStamps': ctStamps,
        # 'ciStamps': ciStamps,
        'ctStamps': ctSa,
        'ciStamps': ciSa,
        'tRData': tRData1d.reshape(rPixY, rPixX),
        'iRData': iRData1d.reshape(rPixY, rPixX),
    })
    return result
    # return {
    #     'oRData': oRData1d.reshape(rPixY, rPixX),
    #     'noiseData': noiseData1d.reshape(rPixY, rPixX),
    #     'mRData': mRData1d.reshape(rPixY, rPixX),
    #     'sumKernel': sumKernel,
    #     'meansigSubstamps': meansigSubstamps,
    #     'scatterSubstamps': scatterSubstamps,
    #     'NskippedSubstamps': NskippedSubstamps,
    #     'convTmpl': convTmpl,
    #     'nS': nS,
    #     'localForceConvolve': localForceConvolve,
    #     'kerSol': kerSol,
    #     'savexy_entries': savexy_entries,
    #     'savexyXmin': rXBMin + (0 if convTmpl else 1),
    #     'savexyYmin': rYBMin + (0 if convTmpl else 1),
    #     # 'ctStamps': ctStamps,
    #     # 'ciStamps': ciStamps,
    #     'ctStamps': ctSa,
    #     'ciStamps': ciSa,
    #     'tRData': tRData1d.reshape(rPixY, rPixX),
    #     'iRData': iRData1d.reshape(rPixY, rPixX),
    # }

def region_output_numpy(convolve_result, setup_result, fit_result,
                         diff_out, noise_out, conv_out, mask_out,
                         ctx_info, params_info, region_idx, stats_list,
                         logger=None):
    import sys
    # import time
    logger.debug("[region %d] region_output_numpy start", region_idx)
    # start_time = time.time()

    nCompKer = ctx_info['nCompKer']
    nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    nStamps = ctx_info['nStamps']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']

    hwKernel = params_info['hwKernel']
    kerOrder = params_info['kerOrder']
    bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    verbose = params_info.get('verbose', 0)
    statSig = params_info['statSig']
    fillVal = params_info['fillVal']
    fillValNoise = params_info['fillValNoise']
    photNormalize = params_info['photNormalize']
    figMerit = params_info['figMerit']
    rescaleOK = params_info.get('rescaleOK', 0)
    kfSpreadMask2 = params_info.get('kfSpreadMask2', -1.0)

    rPixX = setup_result['rPixX']
    rPixY = setup_result['rPixY']
    rXMin = setup_result['rXMin']
    rYMin = setup_result['rYMin']
    rXMax = setup_result['rXMax']
    rYMax = setup_result['rYMax']
    xBufLo = setup_result['xBufLo']
    yBufLo = setup_result['yBufLo']
    xBufHi = setup_result['xBufHi']
    yBufHi = setup_result['yBufHi']
    fpixelOutX = setup_result['fpixelOutX']
    fpixelOutY = setup_result['fpixelOutY']
    lpixelOutX = setup_result['lpixelOutX']
    lpixelOutY = setup_result['lpixelOutY']
    rXBMin = setup_result['rXBMin']
    rYBMin = setup_result['rYBMin']

    convTmpl = convolve_result['convTmpl']
    nS = convolve_result['nS']
    sumKernel = convolve_result['sumKernel']
    meansigSubstamps = convolve_result['meansigSubstamps']
    scatterSubstamps = convolve_result['scatterSubstamps']
    NskippedSubstamps = convolve_result['NskippedSubstamps']

    oRData = convolve_result['oRData'].copy()
    noiseData = convolve_result['noiseData'].copy()
    mRData = convolve_result['mRData'].copy()
    tRData = convolve_result['tRData']
    iRData = convolve_result['iRData']
    ctSa = convolve_result.get('ctStamps')
    ciSa = convolve_result.get('ciStamps')
    # 扁平数组字段（从 ctSa/ciSa dict 读取）
    ctSscntOut = ctSa['sscnt'] if ctSa is not None else None
    ctNssOut = ctSa['nss'] if ctSa is not None else None
    ctXssOut = ctSa['xss'] if ctSa is not None else None
    ctYssOut = ctSa['yss'] if ctSa is not None else None
    ciSscntOut = ciSa['sscnt'] if ciSa is not None else None
    ciNssOut = ciSa['nss'] if ciSa is not None else None
    ciXssOut = ciSa['xss'] if ciSa is not None else None
    ciYssOut = ciSa['yss'] if ciSa is not None else None
    # (now also exposing flattened fields from convolve_result)

    oRData1d = oRData.ravel()
    noiseData1d = noiseData.ravel()
    mRData1d = mRData.ravel()
    tRData1d = tRData.ravel()
    iRData1d = iRData.ravel()

    mRData2d = mRData1d.reshape(rPixY, rPixX)
    mRData2d[:, :hwKernel] |= FLAG_OUTPUT_ISBAD
    mRData2d[:, rPixX - hwKernel:rPixX] |= FLAG_OUTPUT_ISBAD
    mRData2d[:hwKernel, hwKernel:rPixX - hwKernel] |= FLAG_OUTPUT_ISBAD
    mRData2d[rPixY - hwKernel:, hwKernel:rPixX - hwKernel] |= FLAG_OUTPUT_ISBAD

    # tm1 = time.time(); logger.debug("[out] pre-output setup done, %.3fs", tm1 - start_time)

    sys.stderr.write(" Creating and writing output images...\n")

    inv1 = 1.0 / sumKernel

    if conv_out is not None:
        inner_sy = slice(hwKernel, rPixY - hwKernel)
        inner_sx = slice(hwKernel, rPixX - hwKernel)
        norm_ok = (photNormalize[0:1] != "u") and \
                  ((convTmpl and photNormalize[0:1] == "t") or
                   (not convTmpl and photNormalize[0:1] == "i"))
        if norm_ok:
            oRData2d = oRData1d.reshape(rPixY, rPixX)
            oRData2d[inner_sy, inner_sx] *= inv1

        insert_subregion_flt_numpy(
            oRData, conv_out, fpixelOutX, fpixelOutY,
            lpixelOutX, lpixelOutY, xBufLo, yBufLo)

        if norm_ok:
            oRData2d[inner_sy, inner_sx] *= sumKernel

    meansigSubstampsF = 0.0
    scatterSubstampsF = 0.0
    inner_sy = slice(hwKernel, rPixY - hwKernel)
    inner_sx = slice(hwKernel, rPixX - hwKernel)

    if convTmpl:
        oRData2d = oRData1d.reshape(rPixY, rPixX)
        iRData2d = iRData1d.reshape(rPixY, rPixX)
        oRData2d[inner_sy, inner_sx] -= iRData2d[inner_sy, inner_sx]
        if (photNormalize[0:1] != "u") and (photNormalize[0:1] == "t"):
            oRData2d[inner_sy, inner_sx] *= inv1
            noiseData2d = noiseData1d.reshape(rPixY, rPixX)
            noiseData2d[inner_sy, inner_sx] *= inv1
        oRData2d[inner_sy, inner_sx] *= -1.0

        if figMerit[0:1] == "v":
            # tm2 = time.time(); logger.debug("[out] final_stamp_sig loop start")
            temp2 = np.zeros(nS, dtype=np.float32)
            kk = 0
            for l in range(nS):
                # if ctStamps[l]['sscnt'] < ctStamps[l]['nss']:
                if ctSscntOut is not None and ctNssOut is not None and ctXssOut is not None and ctYssOut is not None:
                    if ctSscntOut[l] < ctNssOut[l]:
                        # sig = get_final_stamp_sig_numpy(
                        #     ctStamps[l], oRData1d, noiseData1d,
                        #     fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        sig = get_final_stamp_sig_numpy(
                            ctXssOut, ctYssOut, ctSscntOut, l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        kk += 1
                elif ctSa is not None:
                    if ctSa['sscnt'][l] < ctSa['nss'][l]:
                        sig = get_final_stamp_sig_numpy(
                            ctSa['xss'], ctSa['yss'], ctSa['sscnt'], l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        kk += 1
            # tm3 = time.time(); logger.debug("[out] final_stamp_sig loop done, %.3fs", tm3 - tm2)
            mean_f, stdev_f, rc_f = sigma_clip_numpy(temp2[:kk], 10, statSig)
            meansigSubstampsF = mean_f
            scatterSubstampsF = stdev_f
            sys.stderr.write("   FINAL Mean sig: %6.3f stdev: %6.3f\n" % (meansigSubstampsF, scatterSubstampsF))

    else:
        oRData2d = oRData1d.reshape(rPixY, rPixX)
        tRData2d = tRData1d.reshape(rPixY, rPixX)
        oRData2d[inner_sy, inner_sx] -= tRData2d[inner_sy, inner_sx]
        if (photNormalize[0:1] != "u") and (photNormalize[0:1] == "i"):
            oRData2d[inner_sy, inner_sx] *= inv1
            noiseData2d = noiseData1d.reshape(rPixY, rPixX)
            noiseData2d[inner_sy, inner_sx] *= inv1

        if figMerit[0:1] == "v":
            # tm2 = time.time(); logger.debug("[out] final_stamp_sig loop start")
            temp2 = np.zeros(nS, dtype=np.float32)
            kk = 0
            for l in range(nS):
                # if ciStamps[l]['sscnt'] < ciStamps[l]['nss']:
                if ciSscntOut is not None and ciNssOut is not None and ciXssOut is not None and ciYssOut is not None:
                    if ciSscntOut[l] < ciNssOut[l]:
                        # sig = get_final_stamp_sig_numpy(
                        #     ciStamps[l], oRData1d, noiseData1d,
                        #     fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        sig = get_final_stamp_sig_numpy(
                            ciXssOut, ciYssOut, ciSscntOut, l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        kk += 1
                elif ciSa is not None:
                    if ciSa['sscnt'][l] < ciSa['nss'][l]:
                        sig = get_final_stamp_sig_numpy(
                            ciSa['xss'], ciSa['yss'], ciSa['sscnt'], l, oRData1d, noiseData1d,
                            fwKSStamp, hwKSStamp, rPixX, mRData1d)
                        temp2[kk] = sig
                        kk += 1
            # tm3 = time.time(); logger.debug("[out] final_stamp_sig loop done, %.3fs", tm3 - tm2)
            mean_f, stdev_f, rc_f = sigma_clip_numpy(temp2[:kk], 10, statSig)
            meansigSubstampsF = mean_f
            scatterSubstampsF = stdev_f
            sys.stderr.write("    FINAL Mean sig: %6.3f stdev: %6.3f\n" % (meansigSubstampsF, scatterSubstampsF))

    oRData_2d = oRData1d.reshape(rPixY, rPixX)
    noiseData_2d = noiseData1d.reshape(rPixY, rPixX)
    mRData_2d = mRData1d.reshape(rPixY, rPixX)

    # tm4 = time.time(); logger.debug("[out] get_stamp_stats3 start")

    sys.stderr.write(" Getting diffim stats for GOOD pixels : \n")
    res_good = get_stamp_stats3_numpy(
        oRData_2d, 0, 0, rPixX, rPixY,
        0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
    mean_val = res_good[1]
    sd_val = res_good[4]
    sys.stderr.write("   Mean   : %.2f\n" % res_good[1])
    sys.stderr.write("   Median : %.2f\n" % res_good[2])
    sys.stderr.write("   Mode   : %.2f\n" % res_good[3])
    sys.stderr.write("   Stdev  : %.2f\n" % res_good[4])

    sys.stderr.write(" Getting noiseim stats for GOOD pixels : \n")
    nres_good = get_stamp_stats3_numpy(
        noiseData_2d, 0, 0, rPixX, rPixY,
        0x0, 0xffff, 5, rPixX, mRData_2d, statSig)
    nmean_val = nres_good[1]

    x2norm, nx2norm = get_noise_stats3_numpy(
        oRData1d, noiseData1d, 0x0, 0xffff, rPixX, rPixY, mRData1d)

    sys.stderr.write("   Mean   : %.2f\n" % nres_good[1])
    sys.stderr.write("   Median : %.2f\n" % nres_good[2])
    sys.stderr.write("   Mode   : %.2f\n" % nres_good[3])
    sys.stderr.write("   Stdev  : %.2f\n" % nres_good[4])
    sys.stderr.write(" Emperical / Expected Noise for GOOD pixels = %.2f\n" % (sd_val / nmean_val if nmean_val != 0 else 0.0))
    sys.stderr.write(" X2NORM = %.2f\n\n" % x2norm)

    diffrat = sd_val / nmean_val if nmean_val != 0 else 0.0

    sys.stderr.write(" Getting diffim stats for OK pixels : \n")
    res_ok = get_stamp_stats3_numpy(
        oRData_2d, 0, 0, rPixX, rPixY,
        0xff, FLAG_OUTPUT_ISBAD, 5, rPixX, mRData_2d, statSig)
    meanm_val = res_ok[1]
    sdm_val = res_ok[4]
    sys.stderr.write("   Mean   : %.2f\n" % res_ok[1])
    sys.stderr.write("   Median : %.2f\n" % res_ok[2])
    sys.stderr.write("   Mode   : %.2f\n" % res_ok[3])
    sys.stderr.write("   Stdev  : %.2f\n" % res_ok[4])

    sys.stderr.write(" Getting noiseim stats for OK pixels : \n")
    nres_ok = get_stamp_stats3_numpy(
        noiseData_2d, 0, 0, rPixX, rPixY,
        0xff, FLAG_OUTPUT_ISBAD, 5, rPixX, mRData_2d, statSig)
    nmeanm_val = nres_ok[1]
    sys.stderr.write("   Mean   : %.2f\n" % nres_ok[1])
    sys.stderr.write("   Median : %.2f\n" % nres_ok[2])
    sys.stderr.write("   Mode   : %.2f\n" % nres_ok[3])
    sys.stderr.write("   Stdev  : %.2f\n" % nres_ok[4])

    sys.stderr.write(" Emperical / Expected Noise for OK pixels = %.2f\n\n" % (sdm_val / nmeanm_val if nmeanm_val != 0 else 0.0))

    # tm5 = time.time(); logger.debug("[out] get_stamp_stats3 done, %.3fs", tm5 - tm4)

    if rescaleOK:
        if diffrat != 0:
            diffrat = (sdm_val / nmeanm_val if nmeanm_val != 0 else 0.0) / diffrat
        if diffrat > 1:
            sys.stderr.write(" Scale OK pixel noise by = %.2f\n" % diffrat)
            ok_mask = (mRData1d & 0xff).astype(np.bool_) & ~((mRData1d & FLAG_OUTPUT_ISBAD).astype(np.bool_))
            noiseData1d[ok_mask] *= diffrat
        else:
            sys.stderr.write(" Leave OK pixel noise as-is\n")

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

    stats = {
        'convTmpl': convTmpl,
        'sumKernel': sumKernel,
        'meansigSubstamps': meansigSubstamps,
        'scatterSubstamps': scatterSubstamps,
        'meansigSubstampsF': meansigSubstampsF,
        'scatterSubstampsF': scatterSubstampsF,
        'x2norm': x2norm,
        'nx2norm': nx2norm,
        'mean': mean_val,
        'sd': sd_val,
        'nmean': nmean_val,
        'meanm': meanm_val,
        'sdm': sdm_val,
        'nmeanm': nmeanm_val,
        'diffrat': diffrat,
        'kerSol': kerSol,
    }

    if stats_list is not None and region_idx < len(stats_list):
        stats_list[region_idx] = stats

    sys.stderr.write("Region %i finished\n\n" % region_idx)

    logger.debug("[region %d] region_output_numpy done", region_idx)
    return stats

def hotpants(
    inim, tmplim,
    tni=None, ini=None, tmi=None, imi=None,
    tu=25000., tuk=None, tl=0., tg=1., tr=0., tp=0.,
    iu=25000., iuk=None, il=0., ig=1., ir=0., ip=0.,
    r=10, ko=2, bgo=1,
    ng=3, ng_deg=None, ng_sig=None,
    pca=None,
    nrx=1, nry=1, rf=None,
    nsx=10, nsy=10, ssf=None, afssc=1, nss=3, rss=15,
    ft=20.0, sft=0.5, nft=0.1,
    ssig=3.0, ks=2.0, kfm=0.99,
    mins=1.0, mous=1.0,
    fi=1e-30, fin=0.,
    c='b', n='t', fom='v',
    sconv=0, okn=0, convvar=0,
    v=1, kcs=0,
    uss=0, savexy=0,
    dump_dir=None,
    logger=None,
):
    import sys, os, struct, copy
    import time as tm
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG, stream=sys.stderr)
    if logger is None:
        logger = logging.getLogger('hotpants')
    if ng_deg is None:
        ng_deg = [6, 4, 2]
    if ng_sig is None:
        ng_sig = [0.7, 1.5, 3.0]

    tmpl_arr = np.ascontiguousarray(tmplim, dtype=np.float32)
    sci_arr = np.ascontiguousarray(inim, dtype=np.float32)
    tNx = tmpl_arr.shape[1]
    tNy = tmpl_arr.shape[0]
    iNx = sci_arr.shape[1]
    iNy = sci_arr.shape[0]

    tni_arr = None
    if tni is not None:
        tni_arr = np.ascontiguousarray(tni, dtype=np.float32)

    ini_arr = None
    if ini is not None:
        ini_arr = np.ascontiguousarray(ini, dtype=np.float32)

    tmi_arr = None
    if tmi is not None:
        tmi_arr = np.ascontiguousarray(tmi, dtype=np.int32)

    imi_arr = None
    if imi is not None:
        imi_arr = np.ascontiguousarray(imi, dtype=np.int32)

    tuk_val = float(tu) if tuk is None else float(tuk)
    iuk_val = float(iu) if iuk is None else float(iuk)

    sig_arr = np.array([1.0 / (2.0 * s * s) for s in ng_sig], dtype=np.float32)
    deg_arr = np.array(ng_deg, dtype=np.int32)

    use_pca = 0
    pca_arrs = []
    if pca is not None:
        use_pca = 1
        pca_ng = len(pca)
        pca_arrs = [np.ascontiguousarray(pca[pi], dtype=np.float32) for pi in range(pca_ng)]
        r = pca[0].shape[0] // 2
        deg_arr = np.zeros(pca_ng, dtype=np.int32)
        sig_arr = np.full(pca_ng, -1.0, dtype=np.float32)
        ng = pca_ng

    xMin = 0
    yMin = 0
    xMax = min(tNx, iNx) - 1
    yMax = min(tNy, iNy) - 1

    if rf is not None:
        nR = len(rf)
        rxmins = np.array([reg[0] for reg in rf], dtype=np.int32)
        rxmaxs = np.array([reg[1] for reg in rf], dtype=np.int32)
        rymins = np.array([reg[2] for reg in rf], dtype=np.int32)
        rymaxs = np.array([reg[3] for reg in rf], dtype=np.int32)
    else:
        nR = nrx * nry
        rxmins_l, rxmaxs_l, rymins_l, rymaxs_l = [], [], [], []
        for j in range(nry):
            for i in range(nrx):
                rxmins_l.append(xMin + i * xMax // nrx)
                rymins_l.append(yMin + j * yMax // nry)
                rxmaxs_l.append(min((i + 1) * xMax // nrx, xMax))
                rymaxs_l.append(min((j + 1) * yMax // nry, yMax))
        rxmins = np.array(rxmins_l, dtype=np.int32)
        rxmaxs = np.array(rxmaxs_l, dtype=np.int32)
        rymins = np.array(rymins_l, dtype=np.int32)
        rymaxs = np.array(rymaxs_l, dtype=np.int32)

    xcmp_arr = None
    ycmp_arr = None
    ncmp = 0
    if ssf is not None:
        xcmp_arr = np.array([p[0] - 1 for p in ssf], dtype=np.float32)
        ycmp_arr = np.array([p[1] - 1 for p in ssf], dtype=np.float32)
        ncmp = len(ssf)

    c_bytes = c.encode('ascii')
    n_bytes = n.encode('ascii')
    fom_bytes = fom.encode('ascii')

    oNx = max(tNx, iNx)
    oNy = max(tNy, iNy)
    diff_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    noise_out = np.full((oNy, oNx), float(fin), dtype=np.float32)
    conv_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    mask_out = np.zeros((oNy, oNx), dtype=np.int32)

    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        _path = os.path.join(dump_dir, "py_input.bin")
        with open(_path, "wb") as _f:
            def _de(nm, db):
                nb = nm.encode('ascii')
                _f.write(struct.pack('i', len(nb)))
                _f.write(nb)
                dl = len(db) if db else 0
                _f.write(struct.pack('l', dl))
                if dl > 0: _f.write(db)
            def _di(nm, val): _de(nm, struct.pack('i', int(val)))
            def _dl(nm, val): _de(nm, struct.pack('l', int(val)))
            def _df(nm, val): _de(nm, struct.pack('f', float(val)))
            def _ds(nm, s):
                if s is None: _de(nm, None)
                else: _de(nm, s.encode('ascii') + b'\x00')
            _dl("tNx",tNx); _dl("tNy",tNy); _dl("iNx",iNx); _dl("iNy",iNy)
            _dl("oNx",oNx); _dl("oNy",oNy)
            _di("nR",nR); _di("hwKernel",r); _di("ngauss",ng)
            _di("kerOrder",ko); _di("bgOrder",bgo)
            _di("nStampX",nsx); _di("nStampY",nsy)
            _di("nKSStamps",nss); _di("hwKSStamp",rss)
            _di("useFullSS",uss); _di("findSSC",afssc)
            _df("kerFitThresh",ft); _df("scaleFitThresh",sft)
            _df("minFracGoodStamps",nft)
            _df("statSig",ssig); _df("kerSigReject",ks); _df("kerFracMask",kfm)
            _df("tUThresh",tu); _df("tLThresh",tl)
            _df("tGain",tg); _df("tRdnoise",tr); _df("tPedestal",tp)
            _df("iUThresh",iu); _df("iLThresh",il)
            _df("iGain",ig); _df("iRdnoise",ir); _df("iPedestal",ip)
            _df("tUKThresh",tuk_val); _df("iUKThresh",iuk_val)
            _df("kfSpreadMask1",mins); _df("kfSpreadMask2",mous)
            _df("fillVal",fi); _df("fillValNoise",fin)
            _di("sameConv",sconv); _di("rescaleOK",okn)
            _di("convolveVariance",convvar)
            _di("usePCA",use_pca); _di("Ncmp",ncmp)
            _di("verbose",v); _di("kcStep",kcs); _di("savexyflag",savexy)
            _ds("forceConvolve",c); _ds("photNormalize",n); _ds("figMerit",fom)
            _de("deg_fixe", bytes(np.ascontiguousarray(deg_arr)))
            _de("sigma_gauss", bytes(np.ascontiguousarray(sig_arr)))
            _de("tFullData", bytes(np.ascontiguousarray(tmpl_arr)))
            _de("iFullData", bytes(np.ascontiguousarray(sci_arr)))
            _de("tNoiseFullData", bytes(np.ascontiguousarray(tni_arr)) if tni_arr is not None else None)
            _de("iNoiseFullData", bytes(np.ascontiguousarray(ini_arr)) if ini_arr is not None else None)
            _de("tMaskFullData", bytes(np.ascontiguousarray(tmi_arr)) if tmi_arr is not None else None)
            _de("iMaskFullData", bytes(np.ascontiguousarray(imi_arr)) if imi_arr is not None else None)
            _de("rXMins", bytes(np.ascontiguousarray(rxmins)))
            _de("rXMaxs", bytes(np.ascontiguousarray(rxmaxs)))
            _de("rYMins", bytes(np.ascontiguousarray(rymins)))
            _de("rYMaxs", bytes(np.ascontiguousarray(rymaxs)))
            _de("xcmp", bytes(np.ascontiguousarray(xcmp_arr)) if ssf is not None else None)
            _de("ycmp", bytes(np.ascontiguousarray(ycmp_arr)) if ssf is not None else None)
            _de("diffOut", bytes(np.ascontiguousarray(diff_out)))
            _de("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            _de("convOut", bytes(np.ascontiguousarray(conv_out)))
            _de("maskOut", bytes(np.ascontiguousarray(mask_out)))

    ctx_info = compute_ctx_info(
        int(r), int(ng), deg_arr, sig_arr,
        int(ko), int(bgo),
        int(nsx), int(nsy), int(nss), int(rss),
        int(uss), float(ft), int(kcs),
        tNx, tNy, iNx, iNy, nR)

    params_info = {
        'hwKernel': int(r), 'ngauss': int(ng),
        'deg_fixe': deg_arr, 'sigma_gauss': sig_arr,
        'kerOrder': int(ko), 'bgOrder': int(bgo),
        'hwKSStamp': int(rss), 'nKSStamps': int(nss),
        'scaleFitThresh': float(sft), 'minFracGoodStamps': float(nft),
        'tUKThresh': float(tuk_val), 'iUKThresh': float(iuk_val),
        'tUThresh': float(tu), 'tLThresh': float(tl),
        'tGain': float(tg), 'tRdnoise': float(tr),
        'iUThresh': float(iu), 'iLThresh': float(il),
        'iGain': float(ig), 'iRdnoise': float(ir),
        'verbose': int(v), 'usePCA': int(use_pca),
        'PCA': pca_arrs if use_pca else None,
        'xcmp': np.asarray(xcmp_arr) if ssf is not None else None,
        'ycmp': np.asarray(ycmp_arr) if ssf is not None else None,
        'Ncmp': ncmp,
        'statSig': float(ssig), 'kerSigReject': float(ks), 'kerFracMask': float(kfm),
        'fillVal': float(fi), 'fillValNoise': float(fin),
        'figMerit': fom, 'photNormalize': n,
        'convolveVariance': int(convvar), 'sameConv': int(sconv),
        'savexyflag': int(savexy), 'rescaleOK': int(okn),
        'kfSpreadMask2': float(mous), 'findSSC': int(afssc),
        'tNoiseFullData': tni_arr,
        'iNoiseFullData': ini_arr,
    }

    localFC_py = c
    stats_list_py = [None] * nR

    for ri in range(nR):
        t0 = tm.time()
        logger.debug("  [%d] setup start", ri)
        py = region_setup_numpy(
            tmpl_arr, sci_arr,
            tni_arr, ini_arr, tmi_arr, imi_arr,
            ri, rxmins, rxmaxs, rymins, rymaxs, nR,
            r, ctx_info['fwStamp'], ctx_info['sBorder'],
            ctx_info['xMin'], ctx_info['yMin'], ctx_info['xMax'], ctx_info['yMax'],
            fi, fin,
            tp, ip,
            tg, tr, ig, ir,
            tu, tl, iu, il,
            mins, logger=logger)
        sys.stderr.write(f"  [{ri}] setup: {tm.time()-t0:.3f}s\n"); sys.stderr.flush()

        t1 = tm.time()
        bs_result = region_buildstamps_numpy(py, ctx_info, params_info, localFC_py, logger=logger)
        sys.stderr.write(f"  [{ri}] buildstamps: {tm.time()-t1:.3f}s\n"); sys.stderr.flush()
        if bs_result['status'] != 0:
            continue

        t1 = tm.time()
        fit_result = region_fit_numpy(bs_result, py, ctx_info, params_info, localFC_py, logger=logger)
        sys.stderr.write(f"  [{ri}] fit: {tm.time()-t1:.3f}s\n"); sys.stderr.flush()

        t1 = tm.time()
        conv_result = region_convolve_diff_numpy(
            fit_result, py, bs_result, ctx_info, params_info, ri, localFC_py, logger=logger)
        sys.stderr.write(f"  [{ri}] convolve_diff: {tm.time()-t1:.3f}s\n"); sys.stderr.flush()
        localFC_py = conv_result.get('localForceConvolve', localFC_py)

        t1 = tm.time()
        stats_entry = region_output_numpy(
            conv_result, py, fit_result,
            diff_out, noise_out, conv_out, mask_out,
            ctx_info, params_info, ri, None, logger=logger)
        sys.stderr.write(f"  [{ri}] output: {tm.time()-t1:.3f}s\n"); sys.stderr.flush()
        stats_list_py[ri] = stats_entry

    if dump_dir is not None:
        _opath = os.path.join(dump_dir, "py_output.bin")
        with open(_opath, "wb") as _f:
            def _de2(nm, db):
                nb = nm.encode('ascii')
                _f.write(struct.pack('i', len(nb)))
                _f.write(nb)
                dl2 = len(db) if db else 0
                _f.write(struct.pack('l', dl2))
                if dl2 > 0: _f.write(db)
            _de2("diffOut", bytes(np.ascontiguousarray(diff_out)))
            _de2("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            _de2("convOut", bytes(np.ascontiguousarray(conv_out)))
            _de2("maskOut", bytes(np.ascontiguousarray(mask_out)))

    stats_list = []
    for si in range(nR):
        if stats_list_py[si] is not None:
            entry = stats_list_py[si]
            stats_list.append({
                'conv_tmpl': entry['convTmpl'],
                'sum_kernel': entry['sumKernel'],
                'mean_sig': entry['meansigSubstamps'],
                'scatter_sig': entry['scatterSubstamps'],
                'final_mean_sig': entry['meansigSubstampsF'],
                'final_scatter_sig': entry['scatterSubstampsF'],
                'x2norm': entry['x2norm'],
                'nx2norm': entry['nx2norm'],
                'diff_mean': entry['mean'],
                'diff_sd': entry['sd'],
                'noise_mean': entry['nmean'],
                'diff_mean_ok': entry['meanm'],
                'diff_sd_ok': entry['sdm'],
                'noise_mean_ok': entry['nmeanm'],
                'diffrat': entry['diffrat'],
            })
        else:
            stats_list.append({
                'conv_tmpl': 0, 'sum_kernel': 0.0,
                'mean_sig': 0.0, 'scatter_sig': 0.0,
                'final_mean_sig': 0.0, 'final_scatter_sig': 0.0,
                'x2norm': 0.0, 'nx2norm': 0,
                'diff_mean': 0.0, 'diff_sd': 0.0, 'noise_mean': 0.0,
                'diff_mean_ok': 0.0, 'diff_sd_ok': 0.0, 'noise_mean_ok': 0.0,
                'diffrat': 0.0,
            })

    return diff_out, noise_out, conv_out, mask_out, stats_list


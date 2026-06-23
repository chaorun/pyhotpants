import numpy as np
import sys, os, struct
import time as tm
import logging
from typing import List, Dict, Tuple, Optional
from ..functions import FLAG_BAD_PIXVAL, FLAG_SAT_PIXEL, FLAG_LOW_PIXEL
from ..functions import FLAG_INPUT_ISBAD, FLAG_OUTPUT_ISBAD
from ..functions import sigma_clip_numpy, get_noise_stats3_numpy
from ..functions import insert_subregion_flt_numpy, insert_subregion_int_numpy
from ..functions import get_stamp_stats3_numpy, buildStampsNumba
from ..functions import make_noise_image4_numpy
# ZEROVAL, MAXVAL,
# bin_quartile_numpy,
#    get_stamp_stats3_fast_numpy, cut_sstamp_numpy,
#    check_psf_center_numba, quick_sort_recurse,
#    psfCentersJit, cut_stamp_numpy, Ran1,
# BUILD_STAMP_FLAT_CACHE, LAST_SIGMA_CLIP, LAST_N, LAST_MEDIAN,

from .alard import background_loop_jit, get_final_stamp_sig_numpy
from .alard import make_kernel_numpy, get_kernel_vec_numpy
from .alard import fill_stamp_numba, spatial_convolve_fast_numpy
from .alard import check_stamps_numpy, fit_kernel_numpy
#kernel_vector_pca_numpy, kernel_vector_numpy,  build_matrix_jit, build_scprod_jit, build_matrix_numpy, build_scprod_numpy,
#    fill_stamp_numba_kernel_local, fill_stamp_numpy, get_stamp_sig_batch_jit, get_stamp_sig_jit,spatial_convolve_jit_kernel, buildAllKernels,
#check_again_numpy,

def compute_ctx_info(hwKernel: int, ngauss: int, deg_fixe: List[int], kerOrder: int, bgOrder: int,
                     nStampX_in: int, nStampY_in: int, hwKSStamp: int,
                     useFullSS: int, kerFitThresh: float, kcStep_in: int,
                     tNx: int, tNy: int, iNx: int, iNy: int, nR: int) -> Dict:
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

    return {'nCompKer': nCompKer, 'nComp': nComp, 'nC': nC,
        'nCompBG': nCompBG, 'nBGVectors': nBGVectors, 'nCompTotal': nCompTotal,
        'fwKernel': fwKernel, 'fwStamp': fwStamp, 'fwKSStamp': fwKSStamp,
        'sBorder': sBorder, 'nStamps': nStamps,
        'nStampX': nStampX, 'nStampY': nStampY,
        'xMin': xMin, 'yMin': yMin, 'xMax': xMax, 'yMax': yMax,
        'fitThresh': fitThresh, 'kcStep': kcStep}

def region_setup_numpy(tmpl_2d: np.ndarray, sci_2d: np.ndarray, tnoise_2d: Optional[np.ndarray], inoise_2d: Optional[np.ndarray], tmask_2d: Optional[np.ndarray], imask_2d: Optional[np.ndarray],
                        ri: int, rxmins_np: np.ndarray, rxmaxs_np: np.ndarray, rymins_np: np.ndarray, rymaxs_np: np.ndarray, nR: int,
                        hwKernel: int, fwStamp: int, sBorder: int,
                        xMin: int, yMin: int, xMax: int, yMax: int,
                        fillVal: float, fillValNoise: float,
                        tPedestal: float, iPedestal: float,
                        tGain: float, tRdnoise: float, iGain: float, iRdnoise: float,
                        tUThresh: float, tLThresh: float, iUThresh: float, iLThresh: float,
                        kfSpreadMask1: float, logger: Optional[logging.Logger] = None) -> Dict:
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
        qquad = np.float32(quad_f) * np.float32(quad_f)
        oRData_py = (np.abs(iRData_py.astype(np.float32)) * np.float32(invGain_f) + qquad).astype(np.float32)

    if tnoise_2d is not None:
        eRData_py[:, :] = tnoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        eRData_py *= eRData_py
    else:
        invGain_t = np.float32(1.0 / float(tGain))
        quad_t = np.float32(float(tRdnoise) / float(tGain))
        qquad_t = np.float32(quad_t) * np.float32(quad_t)
        eRData_py = (np.abs(tRData_py.astype(np.float32)) * np.float32(invGain_t) + qquad_t).astype(np.float32)

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
    return {'tRData': tRData_py, 'iRData': iRData_py,
            'oRData': oRData_py, 'eRData': eRData_py,
            'mRData': mRData_py, 'misRData': misRData_py, 'mtsRData': mtsRData_py,
            'rXMin': rXMin, 'rYMin': rYMin, 'rXMax': rXMax, 'rYMax': rYMax,
            'rXBMin': rXBMin, 'rYBMin': rYBMin, 'rXBMax': rXBMax, 'rYBMax': rYBMax,
            'xBufLo': xBufLo, 'xBufHi': xBufHi, 'yBufLo': yBufLo, 'yBufHi': yBufHi,
            'fpixelOutX': fpixelOutX, 'fpixelOutY': fpixelOutY,
            'lpixelOutX': lpixelOutX, 'lpixelOutY': lpixelOutY,
            'rPixX': rPixX, 'rPixY': rPixY}


def region_buildstamps_numpy(setup_result: Dict, ctx_info: Dict, params_info: Dict, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:
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
    ctSa = None
    ciSa = None

    while (status <= 2) and flag:
        flag = 0
        niS = 0
        ntS = 0

        if localForceConvolve != "i":

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
            ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float32)
            ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float32)
            ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float32)
            ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float32)
            ct['chi2'] = np.zeros(nStamps, dtype=np.float32)
            ct['norm'] = np.zeros(nStamps, dtype=np.float32)
            ct['diff'] = np.zeros(nStamps, dtype=np.float32)
            ct['sum_val'] = np.zeros(nStamps, dtype=np.float32)
            ct['mean_val'] = np.zeros(nStamps, dtype=np.float32)
            ct['median'] = np.zeros(nStamps, dtype=np.float32)
            ct['mode'] = np.zeros(nStamps, dtype=np.float32)
            ct['sd'] = np.zeros(nStamps, dtype=np.float32)
            ct['fwhm'] = np.zeros(nStamps, dtype=np.float32)
            ct['lfwhm'] = np.zeros(nStamps, dtype=np.float32)
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
            ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float32)
            ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float32)
            ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float32)
            ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float32)
            ci['chi2'] = np.zeros(nStamps, dtype=np.float32)
            ci['norm'] = np.zeros(nStamps, dtype=np.float32)
            ci['diff'] = np.zeros(nStamps, dtype=np.float32)
            ci['sum_val'] = np.zeros(nStamps, dtype=np.float32)
            ci['mean_val'] = np.zeros(nStamps, dtype=np.float32)
            ci['median'] = np.zeros(nStamps, dtype=np.float32)
            ci['mode'] = np.zeros(nStamps, dtype=np.float32)
            ci['sd'] = np.zeros(nStamps, dtype=np.float32)
            ci['fwhm'] = np.zeros(nStamps, dtype=np.float32)
            ci['lfwhm'] = np.zeros(nStamps, dtype=np.float32)
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
            ctSd = ctSa['sd']; ctFwhm = ctSa['fwhm']; ctLfwhm = ctSa['lfwhm']; ctXss = ctSa['xss']; ctYss = ctSa['yss']  # ctSscnt = ctSa['sscnt']
        else:
            ctNss = np.zeros(1, dtype=np.int32); ctX0 = np.zeros(1, dtype=np.int32); ctY0 = np.zeros(1, dtype=np.int32)
            ctX = np.zeros(1, dtype=np.int32); ctY = np.zeros(1, dtype=np.int32)
            ctSumVal = np.zeros(1, dtype=np.float32); ctMeanVal = np.zeros(1, dtype=np.float32); ctMedian = np.zeros(1, dtype=np.float32)
            ctMode = np.zeros(1, dtype=np.float32); ctSd = np.zeros(1, dtype=np.float32); ctFwhm = np.zeros(1, dtype=np.float32)
            ctLfwhm = np.zeros(1, dtype=np.float32); ctXss = np.zeros((1, 1), dtype=np.int32); ctYss = np.zeros((1, 1), dtype=np.int32)
            ctSscnt = np.zeros(1, dtype=np.int32)
        if ciSa is not None:
            ciNss = ciSa['nss']; ciX0 = ciSa['x0']; ciY0 = ciSa['y0']; ciX = ciSa['x']; ciY = ciSa['y']
            ciSumVal = ciSa['sum_val']; ciMeanVal = ciSa['mean_val']; ciMedian = ciSa['median']; ciMode = ciSa['mode']
            ciSd = ciSa['sd']; ciFwhm = ciSa['fwhm']; ciLfwhm = ciSa['lfwhm']; ciXss = ciSa['xss']; ciYss = ciSa['yss']  # ciSscnt = ciSa['sscnt']
        else:
            ciNss = np.zeros(1, dtype=np.int32); ciX0 = np.zeros(1, dtype=np.int32); ciY0 = np.zeros(1, dtype=np.int32)
            ciX = np.zeros(1, dtype=np.int32); ciY = np.zeros(1, dtype=np.int32)
            ciSumVal = np.zeros(1, dtype=np.float32); ciMeanVal = np.zeros(1, dtype=np.float32); ciMedian = np.zeros(1, dtype=np.float32)
            ciMode = np.zeros(1, dtype=np.float32); ciSd = np.zeros(1, dtype=np.float32); ciFwhm = np.zeros(1, dtype=np.float32)
            ciLfwhm = np.zeros(1, dtype=np.float32); ciXss = np.zeros((1, 1), dtype=np.int32); ciYss = np.zeros((1, 1), dtype=np.int32)
            ciSscnt = np.zeros(1, dtype=np.int32)

        for l in range(nStampY):
            for k in range(nStampX):
                logger.info("Build stamp  : t %4d i %4d (grid coord %2d %2d)", ntS, niS, k, l)

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
                        logger.info("Adding centers manually")
                    for m in range(Ncmp):
                        if (xcmp[m] > sXMin + hwKernel + 1) and (xcmp[m] < sXMax - hwKernel - 1) and \
                           (ycmp[m] > sYMin + hwKernel + 1) and (ycmp[m] < sYMax - hwKernel - 1):

                            (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                             lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                             lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          sYMax, 0, rXBMin,
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
                                                          iRData1d, tRData1d,
                                                          xcmp[m] - rXBMin,
                                                          ycmp[m] - rYBMin,
                                                          localForceConvolve,
                                                          rPixX, rPixY,
                                                          tUKThresh, iUKThresh,
                                                          hwKSStamp, fwStamp,
                                                          nKSStamps,
                                                          kerFitThresh,
                                                          mRData1d.copy(),
                                                          statSig)
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

                    if findSSC:
                        logger.debug("Automatically finding additional centers")

                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          sYMax, 1, rXBMin,
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
                                                          0, localForceConvolve,
                                                          rPixX, rPixY,
                                                          tUKThresh, iUKThresh,
                                                          hwKSStamp, fwStamp,
                                                          nKSStamps,
                                                          kerFitThresh,
                                                          mRData1d.copy(),
                                                          statSig)
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

                else:
                    if useFullSS:

                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal,
                         lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal,
                         lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          sYMax, 1, rXBMin,
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
                                                          0, localForceConvolve,
                                                          rPixX, rPixY,
                                                          tUKThresh, iUKThresh,
                                                          hwKSStamp, fwStamp,
                                                          nKSStamps,
                                                          kerFitThresh,
                                                          mRData1d.copy(),
                                                          statSig)
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


                    else:

                        (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, 
                         lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
                         lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal,
                         lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
                         lmRData) = buildStampsNumba(sXMin, sXMax, sYMin,
                                                          sYMax, 1, rXBMin,
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
                                                          0, localForceConvolve,
                                                          rPixX, rPixY,
                                                          tUKThresh, iUKThresh,
                                                          hwKSStamp, fwStamp,
                                                          nKSStamps,
                                                          kerFitThresh,
                                                          mRData1d.copy(),
                                                          statSig)
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

                if localForceConvolve != "i":
                    logger.debug("    templ: %d substamps", ctSa['nss'][ntS])
                    # if ctStamps[ntS]['nss'] > 0:
                    if ctSa['nss'][ntS] > 0:
                        ntS += 1
                if localForceConvolve != "t":
                    logger.debug("    image: %d substamps", ciSa['nss'][niS])
                    # if ciStamps[niS]['nss'] > 0:
                    if ciSa['nss'][niS] > 0:
                        niS += 1

        iSFrac = niS / float(nStamps)
        tSFrac = ntS / float(nStamps)

        if localForceConvolve == "i":
            logger.info("%d stamps built (%.2f%%)", niS, iSFrac)
            if iSFrac < minFracGoodStamps:
                flag = 1
        elif localForceConvolve == "t":
            logger.info("%d stamps built (%.2f%%)", ntS, tSFrac)
            if tSFrac < minFracGoodStamps:
                flag = 1
        elif (iSFrac < minFracGoodStamps) or (tSFrac < minFracGoodStamps):
            logger.info("%d and %d stamps built (%.2f%%, %.2f%%)", ntS, niS, tSFrac, iSFrac)
            flag = 1
        else:
            logger.info("%d and %d stamps built (%.2f%%, %.2f%%)", ntS, niS, tSFrac, iSFrac)
            break

        if flag and (status <= 1) and (scaleFitThresh < 1.0):
            kerFitThresh *= scaleFitThresh
            logger.info("Too few stamps were fit, scaling down fitting threshold to %.2f", kerFitThresh)

            if localForceConvolve != "i":

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
                ct['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float32)
                ct['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float32)
                ct['scprod'] = np.zeros((nStamps, nC), dtype=np.float32)
                ct['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float32)
                ct['chi2'] = np.zeros(nStamps, dtype=np.float32)
                ct['norm'] = np.zeros(nStamps, dtype=np.float32)
                ct['diff'] = np.zeros(nStamps, dtype=np.float32)
                ct['sum_val'] = np.zeros(nStamps, dtype=np.float32)
                ct['mean_val'] = np.zeros(nStamps, dtype=np.float32)
                ct['median'] = np.zeros(nStamps, dtype=np.float32)
                ct['mode'] = np.zeros(nStamps, dtype=np.float32)
                ct['sd'] = np.zeros(nStamps, dtype=np.float32)
                ct['fwhm'] = np.zeros(nStamps, dtype=np.float32)
                ct['lfwhm'] = np.zeros(nStamps, dtype=np.float32)
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
                ci['vectors'] = np.zeros((nStamps, nVec, fwSq), dtype=np.float32)
                ci['mat'] = np.zeros((nStamps, nC, nC), dtype=np.float32)
                ci['scprod'] = np.zeros((nStamps, nC), dtype=np.float32)
                ci['krefArea'] = np.zeros((nStamps, fwSq), dtype=np.float32)
                ci['chi2'] = np.zeros(nStamps, dtype=np.float32)
                ci['norm'] = np.zeros(nStamps, dtype=np.float32)
                ci['diff'] = np.zeros(nStamps, dtype=np.float32)
                ci['sum_val'] = np.zeros(nStamps, dtype=np.float32)
                ci['mean_val'] = np.zeros(nStamps, dtype=np.float32)
                ci['median'] = np.zeros(nStamps, dtype=np.float32)
                ci['mode'] = np.zeros(nStamps, dtype=np.float32)
                ci['sd'] = np.zeros(nStamps, dtype=np.float32)
                ci['fwhm'] = np.zeros(nStamps, dtype=np.float32)
                ci['lfwhm'] = np.zeros(nStamps, dtype=np.float32)
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
        result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
        return result
    if localForceConvolve == "i":
        if niS == 0:
            result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
                  'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
            return result
    if localForceConvolve == "t":
        if ntS == 0:
            result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
                  'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': -1}
            return result

    filter_x = np.zeros(fwKernel * nCompKer, dtype=np.float32)
    filter_y = np.zeros(fwKernel * nCompKer, dtype=np.float32)
    kernel_vec = get_kernel_vec_numpy(ngauss, deg_fixe, usePCA, fwKernel, hwKernel,
                                      sigma_gauss, filter_x, filter_y, PCA)

    logger.debug("region_buildstamps_numpy done")
    result = {'niS': niS, 'ntS': ntS,'ctStamps': ctSa,'ciStamps': ciSa,
              'kernel_vec': kernel_vec,'filter_x': filter_x,'filter_y': filter_y,'status': 0}
    return result


def region_fit_numpy(buildstamps_result: Dict, setup_result: Dict, ctx_info: Dict, params_info: Dict, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:
    logger.debug("region_fit_numpy start")

    nCompKer = ctx_info['nCompKer']
    # nBGVectors = ctx_info['nBGVectors']
    nC = ctx_info['nC']
    # nCompTotal = ctx_info['nCompTotal']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']

    hwKernel = params_info['hwKernel']
    ngauss = params_info['ngauss']
    deg_fixe = params_info['deg_fixe']
    kerOrder = params_info['kerOrder']
    bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    # verbose = params_info.get('verbose', 0)
    # usePCA = params_info.get('usePCA', 0)
    # PCA = params_info.get('PCA', None)
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
    if ctSa is not None:
        ctSscnt = ctSa['sscnt']
        ctNss = ctSa['nss']
        ctVectors = ctSa['vectors']
        ctMat = ctSa['mat']
        ctScprod = ctSa['scprod']
        ctXss = ctSa['xss']
        ctYss = ctSa['yss']
        ctKrefArea = ctSa['krefArea']
        ctSumVal = ctSa['sum_val']
        ctNorm = ctSa['norm']
        ctDiff = ctSa['diff']
        ctNKSStamps = ctSa['nKSStamps']
    else:
        ctSscnt = None
        ctNss = None
        ctVectors = None
        ctMat = None
        ctScprod = None
        ctXss = None
        ctYss = None
        ctKrefArea = None
        ctSumVal = None
        ctNorm = None
        ctDiff = None
        ctNKSStamps = params_info.get('nKSStamps')
    if ciSa is not None:
        ciSscnt = ciSa['sscnt']
        ciNss = ciSa['nss']
        ciVectors = ciSa['vectors']
        ciMat = ciSa['mat']
        ciScprod = ciSa['scprod']
        ciXss = ciSa['xss']
        ciYss = ciSa['yss']
        ciKrefArea = ciSa['krefArea']
        ciSumVal = ciSa['sum_val']
        ciNorm = ciSa['norm']
        ciDiff = ciSa['diff']
        ciNKSStamps = ciSa['nKSStamps']
    else:
        ciSscnt = None
        ciNss = None
        ciVectors = None
        ciMat = None
        ciScprod = None
        ciXss = None
        ciYss = None
        ciKrefArea = None
        ciSumVal = None
        ciNorm = None
        ciDiff = None
        ciNKSStamps = params_info.get('nKSStamps')

    tMerit = 0.0
    iMerit = 0.0
    convTmpl = 0

    logger.info("Filling Template sub-stamps")
    if localForceConvolve != "i":
        for k in range(ntS):
            ctSa['sscnt'][k] = 0
        # 批量收集有效 stamp 索引
        ct_si_list = [k for k in range(ntS) if ctSscnt[k] < ctNss[k]]
        if ct_si_list:
            # 批量调用 fill_stamp_numba
            ct_out = fill_stamp_numba(ctXss, ctYss, ctSscnt, ctNss, ct_si_list,
                                       tRData1d, iRData1d, rPixX, rPixY, ngauss, deg_fixe,
                                       hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                                       filter_x, filter_y, fillVal, mRData1d)
            ct_out_v, ct_out_k, ct_out_m, ct_out_s, ct_out_sum = ct_out
            # 写回每个 stamp
            nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
            fwSq = fwKSStamp * fwKSStamp
            nC = nCompKer + 1
            for bi, si in enumerate(ct_si_list):
                ctVectors[si, :nCompKer + nbg, :fwSq] = ct_out_v[bi]
                ctKrefArea[si, :] = ct_out_k[bi]
                ctMat[si, :nC + 1, :nC + 1] = ct_out_m[bi]
                ctScprod[si, :nC + 1] = ct_out_s[bi]
                ctSumVal[si] = ct_out_sum[bi]
        if localForceConvolve == "b":
            logger.info("Trying to convolve the TEMPLATE to fit IMAGE")
            tMerit = check_stamps_numpy(ctScprod, ctMat, ctNorm, ctDiff, ctSscnt,
                ctNss, ctXss, ctYss, ctVectors, ctKrefArea, ntS, iRData1d, oRData1d,
                nCompKer, kerOrder, bgOrder,
                localForceConvolve, figMerit, kerSigReject, statSig,
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                kernel_vec, mRData1d, nKSStamps=ctNKSStamps)
            logger.info("    Result : merit = %.3f", tMerit)
        else:
            tMerit = 0.0
            iMerit = 0.0

    logger.info("Filling Image sub-stamps")
    if localForceConvolve != "t":
        for k in range(niS):
            ciSa['sscnt'][k] = 0
        # 批量收集有效 stamp 索引
        ci_si_list = [k for k in range(niS) if ciSscnt[k] < ciNss[k]]
        if ci_si_list:
            ci_out = fill_stamp_numba(ciXss, ciYss, ciSscnt, ciNss, ci_si_list,
                                       iRData1d, tRData1d, rPixX, rPixY, ngauss, deg_fixe,
                                       hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                                       filter_x, filter_y, fillVal, mRData1d)
            ci_out_v, ci_out_k, ci_out_m, ci_out_s, ci_out_sum = ci_out
            nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2
            fwSq = fwKSStamp * fwKSStamp
            nC = nCompKer + 1
            for bi, si in enumerate(ci_si_list):
                ciVectors[si, :nCompKer + nbg, :fwSq] = ci_out_v[bi]
                ciKrefArea[si, :] = ci_out_k[bi]
                ciMat[si, :nC + 1, :nC + 1] = ci_out_m[bi]
                ciScprod[si, :nC + 1] = ci_out_s[bi]
                ciSumVal[si] = ci_out_sum[bi]
        if localForceConvolve == "b":
            logger.info("Trying to convolve the IMAGE to fit TEMPLATE")

            iMerit = check_stamps_numpy(ciScprod, ciMat, ciNorm, ciDiff, ciSscnt,
                ciNss, ciXss, ciYss, ciVectors, ciKrefArea, niS, tRData1d, oRData1d,
                nCompKer, kerOrder, bgOrder,
                localForceConvolve, figMerit, kerSigReject, statSig,
                fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel,
                kernel_vec, mRData1d, nKSStamps=ciNKSStamps)
            logger.info("    Result : merit = %.3f", iMerit)
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
    result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': 0}
    result['convTmpl'] = convTmpl
    result['tMerit'] = tMerit
    result['iMerit'] = iMerit
    return result


def region_convolve_diff_numpy(fit_result: Dict, setup_result: Dict, buildstamps_result: Dict,
                                 ctx_info: Dict, params_info: Dict, region_idx: int, localForceConvolve: str, logger: Optional[logging.Logger] = None) -> Dict:


    nCompKer = ctx_info['nCompKer']
    # nBGVectors = ctx_info['nBGVectors']
    # nC = ctx_info['nC']
    # nStamps = ctx_info['nStamps']
    fwKSStamp = ctx_info['fwKSStamp']
    fwKernel = ctx_info['fwKernel']
    kcStep = ctx_info['kcStep']

    hwKernel = params_info['hwKernel']
    ngauss = params_info['ngauss']
    deg_fixe = params_info['deg_fixe']
    kerOrder = params_info['kerOrder']
    bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    # verbose = params_info.get('verbose', 0)
    # usePCA = params_info.get('usePCA', 0)
    # PCA = params_info.get('PCA', None)
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
    # ctSscnt = ctSa['sscnt'] if ctSa is not None else None
    # ctNss = ctSa['nss'] if ctSa is not None else None
    # ctXss = ctSa['xss'] if ctSa is not None else None
    # ctYss = ctSa['yss'] if ctSa is not None else None
    # ciSscnt = ciSa['sscnt'] if ciSa is not None else None
    # ciNss = ciSa['nss'] if ciSa is not None else None
    # ciXss = ciSa['xss'] if ciSa is not None else None
    # ciYss = ciSa['yss'] if ciSa is not None else None

    ntS = buildstamps_result['ntS']
    niS = buildstamps_result['niS']
    kernel_vec = buildstamps_result['kernel_vec']
    filter_x = buildstamps_result['filter_x']
    filter_y = buildstamps_result['filter_y']

    kernel_coeffs = np.zeros(nCompKer, dtype=np.float32)
    kernel = np.zeros(fwKernel * fwKernel, dtype=np.float32)

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
        logger.info("Region %d:%d,%d:%d : Convolving TEMPLATE", rXMin, rXMax, rYMin, rYMax)
        nS = ntS

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        oRData1d = setup_result['oRData'].ravel().copy()
        fit_result_k = fit_kernel_numpy(ctSa, iRData1d, tRData1d, oRData1d,
            nCompKer, kerOrder, bgOrder, nS,
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            hwKernel, fwKernel, filter_x, filter_y, fillVal,
            logger=logger)
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
            eRData1d = make_noise_image4_numpy(tRData1d, 1.0 / tGain, tRdnoise / tGain)

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        logger.info("Convolving...")  # Template


        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
            tRData1d, eRData1d, rPixX, rPixY, tKerSol, mtsRData1d,
            kcStep, hwKernel, fwKernel,
            convolveVariance, kerFracMask,
            rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
        oRData1d[:] = oRData1d_new.ravel()
        mRData1d[:] = mRData1d_new.ravel()

        if vData is not None:
            eRData1d = vData

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        logger.debug("[region %d] convolve_diff: background start", region_idx)

        background_loop_jit(oRData1d, tKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)

        logger.debug("[region %d] convolve_diff: background done", region_idx)
        logger.debug("[region %d] convolve_diff: make_kernel start", region_idx)
        sumKernel = make_kernel_numpy(rXMin, rYMin, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Sum Kernel at %d,%d: %f", rXMin, rYMin, sumKernel)
        sumKernel = make_kernel_numpy(rXMax, rYMax, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Sum Kernel at %d,%d: %f", rXMax, rYMax, sumKernel)
        sumKernel = make_kernel_numpy(rPixX // 2, rPixY // 2, tKerSol, rPixX, rPixY,
                                      nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel)
        logger.info(" Using Kernel Sum = %f", sumKernel)

        logger.debug("[region %d] convolve_diff: make_kernel done", region_idx)
        logger.debug("[region %d] convolve_diff: noise_combine start", region_idx)
        tRData1d[:] = fillValNoise
        if inoise_2d is not None:
            tRData1d[:] = inoise_2d[rYBMin:rYBMax + 1, rXBMin:rXBMax + 1].ravel()
            tRData1d[:] = tRData1d * tRData1d
        else:
            tRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain)
            
        tRData1d = np.sqrt(tRData1d + eRData1d)

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
                    entry = {'x': int(ctSa['xss'][si, sc]), 'y': int(ctSa['yss'][si, sc])}
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
        logger.info("Region %d:%d %d,%d : Convolving IMAGE", rXMin, rXMax, rYMin, rYMax)
        nS = niS

        logger.debug("[region %d] convolve_diff: fitKernel start", region_idx)
        oRData1d = setup_result['oRData'].ravel().copy()
        # fit_result_k = fit_kernel_numpy(
        #     ciStamps, tRData1d, iRData1d, oRData1d,
        #     nCompKer, kerOrder, bgOrder, verbose, nS,
        #     fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
        #     kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
        #     hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal)
        fit_result_k = fit_kernel_numpy(ciSa, tRData1d, iRData1d, oRData1d,
            nCompKer, kerOrder, bgOrder, nS,
            fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
            kerSigReject, statSig, mRData1d, ngauss, deg_fixe,
            hwKernel, fwKernel, filter_x, filter_y, fillVal,
            logger=logger)
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
            eRData1d = make_noise_image4_numpy(iRData1d, 1.0 / iGain, iRdnoise / iGain)

        logger.debug("[region %d] convolve_diff: noise rebuild done", region_idx)
        logger.debug("[region %d] convolve_diff: spatial_convolve start", region_idx)
        logger.info("Convolving...")  # Image

        vData, oRData1d_new, mRData1d_new = spatial_convolve_fast_numpy(
            iRData1d, eRData1d, rPixX, rPixY, iKerSol, misRData1d,
            kcStep, hwKernel, fwKernel,
            convolveVariance, kerFracMask,
            rPixX, rPixY, nCompKer, kerOrder, kernel_vec)
        oRData1d[:] = oRData1d_new.ravel()
        mRData1d[:] = mRData1d_new.ravel()
        if vData is not None:
            eRData1d = vData

        logger.debug("[region %d] convolve_diff: spatial_convolve done", region_idx)
        logger.debug("[region %d] convolve_diff: background start", region_idx)

        background_loop_jit(oRData1d, iKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY, hwKernel)

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
            localForceConvolve = "i"
            # ctStamps = None
            ctSa = None

    noiseData1d = tRData1d if convTmpl else iRData1d

    # result = build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, None, None, None, 0)
    result = {'niS': niS, 'ntS': ntS, 'ctStamps': ctSa, 'ciStamps': ciSa,
              'kernel_vec': None, 'filter_x': None, 'filter_y': None, 'status': 0}
    result.update({'oRData': oRData1d.reshape(rPixY, rPixX),
        'noiseData': noiseData1d.reshape(rPixY, rPixX), 'mRData': mRData1d.reshape(rPixY, rPixX),
        'sumKernel': sumKernel, 'meansigSubstamps': meansigSubstamps,
        'scatterSubstamps': scatterSubstamps, 'NskippedSubstamps': NskippedSubstamps,
        'convTmpl': convTmpl, 'nS': nS, 'localForceConvolve': localForceConvolve,
        'kerSol': kerSol, 'savexy_entries': savexy_entries,
        'savexyXmin': rXBMin + (0 if convTmpl else 1),
        'savexyYmin': rYBMin + (0 if convTmpl else 1),
        # 'ctStamps': ctStamps,
        # 'ciStamps': ciStamps,
        'ctStamps': ctSa, 'ciStamps': ciSa,
        'tRData': tRData1d.reshape(rPixY, rPixX), 'iRData': iRData1d.reshape(rPixY, rPixX)})
    return result


def region_output_numpy(convolve_result: Dict, setup_result: Dict,
                         diff_out: np.ndarray, noise_out: np.ndarray, conv_out: np.ndarray, mask_out: np.ndarray,
                         ctx_info: Dict, params_info: Dict, region_idx: int, stats_list: List,
                         logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    # import time
    logger.debug("[region %d] region_output_numpy start", region_idx)
    # start_time = time.time()

    # nCompKer = ctx_info['nCompKer']
    # nBGVectors = ctx_info['nBGVectors']
    # nC = ctx_info['nC']
    # nStamps = ctx_info['nStamps']
    fwKSStamp = ctx_info['fwKSStamp']
    # fwKernel = ctx_info['fwKernel']

    hwKernel = params_info['hwKernel']
    # kerOrder = params_info['kerOrder']
    # bgOrder = params_info['bgOrder']
    hwKSStamp = params_info['hwKSStamp']
    # verbose = params_info.get('verbose', 0)
    statSig = params_info['statSig']
    fillVal = params_info['fillVal']
    fillValNoise = params_info['fillValNoise']
    photNormalize = params_info['photNormalize']
    figMerit = params_info['figMerit']
    rescaleOK = params_info.get('rescaleOK', 0)
    kfSpreadMask2 = params_info.get('kfSpreadMask2', -1.0)

    rPixX = setup_result['rPixX']
    rPixY = setup_result['rPixY']
    # rXMin = setup_result['rXMin']
    # rYMin = setup_result['rYMin']
    # rXMax = setup_result['rXMax']
    # rYMax = setup_result['rYMax']
    xBufLo = setup_result['xBufLo']
    yBufLo = setup_result['yBufLo']
    # xBufHi = setup_result['xBufHi']
    # yBufHi = setup_result['yBufHi']
    fpixelOutX = setup_result['fpixelOutX']
    fpixelOutY = setup_result['fpixelOutY']
    lpixelOutX = setup_result['lpixelOutX']
    lpixelOutY = setup_result['lpixelOutY']
    # rXBMin = setup_result['rXBMin']
    # rYBMin = setup_result['rYBMin']

    convTmpl = convolve_result['convTmpl']
    nS = convolve_result['nS']
    sumKernel = convolve_result['sumKernel']
    meansigSubstamps = convolve_result['meansigSubstamps']
    scatterSubstamps = convolve_result['scatterSubstamps']
    # NskippedSubstamps = convolve_result['NskippedSubstamps']

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

    logger.info(" Creating and writing output images...")

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
            mean_f, stdev_f = sigma_clip_numpy(temp2[:kk], 10, statSig)[:2]
            meansigSubstampsF = mean_f
            scatterSubstampsF = stdev_f
            logger.info("   FINAL Mean sig: %6.3f stdev: %6.3f", meansigSubstampsF, scatterSubstampsF)

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
            mean_f, stdev_f = sigma_clip_numpy(temp2[:kk], 10, statSig)[:2]
            meansigSubstampsF = mean_f
            scatterSubstampsF = stdev_f
            logger.info("    FINAL Mean sig: %6.3f stdev: %6.3f", meansigSubstampsF, scatterSubstampsF)

    oRData_2d = oRData1d.reshape(rPixY, rPixX)
    noiseData_2d = noiseData1d.reshape(rPixY, rPixX)
    mRData_2d = mRData1d.reshape(rPixY, rPixX)

    # tm4 = time.time(); logger.debug("[out] get_stamp_stats3 start")

    logger.info(" Getting diffim stats for GOOD pixels : ")
    res_good = get_stamp_stats3_numpy(
        oRData_2d, 0, 0, rPixX, rPixY,
        0x0, 0xffff,          5, mRData_2d, statSig)
    mean_val = res_good[1]
    sd_val = res_good[4]
    logger.info("   Mean   : %.2f", res_good[1])
    logger.info("   Median : %.2f", res_good[2])
    logger.info("   Mode   : %.2f", res_good[3])
    logger.info("   Stdev  : %.2f", res_good[4])

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
    tni: Optional[np.ndarray] = None, ini: Optional[np.ndarray] = None,
    tmi: Optional[np.ndarray] = None, imi: Optional[np.ndarray] = None,
    tu: float = 25000., tuk: Optional[float] = None, tl: float = 0.,
    tg: float = 1., tr: float = 0., tp: float = 0.,
    iu: float = 25000., iuk: Optional[float] = None, il: float = 0.,
    ig: float = 1., ir: float = 0., ip: float = 0.,
    r: int = 10, ko: int = 2, bgo: int = 1,
    ng: int = 3, ng_deg: Optional[List[int]] = None, ng_sig: Optional[List[float]] = None,
    pca: Optional[np.ndarray] = None,
    nrx: int = 1, nry: int = 1, rf: Optional[int] = None,
    nsx: int = 10, nsy: int = 10, ssf: Optional[List] = None,
    afssc: int = 1, nss: int = 3, rss: int = 15,
    ft: float = 20.0, sft: float = 0.5, nft: float = 0.1,
    ssig: float = 3.0, ks: float = 2.0, kfm: float = 0.99,
    mins: float = 1.0, mous: float = 1.0,
    fi: float = 1e-30, fin: float = 0.,
    c: str = 'b', n: str = 't', fom: str = 'v',
    sconv: int = 0, okn: int = 0, convvar: int = 0,
    v: int = 1, kcs: int = 0,
    uss: int = 0, savexy: int = 0,
    dump_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
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

    # c_bytes = c.encode('ascii')
    # n_bytes = n.encode('ascii')
    # fom_bytes = fom.encode('ascii')

    oNx = max(tNx, iNx)
    oNy = max(tNy, iNy)
    diff_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    noise_out = np.full((oNy, oNx), float(fin), dtype=np.float32)
    conv_out = np.full((oNy, oNx), float(fi), dtype=np.float32)
    mask_out = np.zeros((oNy, oNx), dtype=np.int32)

    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "py_input.bin")
        with open(path, "wb") as f:
            def de(nm, db):
                nb = nm.encode('ascii')
                f.write(struct.pack('i', len(nb)))
                f.write(nb)
                dl = len(db) if db else 0
                f.write(struct.pack('l', dl))
                if dl > 0: f.write(db)
            def di(nm, val): de(nm, struct.pack('i', int(val)))
            def dl(nm, val): de(nm, struct.pack('l', int(val)))
            def df(nm, val): de(nm, struct.pack('f', float(val)))
            def ds(nm, s):
                if s is None: de(nm, None)
                else: de(nm, s.encode('ascii') + b'\x00')
            dl("tNx",tNx); dl("tNy",tNy); dl("iNx",iNx); dl("iNy",iNy)
            dl("oNx",oNx); dl("oNy",oNy)
            di("nR",nR); di("hwKernel",r); di("ngauss",ng)
            di("kerOrder",ko); di("bgOrder",bgo)
            di("nStampX",nsx); di("nStampY",nsy)
            di("nKSStamps",nss); di("hwKSStamp",rss)
            di("useFullSS",uss); di("findSSC",afssc)
            df("kerFitThresh",ft); df("scaleFitThresh",sft)
            df("minFracGoodStamps",nft)
            df("statSig",ssig); df("kerSigReject",ks); df("kerFracMask",kfm)
            df("tUThresh",tu); df("tLThresh",tl)
            df("tGain",tg); df("tRdnoise",tr); df("tPedestal",tp)
            df("iUThresh",iu); df("iLThresh",il)
            df("iGain",ig); df("iRdnoise",ir); df("iPedestal",ip)
            df("tUKThresh",tuk_val); df("iUKThresh",iuk_val)
            df("kfSpreadMask1",mins); df("kfSpreadMask2",mous)
            df("fillVal",fi); df("fillValNoise",fin)
            di("sameConv",sconv); di("rescaleOK",okn)
            di("convolveVariance",convvar)
            di("usePCA",use_pca); di("Ncmp",ncmp)
            di("verbose",v); di("kcStep",kcs); di("savexyflag",savexy)
            ds("forceConvolve",c); ds("photNormalize",n); ds("figMerit",fom)
            de("deg_fixe", bytes(np.ascontiguousarray(deg_arr)))
            de("sigma_gauss", bytes(np.ascontiguousarray(sig_arr)))
            de("tFullData", bytes(np.ascontiguousarray(tmpl_arr)))
            de("iFullData", bytes(np.ascontiguousarray(sci_arr)))
            de("tNoiseFullData", bytes(np.ascontiguousarray(tni_arr)) if tni_arr is not None else None)
            de("iNoiseFullData", bytes(np.ascontiguousarray(ini_arr)) if ini_arr is not None else None)
            de("tMaskFullData", bytes(np.ascontiguousarray(tmi_arr)) if tmi_arr is not None else None)
            de("iMaskFullData", bytes(np.ascontiguousarray(imi_arr)) if imi_arr is not None else None)
            de("rXMins", bytes(np.ascontiguousarray(rxmins)))
            de("rXMaxs", bytes(np.ascontiguousarray(rxmaxs)))
            de("rYMins", bytes(np.ascontiguousarray(rymins)))
            de("rYMaxs", bytes(np.ascontiguousarray(rymaxs)))
            de("xcmp", bytes(np.ascontiguousarray(xcmp_arr)) if ssf is not None else None)
            de("ycmp", bytes(np.ascontiguousarray(ycmp_arr)) if ssf is not None else None)
            de("diffOut", bytes(np.ascontiguousarray(diff_out)))
            de("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            de("convOut", bytes(np.ascontiguousarray(conv_out)))
            de("maskOut", bytes(np.ascontiguousarray(mask_out)))

    ctx_info = compute_ctx_info(
        int(r), int(ng), deg_arr,
        int(ko), int(bgo),
        int(nsx), int(nsy), int(rss),
        int(uss), float(ft), int(kcs),
        tNx, tNy, iNx, iNy, nR)

    params_info = {'hwKernel': int(r), 'ngauss': int(ng),'deg_fixe': deg_arr, 'sigma_gauss': sig_arr,'kerOrder': int(ko), 'bgOrder': int(bgo),
        'hwKSStamp': int(rss), 'nKSStamps': int(nss),'scaleFitThresh': float(sft), 'minFracGoodStamps': float(nft),
        'tUKThresh': float(tuk_val), 'iUKThresh': float(iuk_val),'tUThresh': float(tu), 'tLThresh': float(tl),
        'tGain': float(tg), 'tRdnoise': float(tr),'iUThresh': float(iu), 'iLThresh': float(il),'iGain': float(ig), 'iRdnoise': float(ir),
        'verbose': int(v), 'usePCA': int(use_pca),
        'PCA': pca_arrs if use_pca else None,
        'xcmp': np.asarray(xcmp_arr) if ssf is not None else None,
        'ycmp': np.asarray(ycmp_arr) if ssf is not None else None,
        'Ncmp': ncmp,'statSig': float(ssig), 'kerSigReject': float(ks), 'kerFracMask': float(kfm),
        'fillVal': float(fi), 'fillValNoise': float(fin),'figMerit': fom, 'photNormalize': n,
        'convolveVariance': int(convvar), 'sameConv': int(sconv),'savexyflag': int(savexy), 'rescaleOK': int(okn),
        'kfSpreadMask2': float(mous), 'findSSC': int(afssc),'tNoiseFullData': tni_arr,'iNoiseFullData': ini_arr}

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
        logger.info("[%d] setup: %.3fs", ri, tm.time()-t0)

        t1 = tm.time()
        bs_result = region_buildstamps_numpy(py, ctx_info, params_info, localFC_py, logger=logger)
        logger.info("[%d] buildstamps: %.3fs", ri, tm.time()-t1)
        if bs_result['status'] != 0:
            continue

        t1 = tm.time()
        fit_result = region_fit_numpy(bs_result, py, ctx_info, params_info, localFC_py, logger=logger)
        logger.info("[%d] fit: %.3fs", ri, tm.time()-t1)

        t1 = tm.time()
        conv_result = region_convolve_diff_numpy(
            fit_result, py, bs_result, ctx_info, params_info, ri, localFC_py, logger=logger)
        logger.info("[%d] convolve_diff: %.3fs", ri, tm.time()-t1)
        localFC_py = conv_result.get('localForceConvolve', localFC_py)

        t1 = tm.time()
        stats_entry = region_output_numpy(
            conv_result, py,
            diff_out, noise_out, conv_out, mask_out,
            ctx_info, params_info, ri, None, logger=logger)
        logger.info("[%d] output: %.3fs", ri, tm.time()-t1)
        stats_list_py[ri] = stats_entry

    if dump_dir is not None:
        opath = os.path.join(dump_dir, "py_output.bin")
        with open(opath, "wb") as fo:
            def de2(nm, db):
                nb = nm.encode('ascii')
                fo.write(struct.pack('i', len(nb)))
                fo.write(nb)
                dl2 = len(db) if db else 0
                fo.write(struct.pack('l', dl2))
                if dl2 > 0: fo.write(db)
            de2("diffOut", bytes(np.ascontiguousarray(diff_out)))
            de2("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            de2("convOut", bytes(np.ascontiguousarray(conv_out)))
            de2("maskOut", bytes(np.ascontiguousarray(mask_out)))

    stats_list = []
    for si in range(nR):
        if stats_list_py[si] is not None:
            entry = stats_list_py[si]
            stats_list.append({'conv_tmpl': entry['convTmpl'],
                'sum_kernel': entry['sumKernel'],
                'mean_sig': entry['meansigSubstamps'], 'scatter_sig': entry['scatterSubstamps'],
                'final_mean_sig': entry['meansigSubstampsF'], 'final_scatter_sig': entry['scatterSubstampsF'],
                'x2norm': entry['x2norm'], 'nx2norm': entry['nx2norm'],
                'diff_mean': entry['mean'], 'diff_sd': entry['sd'], 'noise_mean': entry['nmean'],
                'diff_mean_ok': entry['meanm'], 'diff_sd_ok': entry['sdm'], 'noise_mean_ok': entry['nmeanm'],
                'diffrat': entry['diffrat']})
        else:
            stats_list.append({'conv_tmpl': 0, 'sum_kernel': 0.0,
                'mean_sig': 0.0, 'scatter_sig': 0.0,
                'final_mean_sig': 0.0, 'final_scatter_sig': 0.0,
                'x2norm': 0.0, 'nx2norm': 0,
                'diff_mean': 0.0, 'diff_sd': 0.0, 'noise_mean': 0.0,
                'diff_mean_ok': 0.0, 'diff_sd_ok': 0.0, 'noise_mean_ok': 0.0,
                'diffrat': 0.0})

    return diff_out, noise_out, conv_out, mask_out, stats_list


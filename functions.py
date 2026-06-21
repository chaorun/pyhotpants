import numpy as np
import numba
import math
import logging
from scipy.ndimage import maximum_filter
logger = logging.getLogger('hotpants')

ZEROVAL = 1e-10
MAXVAL = 1e10
FLAG_BAD_PIXVAL = 0x01
FLAG_SAT_PIXEL = 0x02
FLAG_NEG_PIXEL = 0x04
FLAG_SPREAD = 0x40
FLAG_MASKED = 0x80
FLAG_BORDER = 0x100
FLAG_SUBREGION = 0x200
FLAG_INVALID = 0x400

FLAG_ISNAN = 0x08
FLAG_INPUT_ISBAD = 0x80
FLAG_T_BAD = 0x100
FLAG_T_SKIP = 0x200
FLAG_I_BAD = 0x400
FLAG_I_SKIP = 0x800

BUILD_STAMP_FLAT_CACHE = {}
LAST_SIGMA_CLIP = None
LAST_N = None
LAST_MEDIAN = None


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
    count = len(data)
    if count == 0:
        return (0.0, MAXVAL, 1)

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


@numba.jit(nopython=True)
def get_noise_stats3_numpy(data, noise, umask, smask, rPixX, rPixY, mRData):
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
    pixX = lpixelX - fpixelX + 1
    pixY = lpixelY - fpixelY + 1
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]


def insert_subregion_int_numpy(sub_2d, full_2d, fpixelX, fpixelY, lpixelX, lpixelY, xBufLo, yBufLo):
    pixX = lpixelX - fpixelX + 1
    pixY = lpixelY - fpixelY + 1
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]


def cut_stamp_numpy(data_1d, dxLen, xMin, yMin, xMax, yMax):
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
    nstat = 100
    ufstat = 0.9
    mfstat = 0.5

    npts = nPixX * nPixY
    if npts < nstat:
        return {'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
                'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 4}

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

    nan_mask = np.isnan(data_flat)
    skip_all |= nan_mask

    if nan_mask.any():
        nan_y, nan_x = np.where(nan_mask.reshape(nPixY, nPixX))
        mRData_2d[nan_y + y0Reg, nan_x + x0Reg] |= (FLAG_INPUT_ISBAD | FLAG_ISNAN)

    sdat = np.asarray(data_flat[~skip_all], dtype=np.float32)
    logger.debug("[stats3] mask applied %d/%d pixels, sdat=%d", (~skip_all).sum(), ntotal, len(sdat))
    if len(sdat) == 0:
        mode_val = 0.0
        if nfound > 0:
            mode_val = work[int(mfstat * nfound)]
        return {'sum': 0.0, 'mean': 0.0, 'median': mode_val, 'mode': mode_val,
                'sd': MAXVAL, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 5}

    mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat, maxiter, statSig)
    logger.debug("[stats3] sigma_clip mean=%.4f sd=%.4f rc=%d", mean_val, sd_val, sc_rc)
    if sc_rc != 0:
        return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
                'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 5}

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
            return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 1}

        if len(sdat_clipped) == 0:
            mode_val = 0.0
            if nfound > 0:
                mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 2}

        if current_binsize == 0.0:
            mode_val = 0.0
            if nfound > 0:
                mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 3}

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

    fwhm_val = current_binsize * (upper_val - lower_val) / 1.35

    median_target = goodcnt_h / 2.0
    median_val = bin_quartile_numpy(bins, median_target)
    lfwhm_val = current_binsize * (median_val - lower_val) * 2.0 / 1.35
    median_val = current_bin1 + current_binsize * (median_val - 1.0)

    logger.debug("[stats3] done mean=%.4f median=%.4f mode=%.4f sd=%.4f fwhm=%.4f lfwhm=%.4f",
                 mean_val, median_val, mode_val, sd_val, fwhm_val, lfwhm_val)
    return {'sum': ssum_val, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
            'sd': sd_val, 'fwhm': fwhm_val, 'lfwhm': lfwhm_val, 'return_code': 0}


def cut_sstamp_numpy(saKrefArea, saXss, saYss, saX0, saY0, saNss, saSscnt, saSumVal, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose=0):
    saKrefArea[si, :] = fillVal

    nss = saNss[si]
    sscnt = saSscnt[si]
    xStamp = int(saXss[si, sscnt]) - saX0[si]
    yStamp = int(saYss[si, sscnt]) - saY0[si]

    if sscnt >= nss:
        return 1

    sumVal = 0.0
    for j in range(yStamp - hwKSStamp, yStamp + hwKSStamp + 1):
        y = j - (yStamp - hwKSStamp)
        dy = j + saY0[si]
        for i in range(xStamp - hwKSStamp, xStamp + hwKSStamp + 1):
            x = i - (xStamp - hwKSStamp)
            k = i + saX0[si] + rPixX * dy
            dpt = float(iData[k])
            saKrefArea[si, x + y * fwKSStamp] = dpt
            if not (int(mRData[k]) & FLAG_INPUT_ISBAD):
                sumVal += abs(dpt)

    saSumVal[si] = sumVal
    return 0


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
                            rPixX, hwKSStamp, mRData, kerFitThresh):
    sky = float(np.float32(sky))
    invdsky = float(np.float32(invdsky))
    kerFitThresh = float(np.float32(kerFitThresh))
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

            if int(mRData[nr2]) & bbit:
                brk = 1
                dmax2 = 0.0
                break

            dpt2 = float(iData[nr2])

            if dpt2 >= hiThresh:
                mRData[nr2] = int(mRData[nr2]) | bbit1
                brk = 1
                dmax2 = 0.0
                break

            if ((dpt2 - sky) * invdsky) > kerFitThresh:
                dmax2 += dpt2

        if brk == 1:
            break

    return dmax2


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
    kerFitThresh = float(np.float32(kerFitThresh))
    logger.debug("  psf_centers: si=%d nss=%d nKS=%d hwKS=%d", si, sa.nss[si], nKSStamps, hwKSStamp)
    dfrac = 0.9

    if sa.nss[si] >= nKSStamps:
        return 0

    bbit = bbit1 | bbit2 | 0xbf
    sky = sa.mode[si]
    invdsky = 1.0 / sa.fwhm[si]

    sx0 = sa.x0[si]
    sy0 = sa.y0[si]

    xbuffer = 0
    ybuffer = 0

    floorVal = sky + kerFitThresh * sa.fwhm[si]

    allocSize = max(1, (xLen * yLen) // hwKSStamp)
    xloc = np.zeros(allocSize, dtype=np.int32)
    yloc = np.zeros(allocSize, dtype=np.int32)
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
        qs = np.argsort(peaks[:pcnt])
        nssOrig = sa.nss[si]
        idx = nssOrig
        jj = 0
        while jj < pcnt and idx < nKSStamps:
            sa.xss[si, idx] = xloc[qs[pcnt - jj - 1]] + sx0
            sa.yss[si, idx] = yloc[qs[pcnt - jj - 1]] + sy0
            sa.nss[si] += 1
            idx += 1
            jj += 1
        return 0


def psfCentersVectorized(iData1d, mRData1d, sPixX, sPixY, hwKSStamp, rPixX,
                         hiThresh, kerFitThreshVal, sky_val, invdsky_val,
                         sx0_val, sy0_val, nKSStamps, bbit, bbit1, bbit2,
                         xloc_out, yloc_out, peaks_out):
    dfrac = 0.9
    floorVal = sky_val + kerFitThreshVal / invdsky_val
    xbuffer = 0
    ybuffer = 0
    fcnt = 2 * nKSStamps
    pcnt = 0

    iData2d = iData1d.reshape(sPixY, sPixX).astype(np.float64)
    mRData2d = mRData1d.reshape(sPixY, sPixX)

    bad_mask = (mRData2d & bbit) != 0
    hi_mask = iData2d >= hiThresh
    lo_mask = ((iData2d - sky_val) * invdsky_val) < kerFitThreshVal
    border_mask = np.zeros((sPixY, sPixX), dtype=bool)
    if ybuffer > 0:
        border_mask[:ybuffer, :] = True
        border_mask[-ybuffer:, :] = True
    if xbuffer > 0:
        border_mask[:, :xbuffer] = True
        border_mask[:, -xbuffer:] = True

    base_valid = ~bad_mask & ~hi_mask & ~lo_mask & ~border_mask

    footprint = np.ones((2*hwKSStamp+1, 2*hwKSStamp+1), dtype=bool)

    while pcnt < fcnt:
        loPsf = sky_val + (hiThresh - sky_val) * dfrac
        loPsf = max(loPsf, floorVal)

        above_lo = iData2d >= loPsf
        valid = base_valid & above_lo

        if not valid.any():
            if loPsf == floorVal:
                break
            dfrac -= 0.2
            continue

        padded = np.full((sPixY + 2*hwKSStamp, sPixX + 2*hwKSStamp), -np.inf, dtype=np.float64)
        padded[hwKSStamp:-hwKSStamp, hwKSStamp:-hwKSStamp] = np.where(valid, iData2d, -np.inf)
        local_max = maximum_filter(padded, footprint=footprint, mode='constant', cval=-np.inf)

        is_peak = (padded == local_max) & (padded > -np.inf)
        is_peak[:hwKSStamp, :] = False; is_peak[-hwKSStamp:, :] = False
        is_peak[:, :hwKSStamp] = False; is_peak[:, -hwKSStamp:] = False

        if not is_peak.any():
            if loPsf == floorVal:
                break
            dfrac -= 0.2
            continue

        py, px = np.where(is_peak)
        peak_vals = iData2d[py, px]
        order = np.argsort(peak_vals)[::-1]
        py, px = py[order], px[order]

        for idx in range(len(px)):
            if pcnt >= fcnt:
                break
            ip = px[idx]
            jp = py[idx]
            dmax_val = check_psf_center_numba(
                iData1d, ip, jp, sPixX, sPixY,
                sx0_val, sy0_val, hiThresh, sky_val, invdsky_val,
                xbuffer, ybuffer, bbit, bbit1,
                rPixX, hwKSStamp, mRData1d, kerFitThreshVal)
            if dmax_val == 0.0:
                continue
            xloc_out[pcnt] = ip
            yloc_out[pcnt] = jp
            peaks_out[pcnt] = dmax_val
            pcnt += 1
            jlo = max(0, jp - hwKSStamp); jhi = min(sPixY, jp + hwKSStamp + 1)
            ilo = max(0, ip - hwKSStamp); ihi = min(sPixX, ip + hwKSStamp + 1)
            base_valid[jlo:jhi, ilo:ihi] = False
            mRData1d_2d = mRData2d[jlo:jhi, ilo:ihi]
            mRData2d[jlo:jhi, ilo:ihi] = mRData1d_2d | bbit2

        if pcnt >= fcnt:
            break
        if loPsf == floorVal:
            break
        dfrac -= 0.2

    return pcnt

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
    logger.debug("  build_stamps: niS=%d ntS=%d sXMin=%d sYMin=%d sPix=%dx%d fc=%s getCenters=%d",
                 niS, ntS, sXMin, sYMin, sXMax - sXMin + 1, sYMax - sYMin + 1,
                 forceConvolve, getCenters)
    sPixX = sXMax - sXMin + 1
    sPixY = sYMax - sYMin + 1

    # 读入本地变量 (ntS/niS 位置)
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

    mRData2d = lmRData.reshape(rPixY, rPixX)

    bbitt1 = FLAG_T_BAD
    bbitt2 = FLAG_T_SKIP
    bbiti1 = FLAG_I_BAD
    bbiti2 = FLAG_I_SKIP

    if forceConvolve != "i":
        if lctNss == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                tRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            lctX0 = x0
            lctY0 = y0
            lctX = cx
            lctY = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                refArea2d, lctX0, lctY0,
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result['return_code'] == 0:
                lctSumVal = result['sum']
                lctMeanVal = result['mean']
                lctMedian = result['median']
                lctMode = result['mode']
                lctSd = result['sd']
                lctFwhm = result['fwhm']
                lctLfwhm = result['lfwhm']

    if forceConvolve != "t":
        if lciNss == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                iRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            lciX0 = x0
            lciY0 = y0
            lciX = cx
            lciY = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                refArea2d, lciX0, lciY0,
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result['return_code'] == 0:
                lciSumVal = result['sum']
                lciMeanVal = result['mean']
                lciMedian = result['median']
                lciMode = result['mode']
                lciSd = result['sd']
                lciFwhm = result['fwhm']
                lciLfwhm = result['lfwhm']

    if forceConvolve != "i":
        nss = lctNss
        if getCenters:
            kerFitThresh_t = float(np.float32(kerFitThresh))
            if lctNss < nKSStamps:
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                pcnt = psfCentersVectorized(
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
                    xmax - lctX0,
                    ymax - lctY0,
                    sPixX, sPixY,
                    lctX0, lctY0,
                    tUKThresh,
                    lctMode,
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

                    lctXss[nss] = xmax
                    lctYss[nss] = ymax
                    lctNss += 1

    if forceConvolve != "t":
        nss = lciNss
        if getCenters:
            kerFitThresh_i = float(np.float32(kerFitThresh))
            if lciNss < nKSStamps:
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                pcnt = psfCentersVectorized(
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
                    xmax - lciX0,
                    ymax - lciY0,
                    sPixX, sPixY,
                    lciX0, lciY0,
                    iUKThresh,
                    lciMode,
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

                    lciXss[nss] = xmax
                    lciYss[nss] = ymax
                    lciNss += 1

    return (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
            lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
            lmRData)


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


def make_noise_image4_numpy(data1d, invGain, quad, rPixX, rPixY):
    qquad = float(quad) * float(quad)
    nData = np.abs(data1d.astype(np.float64)) * float(invGain) + qquad
    return nData.astype(np.float32)


# def build_stamps_flatten_helper(ctSa, ciSa, ntS, niS, kernel_vec, filter_x, filter_y, status):
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

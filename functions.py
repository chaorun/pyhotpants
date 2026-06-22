import numpy as np
import numba

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

BUILD_STAMP_FLAT_CACHE = {}
LAST_SIGMA_CLIP = None
LAST_N = None
LAST_MEDIAN = None



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

def get_stamp_stats3_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                            umask, smask, maxiter, mRData_2d, statSig):
    return get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                                        umask, smask, maxiter, mRData_2d, statSig)

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
                                 umask, smask, maxiter, mRData_2d, statSig):
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

    nan_mask = np.isnan(data_flat)
    skip_all |= nan_mask

    # 收集 NaN 像素标记信息，不再原地修改 mRData_2d
    nan_updates = None
    if nan_mask.any():
        nan_y, nan_x = np.where(nan_mask.reshape(nPixY, nPixX))
        nan_updates = (nan_y, nan_x)


    sdat = np.asarray(data_flat[~skip_all], dtype=np.float32)

    if len(sdat) == 0:
        mode_val = 0.0
        if nfound > 0:
            mode_val = work[int(mfstat * nfound)]
        return (0.0, 0.0, mode_val, mode_val, MAXVAL, 0.0, 0.0, 5, nan_updates)


    mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat, maxiter, statSig)

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
        
    fwhm_val = current_binsize * (upper_val - lower_val) / 1.35

    median_target = goodcnt_h / 2.0
    median_val = bin_quartile_numpy(bins, median_target)
    lfwhm_val = current_binsize * (median_val - lower_val) * 2.0 / 1.35
    median_val = current_bin1 + current_binsize * (median_val - 1.0)

    return (ssum_val, mean_val, median_val, mode_val, sd_val, fwhm_val, lfwhm_val, 0, nan_updates)

def cut_sstamp_numpy(saXss, saYss, saX0, saY0, saNss, saSscnt, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData):
    # def cut_sstamp_numpy(sa, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData):
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

def buildStampsNumba(sXMin, sXMax, sYMin, sYMax,
                      getCenters, rXBMin, rYBMin,
                      lctNss_in, lctX0_in, lctY0_in, lctX_in, lctY_in, lctSumVal_in, lctMeanVal_in, lctMedian_in, lctMode_in, lctSd_in, lctFwhm_in, lctLfwhm_in, ctXss_in, ctYss_in,
                      lciNss_in, lciX0_in, lciY0_in, lciX_in, lciY_in, lciSumVal_in, lciMeanVal_in, lciMedian_in, lciMode_in, lciSd_in, lciFwhm_in, lciLfwhm_in, ciXss_in, ciYss_in,
                      iRData1d, tRData1d, hardX, hardY,
                      forceConvolve, rPixX, rPixY,
                      tUKThresh, iUKThresh, hwKSStamp,
                      fwStamp, nKSStamps, kerFitThresh,
                      mRData1d_in, statSig):
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
                sPixX, sPixY, 0x0, 0xffff, 3, mRData2d, statSig)

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
                sPixX, sPixY, 0x0, 0xffff, 3, mRData2d, statSig)

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

                check = check_psf_center_numba(tRData1d, xmax - lctX0, ymax - lctY0, sPixX, sPixY,
                    lctX0, lctY0, tUKThresh, lctMode, 1.0 / lctFwhm, 0, 0,
                    bbitt1 | bbitt2 | 0xbf, bbitt1, rPixX, hwKSStamp, lmRData, kerFitThresh)

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

                check = check_psf_center_numba(iRData1d, xmax - lciX0, ymax - lciY0, sPixX, sPixY,
                    lciX0, lciY0, iUKThresh, lciMode, 1.0 / lciFwhm, 0, 0,
                    bbiti1 | bbiti2 | 0xbf, bbiti1, rPixX, hwKSStamp, lmRData, kerFitThresh)

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

def make_noise_image4_numpy(data1d, invGain, quad):
    qquad = float(quad) * float(quad)
    nData = np.abs(data1d.astype(np.float64)) * float(invGain) + qquad
    return nData.astype(np.float32)


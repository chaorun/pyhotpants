import numpy as np
import math

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


def sigma_clip_numpy(data, maxiter=10, stat_sig=3.0):
    """返回 (mean, stdev, return_code)"""
    count = len(data)
    if count == 0:
        return (0.0, MAXVAL, 1)

    smask = [False] * count
    cnt = 0
    ncnt = count
    iternum = 0
    mean_val = 0.0
    stdev_val = 0.0

    while (ncnt != cnt) and (iternum < maxiter):
        cnt = ncnt
        mean_val = 0.0
        stdev_val = 0.0
        for i in range(count):
            if not smask[i]:
                d = np.float32(data[i])
                mean_val += float(d)
                stdev_val += float(d * d)

        if ncnt > 0:
            mean_val /= ncnt
        else:
            return (0.0, MAXVAL, 2)

        if ncnt > 1:
            stdev_val = stdev_val - ncnt * mean_val * mean_val
            stdev_val = math.sqrt(stdev_val / float(ncnt - 1))
        else:
            return (mean_val, MAXVAL, 3)

        ncnt = 0
        istdev = 1.0 / stdev_val
        for i in range(count):
            if not smask[i]:
                if (abs(float(data[i]) - mean_val) * istdev) > stat_sig:
                    smask[i] = True
                else:
                    ncnt += 1
        iternum += 1

    return (mean_val, stdev_val, 0)


def get_noise_stats3_numpy(data, noise, umask, smask, rPixX, rPixY, mRData):
    """返回 (nnorm, nncount)"""
    nsum = 0.0
    n = 0
    total = rPixX * rPixY
    data_flat = data.ravel()
    noise_flat = noise.ravel()
    mRData_flat = mRData.ravel()

    for i in range(total - 1, -1, -1):
        ddat = np.float32(data_flat[i])
        mdat = int(mRData_flat[i])

        if ((umask > 0) and not (mdat & umask)) or \
           ((smask > 0) and (mdat & smask)) or \
           (abs(float(ddat)) <= ZEROVAL):
            continue

        ndat = np.float32(1.0 / float(noise_flat[i]))
        n += 1
        prod = np.float32(np.float32(np.float32(ddat * ddat) * ndat) * ndat)
        nsum += float(prod)

    if n > 1:
        return (nsum / float(n), n)
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


def get_stamp_stats3_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                            umask, smask, maxiter, rPixX, mRData_2d, statSig):
    nstat = 100
    ufstat = 0.9
    mfstat = 0.5

    npts = nPixX * nPixY
    if npts < nstat:
        return {'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
                'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 4}

    rng = Ran1(-666)
    tries = 0

    goodcnt = 0
    work = [0.0] * nstat
    i = 0
    while (i < nstat) and (goodcnt < npts):
        xr = int(math.floor(rng() * nPixX))
        yr = int(math.floor(rng() * nPixY))

        rdat = float(data_2d[yr, xr])
        mdat = int(mRData_2d[yr + y0Reg, xr + x0Reg])

        if ((umask > 0) and not (mdat & umask)) or \
           ((smask > 0) and (mdat & smask)) or \
           (abs(rdat) <= ZEROVAL):
            pass
        else:
            work[i] = rdat
            i += 1
        goodcnt += 1

    work[:i] = sorted(work[:i])
    npts = i

    binsize = (work[int(ufstat * npts)] - work[int(mfstat * npts)]) / float(nstat)
    bin1 = work[int(mfstat * npts)] - 128.0 * binsize

    goodcnt = 0
    sdat = []
    for j in range(nPixY):
        for ci in range(nPixX):
            rdat = float(data_2d[j, ci])
            mdat = int(mRData_2d[j + y0Reg, ci + x0Reg])

            if ((umask > 0) and not (mdat & umask)) or \
               ((smask > 0) and (mdat & smask)) or \
               (abs(rdat) <= ZEROVAL):
                continue

            if rdat * 0.0 != 0.0:
                mRData_2d[j + y0Reg, ci + x0Reg] = int(mRData_2d[j + y0Reg, ci + x0Reg]) | (FLAG_INPUT_ISBAD | FLAG_ISNAN)
                continue

            sdat.append(rdat)
            goodcnt += 1

    sdat_arr = np.array(sdat, dtype=np.float32)
    mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat_arr, maxiter, statSig)
    if sc_rc != 0:
        return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
                'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 5}

    isd = 1.0 / sd_val

    repeat = 1
    ssum = 0.0
    lower_val = 0.0
    upper_val = 0.0
    mode_val = 0.0
    while repeat:
        if tries >= 5:
            return {'sum': 0.0, 'mean': mean_val, 'median': 0.0, 'mode': 0.0,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 1}

        bins = [0] * 256

        ssum = 0.0
        sumx = 0.0
        sumxx = 0.0
        goodcnt = 0

        for j in range(nPixY):
            for ci in range(nPixX):
                rdat = float(data_2d[j, ci])
                mdat = int(mRData_2d[j + y0Reg, ci + x0Reg])

                if ((umask > 0) and not (mdat & umask)) or \
                   ((smask > 0) and (mdat & smask)) or \
                   (abs(rdat) <= ZEROVAL):
                    continue

                if rdat * 0.0 != 0.0:
                    mRData_2d[j + y0Reg, ci + x0Reg] = int(mRData_2d[j + y0Reg, ci + x0Reg]) | (FLAG_INPUT_ISBAD | FLAG_ISNAN)
                    continue

                if (abs(rdat - mean_val) * isd) > statSig:
                    continue

                index = int(math.floor((rdat - bin1) / binsize)) + 1
                if index < 0:
                    index = 0
                if index > 255:
                    index = 255

                bins[index] += 1
                ssum += abs(rdat)
                goodcnt += 1

        if goodcnt == 0:
            mode_val = work[int(mfstat * npts)]
            median_val = mode_val
            return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 2}

        if binsize == 0.0:
            mode_val = work[int(mfstat * npts)]
            median_val = mode_val
            return {'sum': 0.0, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
                    'sd': sd_val, 'fwhm': 0.0, 'lfwhm': 0.0, 'return_code': 3}

        sumx = 0.0
        maxdens = 0.0
        imax = 0
        ilower = 1
        iupper = 1
        while iupper < 255:
            while (sumx < goodcnt / 10.0) and (iupper < 255):
                sumx += bins[iupper]
                iupper += 1

            if (iupper - ilower) > 0 and sumx / (iupper - ilower) > maxdens:
                maxdens = sumx / (iupper - ilower)
                imax = ilower

            sumx -= bins[ilower]
            ilower += 1

        if imax < 0 or imax > 255:
            imax = 0

        sumxx = 0.0
        sumx = 0.0
        ci = imax
        while (sumx < goodcnt / 10.0) and (ci < 255):
            sumx += bins[ci]
            sumxx += ci * bins[ci]
            ci += 1

        mode_bin = sumxx / sumx + 0.5
        mode_val = bin1 + binsize * (mode_bin - 1.0)

        imax_floor = int(math.floor(mode_bin))
        sumx = 0.0
        for ci in range(imax_floor):
            sumx += bins[ci]
        sumx += bins[imax_floor] * (mode_bin - imax_floor)
        sumx /= goodcnt
        moden = sumx

        lower = goodcnt * 0.25
        upper = goodcnt * 0.75
        sumx = 0.0
        ci = 0
        while sumx < lower:
            sumx += bins[ci]
            ci += 1
        lower_val = ci - (sumx - lower) / bins[ci - 1]

        while sumx < upper:
            sumx += bins[ci]
            ci += 1
        upper_val = ci - (sumx - upper) / bins[ci - 1]

        if (lower_val < 1.0) or (upper_val > 255.0):
            bin1 -= 128.0 * binsize
            binsize *= 2.0
            tries += 1
            repeat = 1
        elif (upper_val - lower_val) < 40.0:
            binsize /= 3.0
            bin1 = mode_val - 128.0 * binsize
            tries += 1
            repeat = 1
        else:
            repeat = 0

    sum_val = ssum

    fwhm_val = binsize * (upper_val - lower_val) / 1.35

    sumx = 0.0
    ci = 0
    while sumx < goodcnt / 2.0:
        sumx += bins[ci]
        ci += 1
    median_val = ci - (sumx - goodcnt / 2.0) / bins[ci - 1]

    lfwhm_val = binsize * (median_val - lower_val) * 2.0 / 1.35

    median_val = bin1 + binsize * (median_val - 1.0)

    return {'sum': sum_val, 'mean': mean_val, 'median': median_val, 'mode': mode_val,
            'sd': sd_val, 'fwhm': fwhm_val, 'lfwhm': lfwhm_val, 'return_code': 0}


def cut_sstamp_numpy(stamp, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData, verbose=0):
    stamp['krefArea'][:] = fillVal

    nss = stamp['nss']
    sscnt = stamp['sscnt']
    xStamp = int(stamp['xss'][sscnt]) - stamp['x0']
    yStamp = int(stamp['yss'][sscnt]) - stamp['y0']

    if sscnt >= nss:
        return 1

    sumVal = 0.0
    for j in range(yStamp - hwKSStamp, yStamp + hwKSStamp + 1):
        y = j - (yStamp - hwKSStamp)
        dy = j + stamp['y0']

        for i in range(xStamp - hwKSStamp, xStamp + hwKSStamp + 1):
            x = i - (xStamp - hwKSStamp)

            k = i + stamp['x0'] + rPixX * dy
            dpt = float(iData[k])

            stamp['krefArea'][x + y * fwKSStamp] = dpt
            if not (int(mRData[k]) & FLAG_INPUT_ISBAD):
                sumVal += abs(dpt)

    stamp['sum'] = sumVal
    return 0


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


def get_psf_centers_numpy(stamp, iData, xLen, yLen, hiThresh, bbit1, bbit2,
                           nKSStamps, hwKSStamp, rPixX, mRData, kerFitThresh,
                           verbose=0):
    kerFitThresh = float(np.float32(kerFitThresh))
    dfrac = 0.9

    if stamp['nss'] >= nKSStamps:
        return 0

    bbit = bbit1 | bbit2 | 0xbf
    sky = stamp['mode']
    invdsky = 1.0 / stamp['fwhm']

    sx0 = stamp['x0']
    sy0 = stamp['y0']

    xbuffer = 0
    ybuffer = 0

    floorVal = sky + kerFitThresh * stamp['fwhm']

    allocSize = max(1, (xLen * yLen) // hwKSStamp)
    xloc = [0] * allocSize
    yloc = [0] * allocSize
    peaks = [0.0] * allocSize

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
        qs = quick_sort_impl(peaks, pcnt)
        nssOrig = stamp['nss']
        idx = nssOrig
        jj = 0
        while jj < pcnt and idx < nKSStamps:
            stamp['xss'][idx] = xloc[qs[pcnt - jj - 1]] + sx0
            stamp['yss'][idx] = yloc[qs[pcnt - jj - 1]] + sy0
            stamp['nss'] += 1
            idx += 1
            jj += 1
        return 0


def build_stamps_numpy(sXMin, sXMax, sYMin, sYMax, niS, ntS,
                        getCenters, rXBMin, rYBMin, ciStamps, ctStamps,
                        iRData1d, tRData1d, hardX, hardY,
                        verbose, forceConvolve, rPixX, rPixY,
                        tUKThresh, iUKThresh, hwKSStamp,
                        fwStamp, nKSStamps, kerFitThresh,
                        mRData1d, statSig):
    sPixX = sXMax - sXMin + 1
    sPixY = sYMax - sYMin + 1

    bbitt1 = FLAG_T_BAD
    bbitt2 = FLAG_T_SKIP
    bbiti1 = FLAG_I_BAD
    bbiti2 = FLAG_I_SKIP

    mRData2d = mRData1d.reshape(rPixY, rPixX)

    if forceConvolve != "i":
        if ctStamps[ntS]['nss'] == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                tRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            ctStamps[ntS]['x0'] = x0
            ctStamps[ntS]['y0'] = y0
            ctStamps[ntS]['x'] = cx
            ctStamps[ntS]['y'] = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                refArea2d, ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result['return_code'] == 0:
                ctStamps[ntS]['sum'] = result['sum']
                ctStamps[ntS]['mean'] = result['mean']
                ctStamps[ntS]['median'] = result['median']
                ctStamps[ntS]['mode'] = result['mode']
                ctStamps[ntS]['sd'] = result['sd']
                ctStamps[ntS]['fwhm'] = result['fwhm']
                ctStamps[ntS]['lfwhm'] = result['lfwhm']

    if forceConvolve != "t":
        if ciStamps[niS]['nss'] == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                iRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            ciStamps[niS]['x0'] = x0
            ciStamps[niS]['y0'] = y0
            ciStamps[niS]['x'] = cx
            ciStamps[niS]['y'] = cy

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                refArea2d, ciStamps[niS]['x0'], ciStamps[niS]['y0'],
                sPixX, sPixY, 0x0, 0xffff, 3, rPixX, mRData2d, statSig)

            if result['return_code'] == 0:
                ciStamps[niS]['sum'] = result['sum']
                ciStamps[niS]['mean'] = result['mean']
                ciStamps[niS]['median'] = result['median']
                ciStamps[niS]['mode'] = result['mode']
                ciStamps[niS]['sd'] = result['sd']
                ciStamps[niS]['fwhm'] = result['fwhm']
                ciStamps[niS]['lfwhm'] = result['lfwhm']

    if forceConvolve != "i":
        nss = ctStamps[ntS]['nss']
        if getCenters:
            get_psf_centers_numpy(
                ctStamps[ntS], tRData1d, sPixX, sPixY,
                tUKThresh, bbitt1, bbitt2, nKSStamps,
                hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
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

                check = check_psf_center_numpy(
                    tRData1d,
                    xmax - ctStamps[ntS]['x0'],
                    ymax - ctStamps[ntS]['y0'],
                    sPixX, sPixY,
                    ctStamps[ntS]['x0'], ctStamps[ntS]['y0'],
                    tUKThresh,
                    ctStamps[ntS]['mode'],
                    1.0 / ctStamps[ntS]['fwhm'],
                    0, 0,
                    bbitt1 | bbitt2 | 0xbf, bbitt1,
                    rPixX, hwKSStamp, mRData1d, kerFitThresh)

                if check != 0.0:
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                            nr2 = l + rPixX * k
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                mRData1d[nr2] = int(mRData1d[nr2]) | bbitt2

                    ctStamps[ntS]['xss'][nss] = xmax
                    ctStamps[ntS]['yss'][nss] = ymax
                    ctStamps[ntS]['nss'] += 1

    if forceConvolve != "t":
        nss = ciStamps[niS]['nss']
        if getCenters:
            get_psf_centers_numpy(
                ciStamps[niS], iRData1d, sPixX, sPixY,
                iUKThresh, bbiti1, bbiti2, nKSStamps,
                hwKSStamp, rPixX, mRData1d, kerFitThresh, verbose)
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

                check = check_psf_center_numpy(
                    iRData1d,
                    xmax - ciStamps[niS]['x0'],
                    ymax - ciStamps[niS]['y0'],
                    sPixX, sPixY,
                    ciStamps[niS]['x0'], ciStamps[niS]['y0'],
                    iUKThresh,
                    ciStamps[niS]['mode'],
                    1.0 / ciStamps[niS]['fwhm'],
                    0, 0,
                    bbiti1 | bbiti2 | 0xbf, bbiti1,
                    rPixX, hwKSStamp, mRData1d, kerFitThresh)

                if check != 0.0:
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                            nr2 = l + rPixX * k
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                mRData1d[nr2] = int(mRData1d[nr2]) | bbiti2

                    ciStamps[niS]['xss'][nss] = xmax
                    ciStamps[niS]['yss'][nss] = ymax
                    ciStamps[niS]['nss'] += 1

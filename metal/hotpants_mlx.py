import mlx.core as mx
import numpy as np

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


def sigma_clip_mlx(data: np.ndarray, maxiter: int = 10, stat_sig: float = 3.0):
    count = len(data)
    if count == 0:
        return (0.0, MAXVAL, 1)
    arr = mx.array(data.astype(np.float32))
    mask = mx.zeros(count, dtype=mx.bool_)
    cnt = 0
    ncnt = count
    iternum = 0
    mean_val = 0.0
    stdev_val = 0.0
    sig = float(stat_sig)
    while ncnt != cnt and iternum < maxiter:
        cnt = ncnt
        valid = mx.logical_not(mask)
        ncnt = int(valid.sum())
        if ncnt <= 1:
            return (0.0, MAXVAL, 2 if ncnt == 0 else 3)
        w = valid.astype(mx.float32)
        mean_val = float((arr * w).sum() / ncnt)
        diff = arr - mean_val
        var = (diff * diff * w).sum() / (ncnt - 1)
        stdev_val = float(mx.sqrt(var))
        if stdev_val < ZEROVAL:
            return (mean_val, MAXVAL, 2)
        deviations = mx.abs(diff) / stdev_val
        new_outliers = deviations > sig
        mask = mx.logical_or(mask, mx.logical_and(new_outliers, valid))
        ncnt = int(mx.logical_not(mask).sum())
        iternum += 1
    return (mean_val, stdev_val, 0)


def get_noise_stats3_mlx(data: np.ndarray, noise: np.ndarray, umask: int, smask: int, rPixX: int, rPixY: int, mRData: np.ndarray):
    total = rPixX * rPixY
    d = mx.array(data[:total].astype(np.float32))
    n = mx.array(noise[:total].astype(np.float32))
    m = mx.array(mRData[:total].astype(np.int32))
    ZVAL = 1e-10
    cond1 = mx.ones(total, dtype=mx.bool_) if umask == 0 else ((m & umask) != 0)
    cond2 = mx.ones(total, dtype=mx.bool_) if smask == 0 else ((m & smask) == 0)
    cond3 = mx.abs(d) > ZVAL
    valid = cond1 & cond2 & cond3
    s = mx.where(valid, d * d / (n * n), mx.zeros_like(d)).sum()
    cnt = int(valid.sum())
    mx.eval(s)
    if cnt > 1:
        return (float(s) / cnt, cnt)
    return (MAXVAL, cnt)


def make_noise_image4_mlx(data1d: np.ndarray, invGain: float, rnoise: float):
    arr = mx.array(data1d.astype(np.float32))
    result = mx.abs(arr) * float(invGain) + float(rnoise) * float(rnoise)
    mx.eval(result)
    return np.array(result, dtype=np.float32)


def check_psf_center_mlx(iData: np.ndarray, imax: int, jmax: int, xLen: int, yLen: int,
                          sx0: int, sy0: int, hiThresh: float, sky: float, invdsky: float,
                          xbuffer: int, ybuffer: int, bbit: int, bbit1: int,
                          rPixX: int, hwKSStamp: int, mRData: np.ndarray, kerFitThresh: float):
    kerFT = np.float32(kerFitThresh)
    dmax2 = 0.0
    iData_mx = mx.array(iData)
    mRData_mx = mx.array(mRData)
    for il in range(jmax - hwKSStamp, jmax + hwKSStamp + 1):
        if il < ybuffer or il >= yLen - ybuffer:
            continue
        yr2 = il + sy0
        k_start = max(imax - hwKSStamp, xbuffer)
        k_end = min(imax + hwKSStamp + 1, xLen - xbuffer)
        if k_start >= k_end:
            continue
        indices_np = np.arange(k_start, k_end, dtype=np.int32) + sx0 + rPixX * yr2
        indices_mx = mx.array(indices_np)
        row_data = mx.take(iData_mx, indices_mx)
        row_mask = mx.take(mRData_mx, indices_mx)
        mx.eval(row_data, row_mask)
        vals_np = np.array(row_data)
        mask_np = np.array(row_mask)
        bad = (mask_np & bbit) != 0
        if bad.any():
            return 0.0
        over = vals_np >= hiThresh
        if over.any():
            first_global = indices_np[np.argmax(over)]
            mRData[first_global] |= bbit1
            return 0.0
        valid = (vals_np - sky) * invdsky > kerFT
        dmax2 += float(vals_np[valid].sum())
    return dmax2


def _local_peak_mlx(iData1d, mRData1d, i, j, hwKSStamp, sPixX, sPixY,
                     sx0_val, sy0_val, rPixX, hiThresh, sky_val, invdsky_val,
                     kerFitThreshVal, bbit, bbit1):
    dmax = -1e30
    imaxVal = i
    jmaxVal = j
    for il in range(j - hwKSStamp, j + hwKSStamp + 1):
        if il < 0 or il >= sPixY:
            continue
        yr2_val = il + sy0_val
        for ik in range(i - hwKSStamp, i + hwKSStamp + 1):
            if ik < 0 or ik >= sPixX:
                continue
            xr2_val = ik + sx0_val
            nr2 = xr2_val + rPixX * yr2_val
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
                imaxVal = ik
                jmaxVal = il
    return dmax, imaxVal, jmaxVal


def _mask_neighborhood_mlx(mRData1d, imaxVal, jmaxVal, hwKSStamp, sPixX, sPixY,
                            sx0_val, sy0_val, rPixX, bbit2):
    for il in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
        yr2_val = il + sy0_val
        for ik in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
            if ik > 0 and ik < sPixX and il > 0 and il < sPixY:
                xr2_val = ik + sx0_val
                nr2 = xr2_val + rPixX * yr2_val
                mRData1d[nr2] = mRData1d[nr2] | bbit2


def psfCenters_mlx(iData1d: np.ndarray, mRData1d: np.ndarray, sPixX: int, sPixY: int,
                    hwKSStamp: int, rPixX: int, hiThresh: float, kerFitThreshVal: float,
                    sky_val: float, invdsky_val: float, sx0_val: int, sy0_val: int,
                    nKSStamps: int, bbit: int, bbit1: int, bbit2: int,
                    xloc_out: np.ndarray, yloc_out: np.ndarray, peaks_out: np.ndarray):
    dfrac = 0.9
    floorVal = sky_val + kerFitThreshVal / invdsky_val
    xbuffer, ybuffer = 0, 0
    pcnt = 0
    fcnt = 2 * nKSStamps
    iData_mx = mx.array(iData1d)
    mRData_mx = mx.array(mRData1d)
    while pcnt < fcnt:
        loPsf = sky_val + (hiThresh - sky_val) * dfrac
        loPsf = max(loPsf, floorVal)
        for jj in range(ybuffer, sPixY - ybuffer):
            yr = jj + sy0_val
            ii_vals = np.arange(xbuffer, sPixX - xbuffer, dtype=np.int32)
            xr_vals = ii_vals + sx0_val
            nr_vals = xr_vals + rPixX * yr
            vals = mx.take(iData_mx, mx.array(nr_vals))
            masks = mx.take(mRData_mx, mx.array(nr_vals))
            mx.eval(vals, masks)
            v_np = np.array(vals).astype(np.float32)
            m_np = np.array(masks)
            ok = np.ones(len(v_np), dtype=bool)
            ok &= (m_np & bbit) == 0
            ok &= v_np < hiThresh
            ok &= ((v_np - sky_val) * invdsky_val) >= kerFitThreshVal
            ok &= v_np > loPsf
            candidates = np.where(ok)[0]
            for cid in candidates:
                if pcnt >= fcnt:
                    break
                c_i = ii_vals[cid]
                c_j = jj
                dmax, imaxVal, jmaxVal = _local_peak_mlx(
                    iData1d, mRData1d, c_i, c_j, hwKSStamp, sPixX, sPixY,
                    sx0_val, sy0_val, rPixX, hiThresh, sky_val, invdsky_val,
                    kerFitThreshVal, bbit, bbit1)
                dmax2 = check_psf_center_mlx(
                    iData1d, imaxVal, jmaxVal, sPixX, sPixY,
                    sx0_val, sy0_val, hiThresh, sky_val, invdsky_val,
                    0, 0, bbit, bbit1,
                    rPixX, hwKSStamp, mRData1d, kerFitThreshVal)
                if dmax2 == 0.0:
                    continue
                xloc_out[pcnt] = imaxVal
                yloc_out[pcnt] = jmaxVal
                peaks_out[pcnt] = dmax2
                pcnt += 1
                _mask_neighborhood_mlx(
                    mRData1d, imaxVal, jmaxVal, hwKSStamp, sPixX, sPixY,
                    sx0_val, sy0_val, rPixX, bbit2)
        if loPsf == floorVal:
            break
        dfrac -= 0.2
    return pcnt


def kernel_vector_mlx(n: int, deg_x: int, deg_y: int, ig: int, usePCA: int, fwKernel: int, hwKernel: int,
                       sigma_gauss: list, filter_x: np.ndarray, filter_y: np.ndarray,
                       kernel_vec: list, PCA=None):
    fwSq = fwKernel * fwKernel
    if usePCA:
        return kernel_vector_pca_mlx(n, fwKernel, PCA, kernel_vec)
    sig = float(sigma_gauss[ig])
    hw = hwKernel
    xs = mx.arange(-hw, hw + 1, dtype=mx.float32)
    qe = mx.exp(-xs * xs * sig)
    qe_np = np.array(qe)
    for ix in range(fwKernel):
        x = float(ix - hwKernel)
        k = ix + n * fwKernel
        filter_x[k] = qe_np[ix] * (x ** deg_x)
        filter_y[k] = qe_np[ix] * (x ** deg_y)
    kernel0 = None
    if n > 0:
        kernel0 = kernel_vec[0].copy()
    sum_x = 1.0 / np.sum(filter_x[n * fwKernel : (n + 1) * fwKernel])
    sum_y = 1.0 / np.sum(filter_y[n * fwKernel : (n + 1) * fwKernel])
    dx = (deg_x // 2) * 2 - deg_x
    dy = (deg_y // 2) * 2 - deg_y
    if dx == 0 and dy == 0:
        for ix in range(fwKernel):
            filter_x[ix + n * fwKernel] *= sum_x
            filter_y[ix + n * fwKernel] *= sum_y
        vector = np.zeros(fwSq, dtype=np.float32)
        for i in range(fwKernel):
            for j in range(fwKernel):
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]
        if n > 0:
            for i in range(fwSq):
                vector[i] -= kernel0[i]
            return vector, 1
        return vector, 0
    else:
        vector = np.zeros(fwSq, dtype=np.float32)
        for i in range(fwKernel):
            for j in range(fwKernel):
                vector[i + fwKernel * j] = filter_x[i + n * fwKernel] * filter_y[j + n * fwKernel]
        return vector, 0


def kernel_vector_pca_mlx(n: int, fwKernel: int, PCA, kernel_vec):
    fwSq = fwKernel * fwKernel
    vector = np.zeros(fwSq, dtype=np.float32)
    for i in range(fwKernel):
        for j in range(fwKernel):
            vector[i + fwKernel * j] = float(PCA[n][i + fwKernel * j])
    ren = 0
    if n > 0:
        k0 = kernel_vec[0]
        for i in range(fwSq):
            vector[i] -= k0[i]
    return vector, ren


def get_kernel_vec_mlx(ngauss: int, deg_fixe: list, usePCA: int, fwKernel: int, hwKernel: int,
                         sigma_gauss: list, filter_x: np.ndarray, filter_y: np.ndarray, PCA=None):
    kernel_vec = []
    nvec = 0
    for ig in range(ngauss):
        for d in range(deg_fixe[ig] + 1):
            for dx in range(d + 1):
                dy = d - dx
                vec, ren = kernel_vector_mlx(nvec, dx, dy, ig, usePCA, fwKernel, hwKernel,
                                              sigma_gauss, filter_x, filter_y, kernel_vec, PCA)
                kernel_vec.append(vec)
                nvec += 1
    return kernel_vec


def make_kernel_mlx(xi: int, yi: int, kernelSol: np.ndarray, rPixX: int, rPixY: int,
                     nCompKer: int, kerOrder: int, fwKernel: int,
                     kernel_vec: list, kernel_coeffs: np.ndarray, kernel: np.ndarray):
    k = 2
    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
    kernel_coeffs[0] = kernelSol[1]
    for ig in range(1, nCompKer):
        coeff = 0.0
        ax = 1.0
        for ix in range(kerOrder + 1):
            ay = 1.0
            for iy in range(kerOrder - ix + 1):
                coeff += kernelSol[k] * ax * ay
                ay *= yf
                k += 1
            ax *= xf
        kernel_coeffs[ig] = coeff
    fwSq = fwKernel * fwKernel
    kernel.fill(0.0)
    sum_kernel = 0.0
    for i in range(fwSq):
        val = 0.0
        for ig in range(nCompKer):
            val += kernel_coeffs[ig] * kernel_vec[ig][i]
        kernel[i] = val
        sum_kernel += val
    return sum_kernel


def buildAllKernels_mlx(kernelSol: np.ndarray, kernelVec2d: np.ndarray, nCompKer: int, kerOrder: int,
                          fwKernel: int, hwKernel: int, kcStep: int, rPixX: int, rPixY: int,
                          xSize: int, ySize: int):
    halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
    nstepsX = int(np.ceil(xSize / kcStep))
    nstepsY = int(np.ceil(ySize / kcStep))
    nBlocks = nstepsX * nstepsY
    i0Arr = np.arange(nstepsX) * kcStep + hwKernel
    j0Arr = np.arange(nstepsY) * kcStep + hwKernel
    iGrid, jGrid = np.meshgrid(i0Arr, j0Arr, indexing='xy')
    xi_mx = mx.array((iGrid + hwKernel).ravel().astype(np.float32))
    yi_mx = mx.array((jGrid + hwKernel).ravel().astype(np.float32))
    xf = (xi_mx - halfX) / halfX
    yf = (yi_mx - halfY) / halfY
    ks_mx = mx.array(kernelSol)
    coeffs_np = np.zeros((nBlocks, nCompKer), dtype=np.float32)
    coeffs_np[:, 0] = kernelSol[1]
    k = 2
    for ig in range(1, nCompKer):
        coeff = mx.zeros(nBlocks, dtype=mx.float32)
        ax = mx.ones(nBlocks, dtype=mx.float32)
        for ix in range(kerOrder + 1):
            ay = mx.ones(nBlocks, dtype=mx.float32)
            for iy in range(kerOrder - ix + 1):
                coeff = coeff + ks_mx[k] * ax * ay
                k += 1
                ay = ay * yf
            ax = ax * xf
        mx.eval(coeff)
        coeffs_np[:, ig] = np.array(coeff)
    allKernels = mx.array(coeffs_np) @ mx.array(kernelVec2d)
    mx.eval(allKernels)
    return np.array(allKernels), nstepsX, nstepsY


def background_loop_mlx(oRData1d: np.ndarray, kernelSol: np.ndarray, nCompKer: int, kerOrder: int,
                          bgOrder: int, rPixX: int, rPixY: int, hwKernel: int):
    nCompForBG = nCompKer - 1
    ncompBG = nCompForBG * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    halfX = 0.5 * rPixX
    halfY = 0.5 * rPixY
    xi = mx.arange(hwKernel, rPixX - hwKernel, dtype=mx.float32)
    yi = mx.arange(hwKernel, rPixY - hwKernel, dtype=mx.float32)
    xg, yg = mx.meshgrid(xi, yi)
    xf = (xg - halfX) / halfX
    yf = (yg - halfY) / halfY
    bg = mx.zeros_like(xf)
    k = 1
    ax = mx.ones_like(xf)
    for idegx in range(bgOrder + 1):
        ay = mx.ones_like(xf)
        for idegy in range(bgOrder - idegx + 1):
            bg = bg + kernelSol[ncompBG + k] * ax * ay
            k += 1
            ay = ay * yf
        ax = ax * xf
    mx.eval(bg)
    bg_np = np.array(bg).ravel()
    ny = rPixY - 2 * hwKernel
    nx = rPixX - 2 * hwKernel
    for j in range(ny):
        for i in range(nx):
            oRData1d[(j + hwKernel) * rPixX + (i + hwKernel)] += bg_np[j * nx + i]


def get_final_stamp_sig_mlx(saXss: np.ndarray, saYss: np.ndarray, saSscnt: np.ndarray, si: int,
                              imDiff: np.ndarray, imNoise: np.ndarray, fwKSStamp: int,
                              hwKSStamp: int, rPixX: int, mRData: np.ndarray):
    FLAG = np.int32(0x80)
    xRegion = int(saXss[si, int(saSscnt[si])])
    yRegion = int(saYss[si, int(saSscnt[si])])
    if xRegion < 0 or yRegion < 0:
        return -1.0
    indices = np.array([
        (xRegion - hwKSStamp + k) + rPixX * (yRegion - hwKSStamp + l)
        for l in range(fwKSStamp) for k in range(fwKSStamp)
    ], dtype=np.int32)
    idat = mx.array(imDiff, dtype=mx.float32)[mx.array(indices)]
    ndat = mx.array(imNoise, dtype=mx.float32)[mx.array(indices)]
    mdat = mx.array(mRData, dtype=mx.int32)[mx.array(indices)]
    valid = (mdat & FLAG) == 0
    sig_sum = mx.where(valid, idat * idat / (ndat * ndat), mx.zeros_like(idat)).sum()
    nsig = int(valid.sum())
    mx.eval(sig_sum)
    if nsig > 0:
        sig = float(sig_sum) / nsig
    else:
        sig = -1.0
    return sig


# =====================================================
# 阶段 7: build_matrix, build_scprod
# =====================================================

def build_matrix_mlx(saMat: np.ndarray, saVectors: np.ndarray, saSscnt: np.ndarray,
                      saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray,
                      nS: int, nCompKer: int, kerOrder: int, bgOrder: int,
                      fwKSStamp: int, rPixX: int, rPixY: int,
                      nKSStamps: int = None):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp
    mat_size = ncomp + nbg_vec + 1

    valid = saSscnt[:nS] < saNss[:nS]
    n_valid = int(valid.sum())

    wxy = np.zeros((nS, ncomp2), dtype=np.float32)
    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float32)

    if nKSStamps is None:
        nKSStamps_val = saXss.shape[1]
    else:
        nKSStamps_val = nKSStamps

    if n_valid == 0:
        return matrix, wxy

    valid_np = np.asarray(valid)
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps_val - 1)
    rg = np.arange(nS)
    all_x_np = np.where(valid_np, saXss[rg, safe_sscnt], 0).astype(np.float32)
    all_y_np = np.where(valid_np, saYss[rg, safe_sscnt], 0).astype(np.float32)

    nC_valid = nCompKer + 1
    nvec_total = nCompKer + nbg_vec
    all_mat = saMat[:nS, :nC_valid, :nC_valid].copy()
    all_vectors = saVectors[:nS, :nvec_total, :].copy()

    wxy.fill(0.0)
    matrix.fill(0.0)

    rPixX2 = np.float32(0.5 * rPixX)
    rPixY2 = np.float32(0.5 * rPixY)

    for s in range(nS):
        if not valid_np[s]:
            continue
        xstamp = all_x_np[s]
        ystamp = all_y_np[s]
        fx = np.float32((xstamp - rPixX2) / rPixX2)
        fy = np.float32((ystamp - rPixY2) / rPixY2)
        kk = 0
        a1 = np.float32(1.0)
        for ideg1 in range(kerOrder + 1):
            a2 = np.float32(1.0)
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2
                kk += 1
                a2 *= fy
            a1 *= fx

        ws = wxy[s]
        wwy = np.outer(ws, ws)
        for i1 in range(ncomp1):
            r0 = i1 * ncomp2 + 2
            for j1 in range(0, i1 + 1):
                c0 = j1 * ncomp2 + 2
                matrix[r0:r0+ncomp2, c0:c0+ncomp2] += wwy * all_mat[s, i1 + 2, j1 + 2]

        matrix[1, 1] += all_mat[s, 1, 1]
        for i1 in range(ncomp1):
            offset_r = i1 * ncomp2 + 2
            matrix[offset_r:offset_r+ncomp2, 1] += ws[:ncomp2] * all_mat[s, i1 + 2, 1]

        for ibg in range(nbg_vec):
            ii = ncomp + ibg + 1
            ivecbg = ncomp1 + ibg + 1
            for i1 in range(1, ncomp1 + 1):
                p0 = np.dot(all_vectors[s, i1, :], all_vectors[s, ivecbg, :])
                matrix[ii + 1, (i1-1)*ncomp2+2 : i1*ncomp2+2] += p0 * ws
            p0 = np.dot(all_vectors[s, 0, :], all_vectors[s, ivecbg, :])
            matrix[ii + 1, 1] += p0
            vec_bg = all_vectors[s, ivecbg, :]
            for jbg in range(ibg + 1):
                q = np.dot(vec_bg, all_vectors[s, ncomp1 + jbg + 1, :])
                matrix[ii + 1, ncomp + jbg + 2] += q

    for i in range(mat_size):
        for j in range(i + 1):
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]

    matrix[0, 0] = 1.0

    return matrix, wxy


def build_scprod_mlx(saScprod: np.ndarray, saVectors: np.ndarray, saSscnt: np.ndarray,
                      saNss: np.ndarray, saXss: np.ndarray, saYss: np.ndarray,
                      nS: int, image: np.ndarray, nCompKer: int, kerOrder: int,
                      bgOrder: int, fwKSStamp: int, hwKSStamp: int, rPixX: int,
                      wxy: np.ndarray, nKSStamps: int = None):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp

    valid = saSscnt[:nS] < saNss[:nS]
    n_valid = int(valid.sum())

    if nKSStamps is None:
        nKSStamps_val = saXss.shape[1]
    else:
        nKSStamps_val = nKSStamps

    if n_valid == 0:
        return np.zeros(ncomp + nbg_vec + 2, dtype=np.float32)

    valid_np = np.asarray(valid)
    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps_val - 1)
    rg = np.arange(nS)
    all_x_np = np.where(valid_np, saXss[rg, safe_sscnt], 0).astype(np.int64)
    all_y_np = np.where(valid_np, saYss[rg, safe_sscnt], 0).astype(np.int64)

    nvec_total = nCompKer + nbg_vec
    all_vectors = saVectors[:nS, :nvec_total, :].copy()
    all_scprod_clip = saScprod[:nS, :nCompKer + 1].copy()
    image_arr = np.asarray(image, dtype=np.float32)
    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float32)

    valid_idx = np.where(valid_np)[0]
    wxy_v = wxy[valid_idx]
    scp_v = all_scprod_clip[valid_idx]
    vec_v = all_vectors[valid_idx]

    kernelSol[1] = scp_v[:, 1].sum()
    for i1 in range(1, ncomp1 + 1):
        p0_vec = scp_v[:, i1 + 1]  # (n_valid,)
        for i2 in range(ncomp2):
            ii = (i1 - 1) * ncomp2 + i2 + 1
            kernelSol[ii + 1] = np.sum(p0_vec * wxy_v[:, i2])

    for s_idx, s in enumerate(valid_idx):
        xi = int(all_x_np[s])
        yi = int(all_y_np[s])
        for ibg in range(nbg_vec):
            q = 0.0
            for xc in range(-hwKSStamp, hwKSStamp + 1):
                for yc in range(-hwKSStamp, hwKSStamp + 1):
                    k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                    q += vec_v[s_idx, ncomp1 + ibg + 1, k] * image_arr[xc + xi + rPixX * (yc + yi)]
            kernelSol[ncomp + ibg + 2] += q

    return kernelSol


# =====================================================
# 阶段 8: get_stamp_sig, get_stamp_sig_batch
# =====================================================

def get_stamp_sig_mlx(vectors: np.ndarray, kernelSol: np.ndarray, imNoise: np.ndarray,
                       mRData1d: np.ndarray, im: np.ndarray,
                       fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
                       nCompKer: int, kerOrder: int, bgOrder: int, xi: int, yi: int,
                       temp: np.ndarray):
    LOCAL_ZEROVAL = 1e-10
    LOCAL_MAXVAL = 1e10
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    LOCAL_FLAG_ISNAN = 0x08

    ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
    fwSq = fwKSStamp * fwKSStamp

    xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
    yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)

    background = 0.0; k = 1; ax = 1.0
    for i in range(bgOrder + 1):
        ay = 1.0
        for j in range(bgOrder - i + 1):
            background += kernelSol[ncompBG + k] * ax * ay
            k += 1; ay *= yf
        ax *= xf

    csModel = np.zeros(fwSq, dtype=np.float32)
    coeff0 = kernelSol[1]
    for i in range(fwSq):
        csModel[i] = coeff0 * vectors[0, i]

    kk = 2
    for i1 in range(1, nCompKer):
        coeff = 0.0; ax = 1.0
        for ix in range(kerOrder + 1):
            ay = 1.0
            for iy in range(kerOrder - ix + 1):
                coeff += kernelSol[kk] * ax * ay
                kk += 1; ay *= yf
            ax *= xf
        for i in range(fwSq):
            csModel[i] += coeff * vectors[i1, i]

    nsig = 0; sig1 = 0.0

    indices = np.array([
        (xi - hwKSStamp + k) + rPixX * (yi - hwKSStamp + l)
        for l in range(fwKSStamp) for k in range(fwKSStamp)
    ], dtype=np.int32)

    ndat = mx.array(imNoise, dtype=mx.float32)[mx.array(indices)]
    mdat = mx.array(mRData1d, dtype=mx.int32)[mx.array(indices)]

    valid = (mdat & LOCAL_FLAG_INPUT_ISBAD) == 0
    not_zero = mx.abs(mx.array(csModel)) > LOCAL_ZEROVAL
    valid = valid & not_zero

    mx.eval(valid)
    valid_np = np.array(valid)
    nsig = int(valid_np.sum())

    for i in range(fwSq):
        if valid_np[i]:
            diff = csModel[i] - im[i] + background
            temp[i] = diff
            sig1 += diff * diff / float(np.array(ndat)[i])

    if nsig > 0:
        sig1 /= nsig
        if sig1 >= LOCAL_MAXVAL:
            sig1 = -1.0
    else:
        sig1 = -1.0

    return (sig1, -1.0, -1.0, nsig)


def get_stamp_sig_batch_mlx(sa_vectors: np.ndarray, sa_krefArea: np.ndarray,
                              sa_sscnt: np.ndarray, sa_nss: np.ndarray,
                              sa_xss: np.ndarray, sa_yss: np.ndarray,
                              kernelSol: np.ndarray, imNoise: np.ndarray,
                              mRData1d: np.ndarray,
                              fwKSStamp: int, hwKSStamp: int, rPixX: int, rPixY: int,
                              nCompKer: int, kerOrder: int, bgOrder: int, nS: int,
                              out_sig1: np.ndarray, out_sig2: np.ndarray, out_sig3: np.ndarray,
                              batched_bg=None, batched_coeffs=None):
    LOCAL_ZEROVAL = 1e-10
    LOCAL_MAXVAL = 1e10
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    fwSq = fwKSStamp * fwKSStamp

    for si in range(nS):
        if sa_sscnt[si] >= sa_nss[si]:
            out_sig1[si] = -1.0; out_sig2[si] = -1.0; out_sig3[si] = -1.0
            continue

        xi = int(sa_xss[si, sa_sscnt[si]])
        yi = int(sa_yss[si, sa_sscnt[si]])

        indices = np.array([
            (xi - hwKSStamp + k) + rPixX * (yi - hwKSStamp + l)
            for l in range(fwKSStamp) for k in range(fwKSStamp)
        ], dtype=np.int32)

        ndat = mx.array(imNoise, dtype=mx.float32)[mx.array(indices)]
        mdat = mx.array(mRData1d, dtype=mx.int32)[mx.array(indices)]
        kref = mx.array(sa_krefArea[si])
        vecs = mx.array(sa_vectors[si])

        valid = (mdat & LOCAL_FLAG_INPUT_ISBAD) == 0
        valid = valid & (mx.abs(kref) > LOCAL_ZEROVAL)
        mx.eval(valid, kref)

        nv = int(valid.sum())
        if nv == 0:
            out_sig1[si] = -1.0; out_sig2[si] = -1.0; out_sig3[si] = -1.0
            continue

        if batched_coeffs is not None:
            coeffs = mx.array(batched_coeffs[si])
        else:
            # 在线计算核系数
            coeffs_np = np.zeros(nCompKer, dtype=np.float32)
            coeffs_np[0] = kernelSol[1]
            xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
            yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
            kk = 2
            for i1 in range(1, nCompKer):
                c = 0.0; ax = 1.0
                for ix in range(kerOrder + 1):
                    ay = 1.0
                    for iy in range(kerOrder - ix + 1):
                        c += kernelSol[kk] * ax * ay
                        kk += 1; ay *= yf
                    ax *= xf
                coeffs_np[i1] = c
            coeffs = mx.array(coeffs_np)

        csModel = mx.zeros(fwSq, dtype=mx.float32)
        for iv in range(nCompKer):
            csModel = csModel + coeffs[iv] * vecs[iv]

        if batched_bg is not None:
            bg = float(batched_bg[si])
        else:
            ncompBG_val = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
            bg = 0.0; k = 1; ax = 1.0
            xf = (xi - 0.5 * rPixX) / (0.5 * rPixX)
            yf = (yi - 0.5 * rPixY) / (0.5 * rPixY)
            for i in range(bgOrder + 1):
                ay = 1.0
                for j in range(bgOrder - i + 1):
                    bg += kernelSol[ncompBG_val + k] * ax * ay
                    k += 1; ay *= yf
                ax *= xf

        diffModel = csModel - kref + bg
        temp_sig = mx.where(valid, diffModel * diffModel / ndat, mx.zeros_like(diffModel))
        sig_sum = temp_sig.sum()
        mx.eval(sig_sum)

        if nv > 0:
            out_sig1[si] = float(sig_sum) / nv
        else:
            out_sig1[si] = -1.0
        out_sig2[si] = -1.0
        out_sig3[si] = -1.0


# =====================================================
# 阶段 9: fill_stamp_kernel
# =====================================================

def fill_stamp_kernel_mlx(image: np.ndarray, imRef: np.ndarray, filterX: np.ndarray, filterY: np.ndarray,
                           xi: np.ndarray, yi: np.ndarray, fwKSStamp: int, hwKSStamp: int,
                           fwKernel: int, hwKernel: int, rPixX: int, rPixY: int,
                           nCompKer: int, bgOrder: int, nvec: int, renFlags_arr: np.ndarray,
                           fillVal: float, mRData1d: np.ndarray,
                           out_vectors: np.ndarray, out_krefArea: np.ndarray, out_mat: np.ndarray,
                           out_scprod: np.ndarray, out_sum_val: np.ndarray, n_stamps: int):
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel
    LOCAL_FLAG_INPUT_ISBAD = 0x80
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2

    kernels_np = np.zeros((nvec, fwSqKernel), dtype=np.float32)
    for n in range(nvec):
        for jc in range(fwKernel):
            for ic in range(fwKernel):
                fy_idx = fwKernel - 1 - jc
                fx_idx = fwKernel - 1 - ic
                kernels_np[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]
    kernels_mx = mx.array(kernels_np)

    stamp_offsets = np.array([
        (ix % fwKSStamp) + rPixX * (ix // fwKSStamp)
        for ix in range(fwSqStamp)
    ], dtype=np.int32)

    kernel_offsets = np.array([
        (kx % fwKernel) + rPixX * (kx // fwKernel)
        for kx in range(fwSqKernel)
    ], dtype=np.int32)

    image_mx = mx.array(image)
    imRef_np = np.asarray(imRef)
    mRData_np = np.asarray(mRData1d)

    rPixX2 = np.float32(0.5 * rPixX)
    rPixY2 = np.float32(0.5 * rPixY)

    for si in range(n_stamps):
        xi_s = xi[si]
        yi_s = yi[si]

        img_base = (xi_s - hwKSStamp - hwKernel) + rPixX * (yi_s - hwKSStamp - hwKernel)
        all_indices = img_base + stamp_offsets[:, None] + kernel_offsets
        all_indices_flat = all_indices.ravel().astype(np.int32)

        img_vals = image_mx[mx.array(all_indices_flat)]
        img_vals_3d = img_vals.reshape(fwSqStamp, fwSqKernel)
        # img_vals_3d: (fwSqStamp, fwSqKernel)
        # kernels_mx: (nvec, fwSqKernel)
        # out: (nvec, fwSqStamp) via matmul
        out_vec_mx = kernels_mx @ img_vals_3d.T  # (nvec, fwSqStamp)
        mx.eval(out_vec_mx)
        out_vectors_np = np.array(out_vec_mx).T  # (fwSqStamp, nvec)

        for n in range(nvec):
            out_vectors[si, n, :] = out_vectors_np[:, n]

        for n in range(nvec):
            if renFlags_arr[n]:
                out_vectors[si, n, :] -= out_vectors[si, 0, :]

        for fsi in range(fwSqStamp):
            out_krefArea[si, fsi] = fillVal

        sumVal = 0.0
        for y_off in range(fwKSStamp):
            img_y = yi_s - hwKSStamp + y_off
            for x_off in range(fwKSStamp):
                img_x = xi_s - hwKSStamp + x_off
                k = img_x + rPixX * img_y
                dpt = imRef_np[k]
                out_krefArea[si, x_off + y_off * fwKSStamp] = dpt
                if (mRData_np[k] & LOCAL_FLAG_INPUT_ISBAD) == 0:
                    sumVal += abs(dpt)
        out_sum_val[si, 0] = sumVal

        nv = nvec
        for y_off in range(fwKSStamp):
            j = yi_s - hwKSStamp + y_off
            yf = (j - rPixY2) / rPixY2
            for x_off in range(fwKSStamp):
                i = xi_s - hwKSStamp + x_off
                xf = (i - rPixX2) / rPixX2
                ipix = x_off + y_off * fwKSStamp
                ax = np.float32(1.0)
                nv = nvec
                for idegx in range(bgOrder + 1):
                    ay = np.float32(1.0)
                    for idegy in range(bgOrder - idegx + 1):
                        out_vectors[si, nv, ipix] = ax * ay
                        ay *= yf
                        nv += 1
                    ax *= xf

        ncomp1_jit = nCompKer

        for i in range(ncomp1_jit):
            for j in range(i + 1):
                q = 0.0
                for k in range(fwSqStamp):
                    q += out_vectors[si, i, k] * out_vectors[si, j, k]
                out_mat[si, i + 1, j + 1] = q

        ivecbg = ncomp1_jit
        for i1 in range(ncomp1_jit):
            p0 = 0.0
            for k in range(fwSqStamp):
                p0 += out_vectors[si, i1, k] * out_vectors[si, ivecbg, k]
            out_mat[si, ncomp1_jit + 1, i1 + 1] = p0

        q = 0.0
        for k in range(fwSqStamp):
            q += out_vectors[si, ivecbg, k] * out_vectors[si, ncomp1_jit, k]
        out_mat[si, ncomp1_jit + 1, ncomp1_jit + 1] = q

        for i1 in range(ncomp1_jit):
            p0 = 0.0
            for xc in range(-hwKSStamp, hwKSStamp + 1):
                for yc in range(-hwKSStamp, hwKSStamp + 1):
                    k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                    p0 += out_vectors[si, i1, k] * out_krefArea[si, k]
            out_scprod[si, i1 + 1] = p0

        p0 = 0.0
        for xc in range(-hwKSStamp, hwKSStamp + 1):
            for yc in range(-hwKSStamp, hwKSStamp + 1):
                k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
                p0 += out_vectors[si, ncomp1_jit, k] * out_krefArea[si, k]
        out_scprod[si, ncomp1_jit + 1] = p0


# Placeholder empty function for spatial_convolve
def spatial_convolve_mlx(image: np.ndarray, variance: np.ndarray, cMask: np.ndarray,
                          cRdata: np.ndarray, vData: np.ndarray, mRData: np.ndarray,
                          kernelSol: np.ndarray,
                          xSize: int, ySize: int, nCompKer: int, kerOrder: int,
                          fwKernel: int, hwKernel: int,
                          kcStep: int, rPixX: int, rPixY: int, kerFracMask: float,
                          dovar: int, convolveVariance: int,
                          kernel_vec_2d: np.ndarray,
                          allKernels=None, nstepsX_in=None):
    fwSq = fwKernel * fwKernel
    FLAG_INPUT_ISBAD = np.int32(0x80)
    FLAG_OK_CONV = np.int32(0x40)
    FLAG_BAD_CONV = np.int32(0x10)

    nstepsX_val = nstepsX_in if nstepsX_in else int(np.ceil(xSize / kcStep))
    nstepsY_val = int(np.ceil(ySize / kcStep))

    if allKernels is None:
        ak, nx, ny = buildAllKernels_mlx(kernelSol, kernel_vec_2d, nCompKer, kerOrder,
                                           fwKernel, hwKernel, kcStep, rPixX, rPixY, xSize, ySize)
    else:
        ak = allKernels

    padded_y = ySize + 2 * hwKernel
    padded_x = xSize + 2 * hwKernel

    img_pad = np.zeros(padded_x * padded_y, dtype=np.float32)
    var_pad = np.zeros(padded_x * padded_y, dtype=np.float32)
    msk_pad = np.zeros(padded_x * padded_y, dtype=np.int32)
    for j in range(ySize):
        s0 = j * rPixX
        d0 = (j + hwKernel) * padded_x + hwKernel
        img_pad[d0:d0+xSize] = image[s0:s0+xSize]
        var_pad[d0:d0+xSize] = variance[s0:s0+xSize]
        msk_pad[d0:d0+xSize] = np.asarray(cMask[s0:s0+xSize], dtype=np.int32)

    img_mx = mx.array(img_pad)
    var_mx = mx.array(var_pad)
    msk_mx = mx.array(msk_pad)

    neighbor_offsets = np.array([
        dx + padded_x * dy
        for dy in range(-hwKernel, hwKernel + 1)
        for dx in range(-hwKernel, hwKernel + 1)
    ], dtype=np.int32)

    ker_mx = mx.array(ak)

    for j1 in range(nstepsY_val):
        j0 = j1 * kcStep + hwKernel
        j_end = min(j0 + kcStep, ySize - hwKernel)
        for i1 in range(nstepsX_val):
            i0 = i1 * kcStep + hwKernel
            i_end = min(i0 + kcStep, xSize - hwKernel)

            kernel_np = ak[j1 * nstepsX_val + i1]
            kernel_flip = np.flip(kernel_np.reshape(fwKernel, fwKernel), axis=(0, 1)).ravel()
            kernel = mx.array(kernel_flip)
            kk_mx = kernel * kernel

            nRows = j_end - j0
            nCols = i_end - i0
            if nRows <= 0 or nCols <= 0:
                continue

            padded_i0 = i0 + hwKernel
            padded_j0 = j0 + hwKernel
            centers = np.arange(padded_i0, padded_i0 + nCols, dtype=np.int32) + padded_x * np.arange(padded_j0, padded_j0 + nRows, dtype=np.int32)[:, None]
            nPixels = nRows * nCols
            neighbors = centers.ravel()[:, None] + neighbor_offsets

            px_vals = img_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
            result = px_vals @ kernel
            mx.eval(result)
            res_np = np.array(result).reshape(nRows, nCols)
            for r in range(nRows):
                cRdata[(j0 + r) * rPixX + i0 : (j0 + r) * rPixX + i_end] = res_np[r]

            if dovar:
                var_vals = var_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
                var_result = var_vals @ kk_mx
                mx.eval(var_result)
                var_np = np.array(var_result).reshape(nRows, nCols)
                for r in range(nRows):
                    vData[(j0 + r) * rPixX + i0 : (j0 + r) * rPixX + i_end] = var_np[r]

            msk_vals = msk_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
            mbit = mx.max(msk_vals, axis=1)
            aks_val = mx.abs(kernel).sum()
            isbad = (msk_vals & FLAG_INPUT_ISBAD) != 0
            uks_vals = mx.where(isbad, mx.zeros_like(msk_vals), mx.abs(kernel)).sum(axis=1)
            bad_frac = mx.where(aks_val > 0, uks_vals / aks_val, mx.ones_like(uks_vals))
            conv_flag = mx.where(bad_frac < kerFracMask,
                                 FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV,
                                 FLAG_OK_CONV)

            self_mask = msk_mx[mx.array(centers.ravel())]
            isbad_self = (self_mask & FLAG_INPUT_ISBAD) != 0
            ok_flag = self_mask | (FLAG_OUTPUT_ISBAD * isbad_self.astype(mx.int32))
            ok_flag = mx.where(mbit != 0, ok_flag | conv_flag, ok_flag)
            mx.eval(ok_flag)
            msk_np = np.array(ok_flag).reshape(nRows, nCols)
            for r in range(nRows):
                mRData[(j0 + r) * rPixX + i0 : (j0 + r) * rPixX + i_end] = msk_np[r]


# =====================================================
# 阶段 11: 编排层
# =====================================================

def check_again_mlx(saSscnt: np.ndarray, saNss: np.ndarray, saChi2: np.ndarray,
                     saXss: np.ndarray, saYss: np.ndarray, saVectors: np.ndarray,
                     saKrefArea: np.ndarray, kernelSol: np.ndarray,
                     imNoise: np.ndarray, nS: int, figMerit: str, kerSigReject: float,
                     statSig: float, fwKSStamp: int, hwKSStamp: int, rPixX: int,
                     rPixY: int, mRData: np.ndarray, nCompKer: int, kerOrder: int,
                     bgOrder: int):
    ss = np.zeros(nS, dtype=np.float32)
    nss_val = 0
    sig = 0.0; check = 0; mean = 0.0; stdev = 0.0
    nskippedSubstamps = 0
    refill_indices = []
    sscnt_local = saSscnt[:nS].copy()
    chi2_local = saChi2[:nS].copy()

    batch_sig1 = None
    if figMerit[0:1] == "v":
        batch_sig1 = np.zeros(nS, dtype=np.float32)
        batch_sig2 = np.zeros(nS, dtype=np.float32)
        batch_sig3 = np.zeros(nS, dtype=np.float32)

        halfX, halfY = 0.5 * rPixX, 0.5 * rPixY
        xf_batch = np.zeros(nS, dtype=np.float32)
        yf_batch = np.zeros(nS, dtype=np.float32)
        bg_batch = np.zeros(nS, dtype=np.float32)
        coeffs_batch = np.zeros((nS, nCompKer), dtype=np.float32)

        for si in range(nS):
            if sscnt_local[si] < saNss[si]:
                xi = float(saXss[si, sscnt_local[si]])
                yi = float(saYss[si, sscnt_local[si]])
                xf = (xi - halfX) / halfX
                yf = (yi - halfY) / halfY
                xf_batch[si] = xf; yf_batch[si] = yf

                ncompBG_val = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) // 2) + 1
                bg = 0.0; k = 1; ax = 1.0
                for i in range(bgOrder + 1):
                    ay = 1.0
                    for j in range(bgOrder - i + 1):
                        bg += kernelSol[ncompBG_val + k] * ax * ay
                        k += 1; ay *= yf
                    ax *= xf
                bg_batch[si] = bg

                coeffs_np = np.zeros(nCompKer, dtype=np.float32)
                coeffs_np[0] = kernelSol[1]
                kk = 2
                for i1 in range(1, nCompKer):
                    c = 0.0; ax = 1.0
                    for ix in range(kerOrder + 1):
                        ay = 1.0
                        for iy in range(kerOrder - ix + 1):
                            c += kernelSol[kk] * ax * ay
                            kk += 1; ay *= yf
                        ax *= xf
                    coeffs_np[i1] = c
                coeffs_batch[si] = coeffs_np

        get_stamp_sig_batch_mlx(
            saVectors, saKrefArea, sscnt_local, saNss, saXss, saYss,
            kernelSol, imNoise, mRData, fwKSStamp, hwKSStamp, rPixX, rPixY,
            nCompKer, kerOrder, bgOrder, nS,
            batch_sig1, batch_sig2, batch_sig3, bg_batch, coeffs_batch)

    for istamp in range(nS):
        if sscnt_local[istamp] < saNss[istamp]:
            if batch_sig1 is not None:
                sig1 = batch_sig1[istamp]
            else:
                sscnt = sscnt_local[istamp]
                xRegion = int(saXss[istamp, sscnt])
                yRegion = int(saYss[istamp, sscnt])
                im_arr = saKrefArea[istamp]
                temp = np.zeros(fwKSStamp * fwKSStamp, dtype=np.float32)
                sig1, sig2, sig3, nsig = get_stamp_sig_mlx(
                    saVectors[istamp], kernelSol, imNoise, mRData, im_arr,
                    fwKSStamp, hwKSStamp, rPixX, rPixY,
                    nCompKer, kerOrder, bgOrder, xRegion, yRegion, temp)

                if figMerit[0:1] != "v":
                    s = get_stamp_stats3_mlx(temp, xRegion - hwKSStamp, yRegion - hwKSStamp,
                                              fwKSStamp, fwKSStamp, 0x0, 0xffff, 5, mRData, statSig)
                    sig2 = s[4] if s[7] == 0 else -1.0
                    sig3 = s[5] if s[7] == 0 else -1.0
                    if sig2 < 0 or sig2 >= MAXVAL: sig2 = -1.0
                    if sig3 < 0 or sig3 >= MAXVAL: sig3 = -1.0

            sig = sig1 if figMerit[0:1] == "v" else (sig2 if figMerit[0:1] == "s" else sig3)
            ok = (figMerit[0:1] == "v" and sig1 != -1) or \
                 (figMerit[0:1] == "s" and sig2 != -1) or \
                 (figMerit[0:1] == "h" and sig3 != -1)

            if ok:
                ss[nss_val] = sig
                nss_val += 1
            else:
                sscnt_local[istamp] += 1
                refill_indices.append(istamp)
                nskippedSubstamps += 1

    if nss_val > 1:
        sarr = ss[:nss_val]
        mean, stdev, rc = sigma_clip_mlx(sarr, 10, statSig)
        sigma_cut = kerSigReject * stdev
        nrejected = 0
        idx = 0
        for istamp in range(nS):
            if sscnt_local[istamp] < saNss[istamp]:
                sig_val = ss[idx]
                idx += 1
                if abs(sig_val - mean) > sigma_cut:
                    sscnt_local[istamp] += 1
                    refill_indices.append(istamp)
                    nrejected += 1
                    check = 1
        meansigSubstamps = mean
        scatterSubstamps = stdev
    elif nss_val == 1:
        meansigSubstamps = ss[0]
        scatterSubstamps = -1.0
    else:
        meansigSubstamps = -1.0
        scatterSubstamps = -1.0

    return (check, meansigSubstamps, scatterSubstamps, nskippedSubstamps,
            refill_indices, sscnt_local, chi2_local)


def get_stamp_stats3_mlx(data_2d, x0Reg, y0Reg, nPixX, nPixY, umask, smask, statSig, mRData_2d, dummy=None):
    from pyhotpants.float32.functions import get_stamp_stats3_fast_numpy
    return get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY, umask, smask, statSig, mRData_2d)


def fill_stamp_numpy_mlx(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss,
                          saKrefArea, saSumVal, si, imConv, imRef, rPixX, rPixY,
                          ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                          bgOrder, nCompKer, filter_x, filter_y, fillVal, mRData):
    from pyhotpants.float32.alard import fill_stamp_numpy as _orig
    return _orig(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss,
                  saKrefArea, saSumVal, si, imConv, imRef, rPixX, rPixY,
                  ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                  bgOrder, nCompKer, filter_x, filter_y, fillVal, mRData)


def fit_kernel_mlx(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
                    nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
                    kerSigReject, statSig, mRData, ngauss, deg_fixe,
                    hwKernel, fwKernel, filter_x, filter_y, fillVal, logger=None):
    from pyhotpants.float32.alard import build_matrix_numpy as _bm, build_scprod_numpy as _bs

    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2

    saMat = sa['mat']; saVectors = sa['vectors']; saSscnt = sa['sscnt']
    saNss = sa['nss']; saXss = sa['xss']; saYss = sa['yss']
    saScprod = sa['scprod']; saKrefArea = sa['krefArea']; saSumVal = sa['sum_val']
    saChi2 = sa['chi2']; nKSStamps = sa['nKSStamps']

    wxy = np.zeros((nS, ncomp2), dtype=np.float32)
    matrix, wxy = build_matrix_mlx(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                    nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, nKSStamps)
    kernelSol = build_scprod_mlx(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                  nS, imRef, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps)

    mat_size = ncomp1 * ncomp2 + nbg_vec + 1
    kernelSol[:-1] = np.linalg.solve(matrix[-2:, -2:], kernelSol[-2:])

    iter_count = 0
    max_iter = 5
    while iter_count < max_iter:
        check, meansigSubstamps, scatterSubstamps, nskippedSubstamps, refill_indices, sscnt_update, chi2_update = \
            check_again_mlx(saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea,
                            kernelSol, imNoise, nS, figMerit, kerSigReject, statSig,
                            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
                            nCompKer, kerOrder, bgOrder)

        for idx in refill_indices:
            fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss,
                             saKrefArea, saSumVal, idx, imConv, imRef, rPixX, rPixY,
                             ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, filter_x, filter_y, fillVal, mRData)

        saSscnt[:nS] = sscnt_update
        saChi2[:nS] = chi2_update

        if check == 0:
            break

        matrix, wxy = build_matrix_mlx(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                        nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, nKSStamps)
        kernelSol = build_scprod_mlx(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                      nS, imRef, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps)
        kernelSol[:-1] = np.linalg.solve(matrix[-2:, -2:], kernelSol[-2:])
        iter_count += 1

    return {'kernelSol': kernelSol, 'meansigSubstamps': meansigSubstamps,
            'scatterSubstamps': scatterSubstamps, 'NskippedSubstamps': nskippedSubstamps,
            'stamps': sa}

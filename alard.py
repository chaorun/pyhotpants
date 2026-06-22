import numpy as np
import numba

from .functions import MAXVAL, sigma_clip_numpy, get_stamp_stats3_numpy


def buildAllKernels(kernelSol, kernelVec2d, nCompKer, kerOrder, fwKernel, hwKernel, kcStep, rPixX, rPixY, xSize, ySize):
    fwSq = fwKernel * fwKernel
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

@numba.jit(nopython=True, parallel=True)
def spatial_convolve_jit_kernel(
    image, variance, cMask,
    cRdata, vData, mRData,
    kernelSol,
    xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
    kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
    kernel_vec_2d,
    allKernels=None,
    nstepsX_in=None):

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

def kernel_vector_pca_numpy(n, fwKernel, PCA, kernel_vec):
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

def kernel_vector_numpy(n, deg_x, deg_y, ig, usePCA, fwKernel, hwKernel,
                        sigma_gauss, filter_x, filter_y, kernel_vec, PCA):
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

def get_kernel_vec_numpy(ngauss, deg_fixe, usePCA, fwKernel, hwKernel,
                         sigma_gauss, filter_x, filter_y, PCA):
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

@numba.jit(nopython=True)
def build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
                     wxy, matrix,
                      nS, kerOrder, rPixX, rPixY,
                      ncomp, ncomp1, ncomp2, nbg_vec, pixStamp):
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

@numba.jit(nopython=True)
def build_scprod_jit(all_vectors, all_scprod, valid_mask, all_x, all_y,
                     wxy, image_flat, kernelSol,
                     nS, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec):
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

def build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, nKSStamps=None):
    
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
                     wxy, matrix, nS, kerOrder, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec,
                     pixStamp)

    for i in range(mat_size):
        for j in range(i + 1):
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]

    return matrix, wxy

def build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss, nS, image, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=None):
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
                     nS, fwKSStamp, hwKSStamp, rPixX,
                     ncomp, ncomp1, ncomp2, nbg_vec)

    return kernelSol

@numba.jit(nopython=True, parallel=True)
def fill_stamp_numba_kernel_local(
    image, imRef, filterX, filterY,
    xi, yi, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
    rPixX, rPixY, nCompKer, bgOrder,
    nvec, renFlags_arr, fillVal, mRData1d,
    out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val,
    n_stamps):

    LOCAL_FLAG_INPUT_ISBAD = 0x80
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel
    nbg = ((bgOrder + 1) * (bgOrder + 2)) // 2

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

def fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, si, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                     filter_x, filter_y, fillVal, mRData):
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

def fill_stamp_numba(saXss, saYss, saSscnt, saNss, si_list, imConv, imRef, rPixX, rPixY, ngauss, deg_fixe,
                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer,
                     filter_x, filter_y, fillVal, mRData):
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

@numba.jit(nopython=True)
def get_stamp_sig_batch_jit(
    sa_vectors, sa_krefArea, sa_sscnt, sa_nss, sa_xss, sa_yss,
    kernelSol, imNoise, mRData1d,
    fwKSStamp, hwKSStamp, rPixX, rPixY,
    nCompKer, kerOrder, bgOrder, nS,
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

@numba.jit(nopython=True)
def get_stamp_sig_jit(vectors, kernelSol, imNoise, mRData1d, im,
                       fwKSStamp, hwKSStamp, rPixX, rPixY,
                       nCompKer, kerOrder, bgOrder, xi, yi,
                       temp):
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

def spatial_convolve_fast_numpy(image, variance, xSize, ySize, kernelSol, cMask, kcStep,
                                hwKernel, fwKernel,
                                convolveVariance, kerFracMask,
                                rPixX, rPixY, nCompKer, kerOrder, kernel_vec):

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

def check_stamps_numpy(saScprod, saMat, saNorm, saDiff, saSscnt, saNss, saXss, saYss,
                       saVectors, saKrefArea, nS, imRef, imNoise, nCompKer, kerOrder, bgOrder,
                       forceConvolve, figMerit,
                       kerSigReject, statSig, fwKSStamp, hwKSStamp,
                       rPixX, rPixY, fwKernel, kernel_vec, mRData,
                       nKSStamps=None):

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

        matrix, _ = build_matrix_numpy(testMat, testVectors, testSscnt, testNss, testXss, testYss,
                                        ntestStamps, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)

        testKerSol = build_scprod_numpy(testScprod, testVectors, testSscnt, testNss, testXss, testYss,
                                        ntestStamps, imRef, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)

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

            figMerit_is_v = 1 if figMerit[0:1] == "v" else 0
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

def check_again_numpy(saSscnt, saNss, saChi2, saXss, saYss, saVectors, saKrefArea, kernelSol,
                      imNoise,
                      nS, figMerit, kerSigReject, statSig,
                      fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
                      nCompKer, kerOrder, bgOrder):

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
                                                    0x0, 0xffff, 5, mRData_2d, statSig)
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

def fit_kernel_numpy(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
                     nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
                      kerSigReject, statSig, mRData, ngauss, deg_fixe,
                      hwKernel, fwKernel, filter_x, filter_y, fillVal,
                      logger=None):
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
    matrix, wxy = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                    nS, nCompKer, kerOrder, bgOrder,
                                    fwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
    kernelSol = build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                   nS, imRef, nCompKer, kerOrder, bgOrder,
                                   fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)
    logger.debug("  fitKernel: build_matrix0+scprod0 done")
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

        matrix, wxy = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                        nS, nCompKer, kerOrder, bgOrder,
                                        fwKSStamp, rPixX, rPixY, nKSStamps=nKSStamps)
        
        kernelSol = build_scprod_numpy(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                                       nS, imRef, nCompKer, kerOrder, bgOrder,
                                       fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=nKSStamps)
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


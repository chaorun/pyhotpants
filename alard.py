import numpy as np
import numba
import math
import logging
logger = logging.getLogger('hotpants')

from .functions import (ZEROVAL, MAXVAL,
    FLAG_BAD_PIXVAL, FLAG_SAT_PIXEL, FLAG_NEG_PIXEL, FLAG_SPREAD,
    FLAG_MASKED, FLAG_BORDER, FLAG_SUBREGION, FLAG_INVALID,
    BUILD_STAMP_FLAT_CACHE, LAST_SIGMA_CLIP, LAST_N, LAST_MEDIAN,
    StampsArray, Ran1,
    sigma_clip_numpy, get_noise_stats3_numpy,
    insert_subregion_flt_numpy, insert_subregion_int_numpy,
    cut_stamp_numpy, get_stamp_stats3_numpy, bin_quartile_numpy,
    get_stamp_stats3_fast_numpy, cut_sstamp_numpy,
    check_psf_center_numba, check_psf_center_numpy, get_psf_centers_numpy,
    quick_sort_impl, quick_sort_recurse,
    psfCentersJit, buildStampsNumba,
    lubksb_numpy, ludcmp_numpy,
    make_noise_image4_numpy, build_stamps_flatten_helper,
)


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
    rPixX2 = np.float64(np.float32(0.5 * rPixX))
    rPixY2 = np.float64(np.float32(0.5 * rPixY))

    for istamp in range(nS):
        if valid_mask[istamp] == 0:
            continue

        xstamp = all_x[istamp]
        ystamp = all_y[istamp]
        fx = np.float64(np.float32(np.float32(xstamp) - np.float32(rPixX2)) / np.float32(rPixX2))
        fy = np.float64(np.float32(np.float32(ystamp) - np.float32(rPixY2)) / np.float32(rPixY2))

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

def build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, verbose, wxy, nC=None, nKSStamps=None):
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
    valid_mask = (saSscnt < saNss).astype(np.int32)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        for i in range(nS):
            for j in range(ncomp2):
                wxy[i, j] = 0.0
        matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        return matrix
    safe_sscnt = np.clip(saSscnt, 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[np.arange(nS), safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_mask, saYss[np.arange(nS), safe_sscnt], 0).astype(np.int64)

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

    wxy.fill(0.0)
    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)

    build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y,
                     wxy, matrix,
                     nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY,
                     ncomp, ncomp1, ncomp2, nbg_vec, mat_size, pixStamp)

    for i in range(mat_size):
        for j in range(i + 1):
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]

    return matrix

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
    valid_mask = (saSscnt < saNss).astype(np.int32)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return np.zeros(ncomp + nbg_vec + 2, dtype=np.float64)
    safe_sscnt = np.clip(saSscnt, 0, nKSStamps - 1)
    all_x = np.where(valid_mask, saXss[np.arange(nS), safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_mask, saYss[np.arange(nS), safe_sscnt], 0).astype(np.int64)

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

@numba.jit(nopython=True)
def fill_stamp_numba_kernel(
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
    rPixX2 = np.float64(np.float32(0.5 * rPixX))
    rPixY2 = np.float64(np.float32(0.5 * rPixY))
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
    return fill_stamp_numba(saVectors, saMat, saScprod, saKrefArea, saSumVal, saXss, saYss, saSscnt, saNss, saX0, saY0, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
                           hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
                           usePCA, filter_x, filter_y, PCA, fillVal, mRData)

def fill_stamp_numba(saVectors, saMat, saScprod, saKrefArea, saSumVal, saXss, saYss, saSscnt, saNss, saX0, saY0, si, imConv, imRef, rPixX, rPixY, verbose, ngauss, deg_fixe,
                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder,
                     usePCA, filter_x, filter_y, PCA, fillVal, mRData):
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
    out_mat = np.zeros((nC, nC), dtype=np.float64)
    out_scprod = np.zeros(nC, dtype=np.float64)
    out_sum_val = np.zeros(1, dtype=np.float64)

    fill_stamp_numba_kernel(
        img_flat, imRef_flat, fx, fy,
        xi, yi, fwKSStamp, hwKSStamp, fwKernel, hwKernel,
        rPixX, rPixY, nCompKer, kerOrder, bgOrder,
        nvec, rflags, fillVal, mRData1d, verbose,
        out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val)

    # for n in range(nvec + nbg):
    #     sa.vectors[si, n, :fwSqStamp] = out_vectors[n]
    saVectors[si, :nvec + nbg, :fwSqStamp] = out_vectors
    # sa.krefArea[si, :] = out_krefArea
    saKrefArea[si, :] = out_krefArea
    # sa.mat[si, :nC, :nC] = out_mat
    saMat[si, :nC, :nC] = out_mat
    # sa.scprod[si, :nC] = out_scprod
    saScprod[si, :nC] = out_scprod
    # sa.sum_val[si] = out_sum_val[0]
    saSumVal[si] = out_sum_val[0]

    return 0

@numba.jit(nopython=True)
def get_stamp_sig_batch_jit(
    sa_vectors, sa_krefArea, sa_sscnt, sa_nss, sa_xss, sa_yss,
    kernelSol, imNoise, mRData1d,
    fwKSStamp, hwKSStamp, rPixX, rPixY,
    nCompKer, kerOrder, bgOrder, nS,
    figMerit_is_v, statSig,
    out_sig1, out_sig2, out_sig3):

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
                csModel[i] += coeff * sa_vectors[si, i1, i]

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

    return (sig1, sig2, sig3)

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

@numba.jit(nopython=True, parallel=True)
def spatial_convolve_jit_kernel(
    image, variance, cMask,
    cRdata, vData, mRData,
    kernelSol,
    xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
    kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
    kernel_vec_2d):

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

            # for ii in range(fwSq):
            #     kernel[ii] = 0.0  # already zeros from np.zeros
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

def spatial_convolve_fast_numpy(image, variance, xSize, ySize, kernelSol, cMask, kcStep,
                                hwKernel, fwKernel, kernel, kernel_coeffs,
                                convolveVariance, kerFracMask,
                                rPixX, rPixY, nCompKer, kerOrder, kernel_vec):
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

    spatial_convolve_jit_kernel(
        image1d, var1d, cMask1d,
        cRdata64, vData64, mRData64,
        kernelSol.astype(np.float64),
        xSize, ySize, nCompKer, kerOrder, fwKernel, hwKernel,
        kcStep, rPixX, rPixY, kerFracMask, dovar, convolveVariance,
        kernel_vec_2d)
    # t2 = time.time(); logger.debug("  sc_fast: jit_kernel %.3fs", t2 - t1)

    if dovar:
        vData = vData64.astype(np.float32)

    cRdata_out = cRdata64.astype(np.float32)
    mRData_out = mRData64
    # t3 = time.time(); logger.debug("  sc_fast: post %.3fs total %.3fs", t3 - t2, t3 - t0)
    return vData, cRdata_out, mRData_out

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
        matrix = build_matrix_numpy(testMat, testVectors, testSscnt, testNss, testXss, testYss,
                                    ntestStamps, nCompKer, kerOrder, bgOrder,
                                    fwKSStamp, rPixX, rPixY, verbose, wxy, nC=nC, nKSStamps=nKSStamps)

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

    # 批量计算所有 stamps 的 sig (figMerit="v" 模式)
    batch_sig1 = None
    if figMerit[0:1] == "v":
        batch_sig1 = np.zeros(nS, dtype=np.float64)
        batch_sig2 = np.zeros(nS, dtype=np.float64)
        batch_sig3 = np.zeros(nS, dtype=np.float64)
        get_stamp_sig_batch_jit(
            np.asarray(saVectors, dtype=np.float64), np.asarray(saKrefArea, dtype=np.float64),
            saSscnt, saNss, saXss, saYss,
            np.asarray(kernelSol, dtype=np.float64),
            np.asarray(imNoise, dtype=np.float64),
            np.asarray(mRData, dtype=np.int32).ravel(),
            fwKSStamp, hwKSStamp, rPixX, rPixY,
            nCompKer, kerOrder, bgOrder, nS,
            1, statSig,
            batch_sig1, batch_sig2, batch_sig3)

    for istamp in range(nS):
        if saSscnt[istamp] < saNss[istamp]:
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
                sscnt = saSscnt[istamp]
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
                saSscnt[istamp] += 1
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

                saChi2[istamp] = sig
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
        if saSscnt[istamp] < saNss[istamp]:
            # if (stamps[istamp]['chi2'] - mean) > kerSigReject * stdev:
            if (saChi2[istamp] - mean) > kerSigReject * stdev:
                # stamps[istamp]['sscnt'] += 1
                saSscnt[istamp] += 1
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

    return {
        'check': check,
        'meansigSubstamps': meansigSubstamps,
        'scatterSubstamps': scatterSubstamps,
        'nskippedSubstamps': nskippedSubstamps,
        'refill_indices': refill_indices,
    }

def fit_kernel_numpy(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
                     verbose, nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
                     kerSigReject, statSig, mRData, ngauss, deg_fixe,
                     hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal):
    # def fit_kernel_numpy(stamps_dicts, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
    #                      verbose, nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
    #                      kerSigReject, statSig, mRData, ngauss, deg_fixe,
    #                      hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    wxy = np.zeros((nS, ncomp2), dtype=np.float64)

    # 从 sa 提取扁平数组（用于后续扁平化的函数调用）
    saMat      = sa.mat
    saVectors  = sa.vectors
    saSscnt    = sa.sscnt
    saNss      = sa.nss
    saXss      = sa.xss
    saYss      = sa.yss
    saScprod   = sa.scprod
    saKrefArea = sa.krefArea
    saSumVal   = sa.sum_val
    saX0       = sa.x0
    saY0       = sa.y0
    saChi2     = sa.chi2
    nC         = sa.nC
    nKSStamps  = sa.nKSStamps

    def do_fill(indices):
        for idx in indices:
            fill_stamp_numpy(saVectors, saMat, saScprod, saXss, saYss, saSscnt, saNss, saKrefArea, saSumVal, saX0, saY0, idx, imConv, imRef, rPixX, rPixY, verbose,
                             ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                             bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y,
                             PCA, fillVal, mRData)

    # import time
    iter_count = 0
    # t_start = time.time()
    logger.debug("  fitKernel: iteration %d start", iter_count)
    # matrix = build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder,
    #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
    # matrix = build_matrix_numpy(sa, nS, nCompKer, kerOrder, bgOrder,
    #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
    # tm = time.time()
    matrix = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                nS, nCompKer, kerOrder, bgOrder,
                                fwKSStamp, rPixX, rPixY, verbose, wxy, nC=nC, nKSStamps=nKSStamps)
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
    ca_result = check_again_numpy(
        saSscnt, saNss, saChi2, saXss, saYss, saVectors, saMat, saScprod, saKrefArea, saSumVal, kernelSol,
        imConv, imRef, imNoise,
        nS, verbose, figMerit, kerSigReject, statSig,
        fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
        nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
        hwKernel, fwKernel, usePCA, filter_x, filter_y,
        PCA, fillVal)
    check = ca_result['check']
    meansigSubstamps = ca_result['meansigSubstamps']
    scatterSubstamps = ca_result['scatterSubstamps']
    nskippedSubstamps = ca_result['nskippedSubstamps']
    do_fill(ca_result['refill_indices'])
    logger.debug("  fitKernel: check_again done, check=%s", check)
    # tm_ca = time.time(); logger.debug("  fitKernel: check_again %.3fs", tm_ca - tm_slv)

    while check:
        iter_count += 1
        logger.debug("  fitKernel: iteration %d start", iter_count)
        wxy = np.zeros((nS, ncomp2), dtype=np.float64)

        # matrix = build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        # matrix = build_matrix_numpy(sa, nS, nCompKer, kerOrder, bgOrder,
        #                             fwKSStamp, rPixX, rPixY, verbose, wxy)
        # tm = time.time()
        matrix = build_matrix_numpy(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                                    nS, nCompKer, kerOrder, bgOrder,
                                    fwKSStamp, rPixX, rPixY, verbose, wxy, nC=nC, nKSStamps=nKSStamps)
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
        ca_result = check_again_numpy(
            saSscnt, saNss, saChi2, saXss, saYss, saVectors, saMat, saScprod, saKrefArea, saSumVal, kernelSol,
            imConv, imRef, imNoise,
            nS, verbose, figMerit, kerSigReject, statSig,
            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData,
            nCompKer, kerOrder, bgOrder, ngauss, deg_fixe,
            hwKernel, fwKernel, usePCA, filter_x, filter_y,
            PCA, fillVal)
        check = ca_result['check']
        meansigSubstamps = ca_result['meansigSubstamps']
        scatterSubstamps = ca_result['scatterSubstamps']
        nskippedSubstamps = ca_result['nskippedSubstamps']
        do_fill(ca_result['refill_indices'])
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


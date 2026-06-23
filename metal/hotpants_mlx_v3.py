import mlx.core as mx
import numpy as np

def build_matrix_mlx_v3(saMat, saVectors, saSscnt, saNss, saXss, saYss,
                         nS, nCompKer, kerOrder, bgOrder,
                         fwKSStamp, rPixX, rPixY, nKSStamps=None):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp
    mat_size = ncomp + nbg_vec + 1
    nC_valid = nCompKer + 1
    nvec_total = nCompKer + nbg_vec

    valid = saSscnt[:nS] < saNss[:nS]
    valid_np = np.asarray(valid)
    n_valid = int(valid_np.sum())

    if nKSStamps is None:
        nKSStamps_val = saXss.shape[1]
    else:
        nKSStamps_val = nKSStamps

    wxy = np.zeros((nS, ncomp2), dtype=np.float32)
    matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float32)

    if n_valid == 0:
        return matrix, wxy

    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps_val - 1)
    rg = np.arange(nS)
    all_x = np.where(valid_np, saXss[rg, safe_sscnt], 0).astype(np.float32)
    all_y = np.where(valid_np, saYss[rg, safe_sscnt], 0).astype(np.float32)

    rPixX2 = np.float32(0.5 * rPixX)
    rPixY2 = np.float32(0.5 * rPixY)

    for s in range(nS):
        if not valid_np[s]: continue
        fx = (all_x[s] - rPixX2) / rPixX2
        fy = (all_y[s] - rPixY2) / rPixY2
        kk = 0; a1 = 1.0
        for ideg1 in range(kerOrder + 1):
            a2 = 1.0
            for ideg2 in range(kerOrder - ideg1 + 1):
                wxy[s, kk] = a1 * a2; kk += 1; a2 *= fy
            a1 *= fx

    all_mat = saMat[:nS, :nC_valid, :nC_valid].copy()
    all_vec = saVectors[:nS, :nvec_total, :].copy()
    valid_idx = np.where(valid_np)[0]
    wxy_v = wxy[valid_idx]
    all_mat_v = all_mat[valid_idx]
    all_vec_v = all_vec[valid_idx]

    # === MLX batch: core blocks ===
    wxy_mx = mx.array(wxy_v)                    # (nV, ncomp2)
    wwy_mx = mx.einsum('si,sj->sij', wxy_mx, wxy_mx)  # (nV, ncomp2, ncomp2)
    am_mx = mx.array(all_mat_v)                 # (nV, nCv, nCv)

    # Accumulate all blocks lazily in MLX, then eval once
    block_results = []
    for i1 in range(ncomp1):
        r0 = i1 * ncomp2 + 2
        for j1 in range(0, i1 + 1):
            c0 = j1 * ncomp2 + 2
            coeff = am_mx[:, i1 + 2, j1 + 2]
            weighted = (wwy_mx * coeff[:, None, None]).sum(axis=0)
            block_results.append(weighted)

    mx.eval(*block_results)

    # Write back
    idx = 0
    for i1 in range(ncomp1):
        r0 = i1 * ncomp2 + 2
        for j1 in range(0, i1 + 1):
            c0 = j1 * ncomp2 + 2
            matrix[r0:r0+ncomp2, c0:c0+ncomp2] += np.array(block_results[idx])
            idx += 1

    # col 1
    matrix[1, 1] += all_mat_v[:, 1, 1].sum()
    for i1 in range(ncomp1):
        offset_r = i1 * ncomp2 + 2
        coeff_c1 = all_mat_v[:, i1 + 2, 1]
        matrix[offset_r:offset_r+ncomp2, 1] += (wxy_v * coeff_c1[:, None]).sum(axis=0)

    # BG terms: all in MLX, matrix as MLX tensor
    mat_mx = mx.array(matrix)  # convert once
    wxy_mx = mx.array(wxy_v)
    av_mx_full = mx.array(all_vec_v)  # (nV, nvt, pixStamp)

    for ibg in range(nbg_vec):
        ii = ncomp + ibg + 1
        ivecbg = ncomp1 + ibg + 1
        vec_bg = av_mx_full[:, ivecbg, :]  # (nV, pixStamp)
        vec_k = av_mx_full[:, 1:ncomp1+1, :]  # (nV, ncomp1, pixStamp)
        # p0_i1[s, i1-1] = dot(vec_bg[s], vec_k[s, i1-1])
        p0_i1 = (vec_bg[:, None, :] * vec_k).sum(axis=2)  # (nV, ncomp1)
        p0_0 = (av_mx_full[:, 0, :] * vec_bg).sum(axis=1)  # (nV,)

        # row contributions: (ii+1, (i1-1)*ncomp2+2 .. i1*ncomp2+2) += sum_s p0_i1[s,i1-1] * wxy[s]
        for i1 in range(1, ncomp1 + 1):
            col_start = (i1 - 1) * ncomp2 + 2
            weighted = (p0_i1[:, i1 - 1, None] * wxy_mx).sum(axis=0)  # (ncomp2,)
            mat_mx[ii + 1, col_start:col_start+ncomp2] = mat_mx[ii + 1, col_start:col_start+ncomp2] + weighted

        # row (ii+1, 1) += sum_s p0_0[s]
        mat_mx[ii + 1, 1] = mat_mx[ii + 1, 1] + p0_0.sum()

        # BG diagonals: (ii+1, ncomp+jbg+2) += dot(vec_bg, vec_bg_jbg) for each jbg
        for jbg in range(ibg + 1):
            vec_bg_jbg = av_mx_full[:, ncomp1 + jbg + 1, :]  # (nV, pixStamp)
            q = (vec_bg * vec_bg_jbg).sum(axis=1)  # (nV,)
            mat_mx[ii + 1, ncomp + jbg + 2] = mat_mx[ii + 1, ncomp + jbg + 2] + q.sum()

    mx.eval(mat_mx)
    matrix = np.array(mat_mx)

    # Symmetry
    for i in range(mat_size):
        for j in range(i + 1):
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]
    matrix[0, 0] = 1.0

    return matrix, wxy


def build_scprod_mlx_v3(saScprod, saVectors, saSscnt, saNss, saXss, saYss,
                         nS, image, nCompKer, kerOrder, bgOrder,
                         fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps=None):
    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    ncomp = ncomp1 * ncomp2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    pixStamp = fwKSStamp * fwKSStamp

    valid = saSscnt[:nS] < saNss[:nS]
    valid_np = np.asarray(valid)
    n_valid = int(valid_np.sum())

    if nKSStamps is None:
        nKSStamps_val = saXss.shape[1]
    else:
        nKSStamps_val = nKSStamps

    if n_valid == 0:
        return np.zeros(ncomp + nbg_vec + 2, dtype=np.float32)

    safe_sscnt = np.clip(saSscnt[:nS], 0, nKSStamps_val - 1)
    rg = np.arange(nS)
    all_x = np.where(valid_np, saXss[rg, safe_sscnt], 0).astype(np.int64)
    all_y = np.where(valid_np, saYss[rg, safe_sscnt], 0).astype(np.int64)

    nvec_total = nCompKer + nbg_vec
    valid_idx = np.where(valid_np)[0]
    wxy_v = wxy[valid_idx]
    scp_v = saScprod[valid_idx, :nCompKer + 1]
    all_vec_v = saVectors[valid_idx, :nvec_total, :]
    image_arr = np.asarray(image, dtype=np.float32)

    kernelSol = np.zeros(ncomp + nbg_vec + 2, dtype=np.float32)

    # scprod batch
    kernelSol[1] = scp_v[:, 1].sum()
    for i1 in range(1, ncomp1 + 1):
        p0_vec = scp_v[:, i1 + 1]
        for i2 in range(ncomp2):
            ii = (i1 - 1) * ncomp2 + i2 + 1
            kernelSol[ii + 1] = np.sum(p0_vec * wxy_v[:, i2])

    # BG terms: batch ALL stamps' image at once
    stamp_offsets = np.array([
        (xc + hwKSStamp) + fwKSStamp * (yc + hwKSStamp)
        for yc in range(-hwKSStamp, hwKSStamp + 1)
        for xc in range(-hwKSStamp, hwKSStamp + 1)
    ], dtype=np.int32)

    all_idxs_list = []
    for s_idx in range(n_valid):
        xi = all_x[valid_idx[s_idx]]; yi = all_y[valid_idx[s_idx]]
        all_idxs_list.append((xi - hwKSStamp + stamp_offsets % fwKSStamp) + rPixX * (yi - hwKSStamp + stamp_offsets // fwKSStamp))
    all_idxs_np = np.array(all_idxs_list, dtype=np.int32)

    img_mx = mx.array(image_arr)
    all_patches = img_mx[mx.array(all_idxs_np.ravel())].reshape(n_valid, pixStamp)
    mx.eval(all_patches)
    all_patches_np = np.array(all_patches)

    for ibg in range(nbg_vec):
        vec_bg = all_vec_v[:, ncomp1 + ibg + 1, :]
        q_all = np.sum(vec_bg * all_patches_np, axis=1)
        kernelSol[ncomp + ibg + 2] = q_all.sum()

    return kernelSol


def fit_kernel_mlx_v3(sa, imRef, imConv, imNoise, nCompKer, kerOrder, bgOrder,
                       nS, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit,
                       kerSigReject, statSig, mRData, ngauss, deg_fixe,
                       hwKernel, fwKernel, filter_x, filter_y, fillVal, logger=None):
    from pyhotpants.float32.alard import fill_stamp_numpy as _orig_fill
    from .hotpants_mlx import check_again_mlx

    ncomp1 = nCompKer - 1
    ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) // 2
    mat_size = ncomp1 * ncomp2 + nbg_vec + 1

    saMat_np = sa['mat']; saVectors_np = sa['vectors']; saSscnt_np = sa['sscnt']
    saNss_np = sa['nss']; saXss_np = sa['xss']; saYss_np = sa['yss']
    saScprod_np = sa['scprod']; nKSStamps = sa['nKSStamps']

    matrix, wxy = build_matrix_mlx_v3(saMat_np, saVectors_np, saSscnt_np, saNss_np, saXss_np, saYss_np,
                                       nS, nCompKer, kerOrder, bgOrder,
                                       fwKSStamp, rPixX, rPixY, nKSStamps)
    kernelSol = build_scprod_mlx_v3(saScprod_np, saVectors_np, saSscnt_np, saNss_np, saXss_np, saYss_np,
                                     nS, imRef, nCompKer, kerOrder, bgOrder,
                                     fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps)
    kernelSol[:mat_size+1] = np.linalg.solve(matrix[:mat_size+1, :mat_size+1], kernelSol[:mat_size+1])

    iter_count = 0
    while iter_count < 50:
        check, meansig, scatter, nskip, refill, sscnt_up, chi2_up = check_again_mlx(
            saSscnt_np, saNss_np, sa['chi2'], saXss_np, saYss_np, saVectors_np, sa['krefArea'],
            kernelSol, imNoise, nS, figMerit, kerSigReject, statSig,
            fwKSStamp, hwKSStamp, rPixX, rPixY, mRData, nCompKer, kerOrder, bgOrder)

        for idx in refill:
            _orig_fill(saVectors_np, saMat_np, saScprod_np, saXss_np, saYss_np, saSscnt_np, saNss_np,
                       sa['krefArea'], sa['sum_val'], idx, imConv, imRef, rPixX, rPixY,
                       ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                       bgOrder, nCompKer, filter_x, filter_y, fillVal, mRData)

        saSscnt_np[:nS] = sscnt_up
        sa['chi2'][:nS] = chi2_up

        if check == 0:
            break

        matrix, wxy = build_matrix_mlx_v3(saMat_np, saVectors_np, saSscnt_np, saNss_np, saXss_np, saYss_np,
                                           nS, nCompKer, kerOrder, bgOrder,
                                           fwKSStamp, rPixX, rPixY, nKSStamps)
        kernelSol = build_scprod_mlx_v3(saScprod_np, saVectors_np, saSscnt_np, saNss_np, saXss_np, saYss_np,
                                         nS, imRef, nCompKer, kerOrder, bgOrder,
                                         fwKSStamp, hwKSStamp, rPixX, wxy, nKSStamps)
        kernelSol[:mat_size+1] = np.linalg.solve(matrix[:mat_size+1, :mat_size+1], kernelSol[:mat_size+1])
        iter_count += 1

    return {'kernelSol': kernelSol, 'meansigSubstamps': meansig,
            'scatterSubstamps': scatter, 'NskippedSubstamps': nskip, 'stamps': sa}

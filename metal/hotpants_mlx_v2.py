import mlx.core as mx
import numpy as np

def _build_kernels(filterX, filterY, fwKernel, nvec):
    fwSqK = fwKernel * fwKernel
    kernels = np.zeros((nvec, fwSqK), dtype=np.float32)
    for n in range(nvec):
        for jc in range(fwKernel):
            for ic in range(fwKernel):
                fy_idx = fwKernel - 1 - jc
                fx_idx = fwKernel - 1 - ic
                kernels[n, jc * fwKernel + ic] = filterY[n * fwKernel + fy_idx] * filterX[n * fwKernel + fx_idx]
    return kernels

def fill_stamp_kernel_mlx_v2(image, imRef, filterX, filterY, xi, yi,
                              fwKSStamp, hwKSStamp, fwKernel, hwKernel,
                              rPixX, rPixY, nCompKer, bgOrder, nvec,
                              renFlags_arr, fillVal, mRData1d,
                              out_vectors, out_krefArea, out_mat, out_scprod, out_sum_val,
                              n_stamps):
    fwSqStamp = fwKSStamp * fwKSStamp
    fwSqKernel = fwKernel * fwKernel
    FLAG_INPUT_ISBAD = 0x80
    ncomp1_jit = nCompKer

    kernels_np = _build_kernels(filterX, filterY, fwKernel, nvec)
    kernels_mx = mx.array(kernels_np)

    stamp_ix = np.arange(fwSqStamp, dtype=np.int32) % fwKSStamp - hwKSStamp
    stamp_iy = np.arange(fwSqStamp, dtype=np.int32) // fwKSStamp - hwKSStamp
    kernel_ix = np.arange(fwSqKernel, dtype=np.int32) % fwKernel - hwKernel
    kernel_iy = np.arange(fwSqKernel, dtype=np.int32) // fwKernel - hwKernel

    xi3 = xi.reshape(n_stamps, 1, 1).astype(np.int32)
    yi3 = yi.reshape(n_stamps, 1, 1).astype(np.int32)
    six3 = stamp_ix.reshape(1, fwSqStamp, 1)
    siy3 = stamp_iy.reshape(1, fwSqStamp, 1)
    kix3 = kernel_ix.reshape(1, 1, fwSqKernel)
    kiy3 = kernel_iy.reshape(1, 1, fwSqKernel)

    x_all = np.clip(xi3 + six3 + kix3, 0, rPixX - 1)
    y_all = np.clip(yi3 + siy3 + kiy3, 0, rPixY - 1)
    idx_all = (x_all + rPixX * y_all).ravel()
    del x_all, y_all, six3, siy3, kix3, kiy3

    img_mx = mx.array(image)
    pixel_vals = img_mx[mx.array(idx_all)].reshape(n_stamps, fwSqStamp, fwSqKernel)
    del idx_all

    out_vec_mx = mx.einsum('spk,vk->svp', pixel_vals, kernels_mx)
    mx.eval(out_vec_mx)
    out_vec_np = np.array(out_vec_mx)
    for s in range(n_stamps):
        for n in range(nvec):
            out_vectors[s, n, :] = out_vec_np[s, n, :]

    for si in range(n_stamps):
        for n in range(nvec):
            if renFlags_arr[n]:
                out_vectors[si, n, :] -= out_vectors[si, 0, :]

    stamp_ix2 = np.arange(fwSqStamp, dtype=np.int32) % fwKSStamp
    stamp_iy2 = np.arange(fwSqStamp, dtype=np.int32) // fwKSStamp
    kref_x = xi.reshape(n_stamps, 1).astype(np.int32) - hwKSStamp + stamp_ix2.reshape(1, fwSqStamp)
    kref_y = yi.reshape(n_stamps, 1).astype(np.int32) - hwKSStamp + stamp_iy2.reshape(1, fwSqStamp)
    kref_idx = (kref_x + rPixX * kref_y).ravel()
    del kref_x, kref_y

    imRef_mx = mx.array(imRef)
    kref_vals = imRef_mx[mx.array(kref_idx)].reshape(n_stamps, fwSqStamp)
    mRData_mx = mx.array(mRData1d.astype(np.int32))
    mref_vals = mRData_mx[mx.array(kref_idx)].reshape(n_stamps, fwSqStamp)
    valid_kref = (mref_vals & FLAG_INPUT_ISBAD) == 0
    sum_vals = mx.where(valid_kref, mx.abs(kref_vals), mx.zeros_like(kref_vals)).sum(axis=1)
    mx.eval(kref_vals, sum_vals)
    del mref_vals, valid_kref

    ok_np = np.array(kref_vals)
    osv_np = np.array(sum_vals)
    for si in range(n_stamps):
        out_krefArea[si, :] = ok_np[si]
        out_sum_val[si, 0] = osv_np[si]

    bg_ix = np.arange(fwKSStamp, dtype=np.int32)
    bg_iy = np.arange(fwKSStamp, dtype=np.int32)
    bg_x = xi.reshape(n_stamps, 1, 1) - hwKSStamp + bg_ix.reshape(1, fwKSStamp, 1)
    bg_y = yi.reshape(n_stamps, 1, 1) - hwKSStamp + bg_iy.reshape(1, 1, fwKSStamp)
    bg_x = np.broadcast_to(bg_x.astype(np.float32), (n_stamps, fwKSStamp, fwKSStamp)).copy()
    bg_y = np.broadcast_to(bg_y.astype(np.float32), (n_stamps, fwKSStamp, fwKSStamp)).copy()
    rPixX2 = np.float32(0.5 * rPixX)
    rPixY2 = np.float32(0.5 * rPixY)
    xf_mx = mx.array((bg_x - rPixX2) / rPixX2)
    yf_mx = mx.array((bg_y - rPixY2) / rPixY2)
    del bg_x, bg_y
    nv = nvec
    ax = mx.ones_like(xf_mx)
    for idegx in range(bgOrder + 1):
        ay = mx.ones_like(xf_mx)
        for idegy in range(bgOrder - idegx + 1):
            bg_vals = (ax * ay).reshape(n_stamps, fwSqStamp)
            bg_np = np.array(bg_vals)
            for si in range(n_stamps):
                out_vectors[si, nv, :] = bg_np[si]
            ay = ay * yf_mx
            nv += 1
        ax = ax * xf_mx

    nc = ncomp1_jit
    full_vecs_mx = mx.array(out_vectors)
    kref_mx = mx.array(out_krefArea)
    vec_core = full_vecs_mx[:, :nc, :]
    mat_mx = vec_core @ vec_core.transpose((0, 2, 1))
    ivecbg_idx = ncomp1_jit
    cross_term = (vec_core * full_vecs_mx[:, ivecbg_idx:ivecbg_idx+1, :]).sum(axis=2)
    bg_dot = (full_vecs_mx[:, ivecbg_idx, :] * full_vecs_mx[:, ivecbg_idx, :]).sum(axis=1)
    sc_mx = (vec_core * kref_mx[:, None, :]).sum(axis=2)
    sc_last = (full_vecs_mx[:, ivecbg_idx, :] * kref_mx).sum(axis=1)
    mx.eval(mat_mx, cross_term, bg_dot, sc_mx, sc_last)
    mat_np = np.array(mat_mx)
    cross_np = np.array(cross_term)
    bg_dot_np = np.array(bg_dot)
    sc_np = np.array(sc_mx)
    sc_last_np = np.array(sc_last)

    for si in range(n_stamps):
        for i in range(nc):
            for j in range(i + 1):
                out_mat[si, i + 1, j + 1] = mat_np[si, i, j]
        for i1 in range(nc):
            out_mat[si, ncomp1_jit + 1, i1 + 1] = cross_np[si, i1]
        out_mat[si, ncomp1_jit + 1, ncomp1_jit + 1] = bg_dot_np[si]
        for i1 in range(nc):
            out_scprod[si, i1 + 1] = sc_np[si, i1]
        out_scprod[si, ncomp1_jit + 1] = sc_last_np[si]


def spatial_convolve_mlx_v2(image, variance, cMask, cRdata, vData, mRData,
                             kernelSol, xSize, ySize, nCompKer, kerOrder,
                             fwKernel, hwKernel, kcStep, rPixX, rPixY,
                             kerFracMask, dovar, convolveVariance,
                             kernel_vec_2d, allKernels=None, nstepsX_in=None):
    fwSq = fwKernel * fwKernel
    FLAG_INPUT_ISBAD = 0x80
    FLAG_OK_CONV = 0x40
    FLAG_BAD_CONV = 0x10
    FLAG_OUTPUT_ISBAD_V = 0x8000

    if allKernels is None:
        from pyhotpants.float32.alard import buildAllKernels
        ak, nx, ny = buildAllKernels(kernelSol, kernel_vec_2d, nCompKer, kerOrder,
                                      fwKernel, hwKernel, kcStep, rPixX, rPixY, xSize, ySize)
    else:
        ak = allKernels

    nstepsX = nstepsX_in if nstepsX_in else int(np.ceil(xSize / kcStep))
    nstepsY = int(np.ceil(ySize / kcStep))

    padded_x = xSize + 2 * hwKernel
    padded_y = ySize + 2 * hwKernel
    img_pad = np.zeros(padded_x * padded_y, dtype=np.float32)
    var_pad = np.zeros(padded_x * padded_y, dtype=np.float32)
    msk_pad = np.zeros(padded_x * padded_y, dtype=np.int32)
    for j in range(ySize):
        s0, d0 = j * rPixX, (j + hwKernel) * padded_x + hwKernel
        img_pad[d0:d0+xSize] = image[s0:s0+xSize]
        var_pad[d0:d0+xSize] = variance[s0:s0+xSize]
        msk_pad[d0:d0+xSize] = np.asarray(cMask[s0:s0+xSize], dtype=np.int32)

    img_mx = mx.array(img_pad)
    var_mx = mx.array(var_pad)
    msk_mx = mx.array(msk_pad)
    del img_pad, var_pad, msk_pad

    neighbor_offs = np.array([dx + padded_x * dy for dy in range(-hwKernel, hwKernel+1)
                              for dx in range(-hwKernel, hwKernel+1)], dtype=np.int32)

    for j1 in range(nstepsY):
        j0 = j1 * kcStep + hwKernel
        j_end = min(j0 + kcStep, ySize - hwKernel)
        if j_end <= j0: continue
        nRows = j_end - j0

        rows_cr = []; rows_var = []; rows_msk = []; row_kernels = []; row_targets = []

        for i1 in range(nstepsX):
            i0 = i1 * kcStep + hwKernel
            i_end = min(i0 + kcStep, xSize - hwKernel)
            if i_end <= i0: continue
            nCols = i_end - i0
            nPixels = nRows * nCols

            kernel = ak[j1 * nstepsX + i1]
            kernel_flip = np.flip(kernel.reshape(fwKernel, fwKernel), axis=(0, 1)).ravel()
            row_kernels.append(mx.array(kernel_flip))

            pi0, pj0 = i0 + hwKernel, j0 + hwKernel
            centers = np.arange(pi0, pi0 + nCols, dtype=np.int32) + padded_x * np.arange(pj0, pj0 + nRows, dtype=np.int32)[:, None]
            neighbors = centers.ravel()[:, None] + neighbor_offs

            px = img_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
            rows_cr.append((px, (j0, i0, j_end, i_end, nRows, nCols)))
            row_targets.append(centers.ravel())

            if dovar:
                var_px = var_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
                rows_var.append(var_px)
            msk_px = msk_mx[mx.array(neighbors.ravel())].reshape(nPixels, fwSq)
            rows_msk.append(msk_px)

        results_cr = [px_data @ row_kernels[i] for i, (px_data, _) in enumerate(rows_cr)]
        mx.eval(*results_cr)
        for idx, (res_mx, (j0b, i0b, j_endb, i_endb, nRowsb, nColsb)) in enumerate(zip(results_cr, [t[1] for t in rows_cr])):
            res_np = np.array(res_mx).reshape(nRowsb, nColsb)
            for r in range(nRowsb):
                cRdata[(j0b+r)*rPixX+i0b:(j0b+r)*rPixX+i_endb] = res_np[r]

        if dovar and rows_var:
            var_results = [var_px @ (row_kernels[i] * row_kernels[i]) for i, var_px in enumerate(rows_var)]
            mx.eval(*var_results)
            for idx, (var_mx, (j0b, i0b, j_endb, i_endb, nRowsb, nColsb)) in enumerate(zip(var_results, [t[1] for t in rows_cr])):
                var_np = np.array(var_mx).reshape(nRowsb, nColsb)
                for r in range(nRowsb):
                    vData[(j0b+r)*rPixX+i0b:(j0b+r)*rPixX+i_endb] = var_np[r]

        if rows_msk:
            msk_results = []
            for i, msk_px in enumerate(rows_msk):
                k_mx = row_kernels[i]
                cents = row_targets[i]
                mbit = mx.max(msk_px, axis=1)
                aks = mx.abs(k_mx).sum()
                isbad = (msk_px & FLAG_INPUT_ISBAD) != 0
                uks = mx.where(isbad, mx.zeros_like(msk_px), mx.abs(k_mx)).sum(axis=1)
                bad_frac = mx.where(aks > 0, uks / aks, mx.ones_like(uks))
                conv_flag = mx.where(bad_frac < kerFracMask, FLAG_OUTPUT_ISBAD_V | FLAG_BAD_CONV, FLAG_OK_CONV)
                self_mask = msk_mx[mx.array(cents)]
                isbad_self = (self_mask & FLAG_INPUT_ISBAD) != 0
                ok = self_mask | (FLAG_OUTPUT_ISBAD_V * isbad_self.astype(mx.int32))
                ok = mx.where(mbit != 0, ok | conv_flag, ok)
                msk_results.append(ok)
            mx.eval(*msk_results)
            for idx, (ok_mx, (j0b, i0b, j_endb, i_endb, nRowsb, nColsb)) in enumerate(zip(msk_results, [t[1] for t in rows_cr])):
                msk_np = np.array(ok_mx).reshape(nRowsb, nColsb)
                for r in range(nRowsb):
                    mRData[(j0b+r)*rPixX+i0b:(j0b+r)*rPixX+i_endb] = msk_np[r]

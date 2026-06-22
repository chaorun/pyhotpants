import numpy as np
import numba
import math
import logging
logger = logging.getLogger('hotpants')

from .functions import (
    ZEROVAL, MAXVAL,
    FLAG_BAD_PIXVAL, FLAG_SAT_PIXEL, FLAG_NEG_PIXEL, FLAG_SPREAD,
    FLAG_MASKED, FLAG_BORDER, FLAG_SUBREGION, FLAG_INVALID,
    BUILD_STAMP_FLAT_CACHE, LAST_SIGMA_CLIP, LAST_N, LAST_MEDIAN,
    Ran1,
)


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

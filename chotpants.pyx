import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, strncmp, memcpy
import os, struct

np.import_array()

def region_setup_numpy(tmpl_2d, sci_2d, tnoise_2d, inoise_2d, tmask_2d, imask_2d,
                        ri, rxmins_np, rxmaxs_np, rymins_np, rymaxs_np, nR,
                        hwKernel, fwStamp, sBorder,
                        xMin, yMin, xMax, yMax,
                        fillVal, fillValNoise,
                        tPedestal, iPedestal,
                        tGain, tRdnoise, iGain, iRdnoise,
                        tUThresh, tLThresh, iUThresh, iLThresh,
                        kfSpreadMask1):
    import sys
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
        qquad = np.float64(quad_f) * np.float64(quad_f)
        oRData_py = (np.abs(iRData_py.astype(np.float64)) * np.float64(invGain_f) + qquad).astype(np.float32)

    if tnoise_2d is not None:
        eRData_py[:, :] = tnoise_2d[rYBMin:rYBMax+1, rXBMin:rXBMax+1]
        eRData_py *= eRData_py
    else:
        invGain_t = np.float32(1.0 / float(tGain))
        quad_t = np.float32(float(tRdnoise) / float(tGain))
        qquad_t = np.float64(quad_t) * np.float64(quad_t)
        eRData_py = (np.abs(tRData_py.astype(np.float64)) * np.float64(invGain_t) + qquad_t).astype(np.float32)

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

    return {
        'tRData': tRData_py, 'iRData': iRData_py,
        'oRData': oRData_py, 'eRData': eRData_py,
        'mRData': mRData_py, 'misRData': misRData_py, 'mtsRData': mtsRData_py,
        'rXMin': rXMin, 'rYMin': rYMin, 'rXMax': rXMax, 'rYMax': rYMax,
        'rXBMin': rXBMin, 'rYBMin': rYBMin, 'rXBMax': rXBMax, 'rYBMax': rYBMax,
        'xBufLo': xBufLo, 'xBufHi': xBufHi, 'yBufLo': yBufLo, 'yBufHi': yBufHi,
        'fpixelOutX': fpixelOutX, 'fpixelOutY': fpixelOutY,
        'lpixelOutX': lpixelOutX, 'lpixelOutY': lpixelOutY,
        'rPixX': rPixX, 'rPixY': rPixY,
    }

cdef dict stamp_c_to_dict(stamp_struct *s, int nCompKer, int nBGVectors, int fwKSStamp, int nC, int nKSStamps):
    cdef int j, k
    cdef int nVec = nCompKer + nBGVectors
    cdef int fwSq = fwKSStamp * fwKSStamp
    xss_arr = np.array([s.xss[j] for j in range(nKSStamps)], dtype=np.int32) if s.xss != NULL else np.zeros(nKSStamps, dtype=np.int32)
    yss_arr = np.array([s.yss[j] for j in range(nKSStamps)], dtype=np.int32) if s.yss != NULL else np.zeros(nKSStamps, dtype=np.int32)
    krefArea_arr = np.array([s.krefArea[j] for j in range(fwSq)], dtype=np.float64) if s.krefArea != NULL else np.zeros(fwSq, dtype=np.float64)
    scprod_arr = np.array([s.scprod[j] for j in range(nC)], dtype=np.float64) if s.scprod != NULL else np.zeros(nC, dtype=np.float64)
    if s.vectors != NULL:
        vecs = np.zeros((nVec, fwSq), dtype=np.float64)
        for j in range(nVec):
            if s.vectors[j] != NULL:
                for k in range(fwSq):
                    vecs[j, k] = s.vectors[j][k]
    else:
        vecs = np.zeros((nVec, fwSq), dtype=np.float64)
    if s.mat != NULL:
        mat_arr = np.zeros((nC, nC), dtype=np.float64)
        for j in range(nC):
            if s.mat[j] != NULL:
                for k in range(nC):
                    mat_arr[j, k] = s.mat[j][k]
    else:
        mat_arr = np.zeros((nC, nC), dtype=np.float64)
    return {
        'x0': s.x0, 'y0': s.y0,
        'x': s.x, 'y': s.y,
        'nx': s.nx, 'ny': s.ny,
        'nss': s.nss, 'sscnt': s.sscnt,
        'xss': xss_arr, 'yss': yss_arr,
        'krefArea': krefArea_arr,
        'scprod': scprod_arr,
        'vectors': vecs,
        'mat': mat_arr,
        'chi2': s.chi2, 'norm': s.norm, 'diff': s.diff,
        'sum': s.sum, 'mean': s.mean, 'median': s.median,
        'mode': s.mode, 'sd': s.sd,
        'fwhm': s.fwhm, 'lfwhm': s.lfwhm,
    }

cdef void dict_to_stamp_c(dict d, stamp_struct *s, int nCompKer, int nBGVectors, int fwKSStamp, int nC, int nKSStamps):
    cdef int j, k
    cdef int nVec = nCompKer + nBGVectors
    cdef int fwSq = fwKSStamp * fwKSStamp
    s.x0 = d['x0']; s.y0 = d['y0']
    s.x = d['x']; s.y = d['y']
    s.nx = d['nx']; s.ny = d['ny']
    s.nss = d['nss']; s.sscnt = d['sscnt']
    s.chi2 = d['chi2']; s.norm = d['norm']; s.diff = d['diff']
    s.sum = d['sum']; s.mean = d['mean']; s.median = d['median']
    s.mode = d['mode']; s.sd = d['sd']
    s.fwhm = d['fwhm']; s.lfwhm = d['lfwhm']
    s.xss = <int*>malloc(nKSStamps * sizeof(int))
    xss_np = np.asarray(d['xss'], dtype=np.int32).ravel()
    for j in range(nKSStamps):
        s.xss[j] = xss_np[j]
    s.yss = <int*>malloc(nKSStamps * sizeof(int))
    yss_np = np.asarray(d['yss'], dtype=np.int32).ravel()
    for j in range(nKSStamps):
        s.yss[j] = yss_np[j]
    s.krefArea = <double*>malloc(fwSq * sizeof(double))
    kref_np = np.asarray(d['krefArea'], dtype=np.float64).ravel()
    for j in range(fwSq):
        s.krefArea[j] = kref_np[j]
    s.scprod = <double*>malloc(nC * sizeof(double))
    scprod_np = np.asarray(d['scprod'], dtype=np.float64).ravel()
    for j in range(nC):
        s.scprod[j] = scprod_np[j]
    s.vectors = <double**>malloc(nVec * sizeof(double*))
    vecs_np = np.asarray(d['vectors'], dtype=np.float64)
    for j in range(nVec):
        s.vectors[j] = <double*>malloc(fwSq * sizeof(double))
        for k in range(fwSq):
            s.vectors[j][k] = vecs_np[j, k]
    s.mat = <double**>malloc(nC * sizeof(double*))
    mat_np = np.asarray(d['mat'], dtype=np.float64)
    for j in range(nC):
        s.mat[j] = <double*>malloc(nC * sizeof(double))
        for k in range(nC):
            s.mat[j][k] = mat_np[j, k]

cdef np.ndarray float_ptr_to_numpy(float *ptr, int n):
    if ptr == NULL or n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(<float[:n]>ptr).copy()

cdef np.ndarray int_ptr_to_numpy(int *ptr, int n):
    if ptr == NULL or n <= 0:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(<int[:n]>ptr).copy()

cdef np.ndarray double_ptr_to_numpy(double *ptr, int n):
    if ptr == NULL or n <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(<double[:n]>ptr).copy()

cdef np.ndarray double_pp_to_numpy2d(double **ptr, int rows, int cols):
    cdef int i, j
    if ptr == NULL or rows <= 0 or cols <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] arr = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        if ptr[i] != NULL:
            for j in range(cols):
                arr[i, j] = ptr[i][j]
    return arr

cdef float* numpy_to_float_ptr(np.ndarray arr):
    cdef int n = arr.size
    cdef float *ptr = <float*>malloc(n * sizeof(float))
    cdef float[:] view = arr.ravel().astype(np.float32)
    memcpy(ptr, &view[0], n * sizeof(float))
    return ptr

cdef int* numpy_to_int_ptr(np.ndarray arr):
    cdef int n = arr.size
    cdef int *ptr = <int*>malloc(n * sizeof(int))
    cdef int[:] view = arr.ravel().astype(np.int32)
    memcpy(ptr, &view[0], n * sizeof(int))
    return ptr

cdef double* numpy_to_double_ptr(np.ndarray arr):
    cdef int n = arr.size
    cdef double *ptr = <double*>malloc(n * sizeof(double))
    cdef double[:] view = arr.ravel().astype(np.float64)
    memcpy(ptr, &view[0], n * sizeof(double))
    return ptr

cdef double** numpy2d_to_double_pp(np.ndarray arr):
    cdef int rows = arr.shape[0], cols = arr.shape[1]
    cdef int i, j
    cdef double **ptr = <double**>malloc(rows * sizeof(double*))
    cdef np.ndarray[double, ndim=2] arr64 = arr.astype(np.float64)
    for i in range(rows):
        ptr[i] = <double*>malloc(cols * sizeof(double))
        for j in range(cols):
            ptr[i][j] = arr64[i, j]
    return ptr

cdef void free_double_pp(double **ptr, int rows):
    cdef int i
    if ptr != NULL:
        for i in range(rows):
            if ptr[i] != NULL:
                free(ptr[i])
        free(ptr)

cdef float** numpy2d_to_float_pp(np.ndarray arr):
    cdef int rows = arr.shape[0], cols = arr.shape[1]
    cdef int i, j
    cdef float **ptr = <float**>malloc(rows * sizeof(float*))
    cdef np.ndarray[float, ndim=2] arr32 = arr.astype(np.float32)
    for i in range(rows):
        ptr[i] = <float*>malloc(cols * sizeof(float))
        for j in range(cols):
            ptr[i][j] = arr32[i, j]
    return ptr

cdef void free_float_pp(float **ptr, int rows):
    cdef int i
    if ptr != NULL:
        for i in range(rows):
            if ptr[i] != NULL:
                free(ptr[i])
        free(ptr)

def test_convert_roundtrip():
    import sys
    ok = True

    orig_f = np.random.randn(100).astype(np.float32)
    cdef float *fp = numpy_to_float_ptr(orig_f)
    rt_f = float_ptr_to_numpy(fp, 100)
    free(fp)
    if not np.array_equal(orig_f, rt_f):
        sys.stderr.write("float roundtrip FAIL\n"); ok = False

    orig_i = np.random.randint(-1000, 1000, 100).astype(np.int32)
    cdef int *ip = numpy_to_int_ptr(orig_i)
    rt_i = int_ptr_to_numpy(ip, 100)
    free(ip)
    if not np.array_equal(orig_i, rt_i):
        sys.stderr.write("int roundtrip FAIL\n"); ok = False

    orig_d = np.random.randn(100).astype(np.float64)
    cdef double *dp = numpy_to_double_ptr(orig_d)
    rt_d = double_ptr_to_numpy(dp, 100)
    free(dp)
    if not np.array_equal(orig_d, rt_d):
        sys.stderr.write("double roundtrip FAIL\n"); ok = False

    orig_dd = np.random.randn(5, 20).astype(np.float64)
    cdef double **dpp = numpy2d_to_double_pp(orig_dd)
    rt_dd = double_pp_to_numpy2d(dpp, 5, 20)
    free_double_pp(dpp, 5)
    if not np.array_equal(orig_dd, rt_dd):
        sys.stderr.write("double** roundtrip FAIL\n"); ok = False

    return ok

def hotpants(
    inim, tmplim,
    tni=None, ini=None, tmi=None, imi=None,
    tu=25000., tuk=None, tl=0., tg=1., tr=0., tp=0.,
    iu=25000., iuk=None, il=0., ig=1., ir=0., ip=0.,
    r=10, ko=2, bgo=1,
    ng=3, ng_deg=None, ng_sig=None,
    pca=None,
    nrx=1, nry=1, rf=None,
    nsx=10, nsy=10, ssf=None, afssc=1, nss=3, rss=15,
    ft=20.0, sft=0.5, nft=0.1,
    ssig=3.0, ks=2.0, kfm=0.99,
    mins=1.0, mous=1.0,
    fi=1e-30, fin=0.,
    c='b', n='t', fom='v',
    sconv=0, okn=0, convvar=0,
    v=1, kcs=0,
    uss=0, savexy=0,
    dump_dir=None,
):
    if ng_deg is None:
        ng_deg = [6, 4, 2]
    if ng_sig is None:
        ng_sig = [0.7, 1.5, 3.0]

    cdef np.ndarray[float, ndim=2, mode="c"] tmpl_arr = np.ascontiguousarray(tmplim, dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode="c"] sci_arr = np.ascontiguousarray(inim, dtype=np.float32)
    cdef long tNx = tmpl_arr.shape[1]
    cdef long tNy = tmpl_arr.shape[0]
    cdef long iNx = sci_arr.shape[1]
    cdef long iNy = sci_arr.shape[0]

    cdef np.ndarray[float, ndim=2, mode="c"] tni_arr
    cdef float *tni_ptr = NULL
    if tni is not None:
        tni_arr = np.ascontiguousarray(tni, dtype=np.float32)
        tni_ptr = &tni_arr[0, 0]

    cdef np.ndarray[float, ndim=2, mode="c"] ini_arr
    cdef float *ini_ptr = NULL
    if ini is not None:
        ini_arr = np.ascontiguousarray(ini, dtype=np.float32)
        ini_ptr = &ini_arr[0, 0]

    cdef np.ndarray[int, ndim=2, mode="c"] tmi_arr
    cdef int *tmi_ptr = NULL
    if tmi is not None:
        tmi_arr = np.ascontiguousarray(tmi, dtype=np.int32)
        tmi_ptr = &tmi_arr[0, 0]

    cdef np.ndarray[int, ndim=2, mode="c"] imi_arr
    cdef int *imi_ptr = NULL
    if imi is not None:
        imi_arr = np.ascontiguousarray(imi, dtype=np.int32)
        imi_ptr = &imi_arr[0, 0]

    cdef float tuk_val = <float>tu if tuk is None else <float>tuk
    cdef float iuk_val = <float>iu if iuk is None else <float>iuk

    cdef np.ndarray[float, ndim=1, mode="c"] sig_arr = np.array(
        [1.0 / (2.0 * s * s) for s in ng_sig], dtype=np.float32)
    cdef np.ndarray[int, ndim=1, mode="c"] deg_arr = np.array(ng_deg, dtype=np.int32)

    cdef int use_pca = 0
    cdef float **pca_ptr = NULL
    cdef list pca_arrs = []
    cdef int pca_ng = ng
    if pca is not None:
        use_pca = 1
        pca_ng = len(pca)
        pca_ptr = <float **>malloc(pca_ng * sizeof(float *))
        for pi in range(pca_ng):
            parr = np.ascontiguousarray(pca[pi], dtype=np.float32)
            pca_arrs.append(parr)
            pca_ptr[pi] = <float *>np.PyArray_DATA(parr)
        r = pca[0].shape[0] // 2
        deg_arr = np.zeros(pca_ng, dtype=np.int32)
        sig_arr = np.full(pca_ng, -1.0, dtype=np.float32)
        ng = pca_ng

    cdef int nR
    cdef np.ndarray[int, ndim=1, mode="c"] rxmins, rxmaxs, rymins, rymaxs
    cdef int xMin = 0
    cdef int yMin = 0
    cdef int xMax = min(tNx, iNx) - 1
    cdef int yMax = min(tNy, iNy) - 1

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

    cdef float *xcmp_ptr = NULL
    cdef float *ycmp_ptr = NULL
    cdef int ncmp = 0
    cdef np.ndarray[float, ndim=1, mode="c"] xcmp_arr, ycmp_arr
    if ssf is not None:
        xcmp_arr = np.array([p[0] - 1 for p in ssf], dtype=np.float32)
        ycmp_arr = np.array([p[1] - 1 for p in ssf], dtype=np.float32)
        ncmp = len(ssf)
        xcmp_ptr = &xcmp_arr[0]
        ycmp_ptr = &ycmp_arr[0]

    c_bytes = c.encode('ascii')
    n_bytes = n.encode('ascii')
    fom_bytes = fom.encode('ascii')
    cdef char *c_str = c_bytes
    cdef char *n_str = n_bytes
    cdef char *fom_str = fom_bytes

    cdef long oNx = max(tNx, iNx)
    cdef long oNy = max(tNy, iNy)
    cdef np.ndarray[float, ndim=2, mode="c"] diff_out = np.full((oNy, oNx), <float>fi, dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode="c"] noise_out = np.full((oNy, oNx), <float>fin, dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode="c"] conv_out = np.full((oNy, oNx), <float>fi, dtype=np.float32)
    cdef np.ndarray[int, ndim=2, mode="c"] mask_out = np.zeros((oNy, oNx), dtype=np.int32)

    cdef region_stats *stats = <region_stats *>calloc(nR, sizeof(region_stats))

    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        import struct as _st
        _path = os.path.join(dump_dir, "py_input.bin")
        with open(_path, "wb") as _f:
            def _de(nm, db):
                nb = nm.encode('ascii')
                _f.write(_st.pack('i', len(nb)))
                _f.write(nb)
                dl = len(db) if db else 0
                _f.write(_st.pack('l', dl))
                if dl > 0: _f.write(db)
            def _di(nm, val): _de(nm, _st.pack('i', int(val)))
            def _dl(nm, val): _de(nm, _st.pack('l', int(val)))
            def _df(nm, val): _de(nm, _st.pack('f', float(val)))
            def _ds(nm, s):
                if s is None: _de(nm, None)
                else: _de(nm, s.encode('ascii') + b'\x00')
            _dl("tNx",tNx); _dl("tNy",tNy); _dl("iNx",iNx); _dl("iNy",iNy)
            _dl("oNx",oNx); _dl("oNy",oNy)
            _di("nR",nR); _di("hwKernel",r); _di("ngauss",ng)
            _di("kerOrder",ko); _di("bgOrder",bgo)
            _di("nStampX",nsx); _di("nStampY",nsy)
            _di("nKSStamps",nss); _di("hwKSStamp",rss)
            _di("useFullSS",uss); _di("findSSC",afssc)
            _df("kerFitThresh",ft); _df("scaleFitThresh",sft)
            _df("minFracGoodStamps",nft)
            _df("statSig",ssig); _df("kerSigReject",ks); _df("kerFracMask",kfm)
            _df("tUThresh",tu); _df("tLThresh",tl)
            _df("tGain",tg); _df("tRdnoise",tr); _df("tPedestal",tp)
            _df("iUThresh",iu); _df("iLThresh",il)
            _df("iGain",ig); _df("iRdnoise",ir); _df("iPedestal",ip)
            _df("tUKThresh",tuk_val); _df("iUKThresh",iuk_val)
            _df("kfSpreadMask1",mins); _df("kfSpreadMask2",mous)
            _df("fillVal",fi); _df("fillValNoise",fin)
            _di("sameConv",sconv); _di("rescaleOK",okn)
            _di("convolveVariance",convvar)
            _di("usePCA",use_pca); _di("Ncmp",ncmp)
            _di("verbose",v); _di("kcStep",kcs); _di("savexyflag",savexy)
            _ds("forceConvolve",c); _ds("photNormalize",n); _ds("figMerit",fom)
            _de("deg_fixe", bytes(np.ascontiguousarray(deg_arr)))
            _de("sigma_gauss", bytes(np.ascontiguousarray(sig_arr)))
            _de("tFullData", bytes(np.ascontiguousarray(tmpl_arr)))
            _de("iFullData", bytes(np.ascontiguousarray(sci_arr)))
            _de("tNoiseFullData", bytes(np.ascontiguousarray(tni_arr)) if tni is not None else None)
            _de("iNoiseFullData", bytes(np.ascontiguousarray(ini_arr)) if ini is not None else None)
            _de("tMaskFullData", bytes(np.ascontiguousarray(tmi_arr)) if tmi is not None else None)
            _de("iMaskFullData", bytes(np.ascontiguousarray(imi_arr)) if imi is not None else None)
            _de("rXMins", bytes(np.ascontiguousarray(rxmins)))
            _de("rXMaxs", bytes(np.ascontiguousarray(rxmaxs)))
            _de("rYMins", bytes(np.ascontiguousarray(rymins)))
            _de("rYMaxs", bytes(np.ascontiguousarray(rymaxs)))
            _de("xcmp", bytes(np.ascontiguousarray(xcmp_arr)) if ssf is not None else None)
            _de("ycmp", bytes(np.ascontiguousarray(ycmp_arr)) if ssf is not None else None)
            _de("diffOut", bytes(np.ascontiguousarray(diff_out)))
            _de("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            _de("convOut", bytes(np.ascontiguousarray(conv_out)))
            _de("maskOut", bytes(np.ascontiguousarray(mask_out)))

    cdef hotpants_context ctx
    memset(&ctx, 0, sizeof(hotpants_context))
    hotpants_init(&ctx, <int>r, <int>ng, &deg_arr[0], &sig_arr[0],
                  <int>ko, <int>bgo, <int>nsx, <int>nsy, <int>nss, <int>rss,
                  <int>uss, <float>ft, <int>kcs, tNx, tNy, iNx, iNy, nR)

    cdef hotpants_params prm
    memset(&prm, 0, sizeof(hotpants_params))
    prm.tFullData = &tmpl_arr[0, 0]; prm.tNx = tNx; prm.tNy = tNy
    prm.iFullData = &sci_arr[0, 0]; prm.iNx = iNx; prm.iNy = iNy
    prm.tNoiseFullData = tni_ptr; prm.iNoiseFullData = ini_ptr
    prm.tMaskFullData = tmi_ptr; prm.iMaskFullData = imi_ptr
    prm.nR = nR
    prm.rXMins = &rxmins[0]; prm.rXMaxs = &rxmaxs[0]
    prm.rYMins = &rymins[0]; prm.rYMaxs = &rymaxs[0]
    prm.hwKernel = <int>r; prm.ngauss = <int>ng
    prm.deg_fixe = &deg_arr[0]; prm.sigma_gauss = &sig_arr[0]
    prm.kerOrder = <int>ko; prm.bgOrder = <int>bgo
    prm.findSSC = <int>afssc
    prm.hwKSStamp = <int>rss; prm.nKSStamps = <int>nss
    prm.kerFitThresh = <float>ft; prm.scaleFitThresh = <float>sft
    prm.minFracGoodStamps = <float>nft
    prm.statSig = <float>ssig; prm.kerSigReject = <float>ks; prm.kerFracMask = <float>kfm
    prm.tUThresh = <float>tu; prm.tLThresh = <float>tl
    prm.tGain = <float>tg; prm.tRdnoise = <float>tr; prm.tPedestal = <float>tp
    prm.iUThresh = <float>iu; prm.iLThresh = <float>il
    prm.iGain = <float>ig; prm.iRdnoise = <float>ir; prm.iPedestal = <float>ip
    prm.tUKThresh = tuk_val; prm.iUKThresh = iuk_val
    prm.kfSpreadMask1 = <float>mins; prm.kfSpreadMask2 = <float>mous
    prm.fillVal = <float>fi; prm.fillValNoise = <float>fin
    prm.forceConvolve = c_str; prm.photNormalize = n_str; prm.figMerit = fom_str
    prm.sameConv = <int>sconv; prm.rescaleOK = <int>okn; prm.convolveVariance = <int>convvar
    prm.usePCA = use_pca; prm.PCA = pca_ptr
    prm.xcmp = xcmp_ptr; prm.ycmp = ycmp_ptr; prm.Ncmp = ncmp
    prm.verbose = <int>v
    prm.savexyflag = <int>savexy
    prm.diffOut = &diff_out[0, 0]; prm.noiseOut = &noise_out[0, 0]
    prm.convOut = &conv_out[0, 0]; prm.maskOut = &mask_out[0, 0]
    prm.oNx = oNx; prm.oNy = oNy
    prm.stats = stats

    cdef char *localFC = c_str
    cdef int ri
    cdef region_state rs_tmp
    cdef int bs_ret
    cdef int npix
    cdef region_state rs_c
    for ri in range(nR):
        memset(&rs_tmp, 0, sizeof(region_state))

        # ========== [旧代码 注释掉] region_setup C 主路径 + numpy 影子 ==========
        # region_setup(&ctx, &prm, ri, &rs_tmp)
        #
        # py_shadow = _region_setup_numpy(
        #     tmpl_arr, sci_arr,
        #     tni_arr if tni is not None else None,
        #     ini_arr if ini is not None else None,
        #     tmi_arr if tmi is not None else None,
        #     imi_arr if imi is not None else None,
        #     ri, np.asarray(rxmins), np.asarray(rxmaxs),
        #     np.asarray(rymins), np.asarray(rymaxs), nR,
        #     r, ctx.fwStamp, ctx.sBorder,
        #     ctx.xMin, ctx.yMin, ctx.xMax, ctx.yMax,
        #     fi, fin,
        #     tp, ip,
        #     tg, tr, ig, ir,
        #     tu, tl, iu, il,
        #     mins)
        # npix_shadow = rs_tmp.rPixX * rs_tmp.rPixY
        # shadow_pairs = [
        #     ('tRData', np.asarray(<float[:npix_shadow]>rs_tmp.tRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['tRData']),
        #     ('iRData', np.asarray(<float[:npix_shadow]>rs_tmp.iRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['iRData']),
        #     ('oRData', np.asarray(<float[:npix_shadow]>rs_tmp.oRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['oRData']),
        #     ('eRData', np.asarray(<float[:npix_shadow]>rs_tmp.eRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['eRData']),
        #     ('mRData', np.asarray(<int[:npix_shadow]>rs_tmp.mRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['mRData']),
        #     ('misRData', np.asarray(<int[:npix_shadow]>rs_tmp.misRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['misRData']),
        #     ('mtsRData', np.asarray(<int[:npix_shadow]>rs_tmp.mtsRData).reshape(rs_tmp.rPixY, rs_tmp.rPixX).copy(), py_shadow['mtsRData']),
        # ]
        # import sys as _sys_shadow
        # shadow_ok = True
        # for _sname, _c_arr, _p_arr in shadow_pairs:
        #     if not np.array_equal(_c_arr, _p_arr):
        #         shadow_ok = False
        #         _diff_idx = np.where(_c_arr != _p_arr)
        #         _ndiff = len(_diff_idx[0])
        #         _sys_shadow.stderr.write(
        #             f"  MISMATCH {_sname} region {ri}: {_ndiff} diffs, "
        #             f"first at ({_diff_idx[0][0]},{_diff_idx[1][0]}): "
        #             f"C={_c_arr[_diff_idx[0][0],_diff_idx[1][0]]}, "
        #             f"py={_p_arr[_diff_idx[0][0],_diff_idx[1][0]]}\n")
        #         _sys_shadow.stderr.flush()
        # if shadow_ok:
        #     _sys_shadow.stderr.write(f"  region_setup shadow OK for region {ri}\n")
        #     _sys_shadow.stderr.flush()
        # else:
        #     _sys_shadow.stderr.write(f"  region_setup shadow FAILED for region {ri}\n")
        #     _sys_shadow.stderr.flush()
        # ========== [旧代码 结束] ==========

        # ========== numpy 主路径 ==========
        py = region_setup_numpy(
            tmpl_arr, sci_arr,
            tni_arr if tni is not None else None,
            ini_arr if ini is not None else None,
            tmi_arr if tmi is not None else None,
            imi_arr if imi is not None else None,
            ri, np.asarray(rxmins), np.asarray(rxmaxs),
            np.asarray(rymins), np.asarray(rymaxs), nR,
            r, ctx.fwStamp, ctx.sBorder,
            ctx.xMin, ctx.yMin, ctx.xMax, ctx.yMax,
            fi, fin,
            tp, ip,
            tg, tr, ig, ir,
            tu, tl, iu, il,
            mins)

        npix = py['rPixX'] * py['rPixY']
        rs_tmp.tRData = <float*>malloc(npix * sizeof(float))
        np.asarray(<float[:npix]>rs_tmp.tRData)[:] = py['tRData'].ravel()
        rs_tmp.iRData = <float*>malloc(npix * sizeof(float))
        np.asarray(<float[:npix]>rs_tmp.iRData)[:] = py['iRData'].ravel()
        rs_tmp.oRData = <float*>malloc(npix * sizeof(float))
        np.asarray(<float[:npix]>rs_tmp.oRData)[:] = py['oRData'].ravel()
        rs_tmp.eRData = <float*>malloc(npix * sizeof(float))
        np.asarray(<float[:npix]>rs_tmp.eRData)[:] = py['eRData'].ravel()
        rs_tmp.mRData = <int*>malloc(npix * sizeof(int))
        np.asarray(<int[:npix]>rs_tmp.mRData)[:] = py['mRData'].ravel()
        rs_tmp.misRData = <int*>malloc(npix * sizeof(int))
        np.asarray(<int[:npix]>rs_tmp.misRData)[:] = py['misRData'].ravel()
        rs_tmp.mtsRData = <int*>malloc(npix * sizeof(int))
        np.asarray(<int[:npix]>rs_tmp.mtsRData)[:] = py['mtsRData'].ravel()

        if strncmp(localFC, b"i", 1) != 0:
            rs_tmp.ctStamps = <stamp_struct*>calloc(ctx.nStamps, sizeof(stamp_struct))
            allocateStamps(rs_tmp.ctStamps, ctx.nStamps, prm.bgOrder, ctx.nCompKer, ctx.fwKSStamp, ctx.nC, prm.nKSStamps)
            rs_tmp.tKerSol = <double*>calloc(ctx.nCompTotal + 1, sizeof(double))
        if strncmp(localFC, b"t", 1) != 0:
            rs_tmp.ciStamps = <stamp_struct*>calloc(ctx.nStamps, sizeof(stamp_struct))
            allocateStamps(rs_tmp.ciStamps, ctx.nStamps, prm.bgOrder, ctx.nCompKer, ctx.fwKSStamp, ctx.nC, prm.nKSStamps)
            rs_tmp.iKerSol = <double*>calloc(ctx.nCompTotal + 1, sizeof(double))

        rs_tmp.rXMin = py['rXMin']; rs_tmp.rYMin = py['rYMin']
        rs_tmp.rXMax = py['rXMax']; rs_tmp.rYMax = py['rYMax']
        rs_tmp.rXBMin = py['rXBMin']; rs_tmp.rYBMin = py['rYBMin']
        rs_tmp.rXBMax = py['rXBMax']; rs_tmp.rYBMax = py['rYBMax']
        rs_tmp.xBufLo = py['xBufLo']; rs_tmp.xBufHi = py['xBufHi']
        rs_tmp.yBufLo = py['yBufLo']; rs_tmp.yBufHi = py['yBufHi']
        rs_tmp.fpixelOutX = py['fpixelOutX']; rs_tmp.fpixelOutY = py['fpixelOutY']
        rs_tmp.lpixelOutX = py['lpixelOutX']; rs_tmp.lpixelOutY = py['lpixelOutY']
        rs_tmp.rPixX = py['rPixX']; rs_tmp.rPixY = py['rPixY']
        rs_tmp.meansigSubstamps = 0.0; rs_tmp.scatterSubstamps = 0.0
        rs_tmp.NskippedSubstamps = 0

        # ========== C 影子验证 ==========
        memset(&rs_c, 0, sizeof(region_state))
        region_setup(&ctx, &prm, ri, &rs_c)

        import sys
        shadow_pairs = [
            ('tRData', np.asarray(<float[:npix]>rs_tmp.tRData).copy(), np.asarray(<float[:npix]>rs_c.tRData).copy()),
            ('iRData', np.asarray(<float[:npix]>rs_tmp.iRData).copy(), np.asarray(<float[:npix]>rs_c.iRData).copy()),
            ('oRData', np.asarray(<float[:npix]>rs_tmp.oRData).copy(), np.asarray(<float[:npix]>rs_c.oRData).copy()),
            ('eRData', np.asarray(<float[:npix]>rs_tmp.eRData).copy(), np.asarray(<float[:npix]>rs_c.eRData).copy()),
            ('mRData', np.asarray(<int[:npix]>rs_tmp.mRData).copy(), np.asarray(<int[:npix]>rs_c.mRData).copy()),
            ('misRData', np.asarray(<int[:npix]>rs_tmp.misRData).copy(), np.asarray(<int[:npix]>rs_c.misRData).copy()),
            ('mtsRData', np.asarray(<int[:npix]>rs_tmp.mtsRData).copy(), np.asarray(<int[:npix]>rs_c.mtsRData).copy()),
        ]
        shadow_ok = True
        for sname, np_arr, c_arr in shadow_pairs:
            if not np.array_equal(np_arr, c_arr):
                shadow_ok = False
                diff_idx = np.where(np_arr != c_arr)
                ndiff = len(diff_idx[0])
                sys.stderr.write(
                    f"  MISMATCH {sname} region {ri}: {ndiff} diffs\n")
                sys.stderr.flush()
        if shadow_ok:
            sys.stderr.write(f"  region_setup numpy-main shadow OK for region {ri}\n")
            sys.stderr.flush()

        region_cleanup_local(&rs_c, 0)

        # ========== 后续步骤用 rs_tmp（numpy 数据） ==========
        bs_ret = region_buildstamps(&ctx, &prm, &rs_tmp, localFC)
        if bs_ret != 0:
            region_cleanup_local(&rs_tmp, rs_tmp.convTmpl)
            continue

        region_fit(&ctx, &prm, &rs_tmp, &localFC)

        region_convolve_diff(&ctx, &prm, ri, &rs_tmp, &localFC)

        region_output(&ctx, &prm, ri, &rs_tmp)

        region_cleanup_local(&rs_tmp, rs_tmp.convTmpl)

    hotpants_cleanup(&ctx)

    if dump_dir is not None:
        import struct as _st
        _opath = os.path.join(dump_dir, "py_output.bin")
        with open(_opath, "wb") as _f:
            def _de2(nm, db):
                nb = nm.encode('ascii')
                _f.write(_st.pack('i', len(nb)))
                _f.write(nb)
                dl2 = len(db) if db else 0
                _f.write(_st.pack('l', dl2))
                if dl2 > 0: _f.write(db)
            _de2("diffOut", bytes(np.ascontiguousarray(diff_out)))
            _de2("noiseOut", bytes(np.ascontiguousarray(noise_out)))
            _de2("convOut", bytes(np.ascontiguousarray(conv_out)))
            _de2("maskOut", bytes(np.ascontiguousarray(mask_out)))

    # if dump_dir is not None:
    #     if 'HOTPANTS_DUMP_DIR' in os.environ:
    #         del os.environ['HOTPANTS_DUMP_DIR']

    stats_list = []
    for si in range(nR):
        stats_list.append({
            'conv_tmpl': stats[si].convTmpl,
            'sum_kernel': stats[si].sumKernel,
            'mean_sig': stats[si].meansigSubstamps,
            'scatter_sig': stats[si].scatterSubstamps,
            'final_mean_sig': stats[si].meansigSubstampsF,
            'final_scatter_sig': stats[si].scatterSubstampsF,
            'x2norm': stats[si].x2norm,
            'nx2norm': stats[si].nx2norm,
            'diff_mean': stats[si].mean,
            'diff_sd': stats[si].sd,
            'noise_mean': stats[si].nmean,
            'diff_mean_ok': stats[si].meanm,
            'diff_sd_ok': stats[si].sdm,
            'noise_mean_ok': stats[si].nmeanm,
            'diffrat': stats[si].diffrat,
        })

    free(stats)
    if pca_ptr != NULL:
        free(pca_ptr)

    return diff_out, noise_out, conv_out, mask_out, stats_list

def test_stamp_roundtrip():
    cdef int nCompKer = 3
    cdef int bgOrder = 1
    cdef int nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int fwKSStamp = 5
    cdef int nC = 6
    cdef int nKSStamps = 10
    cdef int nVec = nCompKer + nBGVectors
    cdef int fwSq = fwKSStamp * fwKSStamp
    cdef int j, k

    cdef stamp_struct *orig = <stamp_struct*>calloc(1, sizeof(stamp_struct))
    allocateStamps(orig, 1, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)

    orig.x0 = 10; orig.y0 = 20
    orig.x = 30; orig.y = 40
    orig.nx = 50; orig.ny = 60
    orig.nss = nKSStamps; orig.sscnt = 7
    orig.chi2 = 1.23; orig.norm = 4.56; orig.diff = 7.89
    orig.sum = 10.1; orig.mean = 11.2; orig.median = 12.3
    orig.mode = 13.4; orig.sd = 14.5
    orig.fwhm = 15.6; orig.lfwhm = 16.7

    for j in range(nKSStamps):
        orig.xss[j] = j * 3 + 1
        orig.yss[j] = j * 5 + 2

    for j in range(fwSq):
        orig.krefArea[j] = <double>(j * 0.1 + 0.01)

    for j in range(nC):
        orig.scprod[j] = <double>(j * 0.5 + 0.05)

    for j in range(nVec):
        for k in range(fwSq):
            orig.vectors[j][k] = <double>(j * 100.0 + k * 0.7)

    for j in range(nC):
        for k in range(nC):
            orig.mat[j][k] = <double>(j * 10.0 + k * 1.1)

    d = stamp_c_to_dict(orig, nCompKer, nBGVectors, fwKSStamp, nC, nKSStamps)

    cdef stamp_struct *rt = <stamp_struct*>calloc(1, sizeof(stamp_struct))
    dict_to_stamp_c(d, rt, nCompKer, nBGVectors, fwKSStamp, nC, nKSStamps)

    cdef bint ok = True
    if orig.x0 != rt.x0 or orig.y0 != rt.y0: ok = False
    if orig.x != rt.x or orig.y != rt.y: ok = False
    if orig.nx != rt.nx or orig.ny != rt.ny: ok = False
    if orig.nss != rt.nss or orig.sscnt != rt.sscnt: ok = False
    if orig.chi2 != rt.chi2 or orig.norm != rt.norm or orig.diff != rt.diff: ok = False
    if orig.sum != rt.sum or orig.mean != rt.mean or orig.median != rt.median: ok = False
    if orig.mode != rt.mode or orig.sd != rt.sd: ok = False
    if orig.fwhm != rt.fwhm or orig.lfwhm != rt.lfwhm: ok = False

    for j in range(nKSStamps):
        if orig.xss[j] != rt.xss[j]: ok = False
        if orig.yss[j] != rt.yss[j]: ok = False

    for j in range(fwSq):
        if orig.krefArea[j] != rt.krefArea[j]: ok = False

    for j in range(nC):
        if orig.scprod[j] != rt.scprod[j]: ok = False

    for j in range(nVec):
        for k in range(fwSq):
            if orig.vectors[j][k] != rt.vectors[j][k]: ok = False

    for j in range(nC):
        for k in range(nC):
            if orig.mat[j][k] != rt.mat[j][k]: ok = False

    for j in range(nVec):
        if orig.vectors[j] != NULL: free(orig.vectors[j])
    free(orig.vectors)
    for j in range(nC):
        if orig.mat[j] != NULL: free(orig.mat[j])
    free(orig.mat)
    free(orig.krefArea); free(orig.scprod)
    free(orig.xss); free(orig.yss)
    free(orig)

    for j in range(nVec):
        if rt.vectors[j] != NULL: free(rt.vectors[j])
    free(rt.vectors)
    for j in range(nC):
        if rt.mat[j] != NULL: free(rt.mat[j])
    free(rt.mat)
    free(rt.krefArea); free(rt.scprod)
    free(rt.xss); free(rt.yss)
    free(rt)

    return ok

def test_functions_batch1():
    import sys
    ok = True

    # --- sigma_clip ---
    np.random.seed(42)
    data_sc = np.random.randn(200).astype(np.float32) * 10 + 50
    cdef float *data_sc_c = numpy_to_float_ptr(data_sc)
    cdef double sc_mean = 0, sc_stdev = 0
    cdef int sc_rc = sigma_clip(data_sc_c, 200, &sc_mean, &sc_stdev, 10, 3.0)
    free(data_sc_c)

    from pyhotpants.numutils import sigma_clip_numpy
    py_mean, py_stdev, py_rc = sigma_clip_numpy(data_sc.copy(), maxiter=10, stat_sig=3.0)

    if abs(sc_mean - py_mean) > 0 or abs(sc_stdev - py_stdev) > 0:
        sys.stderr.write(f"sigma_clip FAIL: c=({sc_mean},{sc_stdev},{sc_rc}), py=({py_mean},{py_stdev},{py_rc})\n")
        ok = False
    else:
        sys.stderr.write("sigma_clip PASS\n")
    sys.stderr.flush()

    # --- getNoiseStats3 ---
    np.random.seed(43)
    cdef int gnPixX = 20, gnPixY = 15
    data_gn = np.random.randn(gnPixX * gnPixY).astype(np.float32) * 5 + 100
    noise_gn = (np.abs(np.random.randn(gnPixX * gnPixY).astype(np.float32)) + 1.0).astype(np.float32)
    mRData_gn = np.zeros(gnPixX * gnPixY, dtype=np.int32)
    mRData_gn[0] = 0x80
    mRData_gn[5] = 0x100

    cdef float *data_gn_c = numpy_to_float_ptr(data_gn)
    cdef float *noise_gn_c = numpy_to_float_ptr(noise_gn)
    cdef int *mRData_gn_c = numpy_to_int_ptr(mRData_gn)
    cdef double gn_nnorm = 0
    cdef int gn_nncount = 0
    getNoiseStats3(data_gn_c, noise_gn_c, &gn_nnorm, &gn_nncount, 0, 0x8000, gnPixX, gnPixY, mRData_gn_c)
    free(data_gn_c)
    free(noise_gn_c)
    free(mRData_gn_c)

    from pyhotpants.numutils import get_noise_stats3_numpy
    py_nnorm, py_nncount = get_noise_stats3_numpy(data_gn, noise_gn, 0, 0x8000, gnPixX, gnPixY, mRData_gn)

    if abs(gn_nnorm - py_nnorm) > 0 or gn_nncount != py_nncount:
        sys.stderr.write(f"getNoiseStats3 FAIL: c=({gn_nnorm},{gn_nncount}), py=({py_nnorm},{py_nncount})\n")
        ok = False
    else:
        sys.stderr.write("getNoiseStats3 PASS\n")
    sys.stderr.flush()

    # --- insert_subregion_flt ---
    np.random.seed(44)
    cdef int isf_subNx = 30, isf_subNy = 25
    cdef long isf_fullNx = 100
    cdef int isf_fullNy = 80
    sub_flt = np.random.randn(isf_subNy * isf_subNx).astype(np.float32)
    full_flt_c = np.zeros(isf_fullNy * isf_fullNx, dtype=np.float32)
    full_flt_py = np.zeros((isf_fullNy, isf_fullNx), dtype=np.float32)
    cdef int isf_fpX = 11, isf_fpY = 6, isf_lpX = 20, isf_lpY = 15
    cdef int isf_xBufLo = 2, isf_yBufLo = 3

    cdef float *sub_flt_c = numpy_to_float_ptr(sub_flt)
    cdef float *full_flt_c_ptr = numpy_to_float_ptr(full_flt_c)
    insert_subregion_flt(sub_flt_c, isf_subNx, full_flt_c_ptr, isf_fullNx, isf_fpX, isf_fpY, isf_lpX, isf_lpY, isf_xBufLo, isf_yBufLo)
    full_flt_c_result = float_ptr_to_numpy(full_flt_c_ptr, isf_fullNy * isf_fullNx)
    free(sub_flt_c)
    free(full_flt_c_ptr)

    from pyhotpants.numutils import insert_subregion_flt_numpy
    sub_flt_2d = sub_flt.reshape(isf_subNy, isf_subNx)
    insert_subregion_flt_numpy(sub_flt_2d, full_flt_py, isf_fpX, isf_fpY, isf_lpX, isf_lpY, isf_xBufLo, isf_yBufLo)
    full_flt_py_result = full_flt_py.ravel()

    if not np.array_equal(full_flt_c_result, full_flt_py_result):
        ndiff = np.sum(full_flt_c_result != full_flt_py_result)
        sys.stderr.write(f"insert_subregion_flt FAIL: {ndiff} diffs\n")
        ok = False
    else:
        sys.stderr.write("insert_subregion_flt PASS\n")
    sys.stderr.flush()

    # --- insert_subregion_int ---
    np.random.seed(45)
    cdef int isi_subNx = 30, isi_subNy = 25
    cdef long isi_fullNx = 100
    cdef int isi_fullNy = 80
    sub_int = np.random.randint(-1000, 1000, isi_subNy * isi_subNx).astype(np.int32)
    full_int_c = np.zeros(isi_fullNy * isi_fullNx, dtype=np.int32)
    full_int_py = np.zeros((isi_fullNy, isi_fullNx), dtype=np.int32)
    cdef int isi_fpX = 11, isi_fpY = 6, isi_lpX = 20, isi_lpY = 15
    cdef int isi_xBufLo = 2, isi_yBufLo = 3

    cdef int *sub_int_c = numpy_to_int_ptr(sub_int)
    cdef int *full_int_c_ptr = numpy_to_int_ptr(full_int_c)
    insert_subregion_int(sub_int_c, isi_subNx, full_int_c_ptr, isi_fullNx, isi_fpX, isi_fpY, isi_lpX, isi_lpY, isi_xBufLo, isi_yBufLo)
    full_int_c_result = int_ptr_to_numpy(full_int_c_ptr, isi_fullNy * isi_fullNx)
    free(sub_int_c)
    free(full_int_c_ptr)

    from pyhotpants.numutils import insert_subregion_int_numpy
    sub_int_2d = sub_int.reshape(isi_subNy, isi_subNx)
    insert_subregion_int_numpy(sub_int_2d, full_int_py, isi_fpX, isi_fpY, isi_lpX, isi_lpY, isi_xBufLo, isi_yBufLo)
    full_int_py_result = full_int_py.ravel()

    if not np.array_equal(full_int_c_result, full_int_py_result):
        ndiff_i = np.sum(full_int_c_result != full_int_py_result)
        sys.stderr.write(f"insert_subregion_int FAIL: {ndiff_i} diffs\n")
        ok = False
    else:
        sys.stderr.write("insert_subregion_int PASS\n")
    sys.stderr.flush()

    # --- cutStamp ---
    np.random.seed(46)
    cdef int cs_dxLen = 50, cs_dyLen = 40
    data_cs = np.random.randn(cs_dxLen * cs_dyLen).astype(np.float32) * 10 + 50
    cdef int cs_xMin = 5, cs_yMin = 8, cs_xMax = 20, cs_yMax = 25
    cdef int cs_sxLen = cs_xMax - cs_xMin + 1
    cdef int cs_syLen = cs_yMax - cs_yMin + 1

    cdef float *data_cs_c = numpy_to_float_ptr(data_cs)
    cdef float *refArea_c = <float*>calloc(cs_sxLen * cs_syLen, sizeof(float))
    cdef stamp_struct cs_stamp
    memset(&cs_stamp, 0, sizeof(stamp_struct))
    cutStamp(data_cs_c, refArea_c, cs_dxLen, cs_xMin, cs_yMin, cs_xMax, cs_yMax, &cs_stamp)
    refArea_result = float_ptr_to_numpy(refArea_c, cs_sxLen * cs_syLen)
    cdef int cs_x0 = cs_stamp.x0, cs_y0 = cs_stamp.y0
    cdef int cs_cx = cs_stamp.x, cs_cy = cs_stamp.y
    free(data_cs_c)
    free(refArea_c)

    from pyhotpants.numutils import cut_stamp_numpy
    py_refArea, py_x0, py_y0, py_cx, py_cy = cut_stamp_numpy(data_cs, cs_dxLen, cs_xMin, cs_yMin, cs_xMax, cs_yMax)

    cs_ok = np.array_equal(refArea_result, py_refArea) and cs_x0 == py_x0 and cs_y0 == py_y0 and cs_cx == py_cx and cs_cy == py_cy
    if not cs_ok:
        sys.stderr.write(f"cutStamp FAIL: ref_eq={np.array_equal(refArea_result, py_refArea)} x0=({cs_x0},{py_x0}) y0=({cs_y0},{py_y0}) cx=({cs_cx},{py_cx}) cy=({cs_cy},{py_cy})\n")
        ok = False
    else:
        sys.stderr.write("cutStamp PASS\n")
    sys.stderr.flush()

    return ok

def test_functions_batch2():
    import sys
    ok = True

    np.random.seed(47)
    cdef int nPixX = 100, nPixY = 100
    cdef int x0Reg = 10, y0Reg = 10
    cdef int gs3rPixX = 120, gs3rPixY = 120
    cdef int gs3umask = 0, gs3smask = 0xffff
    cdef int gs3maxiter = 10
    cdef float gs3statSig = 3.0

    data_gs3 = (np.random.randn(nPixY, nPixX).astype(np.float32) * 10 + 100).astype(np.float32)
    data_gs3[5, 5] = np.float32('nan')
    data_gs3[50, 50] = np.float32('nan')

    mRData_gs3 = np.zeros((gs3rPixY, gs3rPixX), dtype=np.int32)
    mRData_gs3[y0Reg + 3, x0Reg + 3] = 0x80
    mRData_gs3[y0Reg + 7, x0Reg + 7] = 0x100

    data_flat_gs3 = np.ascontiguousarray(data_gs3.ravel(), dtype=np.float32)
    mRData_c_gs3 = np.ascontiguousarray(mRData_gs3.ravel(), dtype=np.int32)
    mRData_py_gs3 = mRData_gs3.copy()

    cdef float *data_c_ptr = numpy_to_float_ptr(data_flat_gs3)
    cdef int *mRData_c_ptr = numpy_to_int_ptr(mRData_c_gs3)

    cdef double c_sum = 0, c_mean = 0, c_median = 0
    cdef double c_mode = 0, c_sd = 0, c_fwhm = 0, c_lfwhm = 0
    cdef int c_rc = getStampStats3(data_c_ptr, x0Reg, y0Reg, nPixX, nPixY,
                                    &c_sum, &c_mean, &c_median,
                                    &c_mode, &c_sd, &c_fwhm, &c_lfwhm,
                                    gs3umask, gs3smask, gs3maxiter,
                                    gs3rPixX, mRData_c_ptr, gs3statSig)

    from pyhotpants.numutils import get_stamp_stats3_numpy
    py_result = get_stamp_stats3_numpy(data_gs3, x0Reg, y0Reg, nPixX, nPixY,
                                        gs3umask, gs3smask, gs3maxiter,
                                        gs3rPixX, mRData_py_gs3, gs3statSig)

    py_rc = py_result['return_code']

    if c_rc != py_rc:
        sys.stderr.write(f"getStampStats3 return_code FAIL: C={c_rc}, py={py_rc}\n")
        ok = False
    else:
        sys.stderr.write(f"getStampStats3 return_code MATCH: {c_rc}\n")

    if c_rc == 0 and py_rc == 0:
        pairs = [
            ('sum', c_sum, py_result['sum']),
            ('mean', c_mean, py_result['mean']),
            ('median', c_median, py_result['median']),
            ('mode', c_mode, py_result['mode']),
            ('sd', c_sd, py_result['sd']),
            ('fwhm', c_fwhm, py_result['fwhm']),
            ('lfwhm', c_lfwhm, py_result['lfwhm']),
        ]
        for name, cval, pyval in pairs:
            if abs(cval - pyval) > 1e-12:
                sys.stderr.write(f"  {name} FAIL: C={cval}, py={pyval}, diff={abs(cval-pyval)}\n")
                ok = False
            else:
                sys.stderr.write(f"  {name} PASS: {cval} (diff={abs(cval-pyval):.2e})\n")

    mRData_c_result = int_ptr_to_numpy(mRData_c_ptr, gs3rPixX * gs3rPixY)
    mRData_py_result = mRData_py_gs3.ravel()
    if not np.array_equal(mRData_c_result, mRData_py_result):
        ndiff_m = np.sum(mRData_c_result != mRData_py_result)
        diff_idx_m = np.where(mRData_c_result != mRData_py_result)[0]
        sys.stderr.write(f"getStampStats3 mRData FAIL: {ndiff_m} diffs, first at flat idx {diff_idx_m[0]}: C={mRData_c_result[diff_idx_m[0]]}, py={mRData_py_result[diff_idx_m[0]]}\n")
        ok = False
    else:
        sys.stderr.write("getStampStats3 mRData PASS\n")

    free(data_c_ptr)
    free(mRData_c_ptr)
    sys.stderr.flush()

    return ok

def test_functions_batch3():
    import sys
    ok = True

    # ===== cutSStamp =====
    np.random.seed(50)
    cdef int css_rPixX = 60
    cdef int css_rPixY = 60
    cdef int css_fwKSStamp = 7, css_hwKSStamp = 3
    cdef float css_fillVal = -999.0
    cdef int css_nKSStamps = 5

    iData_css = np.random.randn(css_rPixX * css_rPixY).astype(np.float32) * 10 + 100
    mRData_css_orig = np.zeros(css_rPixX * css_rPixY, dtype=np.int32)
    mRData_css_orig[23 + 60 * 22] = 0x80

    cdef stamp_struct css_sc
    memset(&css_sc, 0, sizeof(stamp_struct))
    css_sc.x0 = 10; css_sc.y0 = 10
    css_sc.x = 15; css_sc.y = 15
    css_sc.nss = 3; css_sc.sscnt = 0
    css_sc.krefArea = <double*>calloc(css_fwKSStamp * css_fwKSStamp, sizeof(double))
    css_sc.xss = <int*>calloc(css_nKSStamps, sizeof(int))
    css_sc.yss = <int*>calloc(css_nKSStamps, sizeof(int))
    css_sc.xss[0] = 20; css_sc.yss[0] = 20
    css_sc.xss[1] = 22; css_sc.yss[1] = 22
    css_sc.xss[2] = 25; css_sc.yss[2] = 25

    cdef float *iData_css_c = numpy_to_float_ptr(iData_css)
    cdef int *mRData_css_c = numpy_to_int_ptr(mRData_css_orig)

    cdef int css_rc_c = cutSStamp(&css_sc, iData_css_c, css_fwKSStamp, css_hwKSStamp, css_fillVal, css_rPixX, mRData_css_c, 0)

    c_krefArea_css = double_ptr_to_numpy(css_sc.krefArea, css_fwKSStamp * css_fwKSStamp)
    cdef double c_sum_css = css_sc.sum

    free(iData_css_c); free(mRData_css_c)
    free(css_sc.krefArea); free(css_sc.xss); free(css_sc.yss)

    py_stamp_css = {
        'x0': 10, 'y0': 10, 'x': 15, 'y': 15,
        'nss': 3, 'sscnt': 0,
        'xss': np.array([20, 22, 25, 0, 0], dtype=np.int32),
        'yss': np.array([20, 22, 25, 0, 0], dtype=np.int32),
        'krefArea': np.zeros(css_fwKSStamp * css_fwKSStamp, dtype=np.float64),
        'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
        'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0,
        'chi2': 0.0, 'norm': 0.0, 'diff': 0.0,
    }

    from pyhotpants.numutils import cut_sstamp_numpy
    py_rc_css = cut_sstamp_numpy(py_stamp_css, iData_css, css_fwKSStamp, css_hwKSStamp, float(css_fillVal), css_rPixX, mRData_css_orig, 0)

    if css_rc_c != py_rc_css:
        sys.stderr.write(f"cutSStamp rc FAIL: C={css_rc_c}, py={py_rc_css}\n"); ok = False
    elif not np.array_equal(c_krefArea_css, py_stamp_css['krefArea']):
        ndiff = int(np.sum(c_krefArea_css != py_stamp_css['krefArea']))
        sys.stderr.write(f"cutSStamp krefArea FAIL: {ndiff} diffs\n"); ok = False
    elif abs(c_sum_css - py_stamp_css['sum']) > 0:
        sys.stderr.write(f"cutSStamp sum FAIL: C={c_sum_css}, py={py_stamp_css['sum']}\n"); ok = False
    else:
        sys.stderr.write("cutSStamp PASS\n")
    sys.stderr.flush()

    # ===== checkPsfCenter =====
    np.random.seed(51)
    cdef int cpc_rPixX = 50, cpc_rPixY = 50
    cdef int cpc_xLen = 40, cpc_yLen = 40
    cdef int cpc_sx0 = 5, cpc_sy0 = 5
    cdef int cpc_imax = 20, cpc_jmax = 20
    cdef int cpc_hwKS = 3
    cdef double cpc_hiThresh = 50000.0
    cdef float cpc_sky = 100.0
    cdef float cpc_invdsky = 0.1
    cdef int cpc_bbit = 0x3bf, cpc_bbit1 = 0x100
    cdef float cpc_kft = 3.0

    iData_cpc = np.random.randn(cpc_rPixX * cpc_rPixY).astype(np.float32) * 10 + 150
    mRData_cpc_c_arr = np.zeros(cpc_rPixX * cpc_rPixY, dtype=np.int32)
    mRData_cpc_py_arr = mRData_cpc_c_arr.copy()

    cdef float *iData_cpc_c = numpy_to_float_ptr(iData_cpc)
    cdef int *mRData_cpc_c = numpy_to_int_ptr(mRData_cpc_c_arr)

    cdef double cpc_rc = checkPsfCenter(iData_cpc_c, cpc_imax, cpc_jmax, cpc_xLen, cpc_yLen,
                                         cpc_sx0, cpc_sy0, cpc_hiThresh, cpc_sky, cpc_invdsky,
                                         0, 0, cpc_bbit, cpc_bbit1,
                                         cpc_rPixX, cpc_hwKS, mRData_cpc_c, cpc_kft)

    mRData_cpc_c_result = int_ptr_to_numpy(mRData_cpc_c, cpc_rPixX * cpc_rPixY)
    free(iData_cpc_c); free(mRData_cpc_c)

    from pyhotpants.numutils import check_psf_center_numpy
    cpc_rc_py = check_psf_center_numpy(iData_cpc, cpc_imax, cpc_jmax, cpc_xLen, cpc_yLen,
                                        cpc_sx0, cpc_sy0, cpc_hiThresh,
                                        float(cpc_sky), float(cpc_invdsky),
                                        0, 0, cpc_bbit, cpc_bbit1,
                                        cpc_rPixX, cpc_hwKS, mRData_cpc_py_arr, float(cpc_kft))

    if abs(cpc_rc - cpc_rc_py) > 0:
        sys.stderr.write(f"checkPsfCenter retval FAIL: C={cpc_rc}, py={cpc_rc_py}\n"); ok = False
    elif not np.array_equal(mRData_cpc_c_result, mRData_cpc_py_arr):
        ndiff = int(np.sum(mRData_cpc_c_result != mRData_cpc_py_arr))
        sys.stderr.write(f"checkPsfCenter mRData FAIL: {ndiff} diffs\n"); ok = False
    else:
        sys.stderr.write("checkPsfCenter PASS\n")
    sys.stderr.flush()

    # ===== getPsfCenters =====
    cdef int gpc_rPixX = 60, gpc_rPixY = 60
    cdef int gpc_xLen = 40, gpc_yLen = 40
    cdef int gpc_nKSStamps = 3, gpc_hwKS = 3
    cdef double gpc_hiThresh = 500.0
    cdef float gpc_kft = 3.0
    cdef int gpc_bbit1 = 0x100, gpc_bbit2 = 0x200

    iData_gpc = np.full(gpc_rPixX * gpc_rPixY, np.float32(100.0), dtype=np.float32)
    iData_gpc[20 + gpc_rPixX * 20] = np.float32(350.0)
    iData_gpc[30 + gpc_rPixX * 15] = np.float32(350.0)

    mRData_gpc_c_arr = np.zeros(gpc_rPixX * gpc_rPixY, dtype=np.int32)
    mRData_gpc_py_arr = mRData_gpc_c_arr.copy()

    cdef stamp_struct gpc_sc
    memset(&gpc_sc, 0, sizeof(stamp_struct))
    gpc_sc.x0 = 5; gpc_sc.y0 = 5
    gpc_sc.x = 25; gpc_sc.y = 25
    gpc_sc.mode = 100.0; gpc_sc.fwhm = 10.0
    gpc_sc.nss = 0; gpc_sc.sscnt = 0
    gpc_sc.xss = <int*>calloc(gpc_nKSStamps, sizeof(int))
    gpc_sc.yss = <int*>calloc(gpc_nKSStamps, sizeof(int))

    cdef float *iData_gpc_c = numpy_to_float_ptr(iData_gpc)
    cdef int *mRData_gpc_c = numpy_to_int_ptr(mRData_gpc_c_arr)

    cdef int gpc_rc_c = getPsfCenters(&gpc_sc, iData_gpc_c, gpc_xLen, gpc_yLen,
                                       gpc_hiThresh, gpc_bbit1, gpc_bbit2,
                                       gpc_nKSStamps, gpc_hwKS,
                                       gpc_rPixX, mRData_gpc_c, gpc_kft, 0)

    cdef int c_nss_gpc = gpc_sc.nss
    cdef int j_gpc
    c_xss_gpc = np.array([gpc_sc.xss[j_gpc] for j_gpc in range(gpc_nKSStamps)], dtype=np.int32)
    c_yss_gpc = np.array([gpc_sc.yss[j_gpc] for j_gpc in range(gpc_nKSStamps)], dtype=np.int32)
    mRData_gpc_c_result = int_ptr_to_numpy(mRData_gpc_c, gpc_rPixX * gpc_rPixY)
    free(iData_gpc_c); free(mRData_gpc_c)
    free(gpc_sc.xss); free(gpc_sc.yss)

    py_stamp_gpc = {
        'x0': 5, 'y0': 5, 'x': 25, 'y': 25,
        'nss': 0, 'sscnt': 0,
        'mode': 100.0, 'fwhm': 10.0,
        'xss': np.zeros(gpc_nKSStamps, dtype=np.int32),
        'yss': np.zeros(gpc_nKSStamps, dtype=np.int32),
        'krefArea': np.zeros(1, dtype=np.float64),
        'sum': 0.0, 'mean': 0.0, 'median': 0.0,
        'sd': 0.0, 'lfwhm': 0.0,
        'chi2': 0.0, 'norm': 0.0, 'diff': 0.0,
    }

    from pyhotpants.numutils import get_psf_centers_numpy
    gpc_rc_py = get_psf_centers_numpy(py_stamp_gpc, iData_gpc, gpc_xLen, gpc_yLen,
                                       gpc_hiThresh, gpc_bbit1, gpc_bbit2,
                                       gpc_nKSStamps, gpc_hwKS,
                                       gpc_rPixX, mRData_gpc_py_arr, float(gpc_kft), 0)

    gpc_ok = True
    if gpc_rc_c != gpc_rc_py:
        sys.stderr.write(f"getPsfCenters rc FAIL: C={gpc_rc_c}, py={gpc_rc_py}\n"); gpc_ok = False
    if c_nss_gpc != py_stamp_gpc['nss']:
        sys.stderr.write(f"getPsfCenters nss FAIL: C={c_nss_gpc}, py={py_stamp_gpc['nss']}\n"); gpc_ok = False
    if not np.array_equal(c_xss_gpc, py_stamp_gpc['xss']):
        sys.stderr.write(f"getPsfCenters xss FAIL: C={c_xss_gpc}, py={py_stamp_gpc['xss']}\n"); gpc_ok = False
    if not np.array_equal(c_yss_gpc, py_stamp_gpc['yss']):
        sys.stderr.write(f"getPsfCenters yss FAIL: C={c_yss_gpc}, py={py_stamp_gpc['yss']}\n"); gpc_ok = False
    if not np.array_equal(mRData_gpc_c_result, mRData_gpc_py_arr):
        ndiff = int(np.sum(mRData_gpc_c_result != mRData_gpc_py_arr))
        sys.stderr.write(f"getPsfCenters mRData FAIL: {ndiff} diffs\n"); gpc_ok = False
    if gpc_ok:
        sys.stderr.write("getPsfCenters PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    sys.stderr.write("buildStamps -> see test_build_stamps()\n")
    sys.stderr.write("buildSigMask SKIP (not found in functions.c)\n")
    sys.stderr.write("stampStats SKIP (not found in functions.c)\n")
    sys.stderr.flush()

    return ok

def test_build_stamps():
    """buildStamps C vs numpy 影子测试"""
    import sys
    np.random.seed(123)
    ok = True

    cdef int rPixX = 200, rPixY = 200
    cdef int totalPix = rPixX * rPixY
    bg = np.float32(100.0)
    img = np.full(totalPix, bg, dtype=np.float32)

    yy, xx = np.mgrid[0:rPixY, 0:rPixX]
    yyf = yy.ravel().astype(np.float32)
    xxf = xx.ravel().astype(np.float32)
    for sx, sy, flux in [(50, 50, 5000), (120, 80, 3000), (30, 150, 8000), (170, 40, 4000)]:
        img += (flux * np.exp(-((xxf - sx)**2 + (yyf - sy)**2) / (2.0 * 3.0**2))).astype(np.float32)
    img += np.random.randn(totalPix).astype(np.float32) * 10

    tRData_py = img.copy()
    iRData_py = img.copy() + np.random.randn(totalPix).astype(np.float32) * 5
    mRData_py = np.zeros(totalPix, dtype=np.int32)

    cdef int nStamps = 4
    cdef int bgOrder = 2
    cdef int nCompKer = 6
    cdef int fwKSStamp = 15
    cdef int nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int nC = nCompKer + nBGVectors
    cdef int nKSStamps = 10
    cdef int hwKSStamp = fwKSStamp // 2
    cdef int fwStamp = 50

    cdef int sXMin = 25, sXMax = 74
    cdef int sYMin = 25, sYMax = 74
    cdef int rXBMin = 0, rYBMin = 0

    cdef float tUKThresh = 50000.0, iUKThresh = 50000.0
    cdef float kerFitThresh = 20.0, statSig = 3.0
    cdef int findSSC = 1
    cdef int verbose = 0

    cdef float *tRData_c = numpy_to_float_ptr(tRData_py)
    cdef float *iRData_c = numpy_to_float_ptr(iRData_py)
    cdef int *mRData_c = numpy_to_int_ptr(mRData_py)

    cdef stamp_struct *ctStamps_c = <stamp_struct*>calloc(nStamps, sizeof(stamp_struct))
    allocateStamps(ctStamps_c, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)
    cdef stamp_struct *ciStamps_c = <stamp_struct*>calloc(nStamps, sizeof(stamp_struct))
    allocateStamps(ciStamps_c, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)

    cdef int niS_c = 0, ntS_c = 0
    cdef char *lfc = "b"

    buildStamps(sXMin, sXMax, sYMin, sYMax, &niS_c, &ntS_c, findSSC,
                rXBMin, rYBMin, ciStamps_c, ctStamps_c,
                iRData_c, tRData_c, 0, 0,
                verbose, lfc, rPixX, rPixY,
                tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps,
                kerFitThresh, mRData_c, statSig)

    c_ct = stamp_c_to_dict(&ctStamps_c[0], nCompKer, nBGVectors, fwKSStamp, nC, nKSStamps)
    c_ci = stamp_c_to_dict(&ciStamps_c[0], nCompKer, nBGVectors, fwKSStamp, nC, nKSStamps)
    mRData_c_result = int_ptr_to_numpy(mRData_c, totalPix)

    mRData_py_np = mRData_py.copy()

    py_ctStamps = []
    py_ciStamps = []
    cdef int sidx
    for sidx in range(nStamps):
        py_ctStamps.append({
            'x0': 0, 'y0': 0, 'x': 0, 'y': 0,
            'nx': 0, 'ny': 0,
            'nss': 0, 'sscnt': 0,
            'xss': np.zeros(nKSStamps, dtype=np.int32),
            'yss': np.zeros(nKSStamps, dtype=np.int32),
            'krefArea': np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64),
            'vectors': np.zeros((nCompKer + nBGVectors, fwKSStamp * fwKSStamp), dtype=np.float64),
            'mat': np.zeros((nC, nC), dtype=np.float64),
            'scprod': np.zeros(nC, dtype=np.float64),
            'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
            'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0,
            'chi2': 0.0, 'norm': 0.0, 'diff': 0.0,
        })
        py_ciStamps.append({
            'x0': 0, 'y0': 0, 'x': 0, 'y': 0,
            'nx': 0, 'ny': 0,
            'nss': 0, 'sscnt': 0,
            'xss': np.zeros(nKSStamps, dtype=np.int32),
            'yss': np.zeros(nKSStamps, dtype=np.int32),
            'krefArea': np.zeros(fwKSStamp * fwKSStamp, dtype=np.float64),
            'vectors': np.zeros((nCompKer + nBGVectors, fwKSStamp * fwKSStamp), dtype=np.float64),
            'mat': np.zeros((nC, nC), dtype=np.float64),
            'scprod': np.zeros(nC, dtype=np.float64),
            'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'mode': 0.0,
            'sd': 0.0, 'fwhm': 0.0, 'lfwhm': 0.0,
            'chi2': 0.0, 'norm': 0.0, 'diff': 0.0,
        })

    from pyhotpants.numutils import build_stamps_numpy
    build_stamps_numpy(sXMin, sXMax, sYMin, sYMax, 0, 0,
                       findSSC, rXBMin, rYBMin, py_ciStamps, py_ctStamps,
                       iRData_py, tRData_py, 0, 0,
                       verbose, "b", rPixX, rPixY,
                       tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps,
                       kerFitThresh, mRData_py_np, statSig)

    py_ct = py_ctStamps[0]
    py_ci = py_ciStamps[0]

    for field in ['nss', 'sscnt', 'x0', 'y0', 'x', 'y']:
        if c_ct[field] != py_ct[field]:
            sys.stderr.write(f"buildStamps ctStamp.{field} FAIL: C={c_ct[field]}, py={py_ct[field]}\n")
            ok = False
        if c_ci[field] != py_ci[field]:
            sys.stderr.write(f"buildStamps ciStamp.{field} FAIL: C={c_ci[field]}, py={py_ci[field]}\n")
            ok = False

    for field in ['sum', 'mean', 'median', 'mode', 'sd', 'fwhm', 'lfwhm']:
        if abs(c_ct[field] - py_ct[field]) > 1e-4:
            sys.stderr.write(f"buildStamps ctStamp.{field} FAIL: C={c_ct[field]:.6f}, py={py_ct[field]:.6f}\n")
            ok = False
        if abs(c_ci[field] - py_ci[field]) > 1e-4:
            sys.stderr.write(f"buildStamps ciStamp.{field} FAIL: C={c_ci[field]:.6f}, py={py_ci[field]:.6f}\n")
            ok = False

    for prefix, cs, ps in [("ct", c_ct, py_ct), ("ci", c_ci, py_ci)]:
        nssVal = cs['nss']
        if nssVal > 0:
            if not np.array_equal(cs['xss'][:nssVal], ps['xss'][:nssVal]):
                sys.stderr.write(f"buildStamps {prefix}Stamp.xss FAIL: C={cs['xss'][:nssVal]}, py={ps['xss'][:nssVal]}\n")
                ok = False
            if not np.array_equal(cs['yss'][:nssVal], ps['yss'][:nssVal]):
                sys.stderr.write(f"buildStamps {prefix}Stamp.yss FAIL: C={cs['yss'][:nssVal]}, py={ps['yss'][:nssVal]}\n")
                ok = False

    if not np.array_equal(mRData_c_result, mRData_py_np):
        ndiff = int(np.sum(mRData_c_result != mRData_py_np))
        sys.stderr.write(f"buildStamps mRData FAIL: {ndiff} diffs\n")
        ok = False

    if ok:
        sys.stderr.write("buildStamps PASS\n")
    sys.stderr.flush()

    freeStampMem(ctStamps_c, nStamps, nCompKer, nBGVectors, nC)
    free(ctStamps_c)
    freeStampMem(ciStamps_c, nStamps, nCompKer, nBGVectors, nC)
    free(ciStamps_c)
    free(tRData_c)
    free(iRData_c)
    free(mRData_c)

    return ok

def test_alard_batch1():
    import sys
    from pyhotpants.numutils import (get_background_numpy, get_final_stamp_sig_numpy,
                                      make_kernel_numpy, lubksb_numpy,
                                      kernel_vector_pca_numpy)
    ok = True
    np.random.seed(200)

    # ===== get_background =====
    cdef int gb_nCompKer = 4, gb_kerOrder = 2, gb_bgOrder = 2
    cdef int gb_rPixX = 200, gb_rPixY = 200
    cdef int gb_nBGVectors = (gb_bgOrder + 1) * (gb_bgOrder + 2) // 2
    cdef int gb_ncompBG = (gb_nCompKer - 1) * (((gb_kerOrder + 1) * (gb_kerOrder + 2)) // 2) + 1
    cdef int gb_totalSol = gb_ncompBG + 1 + gb_nBGVectors
    gb_kerSol_np = np.random.randn(gb_totalSol).astype(np.float64)
    cdef int gb_xi = 100, gb_yi = 80

    cdef double *gb_kerSol_c = numpy_to_double_ptr(gb_kerSol_np)
    cdef double gb_c_result = get_background(gb_xi, gb_yi, gb_kerSol_c, gb_nCompKer, gb_kerOrder, gb_bgOrder, gb_rPixX, gb_rPixY)
    free(gb_kerSol_c)

    gb_py_result = get_background_numpy(gb_xi, gb_yi, gb_kerSol_np, gb_nCompKer, gb_kerOrder, gb_bgOrder, gb_rPixX, gb_rPixY)

    if abs(gb_c_result - gb_py_result) > 0:
        sys.stderr.write(f"get_background FAIL: C={gb_c_result}, py={gb_py_result}, diff={abs(gb_c_result - gb_py_result)}\n")
        ok = False
    else:
        sys.stderr.write("get_background PASS\n")
    sys.stderr.flush()

    # ===== getFinalStampSig =====
    cdef int fss_fwKSStamp = 11, fss_hwKSStamp = 5
    cdef int fss_rPixX = 100
    cdef int fss_xCenter = 40, fss_yCenter = 30
    cdef int fss_totalPix = fss_rPixX * 50
    cdef int fss_nKSStamps = 3

    fss_imDiff = np.random.randn(fss_totalPix).astype(np.float32) * 10.0
    fss_imNoise = (np.abs(np.random.randn(fss_totalPix).astype(np.float32)) + 1.0).astype(np.float32)
    fss_mRData = np.zeros(fss_totalPix, dtype=np.int32)
    fss_mRData[fss_xCenter + 2 + fss_rPixX * fss_yCenter] = 0x80

    cdef stamp_struct fss_stamp
    memset(&fss_stamp, 0, sizeof(stamp_struct))
    fss_stamp.sscnt = 0
    fss_stamp.xss = <int*>calloc(fss_nKSStamps, sizeof(int))
    fss_stamp.yss = <int*>calloc(fss_nKSStamps, sizeof(int))
    fss_stamp.xss[0] = fss_xCenter
    fss_stamp.yss[0] = fss_yCenter

    cdef float *fss_imDiff_c = numpy_to_float_ptr(fss_imDiff)
    cdef float *fss_imNoise_c = numpy_to_float_ptr(fss_imNoise)
    cdef int *fss_mRData_c = numpy_to_int_ptr(fss_mRData)
    cdef double fss_c_sig = 0.0
    getFinalStampSig(&fss_stamp, fss_imDiff_c, fss_imNoise_c, &fss_c_sig, fss_fwKSStamp, fss_hwKSStamp, fss_rPixX, fss_mRData_c)
    free(fss_imDiff_c)
    free(fss_imNoise_c)
    free(fss_mRData_c)
    free(fss_stamp.xss)
    free(fss_stamp.yss)

    fss_stamp_dict = {
        'xss': np.array([fss_xCenter, 0, 0], dtype=np.int32),
        'yss': np.array([fss_yCenter, 0, 0], dtype=np.int32),
        'sscnt': 0,
    }
    fss_py_sig = get_final_stamp_sig_numpy(fss_stamp_dict, fss_imDiff, fss_imNoise, fss_fwKSStamp, fss_hwKSStamp, fss_rPixX, fss_mRData)

    if abs(fss_c_sig - fss_py_sig) > 0:
        sys.stderr.write(f"getFinalStampSig FAIL: C={fss_c_sig}, py={fss_py_sig}, diff={abs(fss_c_sig - fss_py_sig)}\n")
        ok = False
    else:
        sys.stderr.write("getFinalStampSig PASS\n")
    sys.stderr.flush()

    # ===== make_kernel =====
    cdef int mk_nCompKer = 3, mk_kerOrder = 2, mk_fwKernel = 11
    cdef int mk_rPixX = 200, mk_rPixY = 200
    cdef int mk_xi = 100, mk_yi = 80
    cdef int mk_nTerms = ((mk_kerOrder + 1) * (mk_kerOrder + 2)) // 2
    cdef int mk_totalKerSol = 2 + (mk_nCompKer - 1) * mk_nTerms
    mk_kerSol_np = np.random.randn(mk_totalKerSol).astype(np.float64)
    mk_kv_np = np.random.randn(mk_nCompKer, mk_fwKernel * mk_fwKernel).astype(np.float64)

    cdef double *mk_kerSol_c = numpy_to_double_ptr(mk_kerSol_np)
    cdef double **mk_kv_c = numpy2d_to_double_pp(mk_kv_np)
    cdef double *mk_kcoeff_c = <double*>calloc(mk_nCompKer, sizeof(double))
    cdef double *mk_ker_c = <double*>calloc(mk_fwKernel * mk_fwKernel, sizeof(double))
    cdef double mk_c_sum = make_kernel(mk_xi, mk_yi, mk_kerSol_c, mk_rPixX, mk_rPixY, mk_nCompKer, mk_kerOrder, mk_fwKernel, mk_kv_c, mk_kcoeff_c, mk_ker_c)

    mk_kcoeff_c_result = double_ptr_to_numpy(mk_kcoeff_c, mk_nCompKer)
    mk_ker_c_result = double_ptr_to_numpy(mk_ker_c, mk_fwKernel * mk_fwKernel)
    free(mk_kerSol_c)
    free_double_pp(mk_kv_c, mk_nCompKer)
    free(mk_kcoeff_c)
    free(mk_ker_c)

    mk_kcoeff_py = np.zeros(mk_nCompKer, dtype=np.float64)
    mk_ker_py = np.zeros(mk_fwKernel * mk_fwKernel, dtype=np.float64)
    mk_py_sum = make_kernel_numpy(mk_xi, mk_yi, mk_kerSol_np, mk_rPixX, mk_rPixY, mk_nCompKer, mk_kerOrder, mk_fwKernel, mk_kv_np, mk_kcoeff_py, mk_ker_py)

    mk_ok = True
    if abs(mk_c_sum - mk_py_sum) > 1e-12:
        sys.stderr.write(f"make_kernel sumKernel FAIL: C={mk_c_sum}, py={mk_py_sum}\n")
        mk_ok = False
    if not np.allclose(mk_kcoeff_c_result, mk_kcoeff_py, atol=1e-12, rtol=0):
        sys.stderr.write(f"make_kernel kernel_coeffs FAIL: max diff={np.max(np.abs(mk_kcoeff_c_result - mk_kcoeff_py))}\n")
        mk_ok = False
    if not np.allclose(mk_ker_c_result, mk_ker_py, atol=1e-12, rtol=0):
        sys.stderr.write(f"make_kernel kernel FAIL: max diff={np.max(np.abs(mk_ker_c_result - mk_ker_py))}\n")
        mk_ok = False
    if mk_ok:
        sys.stderr.write("make_kernel PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== lubksb =====
    cdef int lu_n = 6
    lu_a_np = np.zeros((lu_n + 1, lu_n + 1), dtype=np.float64)
    lu_a_np[1:lu_n+1, 1:lu_n+1] = np.random.randn(lu_n, lu_n) * 2.0 + np.eye(lu_n) * 10.0
    lu_b_np = np.zeros(lu_n + 1, dtype=np.float64)
    lu_b_np[1:lu_n+1] = np.random.randn(lu_n)

    cdef double **lu_a_c = numpy2d_to_double_pp(lu_a_np)
    cdef int *lu_indx_c = <int*>calloc(lu_n + 1, sizeof(int))
    cdef double lu_d = 0.0
    cdef double *lu_b_c = NULL
    cdef int lu_rc = ludcmp(lu_a_c, lu_n, lu_indx_c, &lu_d)

    if lu_rc != 0:
        sys.stderr.write("lubksb SKIP: ludcmp returned singular\n")
    else:
        lu_a_lu_np = double_pp_to_numpy2d(lu_a_c, lu_n + 1, lu_n + 1)
        lu_indx_np = int_ptr_to_numpy(lu_indx_c, lu_n + 1)

        lu_b_c = numpy_to_double_ptr(lu_b_np)
        lubksb(lu_a_c, lu_n, lu_indx_c, lu_b_c)
        lu_b_c_result = double_ptr_to_numpy(lu_b_c, lu_n + 1)
        free(lu_b_c)

        lu_b_py = lu_b_np.copy()
        lubksb_numpy(lu_a_lu_np, lu_n, lu_indx_np, lu_b_py)

        lu_maxdiff = np.max(np.abs(lu_b_c_result[1:lu_n+1] - lu_b_py[1:lu_n+1]))
        if lu_maxdiff > 1e-5:
            sys.stderr.write(f"lubksb FAIL: max diff={lu_maxdiff}\n")
            ok = False
        else:
            sys.stderr.write(f"lubksb PASS (max diff={lu_maxdiff:.2e})\n")

    free_double_pp(lu_a_c, lu_n + 1)
    free(lu_indx_c)
    sys.stderr.flush()

    # ===== kernel_vector_PCA =====
    cdef int pca_fwKernel = 7
    cdef int pca_nComp = 3
    pca_PCA_np = np.random.randn(pca_nComp, pca_fwKernel * pca_fwKernel).astype(np.float32)
    pca_kv_np = np.random.randn(pca_nComp, pca_fwKernel * pca_fwKernel).astype(np.float64)

    cdef float **pca_PCA_c = numpy2d_to_float_pp(pca_PCA_np)
    cdef double **pca_kv_c = numpy2d_to_double_pp(pca_kv_np)

    cdef int pca_ren_c0 = 0
    cdef double *pca_vec_c0 = kernel_vector_PCA(0, 0, 0, 0, &pca_ren_c0, pca_fwKernel, pca_PCA_c, pca_kv_c)
    pca_vec_c0_np = double_ptr_to_numpy(pca_vec_c0, pca_fwKernel * pca_fwKernel)
    free(pca_vec_c0)

    pca_vec_py0, pca_ren_py0 = kernel_vector_pca_numpy(0, 0, 0, 0, pca_fwKernel, pca_PCA_np, pca_kv_np)

    pca_ok = True
    if pca_ren_c0 != pca_ren_py0:
        sys.stderr.write(f"kernel_vector_PCA n=0 ren FAIL: C={pca_ren_c0}, py={pca_ren_py0}\n")
        pca_ok = False
    if not np.array_equal(pca_vec_c0_np, pca_vec_py0):
        sys.stderr.write(f"kernel_vector_PCA n=0 vector FAIL: max diff={np.max(np.abs(pca_vec_c0_np - pca_vec_py0))}\n")
        pca_ok = False

    cdef int pca_ren_c1 = 0
    cdef double *pca_vec_c1 = kernel_vector_PCA(1, 0, 0, 0, &pca_ren_c1, pca_fwKernel, pca_PCA_c, pca_kv_c)
    pca_vec_c1_np = double_ptr_to_numpy(pca_vec_c1, pca_fwKernel * pca_fwKernel)
    free(pca_vec_c1)

    pca_vec_py1, pca_ren_py1 = kernel_vector_pca_numpy(1, 0, 0, 0, pca_fwKernel, pca_PCA_np, pca_kv_np)

    if pca_ren_c1 != pca_ren_py1:
        sys.stderr.write(f"kernel_vector_PCA n=1 ren FAIL: C={pca_ren_c1}, py={pca_ren_py1}\n")
        pca_ok = False
    if not np.array_equal(pca_vec_c1_np, pca_vec_py1):
        sys.stderr.write(f"kernel_vector_PCA n=1 vector FAIL: max diff={np.max(np.abs(pca_vec_c1_np - pca_vec_py1))}\n")
        pca_ok = False

    free_float_pp(pca_PCA_c, pca_nComp)
    free_double_pp(pca_kv_c, pca_nComp)

    if pca_ok:
        sys.stderr.write("kernel_vector_PCA PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    return ok

def test_alard_batch2():
    import sys
    from pyhotpants.numutils import (kernel_vector_numpy, get_kernel_vec_numpy,
                                      ludcmp_numpy as ludcmp_numpy_fn,
                                      xy_conv_stamp_numpy as xy_conv_stamp_numpy_fn,
                                      xy_conv_stamp_pca_numpy as xy_conv_stamp_pca_numpy_fn,
                                      build_matrix0_numpy, build_scprod0_numpy,
                                      make_model_numpy)
    ok = True
    np.random.seed(300)

    cdef int fwKernel = 7, hwKernel = 3
    cdef int fwKSStamp = 7, hwKSStamp = 3
    cdef int b2ngauss = 1
    cdef int kerOrder = 1, bgOrder = 1
    cdef int nKSStamps = 3
    cdef int b2rPixX = 30, b2rPixY = 30

    deg_fixe_np = np.array([2], dtype=np.int32)
    sigma_gauss_np = np.array([0.5], dtype=np.float32)

    cdef int b2nCompKer = 0
    cdef int ig_tmp
    for ig_tmp in range(b2ngauss):
        d_tmp = int(deg_fixe_np[ig_tmp])
        b2nCompKer += (d_tmp + 1) * (d_tmp + 2) // 2
    cdef int b2nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int b2nVec = b2nCompKer + b2nBGVectors
    cdef int b2nC = b2nCompKer + b2nBGVectors + 1
    cdef int fwSq = fwKSStamp * fwKSStamp
    cdef int kv_total = b2nCompKer * fwKernel

    # ===== 1. kernel_vector =====
    cdef float *sg_c = numpy_to_float_ptr(sigma_gauss_np)
    cdef double *fx_c = <double*>calloc(kv_total, sizeof(double))
    cdef double *fy_c = <double*>calloc(kv_total, sizeof(double))
    cdef double **kvec_c = <double**>calloc(b2nCompKer, sizeof(double*))
    cdef int ren_c = 0

    fx_py = np.zeros(kv_total, dtype=np.float64)
    fy_py = np.zeros(kv_total, dtype=np.float64)
    kvec_py_list = []

    cdef double *vec0_c = kernel_vector(0, 0, 0, 0, &ren_c, 0, fwKernel, hwKernel, sg_c, fx_c, fy_c, kvec_c, NULL)
    kvec_c[0] = vec0_c
    vec0_c_np = double_ptr_to_numpy(vec0_c, fwKernel * fwKernel)
    fx_c0 = double_ptr_to_numpy(fx_c, kv_total)
    fy_c0 = double_ptr_to_numpy(fy_c, kv_total)

    vec0_py, ren_py0 = kernel_vector_numpy(0, 0, 0, 0, 0, fwKernel, hwKernel, sigma_gauss_np, fx_py, fy_py, kvec_py_list, None)
    kvec_py_list.append(vec0_py)

    kv_ok = True
    if ren_c != ren_py0:
        sys.stderr.write(f"kernel_vector n=0 ren FAIL: C={ren_c}, py={ren_py0}\n"); kv_ok = False
    if not np.allclose(vec0_c_np, vec0_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=0 vector FAIL: max diff={np.max(np.abs(vec0_c_np - vec0_py))}\n"); kv_ok = False
    if not np.allclose(fx_c0, fx_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=0 filter_x FAIL\n"); kv_ok = False
    if not np.allclose(fy_c0, fy_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=0 filter_y FAIL\n"); kv_ok = False

    cdef int ren_c1 = 0
    cdef double *vec1_c = kernel_vector(1, 1, 0, 0, &ren_c1, 0, fwKernel, hwKernel, sg_c, fx_c, fy_c, kvec_c, NULL)
    kvec_c[1] = vec1_c
    vec1_c_np = double_ptr_to_numpy(vec1_c, fwKernel * fwKernel)
    fx_c1 = double_ptr_to_numpy(fx_c, kv_total)
    fy_c1 = double_ptr_to_numpy(fy_c, kv_total)

    vec1_py, ren_py1 = kernel_vector_numpy(1, 1, 0, 0, 0, fwKernel, hwKernel, sigma_gauss_np, fx_py, fy_py, kvec_py_list, None)
    kvec_py_list.append(vec1_py)

    if ren_c1 != ren_py1:
        sys.stderr.write(f"kernel_vector n=1 ren FAIL: C={ren_c1}, py={ren_py1}\n"); kv_ok = False
    if not np.allclose(vec1_c_np, vec1_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=1 vector FAIL: max diff={np.max(np.abs(vec1_c_np - vec1_py))}\n"); kv_ok = False
    if not np.allclose(fx_c1, fx_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=1 filter_x FAIL\n"); kv_ok = False
    if not np.allclose(fy_c1, fy_py, atol=1e-12):
        sys.stderr.write(f"kernel_vector n=1 filter_y FAIL\n"); kv_ok = False

    cdef int kvi
    for kvi in range(b2nCompKer):
        if kvec_c[kvi] != NULL:
            free(kvec_c[kvi])
    free(kvec_c)

    if kv_ok:
        sys.stderr.write("kernel_vector PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== 2. getKernelVec =====
    cdef double *gkv_fx_c = <double*>calloc(kv_total, sizeof(double))
    cdef double *gkv_fy_c = <double*>calloc(kv_total, sizeof(double))
    cdef double **gkv_kvec_c = <double**>calloc(b2nCompKer, sizeof(double*))
    cdef int *deg_fixe_c = numpy_to_int_ptr(deg_fixe_np)

    getKernelVec(b2ngauss, deg_fixe_c, gkv_kvec_c, 0, fwKernel, hwKernel, sg_c, gkv_fx_c, gkv_fy_c, NULL)

    gkv_fx_py = np.zeros(kv_total, dtype=np.float64)
    gkv_fy_py = np.zeros(kv_total, dtype=np.float64)
    gkv_kvec_py = get_kernel_vec_numpy(b2ngauss, deg_fixe_np, 0, fwKernel, hwKernel, sigma_gauss_np, gkv_fx_py, gkv_fy_py, None)

    gkv_ok = True
    gkv_fx_c_np = double_ptr_to_numpy(gkv_fx_c, kv_total)
    gkv_fy_c_np = double_ptr_to_numpy(gkv_fy_c, kv_total)

    if not np.allclose(gkv_fx_c_np, gkv_fx_py, atol=1e-12):
        sys.stderr.write(f"getKernelVec filter_x FAIL: max diff={np.max(np.abs(gkv_fx_c_np - gkv_fx_py))}\n"); gkv_ok = False
    if not np.allclose(gkv_fy_c_np, gkv_fy_py, atol=1e-12):
        sys.stderr.write(f"getKernelVec filter_y FAIL: max diff={np.max(np.abs(gkv_fy_c_np - gkv_fy_py))}\n"); gkv_ok = False

    for kvi in range(b2nCompKer):
        c_vec_gkv = double_ptr_to_numpy(gkv_kvec_c[kvi], fwKernel * fwKernel)
        if not np.allclose(c_vec_gkv, gkv_kvec_py[kvi], atol=1e-12):
            sys.stderr.write(f"getKernelVec vec[{kvi}] FAIL: max diff={np.max(np.abs(c_vec_gkv - gkv_kvec_py[kvi]))}\n")
            gkv_ok = False

    if gkv_ok:
        sys.stderr.write("getKernelVec PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== 3. ludcmp =====
    cdef int lu_n = 8
    lu_a_np = np.zeros((lu_n + 1, lu_n + 1), dtype=np.float64)
    lu_a_np[1:lu_n+1, 1:lu_n+1] = np.random.randn(lu_n, lu_n) * 2.0 + np.eye(lu_n) * 10.0
    lu_a_c_np = lu_a_np.copy()
    lu_a_py_np = lu_a_np.copy()

    cdef double **lu_a_c = numpy2d_to_double_pp(lu_a_c_np)
    cdef int *lu_indx_c = <int*>calloc(lu_n + 1, sizeof(int))
    cdef double lu_d_c = 0.0
    cdef int lu_rc_c = ludcmp(lu_a_c, lu_n, lu_indx_c, &lu_d_c)

    lu_indx_py = np.zeros(lu_n + 1, dtype=np.int32)
    lu_rc_py, lu_d_py = ludcmp_numpy_fn(lu_a_py_np, lu_n, lu_indx_py)

    lu_ok = True
    lu_maxdiff_a = 0.0
    if lu_rc_c != lu_rc_py:
        sys.stderr.write(f"ludcmp rc FAIL: C={lu_rc_c}, py={lu_rc_py}\n"); lu_ok = False
    elif lu_rc_c == 0:
        lu_a_c_result = double_pp_to_numpy2d(lu_a_c, lu_n + 1, lu_n + 1)
        lu_indx_c_result = int_ptr_to_numpy(lu_indx_c, lu_n + 1)
        lu_maxdiff_a = float(np.max(np.abs(lu_a_c_result[1:lu_n+1, 1:lu_n+1] - lu_a_py_np[1:lu_n+1, 1:lu_n+1])))
        lu_indx_match = np.array_equal(lu_indx_c_result[1:lu_n+1], lu_indx_py[1:lu_n+1].astype(np.int32))
        if lu_maxdiff_a > 1e-5:
            sys.stderr.write(f"ludcmp matrix FAIL: max diff={lu_maxdiff_a}\n"); lu_ok = False
        if not lu_indx_match:
            sys.stderr.write(f"ludcmp indx FAIL: C={lu_indx_c_result[1:lu_n+1]}, py={lu_indx_py[1:lu_n+1]}\n"); lu_ok = False
        if abs(lu_d_c - lu_d_py) > 1e-5:
            sys.stderr.write(f"ludcmp d FAIL: C={lu_d_c}, py={lu_d_py}\n"); lu_ok = False

    free_double_pp(lu_a_c, lu_n + 1)
    free(lu_indx_c)

    if lu_ok:
        sys.stderr.write(f"ludcmp PASS (matrix max diff={lu_maxdiff_a:.2e})\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== 4. xy_conv_stamp =====
    cdef int xi_center = 15, yi_center = 15
    image_np = np.random.randn(b2rPixX * b2rPixY).astype(np.float32) * 10 + 100
    cdef float *xc_image_c = numpy_to_float_ptr(image_np)
    cdef int sub_width = fwKSStamp + fwKernel - 1
    cdef float *xc_temp_c = <float*>calloc(sub_width * fwKSStamp, sizeof(float))

    cdef stamp_struct xc_stamp_c
    memset(&xc_stamp_c, 0, sizeof(stamp_struct))
    allocateStamps(&xc_stamp_c, 1, bgOrder, b2nCompKer, fwKSStamp, b2nC, nKSStamps)
    xc_stamp_c.xss[0] = xi_center; xc_stamp_c.yss[0] = yi_center; xc_stamp_c.sscnt = 0

    xc_py = stamp_c_to_dict(&xc_stamp_c, b2nCompKer, b2nBGVectors, fwKSStamp, b2nC, nKSStamps)

    temp_py = np.zeros(sub_width * fwKSStamp, dtype=np.float32)

    xc_ok = True
    cdef int nvec_xc = 0
    cdef int idegx_xc, idegy_xc, dx_xc, dy_xc, ren_xc, ig_xc
    for ig_xc in range(b2ngauss):
        for idegx_xc in range(int(deg_fixe_np[ig_xc]) + 1):
            for idegy_xc in range(int(deg_fixe_np[ig_xc]) - idegx_xc + 1):
                dx_xc = (idegx_xc // 2) * 2 - idegx_xc
                dy_xc = (idegy_xc // 2) * 2 - idegy_xc
                ren_xc = 0
                if dx_xc == 0 and dy_xc == 0 and nvec_xc > 0:
                    ren_xc = 1

                xy_conv_stamp(&xc_stamp_c, xc_image_c, nvec_xc, ren_xc, 0, fwKSStamp, fwKernel, hwKSStamp, hwKernel, b2rPixX, gkv_fx_c, gkv_fy_c, xc_temp_c, NULL)
                xy_conv_stamp_numpy_fn(xc_py, image_np, nvec_xc, ren_xc, 0, fwKSStamp, fwKernel, hwKSStamp, hwKernel, b2rPixX, gkv_fx_c_np, gkv_fy_c_np, temp_py, None)

                c_xc_vec = np.array([xc_stamp_c.vectors[nvec_xc][k] for k in range(fwSq)], dtype=np.float64)
                py_xc_vec = xc_py['vectors'][nvec_xc]
                if not np.allclose(c_xc_vec, py_xc_vec, atol=1e-6):
                    sys.stderr.write(f"xy_conv_stamp vec[{nvec_xc}] FAIL: max diff={np.max(np.abs(c_xc_vec - py_xc_vec)):.2e}\n")
                    xc_ok = False
                nvec_xc += 1

    if xc_ok:
        sys.stderr.write("xy_conv_stamp PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== 5. xy_conv_stamp_PCA =====
    pca_np = np.random.randn(b2nCompKer, fwKernel * fwKernel).astype(np.float32)
    cdef float **pca_c = numpy2d_to_float_pp(pca_np)

    cdef stamp_struct xcp_stamp_c
    memset(&xcp_stamp_c, 0, sizeof(stamp_struct))
    allocateStamps(&xcp_stamp_c, 1, bgOrder, b2nCompKer, fwKSStamp, b2nC, nKSStamps)
    xcp_stamp_c.xss[0] = xi_center; xcp_stamp_c.yss[0] = yi_center; xcp_stamp_c.sscnt = 0

    xcp_py = stamp_c_to_dict(&xcp_stamp_c, b2nCompKer, b2nBGVectors, fwKSStamp, b2nC, nKSStamps)

    xcp_ok = True
    cdef int nvec_xcp = 0
    cdef int idegx_xcp, idegy_xcp, dx_xcp, dy_xcp, ren_xcp, ig_xcp
    for ig_xcp in range(b2ngauss):
        for idegx_xcp in range(int(deg_fixe_np[ig_xcp]) + 1):
            for idegy_xcp in range(int(deg_fixe_np[ig_xcp]) - idegx_xcp + 1):
                dx_xcp = (idegx_xcp // 2) * 2 - idegx_xcp
                dy_xcp = (idegy_xcp // 2) * 2 - idegy_xcp
                ren_xcp = 0
                if dx_xcp == 0 and dy_xcp == 0 and nvec_xcp > 0:
                    ren_xcp = 1

                xy_conv_stamp_PCA(&xcp_stamp_c, xc_image_c, nvec_xcp, ren_xcp, fwKSStamp, hwKSStamp, hwKernel, fwKernel, b2rPixX, pca_c)
                xy_conv_stamp_pca_numpy_fn(xcp_py, image_np, nvec_xcp, ren_xcp, fwKSStamp, hwKSStamp, hwKernel, fwKernel, b2rPixX, pca_np)

                c_xcp_vec = np.array([xcp_stamp_c.vectors[nvec_xcp][k] for k in range(fwSq)], dtype=np.float64)
                py_xcp_vec = xcp_py['vectors'][nvec_xcp]
                if not np.allclose(c_xcp_vec, py_xcp_vec, atol=1e-6):
                    sys.stderr.write(f"xy_conv_stamp_PCA vec[{nvec_xcp}] FAIL: max diff={np.max(np.abs(c_xcp_vec - py_xcp_vec)):.2e}\n")
                    xcp_ok = False
                nvec_xcp += 1

    free_float_pp(pca_c, b2nCompKer)

    if xcp_ok:
        sys.stderr.write("xy_conv_stamp_PCA PASS\n")
    else:
        ok = False
    sys.stderr.flush()

    # ===== 6. build_matrix0 =====
    cdef stamp_struct bm_stamp_c
    memset(&bm_stamp_c, 0, sizeof(stamp_struct))
    allocateStamps(&bm_stamp_c, 1, bgOrder, b2nCompKer, fwKSStamp, b2nC, nKSStamps)
    cdef int bm_j, bm_k
    for bm_j in range(b2nVec):
        for bm_k in range(fwSq):
            bm_stamp_c.vectors[bm_j][bm_k] = np.random.randn()

    bm_py = stamp_c_to_dict(&bm_stamp_c, b2nCompKer, b2nBGVectors, fwKSStamp, b2nC, nKSStamps)

    build_matrix0(&bm_stamp_c, b2nCompKer, kerOrder, bgOrder, fwKSStamp)
    build_matrix0_numpy(bm_py, b2nCompKer, kerOrder, bgOrder, fwKSStamp)

    bm_c_mat = np.zeros((b2nC, b2nC), dtype=np.float64)
    for bm_j in range(b2nC):
        for bm_k in range(b2nC):
            bm_c_mat[bm_j, bm_k] = bm_stamp_c.mat[bm_j][bm_k]

    bm_maxdiff = float(np.max(np.abs(bm_c_mat - bm_py['mat'])))
    if bm_maxdiff > 1e-10:
        sys.stderr.write(f"build_matrix0 FAIL: max diff={bm_maxdiff:.2e}\n")
        ok = False
    else:
        sys.stderr.write(f"build_matrix0 PASS (max diff={bm_maxdiff:.2e})\n")
    sys.stderr.flush()

    # ===== 7. build_scprod0 =====
    cdef stamp_struct bs0_stamp_c
    memset(&bs0_stamp_c, 0, sizeof(stamp_struct))
    allocateStamps(&bs0_stamp_c, 1, bgOrder, b2nCompKer, fwKSStamp, b2nC, nKSStamps)
    bs0_stamp_c.xss[0] = xi_center; bs0_stamp_c.yss[0] = yi_center; bs0_stamp_c.sscnt = 0
    cdef int bs0_j, bs0_k
    for bs0_j in range(b2nVec):
        for bs0_k in range(fwSq):
            bs0_stamp_c.vectors[bs0_j][bs0_k] = np.random.randn()

    bs0_py = stamp_c_to_dict(&bs0_stamp_c, b2nCompKer, b2nBGVectors, fwKSStamp, b2nC, nKSStamps)

    bs0_image_np = np.random.randn(b2rPixX * b2rPixY).astype(np.float32) * 10 + 100
    cdef float *bs0_image_c = numpy_to_float_ptr(bs0_image_np)

    build_scprod0(&bs0_stamp_c, bs0_image_c, b2nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, b2rPixX)
    build_scprod0_numpy(bs0_py, bs0_image_np, b2nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, b2rPixX)

    bs0_c_scprod = np.array([bs0_stamp_c.scprod[bs0_j] for bs0_j in range(b2nC)], dtype=np.float64)
    bs0_maxdiff = float(np.max(np.abs(bs0_c_scprod - bs0_py['scprod'])))
    if bs0_maxdiff > 1e-6:
        sys.stderr.write(f"build_scprod0 FAIL: max diff={bs0_maxdiff:.2e}\n")
        ok = False
    else:
        sys.stderr.write(f"build_scprod0 PASS (max diff={bs0_maxdiff:.2e})\n")
    free(bs0_image_c)
    sys.stderr.flush()

    # ===== 8. make_model =====
    cdef stamp_struct mm_stamp_c
    memset(&mm_stamp_c, 0, sizeof(stamp_struct))
    allocateStamps(&mm_stamp_c, 1, bgOrder, b2nCompKer, fwKSStamp, b2nC, nKSStamps)
    mm_stamp_c.xss[0] = xi_center; mm_stamp_c.yss[0] = yi_center; mm_stamp_c.sscnt = 0
    cdef int mm_j, mm_k
    for mm_j in range(b2nVec):
        for mm_k in range(fwSq):
            mm_stamp_c.vectors[mm_j][mm_k] = np.random.randn()

    mm_py = stamp_c_to_dict(&mm_stamp_c, b2nCompKer, b2nBGVectors, fwKSStamp, b2nC, nKSStamps)

    cdef int mm_nTerms = ((kerOrder + 1) * (kerOrder + 2)) // 2
    cdef int mm_totalKerSol = 2 + (b2nCompKer - 1) * mm_nTerms
    mm_kerSol_np = np.random.randn(mm_totalKerSol).astype(np.float64)
    cdef double *mm_kerSol_c = numpy_to_double_ptr(mm_kerSol_np)
    cdef float *mm_csModel_c = <float*>calloc(fwSq, sizeof(float))

    make_model(&mm_stamp_c, mm_kerSol_c, mm_csModel_c, b2rPixX, b2rPixY, b2nCompKer, kerOrder, fwKSStamp)
    mm_c_result = float_ptr_to_numpy(mm_csModel_c, fwSq)

    mm_py_result = make_model_numpy(mm_py, mm_kerSol_np, b2rPixX, b2rPixY, b2nCompKer, kerOrder, fwKSStamp)

    mm_maxdiff = float(np.max(np.abs(mm_c_result.astype(np.float64) - mm_py_result.astype(np.float64))))
    if mm_maxdiff > 1e-4:
        sys.stderr.write(f"make_model FAIL: max diff={mm_maxdiff:.2e}\n")
        ok = False
    else:
        sys.stderr.write(f"make_model PASS (max diff={mm_maxdiff:.2e})\n")

    free(mm_kerSol_c)
    free(mm_csModel_c)
    sys.stderr.flush()

    # ===== cleanup =====
    freeStampMem(&xc_stamp_c, 1, b2nCompKer, b2nBGVectors, b2nC)
    freeStampMem(&xcp_stamp_c, 1, b2nCompKer, b2nBGVectors, b2nC)
    freeStampMem(&bm_stamp_c, 1, b2nCompKer, b2nBGVectors, b2nC)
    freeStampMem(&bs0_stamp_c, 1, b2nCompKer, b2nBGVectors, b2nC)
    freeStampMem(&mm_stamp_c, 1, b2nCompKer, b2nBGVectors, b2nC)
    free(xc_image_c)
    free(xc_temp_c)
    for kvi in range(b2nCompKer):
        if gkv_kvec_c[kvi] != NULL:
            free(gkv_kvec_c[kvi])
    free(gkv_kvec_c)
    free(gkv_fx_c)
    free(gkv_fy_c)
    free(deg_fixe_c)
    free(sg_c)

    return ok

def test_alard_batch3a():
    import sys
    from pyhotpants.numutils import build_matrix_numpy, build_scprod_numpy
    ok = True

    seeds = [400, 401, 402]

    cdef int nCompKer = 4
    cdef int kerOrder = 2
    cdef int bgOrder = 1
    cdef int fwKSStamp = 7, hwKSStamp = 3
    cdef int rPixX = 200, rPixY = 200
    cdef int nS = 5
    cdef int nKSStamps = 3
    cdef int nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int nC = nCompKer + nBGVectors + 1
    cdef int nVec = nCompKer + nBGVectors
    cdef int fwSq = fwKSStamp * fwKSStamp
    cdef int ncomp1 = nCompKer - 1
    cdef int ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) // 2
    cdef int ncomp = ncomp1 * ncomp2
    cdef int nbg_vec = nBGVectors
    cdef int mat_size = ncomp + nbg_vec + 1
    cdef int totalPix = rPixX * rPixY
    cdef int solSize = ncomp + nbg_vec + 2

    cdef int si, sj, sk, seedIdx
    cdef stamp_struct *stamps_c
    cdef float *image_c
    cdef double **matrix_c
    cdef double **wxy_c
    cdef double *kernelSol_c

    for seedIdx in range(3):
        np.random.seed(seeds[seedIdx])

        stamps_c = <stamp_struct*>calloc(nS, sizeof(stamp_struct))
        for si in range(nS):
            allocateStamps(&stamps_c[si], 1, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)
            for sj in range(nKSStamps):
                stamps_c[si].xss[sj] = 30 + int(np.random.randint(0, 140))
                stamps_c[si].yss[sj] = 30 + int(np.random.randint(0, 140))
            stamps_c[si].nss = nKSStamps
            stamps_c[si].sscnt = 0
            for sj in range(nVec):
                for sk in range(fwSq):
                    stamps_c[si].vectors[sj][sk] = np.random.randn()
            for sj in range(nC):
                for sk in range(nC):
                    stamps_c[si].mat[sj][sk] = np.random.randn()
            for sj in range(nC):
                stamps_c[si].scprod[sj] = np.random.randn()

        stamps_c[1].sscnt = stamps_c[1].nss

        image_np = np.random.randn(totalPix).astype(np.float32) * 10 + 100
        image_c = numpy_to_float_ptr(image_np)

        matrix_c = <double**>malloc((mat_size + 1) * sizeof(double*))
        for si in range(mat_size + 1):
            matrix_c[si] = <double*>calloc(mat_size + 1, sizeof(double))

        wxy_c = <double**>malloc(nS * sizeof(double*))
        for si in range(nS):
            wxy_c[si] = <double*>calloc(ncomp2, sizeof(double))

        build_matrix(stamps_c, nS, matrix_c, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, 0, wxy_c)

        c_matrix = np.zeros((mat_size + 1, mat_size + 1), dtype=np.float64)
        for si in range(mat_size + 1):
            for sj in range(mat_size + 1):
                c_matrix[si, sj] = matrix_c[si][sj]

        c_wxy = np.zeros((nS, ncomp2), dtype=np.float64)
        for si in range(nS):
            for sj in range(ncomp2):
                c_wxy[si, sj] = wxy_c[si][sj]

        stamps_dicts = []
        for si in range(nS):
            stamps_dicts.append(stamp_c_to_dict(&stamps_c[si], nCompKer, nBGVectors, fwKSStamp, nC, nKSStamps))

        py_wxy = np.zeros((nS, ncomp2), dtype=np.float64)
        py_matrix = build_matrix_numpy(stamps_dicts, nS, nCompKer, kerOrder, bgOrder, fwKSStamp, rPixX, rPixY, 0, py_wxy)

        mat_maxdiff = float(np.max(np.abs(c_matrix - py_matrix)))
        sys.stderr.write(f"  seed={seeds[seedIdx]} build_matrix max_diff={mat_maxdiff:.2e}")
        if mat_maxdiff > 1e-10:
            sys.stderr.write(" FAIL\n")
            ok = False
        else:
            sys.stderr.write(" PASS\n")

        wxy_maxdiff = float(np.max(np.abs(c_wxy - py_wxy)))
        sys.stderr.write(f"  seed={seeds[seedIdx]} wxy max_diff={wxy_maxdiff:.2e}")
        if wxy_maxdiff > 1e-10:
            sys.stderr.write(" FAIL\n")
            ok = False
        else:
            sys.stderr.write(" PASS\n")

        kernelSol_c = <double*>calloc(solSize, sizeof(double))
        build_scprod(stamps_c, nS, image_c, kernelSol_c, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, wxy_c)
        c_kernelSol = double_ptr_to_numpy(kernelSol_c, solSize)

        py_kernelSol = build_scprod_numpy(stamps_dicts, nS, image_np, nCompKer, kerOrder, bgOrder, fwKSStamp, hwKSStamp, rPixX, py_wxy)

        sp_maxdiff = float(np.max(np.abs(c_kernelSol - py_kernelSol)))
        sys.stderr.write(f"  seed={seeds[seedIdx]} build_scprod max_diff={sp_maxdiff:.2e}")
        if sp_maxdiff > 1e-6:
            sys.stderr.write(" FAIL\n")
            ok = False
        else:
            sys.stderr.write(" PASS\n")

        sys.stderr.flush()

        free(kernelSol_c)
        free(image_c)
        for si in range(mat_size + 1):
            free(matrix_c[si])
        free(matrix_c)
        for si in range(nS):
            free(wxy_c[si])
        free(wxy_c)
        for si in range(nS):
            freeStampMem(&stamps_c[si], 1, nCompKer, nBGVectors, nC)
        free(stamps_c)

    return ok

def test_alard_batch3b():
    import sys
    from pyhotpants.numutils import fill_stamp_numpy, get_stamp_sig_numpy
    ok = True
    seeds = [500, 501, 502]

    cdef int fwKernel = 7, hwKernel = 3
    cdef int fwKSStamp = 7, hwKSStamp = 3
    cdef int b3ngauss = 1
    cdef int kerOrder = 1, bgOrder = 1
    cdef int nKSStamps = 3
    cdef int b3rPixX = 60, b3rPixY = 60
    cdef int b3usePCA = 0
    cdef float b3fillVal = -999.0
    cdef float b3statSig = 3.0

    deg_fixe_np = np.array([2], dtype=np.int32)
    sigma_gauss_np = np.array([0.5], dtype=np.float32)

    cdef int b3nCompKer = 0
    cdef int ig_tmp3
    for ig_tmp3 in range(b3ngauss):
        b3nCompKer += (int(deg_fixe_np[ig_tmp3]) + 1) * (int(deg_fixe_np[ig_tmp3]) + 2) // 2
    cdef int b3nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int b3nVec = b3nCompKer + b3nBGVectors
    cdef int b3nC = b3nCompKer + b3nBGVectors + 1
    cdef int b3fwSq = fwKSStamp * fwKSStamp
    cdef int b3totalPix = b3rPixX * b3rPixY
    cdef int b3kv_total = b3nCompKer * fwKernel
    cdef int b3sub_width = fwKSStamp + fwKernel - 1
    cdef int b3nTerms = ((kerOrder + 1) * (kerOrder + 2)) // 2
    cdef int b3ncompBG = (b3nCompKer - 1) * b3nTerms + 1
    cdef int b3solSize = b3ncompBG + b3nBGVectors + 2

    cdef int *deg_fixe_c3 = numpy_to_int_ptr(deg_fixe_np)
    cdef float *sg_c3 = numpy_to_float_ptr(sigma_gauss_np)
    cdef double *gkv_fx_c3 = <double*>calloc(b3kv_total, sizeof(double))
    cdef double *gkv_fy_c3 = <double*>calloc(b3kv_total, sizeof(double))
    cdef double **gkv_kvec_c3 = <double**>calloc(b3nCompKer, sizeof(double*))

    getKernelVec(b3ngauss, deg_fixe_c3, gkv_kvec_c3, 0, fwKernel, hwKernel, sg_c3, gkv_fx_c3, gkv_fy_c3, NULL)

    gkv_fx_np3 = double_ptr_to_numpy(gkv_fx_c3, b3kv_total)
    gkv_fy_np3 = double_ptr_to_numpy(gkv_fy_c3, b3kv_total)

    cdef float *temp_c3 = <float*>calloc(b3sub_width * fwKSStamp, sizeof(float))

    cdef int seedIdx3, kvi3
    cdef float *imConv_c3
    cdef float *imRef_c3
    cdef int *mRData_c3
    cdef stamp_struct fs_stamp_c3
    cdef int fs_rc_c3
    cdef double *kernelSol_c3
    cdef float *imNoise_c3
    cdef int *mRData_sig_c3
    cdef int *mRData_sig_c3b
    cdef double sig1_c3, sig2_c3, sig3_c3
    cdef double sig1_c3b, sig2_c3b, sig3_c3b
    cdef float *temp_sig_c3
    cdef float *temp_sig_c3b

    for seedIdx3 in range(3):
        np.random.seed(seeds[seedIdx3])

        imConv_np3 = np.random.randn(b3totalPix).astype(np.float32) * 10 + 100
        imRef_np3 = np.random.randn(b3totalPix).astype(np.float32) * 10 + 100
        mRData_np3 = np.zeros(b3totalPix, dtype=np.int32)

        imConv_c3 = numpy_to_float_ptr(imConv_np3)
        imRef_c3 = numpy_to_float_ptr(imRef_np3)
        mRData_c3 = numpy_to_int_ptr(mRData_np3)

        memset(&fs_stamp_c3, 0, sizeof(stamp_struct))
        allocateStamps(&fs_stamp_c3, 1, bgOrder, b3nCompKer, fwKSStamp, b3nC, nKSStamps)
        fs_stamp_c3.x0 = 20; fs_stamp_c3.y0 = 20
        fs_stamp_c3.x = 30; fs_stamp_c3.y = 30
        fs_stamp_c3.nx = 20; fs_stamp_c3.ny = 20
        fs_stamp_c3.nss = nKSStamps; fs_stamp_c3.sscnt = 0
        fs_stamp_c3.xss[0] = 30; fs_stamp_c3.yss[0] = 30
        fs_stamp_c3.xss[1] = 32; fs_stamp_c3.yss[1] = 32
        fs_stamp_c3.xss[2] = 28; fs_stamp_c3.yss[2] = 28

        fs_py3 = stamp_c_to_dict(&fs_stamp_c3, b3nCompKer, b3nBGVectors, fwKSStamp, b3nC, nKSStamps)
        mRData_py3 = mRData_np3.copy()

        fs_rc_c3 = fillStamp(&fs_stamp_c3, imConv_c3, imRef_c3, b3rPixX, b3rPixY, 0, b3ngauss, deg_fixe_c3,
                             hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, b3nCompKer, kerOrder,
                             b3usePCA, gkv_fx_c3, gkv_fy_c3, temp_c3, NULL, b3fillVal, mRData_c3)

        fs_rc_py3 = fill_stamp_numpy(fs_py3, imConv_np3, imRef_np3, b3rPixX, b3rPixY, 0, b3ngauss, deg_fixe_np,
                                     hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, b3nCompKer, kerOrder,
                                     b3usePCA, gkv_fx_np3, gkv_fy_np3, None, b3fillVal, mRData_py3)

        if fs_rc_c3 != fs_rc_py3:
            sys.stderr.write(f"  seed={seeds[seedIdx3]} fillStamp rc FAIL: C={fs_rc_c3}, py={fs_rc_py3}\n")
            ok = False
            free(imConv_c3); free(imRef_c3); free(mRData_c3)
            freeStampMem(&fs_stamp_c3, 1, b3nCompKer, b3nBGVectors, b3nC)
            continue

        if fs_rc_c3 == 0:
            fs_c_dict3 = stamp_c_to_dict(&fs_stamp_c3, b3nCompKer, b3nBGVectors, fwKSStamp, b3nC, nKSStamps)

            vec_maxdiff3 = float(np.max(np.abs(fs_c_dict3['vectors'] - fs_py3['vectors'])))
            sys.stderr.write(f"  seed={seeds[seedIdx3]} fillStamp vectors max_diff={vec_maxdiff3:.2e}")
            if vec_maxdiff3 > 1e-6:
                sys.stderr.write(" FAIL\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            kref_maxdiff3 = float(np.max(np.abs(fs_c_dict3['krefArea'] - fs_py3['krefArea'])))
            sys.stderr.write(f"  seed={seeds[seedIdx3]} fillStamp krefArea max_diff={kref_maxdiff3:.2e}")
            if kref_maxdiff3 > 1e-6:
                sys.stderr.write(" FAIL\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            mat_maxdiff3 = float(np.max(np.abs(fs_c_dict3['mat'] - fs_py3['mat'])))
            sys.stderr.write(f"  seed={seeds[seedIdx3]} fillStamp mat max_diff={mat_maxdiff3:.2e}")
            if mat_maxdiff3 > 1e-6:
                sys.stderr.write(" FAIL\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            scprod_maxdiff3 = float(np.max(np.abs(fs_c_dict3['scprod'] - fs_py3['scprod'])))
            sys.stderr.write(f"  seed={seeds[seedIdx3]} fillStamp scprod max_diff={scprod_maxdiff3:.2e}")
            if scprod_maxdiff3 > 1e-6:
                sys.stderr.write(" FAIL\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            kernelSol_np3 = np.random.randn(b3solSize).astype(np.float64)
            imNoise_np3 = (np.abs(np.random.randn(b3totalPix).astype(np.float32)) * 0.5 + 1.0).astype(np.float32)

            kernelSol_c3 = numpy_to_double_ptr(kernelSol_np3)
            imNoise_c3 = numpy_to_float_ptr(imNoise_np3)

            mRData_sig_c3 = numpy_to_int_ptr(mRData_np3)
            mRData_sig_py3 = mRData_np3.copy()

            sig1_c3 = 0; sig2_c3 = 0; sig3_c3 = 0
            temp_sig_c3 = <float*>calloc(b3fwSq, sizeof(float))

            getStampSig(&fs_stamp_c3, kernelSol_c3, imNoise_c3, &sig1_c3, &sig2_c3, &sig3_c3,
                        fwKSStamp, hwKSStamp, b3rPixX, b3rPixY, mRData_sig_c3, b"v", temp_sig_c3, b3statSig,
                        b3nCompKer, kerOrder, bgOrder)

            sig1_py3, sig2_py3, sig3_py3 = get_stamp_sig_numpy(fs_c_dict3, kernelSol_np3, imNoise_np3,
                                                                fwKSStamp, hwKSStamp, b3rPixX, b3rPixY,
                                                                mRData_sig_py3, "v", b3statSig,
                                                                b3nCompKer, kerOrder, bgOrder)

            sig1_diff3 = abs(sig1_c3 - sig1_py3)
            sig1_reldiff3 = sig1_diff3 / max(abs(sig1_c3), 1.0)
            sys.stderr.write(f"  seed={seeds[seedIdx3]} getStampSig(v) sig1 diff={sig1_diff3:.2e} rel={sig1_reldiff3:.2e}")
            if sig1_reldiff3 > 1e-6:
                sys.stderr.write(f" FAIL (C={sig1_c3:.6e}, py={sig1_py3:.6e})\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            free(mRData_sig_c3); free(temp_sig_c3)

            mRData_sig_c3b = numpy_to_int_ptr(mRData_np3)
            mRData_sig_py3b = mRData_np3.copy()
            sig1_c3b = 0; sig2_c3b = 0; sig3_c3b = 0
            temp_sig_c3b = <float*>calloc(b3fwSq, sizeof(float))

            getStampSig(&fs_stamp_c3, kernelSol_c3, imNoise_c3, &sig1_c3b, &sig2_c3b, &sig3_c3b,
                        fwKSStamp, hwKSStamp, b3rPixX, b3rPixY, mRData_sig_c3b, b"s", temp_sig_c3b, b3statSig,
                        b3nCompKer, kerOrder, bgOrder)

            sig1_py3b, sig2_py3b, sig3_py3b = get_stamp_sig_numpy(fs_c_dict3, kernelSol_np3, imNoise_np3,
                                                                    fwKSStamp, hwKSStamp, b3rPixX, b3rPixY,
                                                                    mRData_sig_py3b, "s", b3statSig,
                                                                    b3nCompKer, kerOrder, bgOrder)

            sig1s_diff3 = abs(sig1_c3b - sig1_py3b)
            sig1s_reldiff3 = sig1s_diff3 / max(abs(sig1_c3b), 1.0)
            sig2s_diff3 = abs(sig2_c3b - sig2_py3b)
            sig3s_diff3 = abs(sig3_c3b - sig3_py3b)
            sys.stderr.write(f"  seed={seeds[seedIdx3]} getStampSig(s) sig1 diff={sig1s_diff3:.2e} rel={sig1s_reldiff3:.2e}")
            if sig1s_reldiff3 > 1e-6:
                sys.stderr.write(f" FAIL (C={sig1_c3b:.6e}, py={sig1_py3b:.6e})\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")
            sys.stderr.write(f"  seed={seeds[seedIdx3]} getStampSig(s) sig2 diff={sig2s_diff3:.2e}")
            if sig2s_diff3 > 1e-4:
                sys.stderr.write(f" FAIL (C={sig2_c3b:.6e}, py={sig2_py3b:.6e})\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")
            sys.stderr.write(f"  seed={seeds[seedIdx3]} getStampSig(s) sig3 diff={sig3s_diff3:.2e}")
            if sig3s_diff3 > 1e-4:
                sys.stderr.write(f" FAIL (C={sig3_c3b:.6e}, py={sig3_py3b:.6e})\n"); ok = False
            else:
                sys.stderr.write(" PASS\n")

            free(kernelSol_c3); free(imNoise_c3)
            free(mRData_sig_c3b); free(temp_sig_c3b)

        sys.stderr.flush()
        free(imConv_c3); free(imRef_c3); free(mRData_c3)
        freeStampMem(&fs_stamp_c3, 1, b3nCompKer, b3nBGVectors, b3nC)

    for kvi3 in range(b3nCompKer):
        if gkv_kvec_c3[kvi3] != NULL:
            free(gkv_kvec_c3[kvi3])
    free(gkv_kvec_c3)
    free(gkv_fx_c3); free(gkv_fy_c3)
    free(deg_fixe_c3); free(sg_c3)
    free(temp_c3)

    return ok

def test_spatial_convolve():
    import sys
    from pyhotpants.numutils import spatial_convolve_numpy
    ok = True

    seeds = [100, 200, 300]
    dovarList = [0, 1, 1]
    convvarList = [0, 0, 1]

    cdef int xSize = 40, ySize = 40
    cdef int hwKernel = 2
    cdef int fwKernel = 5
    cdef int nCompKer = 3, kerOrder = 1
    cdef int kcStep = 5
    cdef int totalPix = xSize * ySize
    cdef int rPixX = xSize, rPixY = ySize
    cdef float kerFracMask = 0.99

    cdef int nTerms = ((kerOrder + 1) * (kerOrder + 2)) // 2
    cdef int solLen = 2 + (nCompKer - 1) * nTerms

    cdef int seedIdx, dovar, convvar
    cdef float *imageC
    cdef float *varC
    cdef float **varPP
    cdef float *cRdataC
    cdef int *cMaskC
    cdef int *mRDataC
    cdef double *kernelSolC
    cdef double *kernelC
    cdef double *kernelCoeffsC
    cdef double **kernelVecC

    for seedIdx in range(3):
        np.random.seed(seeds[seedIdx])
        dovar = dovarList[seedIdx]
        convvar = convvarList[seedIdx]

        imageNp = np.random.randn(totalPix).astype(np.float32) * 10 + 100
        cMaskNp = np.zeros(totalPix, dtype=np.int32)
        cMaskNp[5 + xSize * 5] = 0x80
        cMaskNp[20 + xSize * 20] = 0x80
        cMaskNp[30 + xSize * 15] = 0x20

        mRDataCNp = np.zeros(totalPix, dtype=np.int32)
        mRDataPyNp = np.zeros(totalPix, dtype=np.int32)

        cRdataCNp = np.zeros(totalPix, dtype=np.float32)
        cRdataPyNp = np.zeros(totalPix, dtype=np.float32)

        kernelSolNp = np.random.randn(solLen).astype(np.float64)
        kernelVecNp = np.random.randn(nCompKer, fwKernel * fwKernel).astype(np.float64)

        if dovar:
            varNp = (np.abs(np.random.randn(totalPix).astype(np.float32)) + 0.5).astype(np.float32)
        else:
            varNp = None

        imageC = numpy_to_float_ptr(imageNp)
        cRdataC = numpy_to_float_ptr(cRdataCNp)
        cMaskC = numpy_to_int_ptr(cMaskNp)
        mRDataC = numpy_to_int_ptr(mRDataCNp)
        kernelSolC = numpy_to_double_ptr(kernelSolNp)
        kernelC = <double*>calloc(fwKernel * fwKernel, sizeof(double))
        kernelCoeffsC = <double*>calloc(nCompKer, sizeof(double))
        kernelVecC = numpy2d_to_double_pp(kernelVecNp)

        if dovar:
            varC = numpy_to_float_ptr(varNp)
        else:
            varC = NULL
        varPP = &varC

        spatial_convolve(imageC, varPP, xSize, ySize, kernelSolC, cRdataC, cMaskC, kcStep,
                         hwKernel, fwKernel, kernelC, kernelCoeffsC,
                         convvar, kerFracMask, mRDataC,
                         rPixX, rPixY, nCompKer, kerOrder, kernelVecC)

        cRdataCResult = float_ptr_to_numpy(cRdataC, totalPix)
        mRDataCResult = int_ptr_to_numpy(mRDataC, totalPix)

        varCResult = None
        if dovar:
            varCResult = float_ptr_to_numpy(varC, totalPix)
            free(varC)

        free(imageC)
        free(cRdataC)
        free(cMaskC)
        free(mRDataC)
        free(kernelSolC)
        free(kernelC)
        free(kernelCoeffsC)
        free_double_pp(kernelVecC, nCompKer)

        kernelPy = np.zeros(fwKernel * fwKernel, dtype=np.float64)
        kernelCoeffsPy = np.zeros(nCompKer, dtype=np.float64)

        varOutPy = spatial_convolve_numpy(
            imageNp, varNp, xSize, ySize, kernelSolNp, cRdataPyNp, cMaskNp, kcStep,
            hwKernel, fwKernel, kernelPy, kernelCoeffsPy,
            convvar, kerFracMask, mRDataPyNp,
            rPixX, rPixY, nCompKer, kerOrder, kernelVecNp)

        seedOk = True

        cRdataMaxdiff = float(np.max(np.abs(cRdataCResult.astype(np.float64) - cRdataPyNp.astype(np.float64))))
        sys.stderr.write(f"  seed={seeds[seedIdx]} cRdata max_diff={cRdataMaxdiff:.2e}")
        if cRdataMaxdiff > 1e-4:
            sys.stderr.write(" FAIL\n"); seedOk = False
        else:
            sys.stderr.write(" PASS\n")

        mRDataEq = np.array_equal(mRDataCResult, mRDataPyNp)
        if not mRDataEq:
            ndiff = int(np.sum(mRDataCResult != mRDataPyNp))
            diffIdx = np.where(mRDataCResult != mRDataPyNp)[0]
            sys.stderr.write(f"  seed={seeds[seedIdx]} mRData FAIL: {ndiff} diffs, first at {diffIdx[0]}: C=0x{mRDataCResult[diffIdx[0]]:x}, py=0x{mRDataPyNp[diffIdx[0]]:x}\n")
            seedOk = False
        else:
            sys.stderr.write(f"  seed={seeds[seedIdx]} mRData PASS\n")

        if dovar:
            if varCResult is not None and varOutPy is not None:
                varMaxdiff = float(np.max(np.abs(varCResult.astype(np.float64) - varOutPy.astype(np.float64))))
                sys.stderr.write(f"  seed={seeds[seedIdx]} variance max_diff={varMaxdiff:.2e}")
                if varMaxdiff > 1e-4:
                    sys.stderr.write(" FAIL\n"); seedOk = False
                else:
                    sys.stderr.write(" PASS\n")

        if not seedOk:
            ok = False
        sys.stderr.flush()

    return ok

def test_check_stamps():
    import sys, copy
    from pyhotpants.numutils import check_stamps_numpy

    seeds = [42, 123, 7]
    all_ok = True

    cdef int fwKernel = 7, hwKernel = 3
    cdef int fwKSStamp = 7, hwKSStamp = 3
    cdef int csngauss = 1
    cdef int kerOrder = 1, bgOrder = 1
    cdef int nKSStamps = 3
    cdef int csrPixX = 60, csrPixY = 60
    cdef float csfillVal = -999.0
    cdef float csstatSig = 3.0
    cdef float cskerSigReject = 2.0

    deg_fixe_np = np.array([2], dtype=np.int32)
    sigma_gauss_np = np.array([0.5], dtype=np.float32)

    cdef int csnCompKer = 6
    cdef int csnBGVectors = 3
    cdef int csnVec = 9
    cdef int csnC = 10
    cdef int csfwSq = 49
    cdef int cstotalPix = 3600
    cdef int cskv_total = 42
    cdef int cssub_width = 13

    cdef int csncomp1 = 5
    cdef int csncomp2 = 3
    cdef int csncomp = 15
    cdef int csnbg_vec = 3
    cdef int csmat_size = 19
    cdef int csnCompTotal = 19
    cdef int csnComps = 7

    cdef int *deg_fixe_c = numpy_to_int_ptr(deg_fixe_np)
    cdef float *sg_c = numpy_to_float_ptr(sigma_gauss_np)
    cdef double *gkv_fx_c = <double*>calloc(cskv_total, sizeof(double))
    cdef double *gkv_fy_c = <double*>calloc(cskv_total, sizeof(double))
    cdef double **gkv_kvec_c = <double**>calloc(csnCompKer, sizeof(double*))

    getKernelVec(csngauss, deg_fixe_c, gkv_kvec_c, 0, fwKernel, hwKernel, sg_c, gkv_fx_c, gkv_fy_c, NULL)

    kv_np = np.zeros((csnCompKer, fwKernel * fwKernel), dtype=np.float64)
    cdef int kvi_idx
    for kvi_idx in range(csnCompKer):
        kv_i = double_ptr_to_numpy(gkv_kvec_c[kvi_idx], fwKernel * fwKernel)
        kv_np[kvi_idx, :] = kv_i

    cdef float *temp_fill_c = <float*>calloc(cssub_width * fwKSStamp, sizeof(float))

    cdef int seedIdx, si
    cdef int nS = 6
    cdef stamp_struct *stamps_c = NULL
    cdef float *imConv_c = NULL
    cdef float *imRef_c = NULL
    cdef float *imNoise_c = NULL
    cdef float *imRef_c2 = NULL
    cdef int *mRData_c = NULL
    cdef int *mRData_c2 = NULL
    cdef double **check_mat_c = NULL
    cdef double *check_vec_c = NULL
    cdef double *check_stack_c = NULL
    cdef int *indx_c = NULL
    cdef double *kernel_coeffs_c = NULL
    cdef double *kernel_c = NULL
    cdef float *temp_cs_c = NULL
    cdef double c_merit
    cdef int xc_pos, yc_pos

    for seedIdx in range(3):
        np.random.seed(seeds[seedIdx])

        imConv_np = np.random.randn(cstotalPix).astype(np.float32) * 10 + 100
        imRef_np = np.random.randn(cstotalPix).astype(np.float32) * 10 + 100
        imNoise_np = (np.abs(np.random.randn(cstotalPix).astype(np.float32)) * 0.5 + 1.0).astype(np.float32)
        mRData_np = np.zeros(cstotalPix, dtype=np.int32)

        stamps_c = <stamp_struct*>calloc(nS, sizeof(stamp_struct))
        imConv_c = numpy_to_float_ptr(imConv_np)
        imRef_c = numpy_to_float_ptr(imRef_np)
        mRData_c = numpy_to_int_ptr(mRData_np)

        for si in range(nS):
            allocateStamps(&stamps_c[si], 1, bgOrder, csnCompKer, fwKSStamp, csnC, nKSStamps)
            xc_pos = 15 + (si % 3) * 12
            yc_pos = 15 + (si // 3) * 12
            stamps_c[si].x0 = xc_pos - hwKSStamp - hwKernel - 2
            stamps_c[si].y0 = yc_pos - hwKSStamp - hwKernel - 2
            stamps_c[si].nss = 1
            stamps_c[si].sscnt = 0
            stamps_c[si].xss[0] = xc_pos
            stamps_c[si].yss[0] = yc_pos

            fillStamp(&stamps_c[si], imConv_c, imRef_c, csrPixX, csrPixY, 0, csngauss, deg_fixe_c,
                      hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, csnCompKer, kerOrder,
                      0, gkv_fx_c, gkv_fy_c, temp_fill_c, NULL, csfillVal, mRData_c)

        stamp_dicts_py = []
        for si in range(nS):
            stamp_dicts_py.append(stamp_c_to_dict(&stamps_c[si], csnCompKer, csnBGVectors, fwKSStamp, csnC, nKSStamps))

        stamp_dicts_py_copy = copy.deepcopy(stamp_dicts_py)

        imRef_c2 = numpy_to_float_ptr(imRef_np)
        imNoise_c = numpy_to_float_ptr(imNoise_np)
        mRData_c2 = numpy_to_int_ptr(mRData_np)

        check_mat_c = <double**>malloc((csmat_size + 1) * sizeof(double*))
        for si in range(csmat_size + 1):
            check_mat_c[si] = <double*>calloc(csmat_size + 1, sizeof(double))

        check_vec_c = <double*>calloc(csmat_size + 1, sizeof(double))
        check_stack_c = <double*>calloc(nS, sizeof(double))
        indx_c = <int*>calloc(csmat_size + 1, sizeof(int))
        kernel_coeffs_c = <double*>calloc(csnCompKer, sizeof(double))
        kernel_c = <double*>calloc(fwKernel * fwKernel, sizeof(double))
        temp_cs_c = <float*>calloc(csfwSq, sizeof(float))

        c_merit = check_stamps(stamps_c, nS, imRef_c2, imNoise_c,
                               csnCompKer, kerOrder, bgOrder, csnCompTotal, 0,
                               indx_c, check_mat_c, check_vec_c, check_stack_c,
                               b"b", b"v", cskerSigReject, csstatSig,
                               fwKSStamp, hwKSStamp, csrPixX, csrPixY, fwKernel,
                               gkv_kvec_c, kernel_coeffs_c, kernel_c,
                               mRData_c2, temp_cs_c)

        mRData_py2 = mRData_np.copy()
        py_merit = check_stamps_numpy(stamp_dicts_py_copy, nS, imRef_np, imNoise_np,
                                      csnCompKer, kerOrder, bgOrder, csnCompTotal, 0,
                                      "b", "v", float(cskerSigReject), float(csstatSig),
                                      fwKSStamp, hwKSStamp, csrPixX, csrPixY, fwKernel,
                                      kv_np, mRData_py2)

        merit_diff = abs(c_merit - py_merit)
        merit_denom = max(abs(c_merit), 1e-30)
        merit_rel = merit_diff / merit_denom
        sys.stderr.write(f"  seed={seeds[seedIdx]} merit C={c_merit:.6e} py={py_merit:.6e} diff={merit_diff:.2e} rel={merit_rel:.2e}")
        if merit_rel > 1e-5:
            sys.stderr.write(" FAIL\n")
            all_ok = False
        else:
            sys.stderr.write(" PASS\n")

        norm_maxrel = 0.0
        diff_maxrel = 0.0
        for si in range(nS):
            c_norm_val = stamps_c[si].norm
            c_diff_val = stamps_c[si].diff
            py_norm_val = stamp_dicts_py_copy[si]['norm']
            py_diff_val = stamp_dicts_py_copy[si]['diff']
            nd = max(abs(c_norm_val), 1e-30)
            dd = max(abs(c_diff_val), 1e-30)
            norm_maxrel = max(norm_maxrel, abs(c_norm_val - py_norm_val) / nd)
            diff_maxrel = max(diff_maxrel, abs(c_diff_val - py_diff_val) / dd)

        sys.stderr.write(f"  seed={seeds[seedIdx]} norm max_reldiff={norm_maxrel:.2e} diff max_reldiff={diff_maxrel:.2e}")
        if norm_maxrel > 1e-5 or diff_maxrel > 1e-5:
            sys.stderr.write(" FAIL\n")
            all_ok = False
        else:
            sys.stderr.write(" PASS\n")

        free(imConv_c); free(imRef_c); free(mRData_c)
        free(imRef_c2); free(imNoise_c); free(mRData_c2)
        for si in range(csmat_size + 1):
            free(check_mat_c[si])
        free(check_mat_c)
        free(check_vec_c); free(check_stack_c); free(indx_c)
        free(kernel_coeffs_c); free(kernel_c); free(temp_cs_c)
        for si in range(nS):
            freeStampMem(&stamps_c[si], 1, csnCompKer, csnBGVectors, csnC)
        free(stamps_c)

        sys.stderr.flush()

    for kvi_idx in range(csnCompKer):
        if gkv_kvec_c[kvi_idx] != NULL:
            free(gkv_kvec_c[kvi_idx])
    free(gkv_kvec_c)
    free(gkv_fx_c); free(gkv_fy_c)
    free(deg_fixe_c); free(sg_c)
    free(temp_fill_c)

    return all_ok

def test_check_again():
    import sys
    from pyhotpants.numutils import check_again_numpy
    ok = True
    seeds = [600, 601, 602]

    cdef int fwKernel = 7, hwKernel = 3
    cdef int fwKSStamp = 7, hwKSStamp = 3
    cdef int ca_ngauss = 1
    cdef int kerOrder = 1, bgOrder = 1
    cdef int nKSStamps = 3
    cdef int ca_rPixX = 60, ca_rPixY = 60
    cdef int ca_usePCA = 0
    cdef float ca_fillVal = -999.0
    cdef float ca_statSig = 3.0
    cdef float ca_kerSigReject = 2.0
    cdef int nS = 5

    deg_fixe_np = np.array([2], dtype=np.int32)
    sigma_gauss_np = np.array([0.5], dtype=np.float32)

    cdef int ca_nCompKer = 0
    cdef int ig_ca
    for ig_ca in range(ca_ngauss):
        ca_nCompKer += (int(deg_fixe_np[ig_ca]) + 1) * (int(deg_fixe_np[ig_ca]) + 2) // 2
    cdef int ca_nBGVectors = (bgOrder + 1) * (bgOrder + 2) // 2
    cdef int ca_nVec = ca_nCompKer + ca_nBGVectors
    cdef int ca_nC = ca_nCompKer + ca_nBGVectors + 1
    cdef int ca_fwSq = fwKSStamp * fwKSStamp
    cdef int ca_totalPix = ca_rPixX * ca_rPixY
    cdef int ca_kv_total = ca_nCompKer * fwKernel
    cdef int ca_sub_width = fwKSStamp + fwKernel - 1
    cdef int ca_nTerms = ((kerOrder + 1) * (kerOrder + 2)) // 2
    cdef int ca_ncompBG = (ca_nCompKer - 1) * ca_nTerms + 1
    cdef int ca_solSize = ca_ncompBG + ca_nBGVectors + 2

    cdef int *deg_fixe_ca = numpy_to_int_ptr(deg_fixe_np)
    cdef float *sg_ca = numpy_to_float_ptr(sigma_gauss_np)
    cdef double *fx_ca = <double*>calloc(ca_kv_total, sizeof(double))
    cdef double *fy_ca = <double*>calloc(ca_kv_total, sizeof(double))
    cdef double **kvec_ca = <double**>calloc(ca_nCompKer, sizeof(double*))
    getKernelVec(ca_ngauss, deg_fixe_ca, kvec_ca, 0, fwKernel, hwKernel, sg_ca, fx_ca, fy_ca, NULL)
    fx_np = double_ptr_to_numpy(fx_ca, ca_kv_total)
    fy_np = double_ptr_to_numpy(fy_ca, ca_kv_total)

    cdef float *temp_ca = <float*>calloc(ca_sub_width * fwKSStamp, sizeof(float))

    cdef int seedIdx_ca, si_ca, sj_ca, kvi_ca
    cdef float *imConv_ca
    cdef float *imRef_ca
    cdef float *imNoise_ca
    cdef float *imConv_ca2
    cdef float *imRef_ca2
    cdef int *mRData_ca
    cdef int *mRData_ca2
    cdef stamp_struct *stamps_c1
    cdef double *kernelSol_ca
    cdef double meansig_c, scatter_c
    cdef int nskipped_c
    cdef char c_check

    stamp_positions = [
        (15, 15), (30, 15), (45, 15), (15, 35), (30, 35)
    ]

    for seedIdx_ca in range(3):
        np.random.seed(seeds[seedIdx_ca])

        imConv_np = np.random.randn(ca_totalPix).astype(np.float32) * 10 + 100
        imRef_np = np.random.randn(ca_totalPix).astype(np.float32) * 10 + 100
        imNoise_np = (np.abs(np.random.randn(ca_totalPix).astype(np.float32)) * 0.5 + 1.0).astype(np.float32)
        mRData_np = np.zeros(ca_totalPix, dtype=np.int32)

        stamps_c1 = <stamp_struct*>calloc(nS, sizeof(stamp_struct))
        imConv_ca = numpy_to_float_ptr(imConv_np)
        imRef_ca = numpy_to_float_ptr(imRef_np)
        mRData_ca = numpy_to_int_ptr(mRData_np)

        for si_ca in range(nS):
            allocateStamps(&stamps_c1[si_ca], 1, bgOrder, ca_nCompKer, fwKSStamp, ca_nC, nKSStamps)
            stamps_c1[si_ca].x0 = stamp_positions[si_ca][0] - hwKSStamp - hwKernel
            stamps_c1[si_ca].y0 = stamp_positions[si_ca][1] - hwKSStamp - hwKernel
            stamps_c1[si_ca].x = stamp_positions[si_ca][0]
            stamps_c1[si_ca].y = stamp_positions[si_ca][1]
            stamps_c1[si_ca].nx = fwKSStamp + fwKernel - 1
            stamps_c1[si_ca].ny = fwKSStamp + fwKernel - 1
            stamps_c1[si_ca].nss = nKSStamps
            stamps_c1[si_ca].sscnt = 0
            for sj_ca in range(nKSStamps):
                stamps_c1[si_ca].xss[sj_ca] = stamp_positions[si_ca][0] + sj_ca * 2
                stamps_c1[si_ca].yss[sj_ca] = stamp_positions[si_ca][1] + sj_ca * 2
            fillStamp(&stamps_c1[si_ca], imConv_ca, imRef_ca, ca_rPixX, ca_rPixY, 0,
                       ca_ngauss, deg_fixe_ca, hwKSStamp, fwKSStamp, hwKernel, fwKernel,
                       bgOrder, ca_nCompKer, kerOrder, ca_usePCA, fx_ca, fy_ca, temp_ca, NULL, ca_fillVal, mRData_ca)

        stamps_c1[2].sscnt = stamps_c1[2].nss

        kernelSol_np = np.random.randn(ca_solSize).astype(np.float64)

        py_stamps = []
        for si_ca in range(nS):
            py_stamps.append(stamp_c_to_dict(&stamps_c1[si_ca], ca_nCompKer, ca_nBGVectors, fwKSStamp, ca_nC, nKSStamps))

        mRData_py_np = int_ptr_to_numpy(mRData_ca, ca_totalPix).copy()
        mRData_ca2 = numpy_to_int_ptr(mRData_py_np)

        kernelSol_ca = numpy_to_double_ptr(kernelSol_np)
        imNoise_ca = numpy_to_float_ptr(imNoise_np)
        imConv_ca2 = numpy_to_float_ptr(imConv_np)
        imRef_ca2 = numpy_to_float_ptr(imRef_np)

        meansig_c = 0; scatter_c = 0; nskipped_c = 0
        c_check = check_again(stamps_c1, kernelSol_ca, imConv_ca2, imRef_ca2, imNoise_ca,
                              &meansig_c, &scatter_c, &nskipped_c,
                              nS, 0, b"v", ca_kerSigReject, ca_statSig,
                              fwKSStamp, hwKSStamp, ca_rPixX, ca_rPixY,
                              mRData_ca, temp_ca, ca_nCompKer, kerOrder, bgOrder,
                              ca_ngauss, deg_fixe_ca, hwKernel, fwKernel,
                              ca_usePCA, fx_ca, fy_ca, NULL, ca_fillVal)

        py_check, py_meansig, py_scatter, py_nskipped = check_again_numpy(
            py_stamps, kernelSol_np, imConv_np, imRef_np, imNoise_np,
            nS, 0, "v", float(ca_kerSigReject), float(ca_statSig),
            fwKSStamp, hwKSStamp, ca_rPixX, ca_rPixY, mRData_py_np,
            ca_nCompKer, kerOrder, bgOrder, ca_ngauss, deg_fixe_np,
            hwKernel, fwKernel, ca_usePCA, fx_np, fy_np, None, float(ca_fillVal))

        seed_ok = True

        if int(c_check) != py_check:
            sys.stderr.write(f"  seed={seeds[seedIdx_ca]} check FAIL: C={int(c_check)}, py={py_check}\n")
            seed_ok = False

        meansig_diff = abs(meansig_c - py_meansig)
        meansig_reldiff = meansig_diff / max(abs(meansig_c), 1.0)
        if meansig_reldiff > 1e-6:
            sys.stderr.write(f"  seed={seeds[seedIdx_ca]} meansig FAIL: C={meansig_c:.6e}, py={py_meansig:.6e}, diff={meansig_diff:.2e}, rel={meansig_reldiff:.2e}\n")
            seed_ok = False

        scatter_diff = abs(scatter_c - py_scatter)
        scatter_reldiff = scatter_diff / max(abs(scatter_c), 1.0)
        if scatter_reldiff > 1e-6:
            sys.stderr.write(f"  seed={seeds[seedIdx_ca]} scatter FAIL: C={scatter_c:.6e}, py={py_scatter:.6e}, diff={scatter_diff:.2e}, rel={scatter_reldiff:.2e}\n")
            seed_ok = False

        if nskipped_c != py_nskipped:
            sys.stderr.write(f"  seed={seeds[seedIdx_ca]} nskipped FAIL: C={nskipped_c}, py={py_nskipped}\n")
            seed_ok = False

        for si_ca in range(nS):
            c_dict = stamp_c_to_dict(&stamps_c1[si_ca], ca_nCompKer, ca_nBGVectors, fwKSStamp, ca_nC, nKSStamps)
            if c_dict['sscnt'] != py_stamps[si_ca]['sscnt']:
                sys.stderr.write(f"  seed={seeds[seedIdx_ca]} stamp[{si_ca}].sscnt FAIL: C={c_dict['sscnt']}, py={py_stamps[si_ca]['sscnt']}\n")
                seed_ok = False
            chi2_diff = abs(c_dict['chi2'] - py_stamps[si_ca]['chi2'])
            chi2_reldiff = chi2_diff / max(abs(c_dict['chi2']), 1.0)
            if chi2_reldiff > 1e-6:
                sys.stderr.write(f"  seed={seeds[seedIdx_ca]} stamp[{si_ca}].chi2 FAIL: C={c_dict['chi2']:.6e}, py={py_stamps[si_ca]['chi2']:.6e}, diff={chi2_diff:.2e}, rel={chi2_reldiff:.2e}\n")
                seed_ok = False

        if seed_ok:
            sys.stderr.write(f"  seed={seeds[seedIdx_ca]} check_again PASS (check={int(c_check)}, meansig={meansig_c:.6f}, scatter={scatter_c:.6f}, nskipped={nskipped_c})\n")
        else:
            ok = False

        sys.stderr.flush()

        free(kernelSol_ca); free(imNoise_ca)
        free(imConv_ca2); free(imRef_ca2)
        free(mRData_ca2)
        free(imConv_ca); free(imRef_ca); free(mRData_ca)
        for si_ca in range(nS):
            freeStampMem(&stamps_c1[si_ca], 1, ca_nCompKer, ca_nBGVectors, ca_nC)
        free(stamps_c1)

    for kvi_ca in range(ca_nCompKer):
        if kvec_ca[kvi_ca] != NULL:
            free(kvec_ca[kvi_ca])
    free(kvec_ca); free(fx_ca); free(fy_ca)
    free(deg_fixe_ca); free(sg_ca); free(temp_ca)

    return ok

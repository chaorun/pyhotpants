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

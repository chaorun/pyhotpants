import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, strncmp
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

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
import os, struct

np.import_array()

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

    hotpants_compute(
        &tmpl_arr[0, 0], tNx, tNy,
        &sci_arr[0, 0], iNx, iNy,
        tni_ptr, ini_ptr, tmi_ptr, imi_ptr,
        nR, &rxmins[0], &rxmaxs[0], &rymins[0], &rymaxs[0],
        <int>r, <int>ng, &deg_arr[0], &sig_arr[0],
        <int>ko, <int>bgo,
        <int>nsx, <int>nsy, <int>nss, <int>rss,
        <int>uss, <int>afssc,
        <float>ft, <float>sft, <float>nft,
        <float>ssig, <float>ks, <float>kfm,
        <float>tu, <float>tl, <float>tg, <float>tr, <float>tp,
        <float>iu, <float>il, <float>ig, <float>ir, <float>ip,
        tuk_val, iuk_val,
        <float>mins, <float>mous,
        <float>fi, <float>fin,
        c_str, n_str, fom_str,
        <int>sconv, <int>okn, <int>convvar,
        use_pca, pca_ptr,
        xcmp_ptr, ycmp_ptr, ncmp,
        <int>v, <int>kcs,
        <int>savexy,
        &diff_out[0, 0], &noise_out[0, 0], &conv_out[0, 0], &mask_out[0, 0],
        oNx, oNy,
        stats
    )

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

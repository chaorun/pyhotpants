cdef extern from "hotpants_globals.h":
    ctypedef struct stamp_struct:
        int x0, y0
        int x, y
        int nx, ny
        int *xss
        int *yss
        int nss
        int sscnt
        double **vectors
        double *krefArea
        double **mat
        double *scprod
        double sum
        double mean
        double median
        double mode
        double sd
        double fwhm
        double lfwhm
        double chi2
        double norm
        double diff

    ctypedef struct savexy_entry:
        int x, y
        int isUsed

    ctypedef struct region_stats:
        int convTmpl
        double sumKernel
        double meansigSubstamps
        double scatterSubstamps
        double meansigSubstampsF
        double scatterSubstampsF
        double x2norm
        int nx2norm
        double mean, sd, nmean
        double meanm, sdm, nmeanm
        double diffrat
        double *kerSol
        int nSavexyEntries
        savexy_entry *savexyEntries
        long savexyXmin, savexyYmin

cdef extern from "hotpants_compute.h":
    ctypedef struct hotpants_context:
        int nCompKer, nComp, nC, nCompBG, nBGVectors, nCompTotal
        int fwKernel, fwStamp, fwKSStamp, sBorder
        int nStamps, nStampX, nStampY
        int xMin, yMin, xMax, yMax
        float fitThresh
        int kcStep
        int *indx
        float *temp
        double *check_stack, *filter_x, *filter_y
        double **kernel_vec
        double *kernel_coeffs, *kernel
        double **check_mat, *check_vec

    int hotpants_init(hotpants_context *ctx,
        int hwKernel, int ngauss, int *deg_fixe, float *sigma_gauss,
        int kerOrder, int bgOrder,
        int nStampX_in, int nStampY_in, int nKSStamps, int hwKSStamp,
        int useFullSS, float kerFitThresh, int kcStep_in,
        long tNx, long tNy, long iNx, long iNy, int nR)

    void hotpants_cleanup(hotpants_context *ctx)

    ctypedef struct hotpants_params:
        float *tFullData
        long tNx, tNy
        float *iFullData
        long iNx, iNy
        float *tNoiseFullData, *iNoiseFullData
        int *tMaskFullData, *iMaskFullData
        int nR
        int *rXMins, *rXMaxs, *rYMins, *rYMaxs
        int hwKernel, ngauss
        int *deg_fixe
        float *sigma_gauss
        int kerOrder, bgOrder
        int findSSC
        int hwKSStamp, nKSStamps
        float kerFitThresh, scaleFitThresh, minFracGoodStamps
        float statSig, kerSigReject, kerFracMask
        float tUThresh, tLThresh, tGain, tRdnoise, tPedestal
        float iUThresh, iLThresh, iGain, iRdnoise, iPedestal
        float tUKThresh, iUKThresh
        float kfSpreadMask1, kfSpreadMask2
        float fillVal, fillValNoise
        char *forceConvolve, *photNormalize, *figMerit
        int sameConv, rescaleOK, convolveVariance
        int usePCA
        float **PCA
        float *xcmp, *ycmp
        int Ncmp
        int verbose
        int savexyflag
        float *diffOut, *noiseOut, *convOut
        int *maskOut
        long oNx, oNy
        region_stats *stats

    int hotpants_process_region(hotpants_context *ctx, hotpants_params *p,
        int region_idx, char **localForceConvolve)

    ctypedef struct region_state:
        float *tRData, *iRData, *oRData, *eRData
        int *mRData, *misRData, *mtsRData
        stamp_struct *ctStamps, *ciStamps
        double *tKerSol, *iKerSol
        int rXMin, rYMin, rXMax, rYMax
        int rXBMin, rYBMin, rXBMax, rYBMax
        int xBufLo, xBufHi, yBufLo, yBufHi
        int fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY
        int rPixX, rPixY
        int nS, niS, ntS
        int convTmpl
        double sumKernel
        double meansigSubstamps, scatterSubstamps
        double meansigSubstampsF, scatterSubstampsF
        int NskippedSubstamps
        double tMerit, iMerit
        double inv1

    int region_setup(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs)
    int region_buildstamps(hotpants_context *ctx, hotpants_params *p, region_state *rs, char *localForceConvolve)
    int region_fit(hotpants_context *ctx, hotpants_params *p, region_state *rs, char **localForceConvolve)
    int region_convolve_diff(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs, char **localForceConvolve)
    void region_output(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs)
    void region_cleanup_local(region_state *rs, int convTmpl)

cdef extern from "functions.h":
    int allocateStamps(stamp_struct *, int, int, int, int, int, int)
    double ran1(int *idum)
    int sigma_clip(float *, int, double *, double *, int, float)
    void getNoiseStats3(float *, float *, double *, int *, int, int, int, int, int *)
    void insert_subregion_flt(float *, int, float *, long, int, int, int, int, int, int)
    void insert_subregion_int(int *, int, int *, long, int, int, int, int, int, int)
    void cutStamp(float *, float *, int, int, int, int, int, stamp_struct *)
    int getStampStats3(float *, int, int, int, int, double *, double *, double *, double *, double *, double *, double *, int, int, int, int, int *, float)
    int cutSStamp(stamp_struct *, float *, int, int, float, int, int *, int)
    double checkPsfCenter(float *, int, int, int, int, int, int, double, float, float, int, int, int, int, int, int, int *, float)
    int getPsfCenters(stamp_struct *, float *, int, int, double, int, int, int, int, int, int *, float, int)
    void dfset(double *, double, int, int)
    void quick_sort(double *, int *, int)
    void buildStamps(int, int, int, int, int *, int *, int, int, int, stamp_struct *, stamp_struct *, float *, float *, float, float, int, char *, int, int, float, float, int, int, int, float, int *, float)
    void freeStampMem(stamp_struct *, int, int, int, int)
    double get_background(int, int, double *, int, int, int, int, int)
    void getFinalStampSig(stamp_struct *, float *, float *, double *, int, int, int, int *)
    double make_kernel(int, int, double *, int, int, int, int, int, double **, double *, double *)
    void lubksb(double **, int, int *, double *)
    int ludcmp(double **, int, int *, double *)
    double *kernel_vector_PCA(int, int, int, int, int *, int, float **, double **)
    void getKernelVec(int, int *, double **, int, int, int, float *, double *, double *, float **)
    double *kernel_vector(int, int, int, int, int *, int, int, int, float *, double *, double *, double **, float **)
    void xy_conv_stamp(stamp_struct *, float *, int, int, int, int, int, int, int, int, double *, double *, float *, float **)
    void xy_conv_stamp_PCA(stamp_struct *, float *, int, int, int, int, int, int, int, float **)
    void build_matrix0(stamp_struct *, int, int, int, int)
    void build_scprod0(stamp_struct *, float *, int, int, int, int, int, int)
    void make_model(stamp_struct *, double *, float *, int, int, int, int, int)

cdef extern from "hotpants_compute.h":
    int hotpants_compute(
        float *tFullData, long tNx, long tNy,
        float *iFullData, long iNx, long iNy,
        float *tNoiseFullData,
        float *iNoiseFullData,
        int *tMaskFullData,
        int *iMaskFullData,
        int nR, int *rXMins, int *rXMaxs, int *rYMins, int *rYMaxs,
        int hwKernel, int ngauss, int *deg_fixe, float *sigma_gauss,
        int kerOrder, int bgOrder,
        int nStampX, int nStampY, int nKSStamps, int hwKSStamp,
        int useFullSS, int findSSC,
        float kerFitThresh, float scaleFitThresh, float minFracGoodStamps,
        float statSig, float kerSigReject, float kerFracMask,
        float tUThresh, float tLThresh, float tGain, float tRdnoise, float tPedestal,
        float iUThresh, float iLThresh, float iGain, float iRdnoise, float iPedestal,
        float tUKThresh, float iUKThresh,
        float kfSpreadMask1, float kfSpreadMask2,
        float fillVal, float fillValNoise,
        char *forceConvolve, char *photNormalize, char *figMerit,
        int sameConv, int rescaleOK, int convolveVariance,
        int usePCA, float **PCA,
        float *xcmp, float *ycmp, int Ncmp,
        int verbose, int kcStep,
        int savexyflag,
        float *diffOut, float *noiseOut, float *convOut, int *maskOut,
        long oNx, long oNy,
        region_stats *stats
    )

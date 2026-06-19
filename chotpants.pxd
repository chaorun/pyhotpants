cdef extern from "hotpants_globals.h":
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

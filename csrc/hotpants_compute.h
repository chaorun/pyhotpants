#ifndef HOTPANTS_COMPUTE_H
#define HOTPANTS_COMPUTE_H

#include "hotpants_globals.h"

typedef struct {
    int nCompKer, nComp, nC, nCompBG, nBGVectors, nCompTotal;
    int fwKernel, fwStamp, fwKSStamp, sBorder;
    int nStamps, nStampX, nStampY;
    int xMin, yMin, xMax, yMax;
    float fitThresh;
    int kcStep;
    int *indx;
    float *temp;
    double *check_stack, *filter_x, *filter_y;
    double **kernel_vec;
    double *kernel_coeffs, *kernel;
    double **check_mat, *check_vec;
} hotpants_context;

int hotpants_init(hotpants_context *ctx,
    int hwKernel, int ngauss, int *deg_fixe, float *sigma_gauss,
    int kerOrder, int bgOrder,
    int nStampX_in, int nStampY_in, int nKSStamps, int hwKSStamp,
    int useFullSS, float kerFitThresh, int kcStep_in,
    long tNx, long tNy, long iNx, long iNy, int nR);

    void hotpants_cleanup(hotpants_context *ctx);

typedef struct {
    float *tFullData; long tNx, tNy;
    float *iFullData; long iNx, iNy;
    float *tNoiseFullData, *iNoiseFullData;
    int *tMaskFullData, *iMaskFullData;
    int nR; int *rXMins, *rXMaxs, *rYMins, *rYMaxs;
    int hwKernel, ngauss; int *deg_fixe; float *sigma_gauss;
    int kerOrder, bgOrder;
    int findSSC;
    int hwKSStamp, nKSStamps;
    float kerFitThresh, scaleFitThresh, minFracGoodStamps;
    float statSig, kerSigReject, kerFracMask;
    float tUThresh, tLThresh, tGain, tRdnoise, tPedestal;
    float iUThresh, iLThresh, iGain, iRdnoise, iPedestal;
    float tUKThresh, iUKThresh;
    float kfSpreadMask1, kfSpreadMask2;
    float fillVal, fillValNoise;
    char *forceConvolve, *photNormalize, *figMerit;
    int sameConv, rescaleOK, convolveVariance;
    int usePCA; float **PCA;
    float *xcmp, *ycmp; int Ncmp;
    int verbose;
    int savexyflag;
    float *diffOut, *noiseOut, *convOut; int *maskOut;
    long oNx, oNy;
    region_stats *stats;
} hotpants_params;

    int hotpants_process_region(hotpants_context *ctx, hotpants_params *p,
        int region_idx, char **localForceConvolve);

typedef struct {
    float *tRData, *iRData, *oRData, *eRData;
    int *mRData, *misRData, *mtsRData;
    stamp_struct *ctStamps, *ciStamps;
    double *tKerSol, *iKerSol;
    int rXMin, rYMin, rXMax, rYMax;
    int rXBMin, rYBMin, rXBMax, rYBMax;
    int xBufLo, xBufHi, yBufLo, yBufHi;
    int fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY;
    int rPixX, rPixY;
    int nS, niS, ntS;
    int convTmpl;
    double sumKernel;
    double meansigSubstamps, scatterSubstamps;
    double meansigSubstampsF, scatterSubstampsF;
    int NskippedSubstamps;
    double tMerit, iMerit;
    double inv1;
} region_state;

int region_setup(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs);
int region_buildstamps(hotpants_context *ctx, hotpants_params *p, region_state *rs, char *localForceConvolve);
int region_fit(hotpants_context *ctx, hotpants_params *p, region_state *rs, char **localForceConvolve);
int region_convolve_diff(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs, char **localForceConvolve);
void region_output(hotpants_context *ctx, hotpants_params *p, int region_idx, region_state *rs);
void region_cleanup_local(region_state *rs, int convTmpl);

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
);

#endif

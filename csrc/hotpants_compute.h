#ifndef HOTPANTS_COMPUTE_H
#define HOTPANTS_COMPUTE_H

#include "hotpants_globals.h"

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

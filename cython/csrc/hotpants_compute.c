#include<stdio.h>
#include<string.h>
#include<math.h>
#include<stdlib.h>

#include "defaults.h"
#include "hotpants_globals.h"
#include "functions.h"

static void dump_entry(FILE *fp, const char *name, const void *data, size_t nbytes) {
    if (!fp) return;
    int nl = (int)strlen(name);
    fwrite(&nl, sizeof(int), 1, fp);
    fwrite(name, 1, nl, fp);
    long dl = data ? (long)nbytes : 0;
    fwrite(&dl, sizeof(long), 1, fp);
    if (data && nbytes > 0) fwrite(data, 1, nbytes, fp);
}
static void dmp_int(FILE *fp, const char *n, int v) { dump_entry(fp,n,&v,sizeof(int)); }
static void dmp_long(FILE *fp, const char *n, long v) { dump_entry(fp,n,&v,sizeof(long)); }
static void dmp_float(FILE *fp, const char *n, float v) { dump_entry(fp,n,&v,sizeof(float)); }
static void dmp_double(FILE *fp, const char *n, double v) { dump_entry(fp,n,&v,sizeof(double)); }
static void dmp_char(FILE *fp, const char *n, const char *s) {
    if (!fp) return;
    int nl = (int)strlen(n);
    fwrite(&nl, sizeof(int), 1, fp);
    fwrite(n, 1, nl, fp);
    long dl = s ? (long)(strlen(s)+1) : 0;
    fwrite(&dl, sizeof(long), 1, fp);
    if (s) fwrite(s, 1, strlen(s)+1, fp);
}
static void dmp_ptr_flag(FILE *fp, const char *n, const void *ptr) { int f = (ptr!=NULL); dmp_int(fp,n,f); }
static void dmp_stamps(FILE *fp, const char *tag, stamp_struct *stamps, int nStamps, int nCompKer, int nC, int fwKSStamp, int nCompTotal) {
    if (!fp || !stamps) return;
    int nl = (int)strlen(tag);
    fwrite(&nl, sizeof(int), 1, fp);
    fwrite(tag, 1, nl, fp);
    long marker = -1;
    fwrite(&marker, sizeof(long), 1, fp);
    fwrite(&nStamps, sizeof(int), 1, fp);
    fwrite(&nCompKer, sizeof(int), 1, fp);
    fwrite(&nC, sizeof(int), 1, fp);
    fwrite(&fwKSStamp, sizeof(int), 1, fp);
    fwrite(&nCompTotal, sizeof(int), 1, fp);
    fwrite(stamps, sizeof(stamp_struct), nStamps, fp);
    int nvec = nCompKer + 2, vecsize = fwKSStamp * fwKSStamp, si, j;
    for (si = 0; si < nStamps; si++) {
        fwrite(&stamps[si].nss, sizeof(int), 1, fp);
        if (stamps[si].xss) fwrite(stamps[si].xss, sizeof(int), stamps[si].nss > 0 ? stamps[si].nss : 1, fp);
        if (stamps[si].yss) fwrite(stamps[si].yss, sizeof(int), stamps[si].nss > 0 ? stamps[si].nss : 1, fp);
        if (stamps[si].vectors) { for (j = 0; j < nvec; j++) if (stamps[si].vectors[j]) fwrite(stamps[si].vectors[j], sizeof(double), vecsize, fp); }
        if (stamps[si].mat) { for (j = 0; j < nC; j++) if (stamps[si].mat[j]) fwrite(stamps[si].mat[j], sizeof(double), nC, fp); }
        if (stamps[si].krefArea) fwrite(stamps[si].krefArea, sizeof(double), vecsize, fp);
        if (stamps[si].scprod) fwrite(stamps[si].scprod, sizeof(double), nCompTotal + 1, fp);
    }
}
static void dmp_check_mat(FILE *fp, const char *name, double **mat, int nC) {
    if (!fp || !mat) return;
    int nl = (int)strlen(name);
    fwrite(&nl, sizeof(int), 1, fp);
    fwrite(name, 1, nl, fp);
    long dl = (long)nC * nC * sizeof(double);
    fwrite(&dl, sizeof(long), 1, fp);
    int ci;
    for (ci = 0; ci < nC; ci++) if (mat[ci]) fwrite(mat[ci], sizeof(double), nC, fp);
}

#define DUMP_ALL(dir, reg, step, phase) do { \
    if (!(dir)) break; \
    char _dp[2048]; \
    snprintf(_dp, sizeof(_dp), "%s/r%03d_s%02d_%s.bin", (dir), (reg), (step), (phase)); \
    FILE *_dfp = fopen(_dp, "wb"); \
    if (!_dfp) break; \
    dmp_int(_dfp,"nR",nR); \
    dmp_int(_dfp,"hwKernel",hwKernel); \
    dmp_int(_dfp,"ngauss",ngauss); \
    dmp_int(_dfp,"kerOrder",kerOrder); \
    dmp_int(_dfp,"bgOrder",bgOrder); \
    dmp_int(_dfp,"nStampX",nStampX); \
    dmp_int(_dfp,"nStampY",nStampY); \
    dmp_int(_dfp,"nKSStamps",nKSStamps); \
    dmp_int(_dfp,"hwKSStamp",hwKSStamp); \
    dmp_int(_dfp,"useFullSS",useFullSS); \
    dmp_int(_dfp,"findSSC",findSSC); \
    dmp_int(_dfp,"nCompKer",nCompKer); \
    dmp_int(_dfp,"nComp",nComp); \
    dmp_int(_dfp,"nCompTotal",nCompTotal); \
    dmp_int(_dfp,"nBGVectors",nBGVectors); \
    dmp_int(_dfp,"fwKernel",fwKernel); \
    dmp_int(_dfp,"fwStamp",fwStamp); \
    dmp_int(_dfp,"fwKSStamp",fwKSStamp); \
    dmp_int(_dfp,"kcStep",kcStep); \
    dmp_int(_dfp,"nStamps",nStamps); \
    dmp_int(_dfp,"sBorder",sBorder); \
    dmp_int(_dfp,"rPixX",rPixX); \
    dmp_int(_dfp,"rPixY",rPixY); \
    dmp_int(_dfp,"rXMin",rXMin); \
    dmp_int(_dfp,"rYMin",rYMin); \
    dmp_int(_dfp,"rXMax",rXMax); \
    dmp_int(_dfp,"rYMax",rYMax); \
    dmp_int(_dfp,"rXBMin",rXBMin); \
    dmp_int(_dfp,"rYBMin",rYBMin); \
    dmp_int(_dfp,"rXBMax",rXBMax); \
    dmp_int(_dfp,"rYBMax",rYBMax); \
    dmp_int(_dfp,"xBufLo",xBufLo); \
    dmp_int(_dfp,"xBufHi",xBufHi); \
    dmp_int(_dfp,"yBufLo",yBufLo); \
    dmp_int(_dfp,"yBufHi",yBufHi); \
    dmp_int(_dfp,"fpixelOutX",fpixelOutX); \
    dmp_int(_dfp,"fpixelOutY",fpixelOutY); \
    dmp_int(_dfp,"lpixelOutX",lpixelOutX); \
    dmp_int(_dfp,"lpixelOutY",lpixelOutY); \
    dmp_int(_dfp,"verbose",verbose); \
    dmp_int(_dfp,"sameConv",sameConv); \
    dmp_int(_dfp,"rescaleOK",rescaleOK); \
    dmp_int(_dfp,"convolveVariance",convolveVariance); \
    dmp_int(_dfp,"usePCA",usePCA); \
    dmp_int(_dfp,"savexyflag",savexyflag); \
    dmp_int(_dfp,"Ncmp",Ncmp); \
    dmp_int(_dfp,"nC",nC); \
    dmp_int(_dfp,"nCompBG",nCompBG); \
    dmp_int(_dfp,"xMin",xMin); \
    dmp_int(_dfp,"yMin",yMin); \
    dmp_int(_dfp,"xMax",xMax); \
    dmp_int(_dfp,"yMax",yMax); \
    dmp_int(_dfp,"nx2norm",nx2norm); \
    dmp_int(_dfp,"nS",nS); \
    dmp_int(_dfp,"ntS",ntS); \
    dmp_int(_dfp,"niS",niS); \
    dmp_int(_dfp,"convTmpl",convTmpl); \
    dmp_int(_dfp,"NskippedSubstamps",NskippedSubstamps); \
    dmp_long(_dfp,"tNx",tNx); \
    dmp_long(_dfp,"tNy",tNy); \
    dmp_long(_dfp,"iNx",iNx); \
    dmp_long(_dfp,"iNy",iNy); \
    dmp_long(_dfp,"oNx",oNx); \
    dmp_long(_dfp,"oNy",oNy); \
    dmp_float(_dfp,"kerFitThresh",kerFitThresh); \
    dmp_float(_dfp,"scaleFitThresh",scaleFitThresh); \
    dmp_float(_dfp,"minFracGoodStamps",minFracGoodStamps); \
    dmp_float(_dfp,"statSig",statSig); \
    dmp_float(_dfp,"kerSigReject",kerSigReject); \
    dmp_float(_dfp,"kerFracMask",kerFracMask); \
    dmp_float(_dfp,"tUThresh",tUThresh); \
    dmp_float(_dfp,"tLThresh",tLThresh); \
    dmp_float(_dfp,"tGain",tGain); \
    dmp_float(_dfp,"tRdnoise",tRdnoise); \
    dmp_float(_dfp,"tPedestal",tPedestal); \
    dmp_float(_dfp,"iUThresh",iUThresh); \
    dmp_float(_dfp,"iLThresh",iLThresh); \
    dmp_float(_dfp,"iGain",iGain); \
    dmp_float(_dfp,"iRdnoise",iRdnoise); \
    dmp_float(_dfp,"iPedestal",iPedestal); \
    dmp_float(_dfp,"tUKThresh",tUKThresh); \
    dmp_float(_dfp,"iUKThresh",iUKThresh); \
    dmp_float(_dfp,"kfSpreadMask1",kfSpreadMask1); \
    dmp_float(_dfp,"kfSpreadMask2",kfSpreadMask2); \
    dmp_float(_dfp,"fillVal",fillVal); \
    dmp_float(_dfp,"fillValNoise",fillValNoise); \
    dmp_float(_dfp,"fitThresh",fitThresh); \
    dmp_double(_dfp,"sumKernel",sumKernel); \
    dmp_double(_dfp,"meansigSubstamps",meansigSubstamps); \
    dmp_double(_dfp,"scatterSubstamps",scatterSubstamps); \
    dmp_double(_dfp,"meansigSubstampsF",meansigSubstampsF); \
    dmp_double(_dfp,"scatterSubstampsF",scatterSubstampsF); \
    dmp_double(_dfp,"inv1",inv1); \
    dmp_double(_dfp,"tMerit",tMerit); \
    dmp_double(_dfp,"iMerit",iMerit); \
    dmp_double(_dfp,"diffrat",diffrat); \
    dmp_double(_dfp,"x2norm",x2norm); \
    dmp_char(_dfp,"forceConvolve",forceConvolve); \
    dmp_char(_dfp,"photNormalize",photNormalize); \
    dmp_char(_dfp,"figMerit",figMerit); \
    dmp_char(_dfp,"localForceConvolve",localForceConvolve); \
    dump_entry(_dfp,"deg_fixe",deg_fixe,ngauss*sizeof(int)); \
    dump_entry(_dfp,"sigma_gauss",sigma_gauss,ngauss*sizeof(float)); \
    dump_entry(_dfp,"tFullData",tFullData,tNx*tNy*sizeof(float)); \
    dump_entry(_dfp,"iFullData",iFullData,iNx*iNy*sizeof(float)); \
    dump_entry(_dfp,"tRData",tRData,rPixX*rPixY*sizeof(float)); \
    dump_entry(_dfp,"iRData",iRData,rPixX*rPixY*sizeof(float)); \
    dump_entry(_dfp,"oRData",oRData,rPixX*rPixY*sizeof(float)); \
    dump_entry(_dfp,"eRData",eRData,rPixX*rPixY*sizeof(float)); \
    dump_entry(_dfp,"mRData",mRData,rPixX*rPixY*sizeof(int)); \
    dump_entry(_dfp,"misRData",misRData,rPixX*rPixY*sizeof(int)); \
    dump_entry(_dfp,"mtsRData",mtsRData,rPixX*rPixY*sizeof(int)); \
    dump_entry(_dfp,"diffOut",diffOut,oNx*oNy*sizeof(float)); \
    dump_entry(_dfp,"noiseOut",noiseOut,oNx*oNy*sizeof(float)); \
    dump_entry(_dfp,"convOut",convOut,oNx*oNy*sizeof(float)); \
    dump_entry(_dfp,"maskOut",maskOut,oNx*oNy*sizeof(int)); \
    dump_entry(_dfp,"filter_x",filter_x,nCompKer*fwKernel*sizeof(double)); \
    dump_entry(_dfp,"filter_y",filter_y,nCompKer*fwKernel*sizeof(double)); \
    dump_entry(_dfp,"kernel",kernel,fwKernel*fwKernel*sizeof(double)); \
    dump_entry(_dfp,"kernel_coeffs",kernel_coeffs,nCompKer*sizeof(double)); \
    dump_entry(_dfp,"indx",indx,(nCompTotal+1+100)*sizeof(int)); \
    dump_entry(_dfp,"temp",temp,(fwKSStamp+fwKernel)*fwKSStamp*sizeof(float)); \
    dump_entry(_dfp,"tKerSol",tKerSol,(nCompTotal+1)*sizeof(double)); \
    dump_entry(_dfp,"iKerSol",iKerSol,(nCompTotal+1)*sizeof(double)); \
    dump_entry(_dfp,"check_vec",check_vec,nC*sizeof(double)); \
    dump_entry(_dfp,"check_stack",check_stack,nStamps*sizeof(double)); \
    dump_entry(_dfp,"kernel_vec",kernel_vec,nCompKer*sizeof(double*)); \
    dump_entry(_dfp,"rXMins",rXMins,nR*sizeof(int)); \
    dump_entry(_dfp,"rXMaxs",rXMaxs,nR*sizeof(int)); \
    dump_entry(_dfp,"rYMins",rYMins,nR*sizeof(int)); \
    dump_entry(_dfp,"rYMaxs",rYMaxs,nR*sizeof(int)); \
    dump_entry(_dfp,"xcmp",xcmp,Ncmp*sizeof(float)); \
    dump_entry(_dfp,"ycmp",ycmp,Ncmp*sizeof(float)); \
    dump_entry(_dfp,"tNoiseFullData",tNoiseFullData,tNx*tNy*sizeof(float)); \
    dump_entry(_dfp,"iNoiseFullData",iNoiseFullData,iNx*iNy*sizeof(float)); \
    dump_entry(_dfp,"tMaskFullData",tMaskFullData,tNx*tNy*sizeof(int)); \
    dump_entry(_dfp,"iMaskFullData",iMaskFullData,iNx*iNy*sizeof(int)); \
    dmp_ptr_flag(_dfp,"tNoiseFullData_flag",tNoiseFullData); \
    dmp_ptr_flag(_dfp,"iNoiseFullData_flag",iNoiseFullData); \
    dmp_ptr_flag(_dfp,"tMaskFullData_flag",tMaskFullData); \
    dmp_ptr_flag(_dfp,"iMaskFullData_flag",iMaskFullData); \
    dmp_ptr_flag(_dfp,"PCA_flag",PCA); \
    dmp_ptr_flag(_dfp,"xcmp_flag",xcmp); \
    if (ctStamps) dmp_stamps(_dfp,"ctStamps",ctStamps,nStamps,nCompKer,nC,fwKSStamp,nCompTotal); \
    if (ciStamps) dmp_stamps(_dfp,"ciStamps",ciStamps,nStamps,nCompKer,nC,fwKSStamp,nCompTotal); \
    dmp_check_mat(_dfp,"check_mat",check_mat,nC); \
    fclose(_dfp); \
} while(0)

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
) {
    const char *dump_dir = getenv("HOTPANTS_DUMP_DIR");
    int         i,j,k,l,m;
    double      sumKernel;
    double      meansigSubstamps,scatterSubstamps;
    double      meansigSubstampsF,scatterSubstampsF;
    // double      meanksumSubstamps,scatterksumSubstamps;
    int         NskippedSubstamps;

    float *tRData = NULL;
    double tMerit=0;
    double *tKerSol = NULL;
    stamp_struct *ctStamps = NULL;

    float *iRData = NULL;
    double iMerit=0;
    double *iKerSol = NULL;
    stamp_struct *ciStamps = NULL;

    float *oRData = NULL;
    float *eRData = NULL;

    int      *misRData = NULL;
    int      *mtsRData = NULL;

    int xMin, yMin, xMax, yMax;

    int rXMin, rYMin, rXMax, rYMax;
    int rXBMin, rYBMin, rXBMax, rYBMax;
    int xBufLo, xBufHi, yBufLo, yBufHi;

    int sXMin, sYMin, sXMax, sYMax;
    int niS, ntS;

    int convTmpl;

    int fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY;
    // int anynul;
    int status = 0;
    // int kInfoNum;
    int nx2norm;

    double sum,  mean,  median,  mode,  sd,  fwhm,  lfwhm, diffrat, x2norm;
    double summ, meanm, medianm, modem, sdm, fwhmm, lfwhmm;
    double nsum,  nmean,  nmedian,  nmode,  nsd,  nfwhm, nlfwhm;
    double nsumm, nmeanm, nmedianm, nmodem, nsdm, nfwhmm, nlfwhmm;

    float iSFrac, tSFrac;
    float fitThresh;
    int flag;
    double inv1;
    int sBorder;

    int rPixX, rPixY;
    int nStamps, nS, nCompKer, nC;
    int nComp, nCompBG, nBGVectors, nCompTotal;
    int fwKernel, fwStamp, fwKSStamp;
    int *indx = NULL;
    float *temp = NULL, *temp2 = NULL;
    double *check_stack = NULL, *filter_x = NULL, *filter_y = NULL;
    double **kernel_vec = NULL;
    double *kernel_coeffs = NULL, *kernel = NULL;
    double **check_mat = NULL, *check_vec = NULL;
    int *mRData = NULL;
    char *localForceConvolve = forceConvolve;

    nCompKer = 0;
    for(i = 0; i < ngauss; i++)
        nCompKer += ((deg_fixe[i] + 1) * (deg_fixe[i] + 2)) / 2;

    nComp       = ((kerOrder + 1) * (kerOrder + 2)) / 2;
    nC          = nCompKer + 2;
    nCompBG     = (nCompKer - 1) * nComp + 1;
    nBGVectors  = ((bgOrder + 1) * (bgOrder + 2)) / 2;
    nCompTotal  = nCompKer * nComp + nBGVectors;

    fwKernel    = hwKernel * 2 + 1;

    if (useFullSS) {
        fwKSStamp   = fwKernel;
        fwStamp     = fwKernel;
        nStampX     = (int)(imin(tNx, iNx) / nR / fwStamp);
        nStampY     = (int)(imin(tNy, iNy) / nR / fwStamp);
        fprintf(stderr, "Using maximial number of stamps : %d x %d\n", nStampX, nStampY);
    }
    else {
        fwKSStamp   = hwKSStamp * 2 + 1;

        fwStamp = imin( imin(tNx, iNx) / (int)sqrt((double)nR) / nStampX,
                        imin(tNy, iNy) / (int)sqrt((double)nR) / nStampY );
        fwStamp -= fwKernel;
        fwStamp -= fwStamp % 2 == 0 ? 1 : 0;

        if (fwStamp < fwKSStamp) {
            fwStamp  = fwKSStamp+fwKernel;
            fwStamp -= fwStamp % 2 == 0 ? 1 : 0;

            nStampX = imin(tNx, iNx) / (int)sqrt((double)nR) / fwStamp;
            nStampY = imin(tNy, iNy) / (int)sqrt((double)nR) / fwStamp;

            fprintf(stderr, "WARNING : too many stamps requested\n");
            fprintf(stderr, "          using nsx = %d, nsy = %d\n", nStampX, nStampY);
        }

    }

    kcStep  = kcStep ? kcStep : fwKernel;

    nStamps = nStampX * nStampY;
    sBorder = hwKSStamp + hwKernel;

    fprintf(stderr, "Mallocing massive amounts of memory...\n");

    if ( !(temp    = (float *)calloc((fwKSStamp+fwKernel)*fwKSStamp, sizeof(float))) ||
         !(indx    = (int *)calloc((nCompTotal+1+100), sizeof(int))) ||

         !(kernel        = (double *)calloc(fwKernel*fwKernel, sizeof(double))) ||
         !(kernel_coeffs = (double *)calloc(nCompKer, sizeof(double))) ||

         !(kernel_vec = (double **)calloc(nCompKer, sizeof(double *))) ||

         !(filter_x   = (double *)calloc(nCompKer*fwKernel, sizeof(double))) ||
         !(filter_y   = (double *)calloc(nCompKer*fwKernel, sizeof(double))) ||

         !(check_mat   = (double **)calloc(nC, sizeof(double *))) ||
         !(check_vec   = (double *)calloc(nC, sizeof(double))) ||
         !(check_stack = (double *)calloc(nStamps, sizeof(double))) ) {
        return 1;
    }
    for (i = 0; i < nC; i++) 
        if ( !(check_mat[i] = (double *)calloc(nC, sizeof(double))) ) {
            return 1;
        }

    xMin = 0;
    yMin = 0;
    xMax = imin(tNx, iNx) - 1;
    yMax = imin(tNy, iNy) - 1;

    fitThresh = kerFitThresh;

    for (i = 0; i < nR; i++) {
        // if (kernelImIn) {
        //     readKernel(kernelImIn, i, &tKerSol, &iKerSol, &rXMin, &rXMax, &rYMin, &rYMax,
        //                &meansigSubstamps, &scatterSubstamps,
        //                &meansigSubstampsF, &scatterSubstampsF,
        //                &diffrat, &NskippedSubstamps,
        //                nCompTotal, &localForceConvolve);
        // }
        // else {
            rXMin = rXMins[i];
            rXMax = rXMaxs[i];
            rYMin = rYMins[i];
            rYMax = rYMaxs[i];
            meansigSubstamps = scatterSubstamps = 0.0;
            NskippedSubstamps = 0;
        // }

        if (nR > 1) {
            rXBMin = imax(xMin, rXMin - fwStamp/2);
            rYBMin = imax(yMin, rYMin - fwStamp/2);
            rXBMax = imin(xMax, rXMax + fwStamp/2);
            rYBMax = imin(yMax, rYMax + fwStamp/2);
        }
        else {
            rXBMin = imax(xMin, rXMin - hwKernel);
            rYBMin = imax(yMin, rYMin - hwKernel); 
            rXBMax = imin(xMax, rXMax + hwKernel);
            rYBMax = imin(yMax, rYMax + hwKernel);
        }

        xBufLo = rXMin - rXBMin;
        xBufHi = rXBMax - rXMax;
        yBufLo = rYMin - rYBMin;
        yBufHi = rYBMax - rYMax;

        rPixX = rXBMax - rXBMin + 1;
        rPixY = rYBMax - rYBMin + 1;

        fpixelOutX = rXBMin + xBufLo + 1;
        fpixelOutY = rYBMin + yBufLo + 1;
        lpixelOutX = fpixelOutX + (rPixX - xBufHi - xBufLo - 1);
        lpixelOutY = fpixelOutY + (rPixY - yBufHi - yBufLo - 1);

        fprintf(stderr, "Region %d buffered             : %d:%d,%d:%d\n"
                , i, rXBMin, rXBMax, rYBMin, rYBMax);
        fprintf(stderr, " Vector Indices (good data): %d:%d,%d:%d\n"
                , rXMin, rXMax, rYMin, rYMax);

        tRData = (float *)calloc(rPixX*rPixY, sizeof(float));
        iRData = (float *)calloc(rPixX*rPixY, sizeof(float));
        oRData = (float *)calloc(rPixX*rPixY, sizeof(float));
        eRData = (float *)calloc(rPixX*rPixY, sizeof(float));
        if (tRData == NULL || iRData == NULL || oRData == NULL || eRData == NULL) {
            fprintf(stderr, "Cannot Allocate Standard Data Arrays\n"); 
            return 1;
        }
        fset(tRData, fillVal, rPixX, rPixY);
        fset(iRData, fillVal, rPixX, rPixY);
        fset(oRData, fillValNoise, rPixX, rPixY);
        fset(eRData, fillValNoise, rPixX, rPixY);

        mRData   = (int *)calloc(rPixX*rPixY, sizeof(int));
        misRData = (int *)calloc(rPixX*rPixY, sizeof(int));
        mtsRData = (int *)calloc(rPixX*rPixY, sizeof(int));
        if (mRData == NULL || misRData == NULL || mtsRData == NULL ) {
            fprintf(stderr, "Cannot Allocate Mask Arrays\n"); 
            return 1;
        }

        // if (!(kernelImIn)) {
            if (!(strncmp(localForceConvolve, "i", 1)==0)) {
                if (verbose>=2) fprintf(stderr,"Allocating stamps...\n");
                if(!(ctStamps = (stamp_struct *)calloc(nStamps, sizeof(stamp_struct)))) {
                    printf("Cannot Allocate Stamp List\n"); 
                    return 1;
                }
                if (allocateStamps(ctStamps, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)) {
                    fprintf(stderr,"Cannot Allocate Stamp Vector\n"); 
                    return 1;
                }
                tKerSol = (double *)calloc((nCompTotal+1), sizeof(double));
            }
            if (!(strncmp(localForceConvolve, "t", 1)==0)) {
                if(!(ciStamps = (stamp_struct *)calloc(nStamps, sizeof(stamp_struct)))) {
                    printf("Cannot Allocate Stamp List\n"); 
                    return 1;
                }
                if (allocateStamps(ciStamps, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)) {
                    fprintf(stderr,"Cannot Allocate Stamp Vector\n"); 
                    return 1;
                }
                iKerSol = (double *)calloc((nCompTotal+1), sizeof(double));
            }
        // }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 1, "pre"); }
        extract_subregion_flt(tFullData, tNx, rXBMin, rYBMin, rXBMax, rYBMax, tRData, rPixX);
        if (dump_dir) { DUMP_ALL(dump_dir, i, 1, "post"); }
        if (dump_dir) { DUMP_ALL(dump_dir, i, 2, "pre"); }
        extract_subregion_flt(iFullData, iNx, rXBMin, rYBMin, rXBMax, rYBMax, iRData, rPixX);
        if (dump_dir) { DUMP_ALL(dump_dir, i, 2, "post"); }

        if (tPedestal != 0. || iPedestal != 0.) {
            for (l = rPixX*rPixY; l--; ) {
                tRData[l] -= tPedestal;
                iRData[l] -= iPedestal;
            }
        }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 3, "pre"); }
        if (iNoiseFullData) {
            extract_subregion_flt(iNoiseFullData, iNx, rXBMin, rYBMin, rXBMax, rYBMax, oRData, rPixX);

            for (l = rPixX*rPixY; l--; )
                oRData[l] *= oRData[l];
        }
        else {
            oRData = makeNoiseImage4(iRData, 1./iGain, iRdnoise/iGain, rPixX, rPixY);
        }
        if (dump_dir) { DUMP_ALL(dump_dir, i, 3, "post"); }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 4, "pre"); }
        if (tNoiseFullData) {
            extract_subregion_flt(tNoiseFullData, tNx, rXBMin, rYBMin, rXBMax, rYBMax, eRData, rPixX);

            for (l = rPixX*rPixY; l--; )
                eRData[l] *= eRData[l];
        }
        else {
            eRData = makeNoiseImage4(tRData, 1./tGain, tRdnoise/tGain, rPixX, rPixY);
        }
        if (dump_dir) { DUMP_ALL(dump_dir, i, 4, "post"); }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 5, "pre"); }
        for (l = rPixX*rPixY; l--; ) 
            oRData[l] += eRData[l];
        if (dump_dir) { DUMP_ALL(dump_dir, i, 5, "post"); }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 6, "pre"); }
        if (iMaskFullData) {
            extract_subregion_int(iMaskFullData, iNx, rXBMin, rYBMin, rXBMax, rYBMax, misRData, rPixX);

            for (l = rPixX*rPixY; l--; ) {
                misRData[l] |= FLAG_INPUT_MASK * (misRData[l] > 0);
                mRData[l]   |= misRData[l];
            }
        }

        if (tMaskFullData) {
            extract_subregion_int(tMaskFullData, tNx, rXBMin, rYBMin, rXBMax, rYBMax, mtsRData, rPixX);

            for (l = rPixX*rPixY; l--; ) {
                mtsRData[l] |= FLAG_INPUT_MASK * (mtsRData[l] > 0);
                mRData[l]   |= mtsRData[l];
            }
        }
        if (dump_dir) { DUMP_ALL(dump_dir, i, 6, "post"); }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 7, "pre"); }
        makeInputMask(tRData, iRData, mRData, rPixX, rPixY, fillVal, tUThresh, iUThresh, tLThresh, iLThresh, hwKernel, kfSpreadMask1);
        if (dump_dir) { DUMP_ALL(dump_dir, i, 7, "post"); }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 8, "pre"); }
        for (l = 0; l < rPixY; l++) {
            for (k = 0; k < sBorder; k++) {
                mRData[k+rPixX*l] |= (FLAG_T_BAD | FLAG_I_BAD); 
            }
            for (k = rPixX-sBorder; k < rPixX; k++) {
                mRData[k+rPixX*l] |= (FLAG_T_BAD | FLAG_I_BAD);
            }
        }
        for (l = 0; l < sBorder; l++) 
            for (k = sBorder; k < rPixX-sBorder; k++) 
                mRData[k+rPixX*l] |= (FLAG_T_BAD | FLAG_I_BAD);
        for (l = rPixY-sBorder; l < rPixY; l++) 
            for (k = sBorder; k < rPixX-sBorder; k++) 
                mRData[k+rPixX*l] |= (FLAG_T_BAD | FLAG_I_BAD);
        if (dump_dir) { DUMP_ALL(dump_dir, i, 8, "post"); }
	    

        status       = 1;
        flag         = 1;
        kerFitThresh = fitThresh;
        // if (!(kernelImIn)) {
            while ((status <= 2) && (flag)) {
                flag = 0;
                niS  = 0;
                ntS  = 0;

                for (l = 0; l < nStampY; l++) {
                    for (k = 0; k < nStampX; k++) {

                        fprintf(stderr, "Build stamp  : t %4d i %4d (grid coord %2d %2d)\n", ntS, niS, k, l);

                        sXMin = rXBMin + k * rPixX / nStampX;
                        sYMin = rYBMin + l * rPixY / nStampY;
                        sXMax = imin(sXMin + fwStamp - 1, rXBMax);
                        sYMax = imin(sYMin + fwStamp - 1, rYBMax);

                        if (!(strncmp(localForceConvolve, "i", 1)==0)) {
                            ctStamps[ntS].sscnt = ctStamps[ntS].nss = 0;
                            ctStamps[ntS].chi2 = 0.0;
                        }
                        if (!(strncmp(localForceConvolve, "t", 1)==0)) {
                            ciStamps[niS].sscnt = ciStamps[niS].nss = 0;
                            ciStamps[niS].chi2 = 0.0;
                        }

                        if (xcmp && Ncmp > 0) {
                            if (verbose >= 2) fprintf(stderr, "Adding centers manually\n");
                            for (m = 0; m < Ncmp; m++) {
                                if ((xcmp[m] > sXMin + hwKernel + 1) && (xcmp[m] < sXMax - hwKernel - 1) &&
                                    (ycmp[m] > sYMin + hwKernel + 1) && (ycmp[m] < sYMax - hwKernel - 1)) {

                                    buildStamps(sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 0, rXBMin, rYBMin,
                                                ciStamps, ctStamps, iRData, tRData,
                                                xcmp[m] - rXBMin, ycmp[m] - rYBMin,
                                                verbose, localForceConvolve, rPixX, rPixY, tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps, kerFitThresh, mRData, statSig);

                                }
                            }
                            if (findSSC) {
                                if (verbose >= 2) fprintf(stderr, "Automatically finding additional centers\n");
                                buildStamps(sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 1, rXBMin, rYBMin,		
                                            ciStamps, ctStamps, iRData, tRData, 0, 0,
                                            verbose, localForceConvolve, rPixX, rPixY, tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps, kerFitThresh, mRData, statSig);
                            }
                        }
                        else {
                            if (useFullSS) {
                                buildStamps(sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 0, rXBMin, rYBMin,		
                                            ciStamps, ctStamps, iRData, tRData, 0, 0,
                                            verbose, localForceConvolve, rPixX, rPixY, tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps, kerFitThresh, mRData, statSig);
                            }
                            else {
                                buildStamps(sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 1, rXBMin, rYBMin,		
                                            ciStamps, ctStamps, iRData, tRData, 0, 0,
                                            verbose, localForceConvolve, rPixX, rPixY, tUKThresh, iUKThresh, hwKSStamp, fwStamp, nKSStamps, kerFitThresh, mRData, statSig);
                            }
                        }


                        if (!(strncmp(localForceConvolve, "i", 1)==0)) {
                            if (verbose>=2) fprintf(stderr,"    templ: %d substamps\n",ctStamps[ntS].nss);
                            if (ctStamps[ntS].nss > 0) ntS += 1;
                        }

                        if (!(strncmp(localForceConvolve, "t", 1)==0)) {
                            if (verbose>=2) fprintf(stderr,"    image: %d substamps\n",ciStamps[niS].nss);
                            if (ciStamps[niS].nss > 0) niS += 1;
                        }

                    }
                }

                iSFrac = niS / (float) nStamps;
                tSFrac = ntS / (float) nStamps;

                if (strncmp(localForceConvolve, "i", 1)==0) {
                    fprintf(stderr, "%d stamps built (%.2f%s)\n\n", niS, iSFrac, "%");
                    if (iSFrac < minFracGoodStamps)
                        flag = 1;
                }
                else if (strncmp(localForceConvolve, "t", 1)==0) {
                    fprintf(stderr, "%d stamps built (%.2f%s)\n\n", ntS, tSFrac, "%");
                    if (tSFrac < minFracGoodStamps)
                        flag = 1;
                }
                else if ((iSFrac < minFracGoodStamps) || (tSFrac < minFracGoodStamps)) {
                    fprintf(stderr, "%d and %d stamps built (%.2f%s, %.2f%s)\n\n", ntS, niS, tSFrac, "%", iSFrac, "%");
                    flag = 1;
                }
                else {
                    fprintf(stderr, "%d and %d stamps built (%.2f%s, %.2f%s)\n\n", ntS, niS, tSFrac, "%", iSFrac, "%");
                    break;
                }

                if ((flag) && (status <= 1) && (scaleFitThresh < 1.)) {

                    kerFitThresh *= scaleFitThresh;

                    fprintf(stderr, "Too few stamps were fit, scaling down fitting threshold to %.2f\n", kerFitThresh);

                    if (ctStamps) {
                        freeStampMem(ctStamps, nStamps, nCompKer, nBGVectors, nC);
                        free(ctStamps);
                        ctStamps = NULL;
                    }
                    if (ciStamps) {
                        freeStampMem(ciStamps, nStamps, nCompKer, nBGVectors, nC);
                        free(ciStamps);
                        ciStamps = NULL;
                    }		  


                    if (!(strncmp(localForceConvolve, "i", 1)==0)) {
                        if(!(ctStamps = (stamp_struct *)calloc(nStamps, sizeof(stamp_struct)))) {
                            printf("Cannot Allocate Stamp List\n"); 
                            return 1;
                        }
                        if (allocateStamps(ctStamps, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)) {
                            fprintf(stderr,"Cannot Allocate Stamp Vector\n"); 
                            return 1;
                        }
                    }

                    if (!(strncmp(localForceConvolve, "t", 1)==0)) {
                        if(!(ciStamps = (stamp_struct *)calloc(nStamps, sizeof(stamp_struct)))) {
                            printf("Cannot Allocate Stamp List\n"); 
                            return 1;
                        }
                        if (allocateStamps(ciStamps, nStamps, bgOrder, nCompKer, fwKSStamp, nC, nKSStamps)) {
                            fprintf(stderr,"Cannot Allocate Stamp Vector\n"); 
                            return 1;
                        }
                    }

                    for (l = rPixX*rPixY; l--; ) 
                        mRData[l] &= ~0xa00;

                }
                status += 1;
            }
            status = 0;

            if ((niS == 0) && (ntS == 0))
                goto region_cleanup;
            if (strncmp(localForceConvolve, "i", 1)==0)
                if (niS == 0)
                    goto region_cleanup;
            if (strncmp(localForceConvolve, "t", 1)==0)
                if (ntS == 0)
                    goto region_cleanup;


            if (dump_dir) { DUMP_ALL(dump_dir, i, 9, "pre"); }
            getKernelVec(ngauss, deg_fixe, kernel_vec, usePCA, fwKernel, hwKernel, sigma_gauss, filter_x, filter_y, PCA); 
            if (dump_dir) { DUMP_ALL(dump_dir, i, 9, "post"); }

            fprintf(stderr, "Filling Template sub-stamps\n");
            if (!(strncmp(localForceConvolve, "i", 1)==0)) {
                if (dump_dir) { DUMP_ALL(dump_dir, i, 10, "pre"); }
                for (k = 0; k < ntS; k++) {
                    ctStamps[k].sscnt = 0;
                    fillStamp(&ctStamps[k], tRData, iRData, rPixX, rPixY, verbose, ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y, temp, PCA, fillVal, mRData);
                }
                if ((strncmp(localForceConvolve, "b", 1)==0)) {
                    fprintf(stderr, "\n\nTrying to convolve the TEMPLATE to fit IMAGE\n");
                    tMerit = check_stamps(ctStamps, ntS, iRData, oRData, nCompKer, kerOrder, bgOrder, nCompTotal, verbose, indx, check_mat, check_vec, check_stack, localForceConvolve, figMerit, kerSigReject, statSig, fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel, kernel_vec, kernel_coeffs, kernel, mRData, temp);
                    fprintf(stderr, "    Result : merit = %.3f\n", tMerit);
                }
                else 
                    tMerit = iMerit = 0;
                if (dump_dir) { DUMP_ALL(dump_dir, i, 10, "post"); }
            }

            fprintf(stderr, "Filling Image sub-stamps\n");
            if (!(strncmp(localForceConvolve, "t", 1)==0)) {
                if (dump_dir) { DUMP_ALL(dump_dir, i, 10, "pre"); }

                for (k = 0; k < niS; k++) {
                    ciStamps[k].sscnt = 0;
                    fillStamp(&ciStamps[k], iRData, tRData, rPixX, rPixY, verbose, ngauss, deg_fixe, hwKSStamp, fwKSStamp, hwKernel, fwKernel, bgOrder, nCompKer, kerOrder, usePCA, filter_x, filter_y, temp, PCA, fillVal, mRData);
                }
                if ((strncmp(localForceConvolve, "b", 1)==0)) {
                    fprintf(stderr, "\n\nTrying to convolve the IMAGE to fit TEMPLATE \n");
                    iMerit = check_stamps(ciStamps, niS, tRData, oRData, nCompKer, kerOrder, bgOrder, nCompTotal, verbose, indx, check_mat, check_vec, check_stack, localForceConvolve, figMerit, kerSigReject, statSig, fwKSStamp, hwKSStamp, rPixX, rPixY, fwKernel, kernel_vec, kernel_coeffs, kernel, mRData, temp);
                    fprintf(stderr, "    Result : merit = %.3f\n", iMerit);
                }
                else
                    iMerit = tMerit = 0;
                if (dump_dir) { DUMP_ALL(dump_dir, i, 10, "post"); }
            }

        // } /* end of if not kernelImIn */
        // else {
        //     status = 0;
        //     getKernelVec(ngauss, deg_fixe, kernel_vec, usePCA, fwKernel, hwKernel, sigma_gauss, filter_x, filter_y, PCA); 
        // }

        if ( (strncmp(localForceConvolve, "t", 1)==0) ||
             ( (tMerit < iMerit) && (!(strncmp(localForceConvolve, "i", 1)==0))) ) {
            convTmpl = 1;

        } else {
            convTmpl = 0;
        }


        if (convTmpl) {
            fprintf(stderr, "\n\n Region %d:%d,%d:%d : Convolving TEMPLATE\n", rXMin, rXMax, rYMin, rYMax);

            freeStampMem(ciStamps, nStamps, nCompKer, nBGVectors, nC);
            /*allocateStamps(ciStamps, nStamps);*/
            nS = ntS;

            // if (!(kernelImIn))
                if (dump_dir) { DUMP_ALL(dump_dir, i, 11, "pre"); }
                fitKernel(ctStamps, iRData, tRData, oRData, tKerSol, &meansigSubstamps, &scatterSubstamps, &NskippedSubstamps, nCompKer, kerOrder, bgOrder, verbose, nS, indx, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit, kerSigReject, statSig, mRData, temp, ngauss, deg_fixe, hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal);
                if (dump_dir) { DUMP_ALL(dump_dir, i, 11, "post"); }

            oRData = (float *)realloc(oRData, rPixX*rPixY*sizeof(float));
            fset(oRData, fillVal, rPixX, rPixY);

            for (l = rPixX*rPixY; l--; ) {
                mtsRData[l] |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tRData[l] == fillVal);
                mtsRData[l] |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL)  * (tRData[l] >= tUThresh);
                mtsRData[l] |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL)  * (tRData[l] <= tLThresh);
            }

            memset(mRData, 0, rPixX*rPixY*sizeof(int));

            eRData = (float *)realloc(eRData, rPixX*rPixY*sizeof(float));
            fset(eRData, fillValNoise, rPixX, rPixY);

            if (tNoiseFullData) {
                extract_subregion_flt(tNoiseFullData, tNx, rXBMin, rYBMin, rXBMax, rYBMax, eRData, rPixX);

                for (l = rPixX*rPixY; l--; )
                    eRData[l] *= eRData[l];
            }
            else {
                eRData = makeNoiseImage4(tRData, 1./tGain, tRdnoise/tGain, rPixX, rPixY);
            }

            fprintf(stderr, "\n Convolving...\n");
            if (dump_dir) { DUMP_ALL(dump_dir, i, 12, "pre"); }
            spatial_convolve(tRData, &eRData, rPixX, rPixY, tKerSol, oRData, mtsRData, kcStep, hwKernel, fwKernel, kernel, kernel_coeffs, convolveVariance, kerFracMask, mRData, rPixX, rPixY, nCompKer, kerOrder, kernel_vec);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 12, "post"); }

            if (dump_dir) { DUMP_ALL(dump_dir, i, 13, "pre"); }
            for (l = hwKernel; l < rPixY - hwKernel; l++) 
                for (k = hwKernel; k < rPixX - hwKernel; k++) 
                    oRData[k+rPixX*l] += get_background(k, l, tKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 13, "post"); }

            if (dump_dir) { DUMP_ALL(dump_dir, i, 14, "pre"); }
            sumKernel = make_kernel(rXMin, rYMin, tKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Sum Kernel at %d,%d: %f\n", rXMin, rYMin, sumKernel);
            sumKernel = make_kernel(rXMax, rYMax, tKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Sum Kernel at %d,%d: %f\n", rXMax, rYMax, sumKernel);
            sumKernel = make_kernel(rPixX/2, rPixY/2, tKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Using Kernel Sum = %f\n\n", sumKernel);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 14, "post"); }


            if (dump_dir) { DUMP_ALL(dump_dir, i, 15, "pre"); }
            fset(tRData, fillValNoise, rPixX, rPixY);
            if (iNoiseFullData) {
                extract_subregion_flt(iNoiseFullData, iNx, rXBMin, rYBMin, rXBMax, rYBMax, tRData, rPixX);

                for (l = rPixX*rPixY; l--; )
                    tRData[l] *= tRData[l];
            }
            else {
                tRData = makeNoiseImage4(iRData, 1./iGain, iRdnoise/iGain, rPixX, rPixY);
            }

            for (l = rPixX*rPixY; l--; ) 
                tRData[l] = sqrt(tRData[l] + eRData[l]);
            free(eRData);
            eRData = NULL;
            if (dump_dir) { DUMP_ALL(dump_dir, i, 15, "post"); }

            for (l = rPixX*rPixY; l--; ) {
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (iRData[l] == fillVal);
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL)  * (iRData[l] >= iUThresh);
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL)  * (iRData[l] <= iLThresh);
                mRData[l] |= misRData[l];
                mRData[l] |= FLAG_OUTPUT_ISBAD * ((misRData[l] & FLAG_INPUT_ISBAD) > 0);
            }

            if (savexyflag){
                int si, sc, idx2, cnt2 = 0;
                for (si = 0; si < ntS; si++) cnt2 += ctStamps[si].nss;
                stats[i].savexyEntries = (savexy_entry *)malloc(cnt2 * sizeof(savexy_entry));
                stats[i].nSavexyEntries = cnt2;
                stats[i].savexyXmin = rXBMin;
                stats[i].savexyYmin = rYBMin;
                idx2 = 0;
                for (si = 0; si < ntS; si++) {
                    for (sc = 0; sc < ctStamps[si].nss; sc++) {
                        stats[i].savexyEntries[idx2].x = ctStamps[si].xss[sc];
                        stats[i].savexyEntries[idx2].y = ctStamps[si].yss[sc];
                        if (sc == ctStamps[si].sscnt) stats[i].savexyEntries[idx2].isUsed = 1;
                        else if (sc < ctStamps[si].sscnt) stats[i].savexyEntries[idx2].isUsed = -1;
                        else stats[i].savexyEntries[idx2].isUsed = 0;
                        idx2++;
                    }
                }
            }

            if (sameConv) {
                localForceConvolve = "t";
                if (ciStamps) free (ciStamps);
                ciStamps = NULL;
            }

        }
        else {
            fprintf(stderr, "\n\n Region %d,%d %d,%d : Convolving IMAGE\n", rXMin, rXMax, rYMin, rYMax);

            freeStampMem(ctStamps, nStamps, nCompKer, nBGVectors, nC);
            /*allocateStamps(ctStamps, nStamps);*/
            nS = niS;

            // if (!(kernelImIn))
                if (dump_dir) { DUMP_ALL(dump_dir, i, 11, "pre"); }
                fitKernel(ciStamps, tRData, iRData, oRData, iKerSol, &meansigSubstamps, &scatterSubstamps, &NskippedSubstamps, nCompKer, kerOrder, bgOrder, verbose, nS, indx, fwKSStamp, hwKSStamp, rPixX, rPixY, figMerit, kerSigReject, statSig, mRData, temp, ngauss, deg_fixe, hwKernel, fwKernel, usePCA, filter_x, filter_y, PCA, fillVal);
                if (dump_dir) { DUMP_ALL(dump_dir, i, 11, "post"); }

            oRData = (float *)realloc(oRData, rPixX*rPixY*sizeof(float));
            fset(oRData, fillVal, rPixX, rPixY);

            for (l = rPixX*rPixY; l--; ) {
                misRData[l] |= (FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (iRData[l] == fillVal);
                misRData[l] |= (FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL)  * (iRData[l] >= iUThresh);
                misRData[l] |= (FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL)  * (iRData[l] <= iLThresh);
            }

            memset(mRData, 0, rPixX*rPixY*sizeof(int));

            eRData = (float *)realloc(eRData, rPixX*rPixY*sizeof(float));
            fset(eRData, fillValNoise, rPixX, rPixY);

            if (iNoiseFullData) {
                extract_subregion_flt(iNoiseFullData, iNx, rXBMin, rYBMin, rXBMax, rYBMax, eRData, rPixX);

                for (l = rPixX*rPixY; l--; )
                    eRData[l] *= eRData[l];
            }
            else {
                eRData = makeNoiseImage4(iRData, 1./iGain, iRdnoise/iGain, rPixX, rPixY);
            }

            fprintf(stderr, "\n Convolving...\n");
            if (dump_dir) { DUMP_ALL(dump_dir, i, 12, "pre"); }
            spatial_convolve(iRData, &eRData, rPixX, rPixY, iKerSol, oRData, misRData, kcStep, hwKernel, fwKernel, kernel, kernel_coeffs, convolveVariance, kerFracMask, mRData, rPixX, rPixY, nCompKer, kerOrder, kernel_vec);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 12, "post"); }

            if (dump_dir) { DUMP_ALL(dump_dir, i, 13, "pre"); }
            for (l = hwKernel; l < rPixY - hwKernel; l++) 
                for (k = hwKernel; k < rPixX - hwKernel; k++) 
                    oRData[k+rPixX*l] += get_background(k, l, iKerSol, nCompKer, kerOrder, bgOrder, rPixX, rPixY);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 13, "post"); }

            if (dump_dir) { DUMP_ALL(dump_dir, i, 14, "pre"); }
            sumKernel = make_kernel(rXMin, rYMin, iKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Sum Kernel at %d,%d: %f\n", rXMin, rYMin, sumKernel);
            sumKernel = make_kernel(rXMax, rYMax, iKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Sum Kernel at %d,%d: %f\n", rXMax, rYMax, sumKernel);
            sumKernel = make_kernel(rPixX/2, rPixY/2, iKerSol, rPixX, rPixY, nCompKer, kerOrder, fwKernel, kernel_vec, kernel_coeffs, kernel);
            fprintf(stderr, " Using Kernel Sum = %f\n\n", sumKernel);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 14, "post"); }


            if (dump_dir) { DUMP_ALL(dump_dir, i, 15, "pre"); }
            fset(iRData, fillValNoise, rPixX, rPixY);
            if (tNoiseFullData) {
                extract_subregion_flt(tNoiseFullData, tNx, rXBMin, rYBMin, rXBMax, rYBMax, iRData, rPixX);

                for (l = rPixX*rPixY; l--; )
                    iRData[l] *= iRData[l];
            }
            else {
                iRData = makeNoiseImage4(tRData, 1./tGain, tRdnoise/tGain, rPixX, rPixY);
            }

            for (l = rPixX*rPixY; l--; ) 
                iRData[l] = sqrt(iRData[l] + eRData[l]);
            free(eRData);
            eRData = NULL;
            if (dump_dir) { DUMP_ALL(dump_dir, i, 15, "post"); }

            for (l = rPixX*rPixY; l--; ) {
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL) * (tRData[l] == fillVal);
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL)  * (tRData[l] >= tUThresh);
                mRData[l] |= (FLAG_OUTPUT_ISBAD | FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL)  * (tRData[l] <= tLThresh);
                mRData[l] |= mtsRData[l];
                mRData[l] |= FLAG_OUTPUT_ISBAD * ((mtsRData[l] & FLAG_INPUT_ISBAD) > 0);
            }

            if (savexyflag){
                int si, sc, idx2, cnt2 = 0;
                for (si = 0; si < niS; si++) cnt2 += ciStamps[si].nss;
                stats[i].savexyEntries = (savexy_entry *)malloc(cnt2 * sizeof(savexy_entry));
                stats[i].nSavexyEntries = cnt2;
                stats[i].savexyXmin = rXBMin + 1;
                stats[i].savexyYmin = rYBMin + 1;
                idx2 = 0;
                for (si = 0; si < niS; si++) {
                    for (sc = 0; sc < ciStamps[si].nss; sc++) {
                        stats[i].savexyEntries[idx2].x = ciStamps[si].xss[sc];
                        stats[i].savexyEntries[idx2].y = ciStamps[si].yss[sc];
                        if (sc == ciStamps[si].sscnt) stats[i].savexyEntries[idx2].isUsed = 1;
                        else if (sc < ciStamps[si].sscnt) stats[i].savexyEntries[idx2].isUsed = -1;
                        else stats[i].savexyEntries[idx2].isUsed = 0;
                        idx2++;
                    }
                }
            }

            if (sameConv) {
                localForceConvolve = "i";
                if (ctStamps) free(ctStamps);
                ctStamps = NULL;
            }

        }

        for (l = 0; l < rPixY; l++) {
            for (k = 0; k < hwKernel; k++)
                mRData[k+rPixX*l] |= FLAG_OUTPUT_ISBAD;
            for (k = rPixX - hwKernel; k < rPixX; k++) 
                mRData[k+rPixX*l] |= FLAG_OUTPUT_ISBAD;
        }
        for (l = 0; l < hwKernel; l++) 
            for (k = hwKernel; k < rPixX - hwKernel; k++)
                mRData[k+rPixX*l] |= FLAG_OUTPUT_ISBAD;
        for (l = rPixY - hwKernel; l < rPixY; l++) 
            for (k = hwKernel; k < rPixX - hwKernel; k++)
                mRData[k+rPixX*l] |= FLAG_OUTPUT_ISBAD;


        fprintf(stderr, " Creating and writing output images...\n");

        inv1 = 1. / sumKernel;

        if (convOut) {
            for (l = hwKernel; l < rPixY - hwKernel; l++) {
                for (k = hwKernel; k < rPixX - hwKernel; k++) {
                    if ( (strncmp(photNormalize, "u", 1)!=0) &&
                         ( ( convTmpl && strncmp(photNormalize, "t", 1)==0 ) ||
                           (!convTmpl && strncmp(photNormalize, "i", 1)==0 ) ) )
                        oRData[k+rPixX*l] *= inv1;
                }
            }

            if (dump_dir) { DUMP_ALL(dump_dir, i, 18, "pre"); }
            insert_subregion_flt(oRData, rPixX, convOut, oNx, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY, xBufLo, yBufLo);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 18, "post"); }

            for (l = hwKernel; l < rPixY - hwKernel; l++) {
                for (k = hwKernel; k < rPixX - hwKernel; k++) {
                    if ( (strncmp(photNormalize, "u", 1)!=0) &&
                         ( ( convTmpl && strncmp(photNormalize, "t", 1)==0 ) ||
                           (!convTmpl && strncmp(photNormalize, "i", 1)==0 ) ) )
                        oRData[k+rPixX*l] *= sumKernel;
                }
            }
        }

        if (convTmpl) {
            if (dump_dir) { DUMP_ALL(dump_dir, i, 16, "pre"); }
            for (l = hwKernel; l < rPixY - hwKernel; l++) {
                for (k = hwKernel; k < rPixX - hwKernel; k++) {
                    m = k+rPixX*l;
                    oRData[m] -= iRData[m];
                    if ((strncmp(photNormalize, "u", 1)!=0) && (strncmp(photNormalize, "t", 1)==0)) {
                        oRData[m] *= inv1;

                        tRData[m] *= inv1;
                    }
                    oRData[m] *= -1.;
                }
            }
            if (dump_dir) { DUMP_ALL(dump_dir, i, 16, "post"); }

            // if (!(kernelImIn)) {
                if (strncmp(figMerit, "v", 1)==0) {
                    if (dump_dir) { DUMP_ALL(dump_dir, i, 17, "pre"); }
                    temp2 = (float *)calloc(nS, sizeof(float));
                    k     = 0;
                    for (l = 0; l < nS; l++) {
                        if (ctStamps[l].sscnt < ctStamps[l].nss) {
                            getFinalStampSig(&ctStamps[l], oRData, tRData, &sum, fwKSStamp, hwKSStamp, rPixX, mRData);
                            temp2[k++] = sum;
                        }
                    }
                    sigma_clip(temp2, k, &meansigSubstampsF, &scatterSubstampsF, 10, statSig);
                    fprintf(stderr, "   FINAL Mean sig: %6.3f stdev: %6.3f\n", meansigSubstampsF, scatterSubstampsF);
                    free(temp2);
                    if (dump_dir) { DUMP_ALL(dump_dir, i, 17, "post"); }
                }
            // }
            freeStampMem(ctStamps, nStamps, nCompKer, nBGVectors, nC); 

        }
        else {
            if (dump_dir) { DUMP_ALL(dump_dir, i, 16, "pre"); }
            for (l = hwKernel; l < rPixY - hwKernel; l++) {
                for (k = hwKernel; k < rPixX - hwKernel; k++) {
                    m = k+rPixX*l;
                    oRData[m] -= tRData[m];
                    if ((strncmp(photNormalize, "u", 1)!=0) && (strncmp(photNormalize, "i", 1)==0)) {
                        oRData[m] *= inv1;

                        iRData[m] *= inv1;
                    }
                }
            }
            if (dump_dir) { DUMP_ALL(dump_dir, i, 16, "post"); }

            // if (!(kernelImIn)) {
                if (strncmp(figMerit, "v", 1)==0) {
                    if (dump_dir) { DUMP_ALL(dump_dir, i, 17, "pre"); }
                    temp2 = (float *)calloc(nS, sizeof(float));
                    k     = 0;
                    for (l = 0; l < nS; l++) {
                        if (ciStamps[l].sscnt < ciStamps[l].nss) {
                            getFinalStampSig(&ciStamps[l], oRData, iRData, &sum, fwKSStamp, hwKSStamp, rPixX, mRData);
                            temp2[k++] = sum;
                        }
                    }
                    sigma_clip(temp2, k, &meansigSubstampsF, &scatterSubstampsF, 10, statSig);
                    fprintf(stderr, "    FINAL Mean sig: %6.3f stdev: %6.3f\n", meansigSubstampsF, scatterSubstampsF);
                    free(temp2);
                    if (dump_dir) { DUMP_ALL(dump_dir, i, 17, "post"); }
                }
            // }
            freeStampMem(ciStamps, nStamps, nCompKer, nBGVectors, nC); 

        }

        fprintf(stderr, " Getting diffim stats for GOOD pixels : \n");
        getStampStats3(oRData, 0, 0, rPixX, rPixY,
                       &sum, &mean, &median,
                       &mode, &sd, &fwhm, &lfwhm, 0x0, 0xffff, 5, rPixX, mRData, statSig);
        fprintf(stderr, "   Mean   : %.2f\n", mean);
        fprintf(stderr, "   Median : %.2f\n", median);
        fprintf(stderr, "   Mode   : %.2f\n", mode);
        fprintf(stderr, "   Stdev  : %.2f\n", sd);
        if (verbose >= 2) fprintf(stderr, "   FWHM   : %.2f\n", fwhm);
        if (verbose >= 2) fprintf(stderr, "   lFWHM  : %.2f\n", lfwhm);

        fprintf(stderr, " Getting noiseim stats for GOOD pixels : \n");
        if (convTmpl) {
            getStampStats3(tRData, 0, 0, rPixX, rPixY,
                           &nsum, &nmean, &nmedian,
                           &nmode, &nsd, &nfwhm, &nlfwhm, 0x0, 0xffff, 5, rPixX, mRData, statSig);
            getNoiseStats3(oRData, tRData, &x2norm, &nx2norm, 0x0, 0xffff, rPixX, rPixY, mRData);
        }

        else {
            getStampStats3(iRData, 0, 0, rPixX, rPixY,
                           &nsum, &nmean, &nmedian,
                           &nmode, &nsd, &nfwhm, &nlfwhm, 0x0, 0xffff, 5, rPixX, mRData, statSig);
            getNoiseStats3(oRData, iRData, &x2norm, &nx2norm, 0x0, 0xffff, rPixX, rPixY, mRData);
        }

        fprintf(stderr, "   Mean   : %.2f\n", nmean);
        fprintf(stderr, "   Median : %.2f\n", nmedian);
        fprintf(stderr, "   Mode   : %.2f\n", nmode);
        fprintf(stderr, "   Stdev  : %.2f\n", nsd);
        if (verbose >= 2) fprintf(stderr, "   FWHM   : %.2f\n", nfwhm);
        if (verbose >= 2) fprintf(stderr, "   lFWHM  : %.2f\n", nlfwhm);

        // if (!(kernelImIn))
            fprintf(stderr, " Emperical / Expected Noise for GOOD pixels = %.2f\n", sd / nmean);
        fprintf(stderr, " X2NORM = %.2f\n\n", x2norm);

        // if (!(kernelImIn)) {
            diffrat = sd / nmean;
        // }

        fprintf(stderr, " Getting diffim stats for OK pixels : \n");
        getStampStats3(oRData, 0, 0, rPixX, rPixY,
                       &summ, &meanm, &medianm,
                       &modem, &sdm, &fwhmm, &lfwhmm, 0xff, FLAG_OUTPUT_ISBAD, 5, rPixX, mRData, statSig);
        fprintf(stderr, "   Mean   : %.2f\n", meanm);
        fprintf(stderr, "   Median : %.2f\n", medianm);
        fprintf(stderr, "   Mode   : %.2f\n", modem);
        fprintf(stderr, "   Stdev  : %.2f\n", sdm);
        if (verbose >= 2) fprintf(stderr, "   FWHM   : %.2f\n", fwhmm);
        if (verbose >= 2) fprintf(stderr, "   lFWHM  : %.2f\n", lfwhmm);

        fprintf(stderr, " Getting noiseim stats for OK pixels : \n");
        if (convTmpl) 
            getStampStats3(tRData, 0, 0, rPixX, rPixY,
                           &nsumm, &nmeanm, &nmedianm,
                           &nmodem, &nsdm, &nfwhmm, &nlfwhmm, 0xff, FLAG_OUTPUT_ISBAD, 5, rPixX, mRData, statSig);
        else 
            getStampStats3(iRData, 0, 0, rPixX, rPixY,
                           &nsumm, &nmeanm, &nmedianm,
                           &nmodem, &nsdm, &nfwhmm, &nlfwhmm, 0xff, FLAG_OUTPUT_ISBAD, 5, rPixX, mRData, statSig);

        fprintf(stderr, "   Mean   : %.2f\n", nmeanm);
        fprintf(stderr, "   Median : %.2f\n", nmedianm);
        fprintf(stderr, "   Mode   : %.2f\n", nmodem);
        fprintf(stderr, "   Stdev  : %.2f\n", nsdm);
        if (verbose >= 2) fprintf(stderr, "   FWHM   : %.2f\n", nfwhmm);
        if (verbose >= 2) fprintf(stderr, "   lFWHM  : %.2f\n", nlfwhmm);

        // if (!(kernelImIn))
            fprintf(stderr, " Emperical / Expected Noise for OK pixels = %.2f\n\n", sdm / nmeanm);

        if (rescaleOK) {
            // if (!(kernelImIn))
                diffrat = (sdm / nmeanm) / diffrat;

            if (diffrat > 1) {
                fprintf(stderr, " Scale OK pixel noise by = %.2f\n", diffrat);
                if (convTmpl)
                    for (l = rPixX*rPixY; l--; ) {
                        if ( (mRData[l] & 0xff) && (!(mRData[l] & FLAG_OUTPUT_ISBAD)) )
                            tRData[l] *= diffrat;
                    }
                else
                    for (l = rPixX*rPixY; l--; ) {
                        if ( (mRData[l] & 0xff) && (!(mRData[l] & FLAG_OUTPUT_ISBAD)) )
                            iRData[l] *= diffrat;
                    }
            }
            else
                fprintf(stderr, " Leave OK pixel noise as-is\n");
        }

        if (kfSpreadMask2 >= 0) {
            if (convTmpl) {
                for (l = hwKernel; l < rPixY - hwKernel; l++) {
                    for (k = hwKernel; k < rPixX - hwKernel; k++) {
                        if (mRData[k+rPixX*l] & FLAG_OUTPUT_ISBAD) {
                            oRData[k+rPixX*l] = fillVal;
                            tRData[k+rPixX*l] = fillValNoise;
                        }
                    }
                }
            }
            else {
                for (l = hwKernel; l < rPixY - hwKernel; l++) {
                    for (k = hwKernel; k < rPixX - hwKernel; k++) {
                        if (mRData[k+rPixX*l] & FLAG_OUTPUT_ISBAD) {
                            oRData[k+rPixX*l] = fillVal;
                            iRData[k+rPixX*l] = fillValNoise;
                        }
                    }
                }
            }
        }

        if (dump_dir) { DUMP_ALL(dump_dir, i, 19, "pre"); }
        insert_subregion_flt(oRData, rPixX, diffOut, oNx, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY, xBufLo, yBufLo);
        if (dump_dir) { DUMP_ALL(dump_dir, i, 19, "post"); }

        if (noiseOut) {
            if (dump_dir) { DUMP_ALL(dump_dir, i, 20, "pre"); }
            if (convTmpl) {
                insert_subregion_flt(tRData, rPixX, noiseOut, oNx, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY, xBufLo, yBufLo);
            }
            else {
                insert_subregion_flt(iRData, rPixX, noiseOut, oNx, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY, xBufLo, yBufLo);
            }
            if (dump_dir) { DUMP_ALL(dump_dir, i, 20, "post"); }
        }

        if (maskOut) {
            if (dump_dir) { DUMP_ALL(dump_dir, i, 21, "pre"); }
            insert_subregion_int(mRData, rPixX, maskOut, oNx, fpixelOutX, fpixelOutY, lpixelOutX, lpixelOutY, xBufLo, yBufLo);
            if (dump_dir) { DUMP_ALL(dump_dir, i, 21, "post"); }
        }

        stats[i].convTmpl            = convTmpl;
        stats[i].sumKernel           = sumKernel;
        stats[i].meansigSubstamps    = meansigSubstamps;
        stats[i].scatterSubstamps    = scatterSubstamps;
        stats[i].meansigSubstampsF   = meansigSubstampsF;
        stats[i].scatterSubstampsF   = scatterSubstampsF;
        stats[i].x2norm              = x2norm;
        stats[i].nx2norm             = nx2norm;
        stats[i].mean                = mean;
        stats[i].sd                  = sd;
        stats[i].nmean               = nmean;
        stats[i].meanm               = meanm;
        stats[i].sdm                 = sdm;
        stats[i].nmeanm              = nmeanm;
        stats[i].diffrat             = diffrat;

        if (convTmpl)
            stats[i].kerSol = tKerSol;
        else
            stats[i].kerSol = iKerSol;
        if (dump_dir) { DUMP_ALL(dump_dir, i, 22, "pre"); }
        if (dump_dir) { DUMP_ALL(dump_dir, i, 22, "post"); }

        fprintf(stderr,"Region %i finished\n\n",i);

region_cleanup:
        free(tRData);   tRData   = NULL;
        free(iRData);   iRData   = NULL;
        free(oRData);   oRData   = NULL;
        if (eRData) { free(eRData); eRData = NULL; }
        free(mRData);   mRData   = NULL;
        free(misRData); misRData = NULL;
        free(mtsRData); mtsRData = NULL;

        if (ctStamps) { free(ctStamps); ctStamps = NULL; }
        if (ciStamps) { free(ciStamps); ciStamps = NULL; }
        if (convTmpl) {
            tKerSol = NULL;
            if (iKerSol) { free(iKerSol); iKerSol = NULL; }
        } else {
            iKerSol = NULL;
            if (tKerSol) { free(tKerSol); tKerSol = NULL; }
        }
    }

    if (temp)          free(temp);
    if (indx)          free(indx);
    if (kernel)        free(kernel);
    if (kernel_coeffs) free(kernel_coeffs);
    if (kernel_vec)    free(kernel_vec);
    if (filter_x)      free(filter_x);
    if (filter_y)      free(filter_y);
    if (check_vec)     free(check_vec);
    if (check_stack)   free(check_stack);

    for (i = 0; i < nC; i++)
        if (check_mat[i]) free(check_mat[i]);
    if (check_mat)       free(check_mat);

    return 0;
}

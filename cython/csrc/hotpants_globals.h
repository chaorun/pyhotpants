#ifndef HOTPANTS_GLOBALS_H
#define HOTPANTS_GLOBALS_H

typedef struct
{
   int       x0,y0;       /* origin of stamp in region coords*/
   int       x,y;         /* center of stamp in region coords*/
   int       nx,ny;       /* size of stamp */
   int       *xss;        /* x location of test substamp centers */
   int       *yss;        /* y location of test substamp centers */
   int       nss;         /* number of detected substamps, 1 .. nss     */
   int       sscnt;       /* represents which nss to use,  0 .. nss - 1 */
   double    **vectors;   /* contains convolved image data */
   double    *krefArea;   /* contains kernel substamp data */
   double    **mat;       /* fitting matrices */
   double    *scprod;     /* kernel sum solution */
   double    sum;         /* sum of fabs, for sigma use in check_stamps */
   double    mean;
   double    median;
   double    mode;        /* sky estimate */
   double    sd;
   double    fwhm;
   double    lfwhm;
   double    chi2;        /* residual in kernel fitting */
   double    norm;        /* kernel sum */
   double    diff;        /* (norm - mean_ksum) * sqrt(sum) */
} stamp_struct;

typedef struct {
   int    x, y;
   int    isUsed;
} savexy_entry;

typedef struct {
   int    convTmpl;
   double sumKernel;
   double meansigSubstamps;
   double scatterSubstamps;
   double meansigSubstampsF;
   double scatterSubstampsF;
   double x2norm;
   int    nx2norm;
   double mean, sd, nmean;
   double meanm, sdm, nmeanm;
   double diffrat;
   double *kerSol;
   int    nSavexyEntries;
   savexy_entry *savexyEntries;
   long   savexyXmin, savexyYmin;
} region_stats;

#endif

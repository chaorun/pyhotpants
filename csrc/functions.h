#include <fitsio.h>
#include "hotpants_globals.h"

/* Alard.c */
void        getKernelVec(int, int *, double **, int, int, int, float *, double *, double *, float **);
int         fillStamp(stamp_struct *, float *, float *, int, int, int, int, int *, int, int, int, int, int, int, int, int, double *, double *, float *, float **, float, int *);
double      *kernel_vector(int, int, int, int, int *, int, int, int, float *, double *, double *, double **, float **);
double      *kernel_vector_PCA(int, int, int, int, int *, int, float **, double **);
void        xy_conv_stamp(stamp_struct *, float *, int, int, int, int, int, int, int, int, double *, double *, float *, float **);
void        xy_conv_stamp_PCA(stamp_struct *, float *, int, int, int, int, int, int, int, float **);
void        fitKernel(stamp_struct *, float *, float *, float *, double *, double *, double *, int *, int, int, int, int, int, int *, int, int, int, int, char *, float, float, int *, float *, int, int *, int, int, int, double *, double *, float **, float);
void        build_matrix0(stamp_struct *, int, int, int, int);
void        build_scprod0(stamp_struct *, float *, int, int, int, int, int, int);
double      check_stamps(stamp_struct *, int, float *, float *, int, int, int, int, int, int *, double **, double *, double *, char *, char *, float, float, int, int, int, int, int, double **, double *, double *, int *, float *);
void        build_matrix(stamp_struct *, int, double **, int, int, int, int, int, int, int, double **);
void        build_scprod(stamp_struct *, int, float *, double *, int, int, int, int, int, int, double **);
void        getStampSig(stamp_struct *, double *, float *, double *, double *, double *, int, int, int, int, int *, char *, float *, float, int, int, int);
void        getFinalStampSig(stamp_struct *, float *, float *, double *, int, int, int, int *);
char        check_again(stamp_struct *, double *, float *, float *, float *, double *, double *, int *, int, int, char *, float, float, int, int, int, int, int *, float *, int, int, int, int, int *, int, int, int, double *, double *, float **, float);
void        spatial_convolve(float *, float **, int, int, double *, float *, int *, int, int, int, double *, double *, int, float, int *, int, int, int, int, double **);
double      make_kernel(int, int, double *, int, int, int, int, int, double **, double *, double *);
double      get_background(int, int, double *, int, int, int, int, int);
void        make_model(stamp_struct *, double *, float *, int, int, int, int, int);
int         ludcmp(double **, int, int *, double *);
void        lubksb(double **, int, int *, double *);

/* Functions.c */
int         allocateStamps(stamp_struct *, int, int, int, int, int, int);
void        buildStamps(int, int, int, int, int *, int *, int, int, int,
                        stamp_struct *, stamp_struct *, float *, float *,
                        float, float,
                        int, char *, int, int, float, float, int, int, int, float, int *, float);
void        cutStamp(float *, float *, int, int, int, int, int, stamp_struct *);
void        buildSigMask(stamp_struct *, int, int, int *);
int         cutSStamp(stamp_struct *, float *, int, int, float, int, int *, int);
double      checkPsfCenter(float *, int, int, int, int, int, int, double, float, float,
			   int, int, int, int, int, int, int *, float);
int         getPsfCenters(stamp_struct *, float *, int, int, double, int, int, int, int, int, int *, float, int);
int         getPsfCentersORIG(stamp_struct *, float *, int, int, double, int, int);
int         getStampStats3(float *, int, int, int, int, double *, double *, double *, double *, double *, double *, double *, int, int, int, int, int *, float);
void        getNoiseStats3(float *, float *, double *, int *, int, int, int, int, int *);
int         stampStats(double *, int *, long, double *, double *, double *, double *, double *, double *, double *);
int         sigma_clip(float *, int, double *, double *, int, float);
float      *calculateAvgNoise(float *, int *, int, int, int, int, int); /* not used? */
void        freeStampMem(stamp_struct *, int, int, int, int);
/*int         makeNoiseImage2(float **, float, float, float *, float, float, int, int, double *);*/
/*int         makeNoiseImage3(float *, float, float, float *, float, float, int, int);*/
float       *makeNoiseImage4(float *, float, float, int, int);
void        getKernelInfo(char *, int *, int *, int *, int *, int **, float **, char **);
void        readKernel(char *, int, double **, double **, int *, int *, int *, int *, double *, double *, double *, double *, double *, int *, int, char **);
void        fits_get_kernel_btbl(fitsfile *, double **, int, int);
void        spreadMask(int *, int, int, int);
void        makeInputMask(float *, float *, int *, int, int, float, float, float, float, float, int, float);
void        makeOutputMask(float *, float, float, float *, float, float, int *, int *, int *);
int         hp_fits_copy_header(fitsfile *, fitsfile *, int *);
void        hp_fits_correct_data(float *, int, float, float, int, int *);
void        hp_fits_correct_data_int(int *, int, float, float, int, int *);
int         hp_fits_write_subset(fitsfile *, long, long, long *,
                                 float *, int *,
                                 int, float, float,
                                 int, int, int, int, int, int,
                                 int, int *);
int         hp_fits_write_subset_int(fitsfile *, long, long, long *,
				     int *, int *,
				     int, float, float,
				     int, int, int, int, int, int,
				     int, int *);
void        fset(float *, double, int, int);
void        dfset(double *, double, int, int);
void        extract_subregion_flt(float *, long, int, int, int, int, float *, int);
void        extract_subregion_int(int *, long, int, int, int, int, int *, int);
void        insert_subregion_flt(float *, int, float *, long, int, int, int, int, int, int);
void        insert_subregion_int(int *, int, int *, long, int, int, int, int, int, int);
void        printError(int);
double      ran1(int *);
void        quick_sort(double *,int *, int);
int         imin(int, int);
int         imax(int, int);

/* armin */
void savexy(stamp_struct *, int, long, long, int, char *, int);
void loadxyfile(char *, int, float **, float **, int *);

/* mysinc.c */
int    swarp_remap(float *, float *, double, double, int, int,
		   int, float *, float *, int);
double luptonD(int, double);
double luptonD_appx(int, double);
void   make_lupton_kernel(double, double *, int);
void   lanczos3(double, double *);
void   lanczos4(double, double *);
void   lanczos(double, double *, int);



/* Vargs.c */
void        vargs(int, char *[],
                  char **, char **, char **,
                  char **, char **, char **, char **,
                  char **, char **, char **,
                  float *, float *, float *, float *, float *,
                  float *, float *, float *, float *, float *,
                  float *, float *,
                  float *, float *, float *, float *,
                  int *, int *, float *, float *, float *,
                  int *, int *, int *,
                  float *, float *,
                  int *, int *, int *, int *, int *, int *,
                  char **, char **, int *,
                  int *, int *, int *, int *, int *,
                  char **, int *, int *,
                  float *, float *, float *,
                  char **, char **, char **,
                  int *, int *, float *, float *,
                  int *, int *, int *, int *,
                  int *, int *, int *, int *,
                  char **, char **, char **, char **,
                  int *, int *, int *,
                  int *, float ***,
                  int *, int **, float **,
                  char *, char *);

/* jtwarp.c */
/*
void         jtrebin(int, int, float *, int, int, int, int,
		     double *, double *, double *, float *, float *,
		     float, float, int, float, float, float, float, float);
int          jtdotri(int, int, double *, double [], double [], double, float *, int []);
void         jtsprinkle(int, int, double *, double, double, double, double,
			double, float, float *, int []);
*/

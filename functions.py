import numpy as np
import numba
import math
import logging
from scipy.ndimage import maximum_filter
logger = logging.getLogger('hotpants')

ZEROVAL = 1e-10
MAXVAL = 1e10
FLAG_BAD_PIXVAL = 0x01
FLAG_SAT_PIXEL = 0x02
FLAG_NEG_PIXEL = 0x04
FLAG_SPREAD = 0x40
FLAG_MASKED = 0x80
FLAG_BORDER = 0x100
FLAG_SUBREGION = 0x200
FLAG_INVALID = 0x400

FLAG_ISNAN = 0x08
FLAG_INPUT_ISBAD = 0x80
FLAG_T_BAD = 0x100
FLAG_T_SKIP = 0x200
FLAG_I_BAD = 0x400
FLAG_I_SKIP = 0x800

BUILD_STAMP_FLAT_CACHE = {}
LAST_SIGMA_CLIP = None
LAST_N = None
LAST_MEDIAN = None


# class StampsArray:
#     def __init__(self, nS, nKSStamps, fwKSStamp, nCompKer, nBGVectors, nC):
#         fwSq = fwKSStamp * fwKSStamp
#         nVec = nCompKer + nBGVectors
#
#         self.nS = nS
#         self.nKSStamps = nKSStamps
#         self.fwKSStamp = fwKSStamp
#         self.nCompKer = nCompKer
#         self.nBGVectors = nBGVectors
#         self.nC = nC
#         self.fwSq = fwSq
#         self.nVec = nVec
#
#         self.sscnt   = np.zeros(nS, dtype=np.int32)
#         self.nss     = np.zeros(nS, dtype=np.int32)
#         self.x0      = np.zeros(nS, dtype=np.int32)
#         self.y0      = np.zeros(nS, dtype=np.int32)
#         self.x       = np.zeros(nS, dtype=np.int32)
#         self.y       = np.zeros(nS, dtype=np.int32)
#
#         self.xss     = np.zeros((nS, nKSStamps), dtype=np.int32)
#         self.yss     = np.zeros((nS, nKSStamps), dtype=np.int32)
#
#         self.vectors  = np.zeros((nS, nVec, fwSq), dtype=np.float64)
#         self.mat      = np.zeros((nS, nC, nC), dtype=np.float64)
#         self.scprod   = np.zeros((nS, nC), dtype=np.float64)
#         self.krefArea = np.zeros((nS, fwSq), dtype=np.float64)
#
#         self.chi2    = np.zeros(nS, dtype=np.float64)
#         self.norm    = np.zeros(nS, dtype=np.float64)
#         self.diff    = np.zeros(nS, dtype=np.float64)
#         self.sum_val = np.zeros(nS, dtype=np.float64)
#         self.mean_val = np.zeros(nS, dtype=np.float64)
#         self.median  = np.zeros(nS, dtype=np.float64)
#         self.mode    = np.zeros(nS, dtype=np.float64)
#         self.sd      = np.zeros(nS, dtype=np.float64)
#         self.fwhm    = np.zeros(nS, dtype=np.float64)
#         self.lfwhm   = np.zeros(nS, dtype=np.float64)
#
#         self.valid   = np.zeros(nS, dtype=np.bool_)
#         self.ntS     = 0
#
#     @staticmethod
#     def subset(src, mask):
#         nSub = mask.sum()
#         if nSub == 0:
#             return None
#         sa = StampsArray(nSub, src.nKSStamps, src.fwKSStamp, src.nCompKer, src.nBGVectors, src.nC)
#         sa.sscnt[:]     = src.sscnt[mask]
#         sa.nss[:]       = src.nss[mask]
#         sa.x0[:]        = src.x0[mask]
#         sa.y0[:]        = src.y0[mask]
#         sa.x[:]         = src.x[mask]
#         sa.y[:]         = src.y[mask]
#         sa.xss[:]       = src.xss[mask]
#         sa.yss[:]       = src.yss[mask]
#         sa.vectors[:]   = src.vectors[mask]
#         sa.mat[:]       = src.mat[mask]
#         sa.scprod[:]    = src.scprod[mask]
#         sa.krefArea[:]  = src.krefArea[mask]
#         sa.chi2[:]      = src.chi2[mask]
#         sa.norm[:]      = src.norm[mask]
#         sa.diff[:]      = src.diff[mask]
#         sa.sum_val[:]   = src.sum_val[mask]
#         sa.mean_val[:]  = src.mean_val[mask]
#         sa.median[:]    = src.median[mask]
#         sa.mode[:]      = src.mode[mask]
#         sa.sd[:]        = src.sd[mask]
#         sa.fwhm[:]      = src.fwhm[mask]
#         sa.lfwhm[:]     = src.lfwhm[mask]
#         sa.valid[:]     = src.valid[mask]
#         sa.ntS          = nSub
#         return sa
#
#     def deepCopy(self):
#         cp = StampsArray(self.nS, self.nKSStamps, self.fwKSStamp, self.nCompKer, self.nBGVectors, self.nC)
#         cp.sscnt[:] = self.sscnt
#         cp.nss[:] = self.nss
#         cp.x0[:] = self.x0
#         cp.y0[:] = self.y0
#         cp.x[:] = self.x
#         cp.y[:] = self.y
#         cp.xss[:] = self.xss
#         cp.yss[:] = self.yss
#         cp.vectors[:] = self.vectors
#         cp.mat[:] = self.mat
#         cp.scprod[:] = self.scprod
#         cp.krefArea[:] = self.krefArea
#         cp.chi2[:] = self.chi2
#         cp.norm[:] = self.norm
#         cp.diff[:] = self.diff
#         cp.sum_val[:] = self.sum_val
#         cp.mean_val[:] = self.mean_val
#         cp.median[:] = self.median
#         cp.mode[:] = self.mode
#         cp.sd[:] = self.sd
#         cp.fwhm[:] = self.fwhm
#         cp.lfwhm[:] = self.lfwhm
#         cp.valid[:] = self.valid
#         cp.ntS = self.ntS
#         return cp


class Ran1:
    M1 = 259200;  IA1 = 7141;  IC1 = 54773;  RM1 = 1.0 / 259200
    M2 = 134456;  IA2 = 8121;  IC2 = 28411;  RM2 = 1.0 / 134456
    M3 = 243000;  IA3 = 4561;  IC3 = 51349

    def __init__(self, idum):
        self.ix1 = 0
        self.ix2 = 0
        self.ix3 = 0
        self.r = [0.0] * 98
        self.iff = 0
        self.idum = int(idum)

    def __call__(self):
        if self.idum < 0 or self.iff == 0:
            self.iff = 1
            self.ix1 = (self.IC1 - self.idum) % self.M1
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            self.ix2 = self.ix1 % self.M2
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            self.ix3 = self.ix1 % self.M3
            for j in range(1, 98):
                self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
                self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
                self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
            self.idum = 1

        self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
        self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
        self.ix3 = (self.IA3 * self.ix3 + self.IC3) % self.M3
        j = 1 + (97 * self.ix3) // self.M3
        temp = self.r[j]
        self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
        return temp

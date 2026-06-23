import numpy as np
# 导入numpy科学计算库
import numba
# 导入numba JIT编译器

ZEROVAL = 1e-10
# 零值判定阈值：绝对值小于此值视为零
MAXVAL = 1e10
# 最大值判定阈值：超过此值视为异常
FLAG_BAD_PIXVAL = 0x01
# 坏像素值标记位（填充值/饱和/过低等）
FLAG_SAT_PIXEL = 0x02
# 饱和像素标记位
FLAG_LOW_PIXEL = 0x04
# 过低像素标记位
FLAG_ISNAN = 0x08
# NaN像素标记位
FLAG_BAD_CONV = 0x10
# 坏卷积像素标记位（核分数不足）
FLAG_INPUT_MASK = 0x20
# 输入遮罩标记位（用户提供的mask）
FLAG_OK_CONV = 0x40
# 正常卷积像素标记位
FLAG_INPUT_ISBAD = 0x80
# 输入坏像素标记位（综合坏像素标志）
FLAG_T_BAD = 0x100
# 模板区域边界标记位
FLAG_T_SKIP = 0x200
# 模板跳过标记位
FLAG_I_BAD = 0x400
# 图像区域边界标记位
FLAG_I_SKIP = 0x800
# 图像跳过标记位
FLAG_OUTPUT_ISBAD = 0x8000
# 输出坏像素标记位

BUILD_STAMP_FLAT_CACHE = {}
# Stamp扁平化缓存字典（键为buildstamps结果id，值为预提取的扁平数组）
LAST_SIGMA_CLIP = None
# 上次sigma_clip的结果缓存
LAST_N = None
# 上次sigma_clip的样本数缓存
LAST_MEDIAN = None
# 上次sigma_clip的中位数缓存



class Ran1:
# Park-Miller随机数生成器，三种LCG组合产生均匀分布[0,1)
    M1 = 259200;  IA1 = 7141;  IC1 = 54773;  RM1 = 1.0 / 259200
    # 第一个LCG参数：模数、乘子、增量、倒数
    M2 = 134456;  IA2 = 8121;  IC2 = 28411;  RM2 = 1.0 / 134456
    # 第二个LCG参数
    M3 = 243000;  IA3 = 4561;  IC3 = 51349
    # 第三个LCG参数

    def __init__(self, idum):
    # 初始化随机数种子和内部状态
        self.ix1 = 0
        # 第一个LCG状态
        self.ix2 = 0
        # 第二个LCG状态
        self.ix3 = 0
        # 第三个LCG状态
        self.r = [0.0] * 98
        # 混洗表（97个元素的缓冲区）
        self.iff = 0
        # 初始化标志
        self.idum = int(idum)
        # 存储种子值

    def __call__(self):
    # 每次调用生成一个[0,1)的均匀随机数
        if self.idum < 0 or self.iff == 0:
        # 首次调用或种子为负数时初始化混洗表
            self.iff = 1
            self.ix1 = (self.IC1 - self.idum) % self.M1
            # 用种子初始化ix1
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            # 生成第一个ix1（抛弃）
            self.ix2 = self.ix1 % self.M2
            # 生成ix2
            self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
            # 生成下一个ix1
            self.ix3 = self.ix1 % self.M3
            # 生成ix3
            for j in range(1, 98):
            # 填充混洗表（跳过0号位置）
                self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
                # 更新ix1
                self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
                # 更新ix2
                self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
                # 组合生成0-1之间的随机数
            self.idum = 1
            # 标记已初始化

        self.ix1 = (self.IA1 * self.ix1 + self.IC1) % self.M1
        # 更新ix1
        self.ix2 = (self.IA2 * self.ix2 + self.IC2) % self.M2
        # 更新ix2
        self.ix3 = (self.IA3 * self.ix3 + self.IC3) % self.M3
        # 更新ix3
        j = 1 + (97 * self.ix3) // self.M3
        # 从ix3计算混洗表索引j（1-97）
        temp = self.r[j]
        # 取出混洗表中的旧值作为本次输出
        self.r[j] = (self.ix1 + self.ix2 * self.RM2) * self.RM1
        # 用新生成的随机数替换混洗表中j位置的值
        return temp
        # 返回本次随机数

def sigma_clip_numpy(data, maxiter=10, stat_sig=3.0):
# Sigma裁剪函数：迭代剔除偏离均值超过stat_sig倍标准差的离群值
    count = len(data)
    # 数据点总数
    if count == 0:
    # 空数据
        return (0.0, MAXVAL, 1)
        # 返回失败

    arr = np.asarray(data, dtype=np.float32).astype(np.float64)
    # 先截断为float32再转float64，模拟C版本的float中间精度
    mask = np.zeros(count, dtype=bool)
    # 创建剔除遮罩数组（True表示被剔除）
    cnt = 0; ncnt = count
    # cnt=当前保留数，ncnt=上一轮保留数
    iternum = 0
    # 迭代计数器
    mean_val = 0.0; stdev_val = 0.0
    # 均值和标准差初始化为0

    while (ncnt != cnt) and (iternum < maxiter):
    # 当保留数不再减少或达到最大迭代次数时停止
        cnt = ncnt
        # 更新上一轮保留数
        good = arr[~mask]
        # 取出未被剔除的数据
        ncnt = len(good)
        # 当前保留数据点数
        if ncnt > 0:
        # 还有数据
            mean_val = float(good.mean())
            # 计算均值
        else:
            return (0.0, MAXVAL, 2)
            # 全部被剔除
        if ncnt > 1:
        # 至少2个数据才能算标准差
            stdev_val = float(good.std(ddof=1))
            # 计算样本标准差(ddof=1)
        else:
            return (mean_val, MAXVAL, 3)
            # 仅1个数据时标准差无效

        istdev = 1.0 / stdev_val
        # 标准差的倒数（用于归一化偏差）
        deviations = np.abs(good - mean_val) * istdev
        # 计算每个数据的标准化偏差|(x-μ)/σ|
        new_outliers = deviations > stat_sig
        # 标记偏差超过阈值的离群值
        good_indices = np.where(~mask)[0]
        # 获取当前保留数据在原数组中的索引
        mask[good_indices[new_outliers]] = True
        # 将新发现的离群值标记为剔除
        ncnt = count - int(np.sum(mask))
        # 计算新的保留数据点数
        iternum += 1
        # 迭代计数加1

    return (mean_val, stdev_val, 0)
    # 返回裁剪后的均值和标准差

@numba.jit(nopython=True)
def get_noise_stats3_numpy(data, noise, umask, smask, rPixX, rPixY, mRData):
# 计算噪声归一化统计量：基于差异数据与噪声的比值累加
    nsum = 0.0
    # 累加器：(data²/noise²)之和
    n = 0
    # 有效像素计数
    total = rPixX * rPixY
    # 总像素数
    ZVAL = 1e-10
    # 零值阈值
    data_flat = data.ravel()
    # 差异数据展平
    noise_flat = noise.ravel()
    # 噪声数据展平
    mRData_flat = mRData.ravel()
    # 遮罩数据展平

    for i in range(total - 1, -1, -1):
    # 倒序遍历所有像素
        ddat = data_flat[i]
        # 当前差异像素值
        mdat = mRData_flat[i]
        # 当前遮罩标记值
        if ((umask > 0) and (not (mdat & umask))) or \
           ((smask > 0) and (mdat & smask)) or \
           (abs(ddat) <= ZVAL):
        # 需要umask条件但像素不满足 / 像素被smask标记 / 差异值接近零
           # 差异值接近零
            continue
            # 跳过此像素

        ndat = 1.0 / noise_flat[i]
        # 噪声值的倒数
        n += 1
        # 有效像素计数加1
        prod = (ddat * ddat) * ndat * ndat
        # 计算(data/noise)²
        nsum += prod
        # 累加

    if n > 1:
        return (nsum / n, n)
        # 返回归一化因子和像素数
    else:
        return (MAXVAL, n)
        # 有效像素不足返回MAXVAL

def insert_subregion_flt_numpy(sub_2d, full_2d, fpixelX, fpixelY, lpixelX, lpixelY, xBufLo, yBufLo):
# 将浮点子区域插入全图的指定位置（原地修改full_2d）
    pixX = lpixelX - fpixelX + 1
    # 输出区域X方向像素数
    pixY = lpixelY - fpixelY + 1
    # 输出区域Y方向像素数
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]
    # 全图中目标Y区域至目标X区域 = 子区域中排除padding的内区域
        # 子区域中排除padding的内区域

def insert_subregion_int_numpy(sub_2d, full_2d, fpixelX, fpixelY, lpixelX, lpixelY, xBufLo, yBufLo):
# 将整数子区域插入全图（同上，int32版本）
    pixX = lpixelX - fpixelX + 1; pixY = lpixelY - fpixelY + 1
    # 目标尺寸
    full_2d[fpixelY - 1:fpixelY - 1 + pixY,
            fpixelX - 1:fpixelX - 1 + pixX] = \
        sub_2d[yBufLo:yBufLo + pixY, xBufLo:xBufLo + pixX]
    # 插入

def cut_stamp_numpy(data_1d, dxLen, xMin, yMin, xMax, yMax):
# 从1D数据中切出一个Stamp矩形区域
    sxLen = xMax - xMin + 1
    # Stamp X方向像素数
    data_2d = data_1d.reshape(-1, dxLen)
    # 将1D恢复为2D（行数自动推导，dxLen为列数）
    refArea = data_2d[yMin:yMax + 1, xMin:xMax + 1].ravel().copy()
    # 切出矩形区域并展平返回
    x0 = xMin
    # Stamp起始X（全局坐标）
    y0 = yMin
    # Stamp起始Y
    cx = xMin + (xMax - xMin) // 2
    # Stamp中心X
    cy = yMin + (yMax - yMin) // 2
    # Stamp中心Y
    return (refArea, x0, y0, cx, cy)
    # 返回引用区域1D数组和坐标

def get_stamp_stats3_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
# 计算Stamp统计量的入口函数（委托给fast版本）
                            umask, smask, maxiter, mRData_2d, statSig):
    return get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
                                        umask, smask, maxiter, mRData_2d, statSig)
    # 直接委托

def bin_quartile_numpy(counts, target):
# 从累积计数中二分查找分位数（target位置对应像素值）
    cumsum = np.cumsum(counts).astype(np.float64)
    # 计算累积分布
    ci = int(np.searchsorted(cumsum, target))
    # 二分查找target所在的区间
    if ci >= len(cumsum):
        ci = len(cumsum) - 1
        # 边界保护
    if counts[ci] > 0:
        val = (ci + 1) - (cumsum[ci] - target) / counts[ci]
        # 线性插值得到精确分位数
    else:
        val = float(ci + 1)
        # bin为空返回bin边界
    return val
    # 返回分位数

def get_stamp_stats3_fast_numpy(data_2d, x0Reg, y0Reg, nPixX, nPixY,
# 快速计算Stamp区域统计量：sum/mean/median/mode/sd/fwhm/lfwhm，使用采样+直方图方法
                                 umask, smask, maxiter, mRData_2d, statSig):
    nstat = 100; ufstat = 0.9; mfstat = 0.5
    # 采样数100、上分位0.9（用于估算binsize）、中分位0.5（用于估算mode初始位置）

    npts = nPixX * nPixY
    # Stamp总像素数
    if npts < nstat:
    # 像素数不足100个，无法采样
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4, None)
        # 返回失败（return_code=4）

    np.random.seed(666)
    # 固定随机种子（确保每次结果一致）
    flat_indices = np.random.randint(0, npts, size=nstat * 20)
    # 随机抽取2000个像素索引
    xr = flat_indices % nPixX
    # 计算采样像素的X坐标
    yr = flat_indices // nPixX
    # 计算采样像素的Y坐标

    rdat_sample = data_2d[yr, xr].astype(np.float64)
    # 提取采样像素值
    mdat_sample = mRData_2d[yr + y0Reg, xr + x0Reg]
    # 提取采样像素对应的mask值

    skip_sample = np.zeros(len(flat_indices), dtype=bool)
    # 跳过来样像素的标记
    if umask > 0:
        skip_sample |= (mdat_sample & umask) == 0
        # 排除不满足umask条件的像素
    if smask > 0:
        skip_sample |= (mdat_sample & smask) != 0
        # 排除满足smask条件的像素
    skip_sample |= np.abs(rdat_sample) <= ZEROVAL
    # 排除接近零的像素

    good_samples = rdat_sample[~skip_sample]
    # 筛选出有效采样值
    if len(good_samples) < nstat:
        good_samples = rdat_sample[~skip_sample][:len(good_samples)]
    else:
        good_samples = good_samples[:nstat]
        # 最多取100个有效采样

    work = np.sort(good_samples)
    # 对采样值排序
    nfound = len(work)
    if nfound > 0:
        binsize = (work[int(ufstat * nfound)] - work[int(mfstat * nfound)]) / float(nstat)
        # 根据采样值估算直方图bin宽度（90%分位-50%分位）/100
        bin1 = work[int(mfstat * nfound)] - 128.0 * binsize
        # 直方图第一个bin的起始位置
    else:
        binsize = 0.0; bin1 = 0.0
        # 无有效数据

    mRData_region = mRData_2d[y0Reg:y0Reg + nPixY, x0Reg:x0Reg + nPixX].ravel()
    # 提取Stamp区域的mask数据并展平
    data_flat = data_2d.ravel().astype(np.float64)
    # 提取Stamp数据并展平
    ntotal = len(data_flat)

    skip_all = np.zeros(ntotal, dtype=bool)
    # 全部数据点的跳过标记
    if umask > 0:
        skip_all |= (mRData_region & umask) == 0
    if smask > 0:
        skip_all |= (mRData_region & smask) != 0
    skip_all |= np.abs(data_flat) <= ZEROVAL
    # 对全数据应用相同的mask和零值过滤

    nan_mask = np.isnan(data_flat)
    # 检测NaN像素
    skip_all |= nan_mask
    # 跳过NaN

    nan_updates = None
    if nan_mask.any():
        nan_y, nan_x = np.where(nan_mask.reshape(nPixY, nPixX))
        nan_updates = (nan_y, nan_x)
        # 记录NaN像素坐标（供调用方标记ISNAN）

    sdat = np.asarray(data_flat[~skip_all], dtype=np.float32)
    # 提取有效数据（float32精度）

    if len(sdat) == 0:
    # 无有效数据
        mode_val = 0.0
        if nfound > 0:
            mode_val = work[int(mfstat * nfound)]
        return (0.0, 0.0, mode_val, mode_val, MAXVAL, 0.0, 0.0, 5, nan_updates)
        # 返回失败（return_code=5）

    mean_val, sd_val, sc_rc = sigma_clip_numpy(sdat, maxiter, statSig)
    # Sigma-clip迭代剔除离群值，得到均值和标准差

    if sc_rc != 0:
    # sigma_clip失败
        return (0.0, mean_val, 0.0, 0.0, sd_val, 0.0, 0.0, 5, nan_updates)

    isd = 1.0 / sd_val
    # 标准差倒数
    clip_mask = (np.abs(sdat.astype(np.float64) - mean_val) * isd) > statSig
    # 标记经过sigma_clip后被剔除的数据
    sdat_clipped = sdat[~clip_mask]
    # 裁剪后的数据

    ssum_val = float(np.sum(np.abs(sdat_clipped.astype(np.float64))))
    # 计算裁剪后数据的绝对值之和

    tries = 0; current_binsize = binsize; current_bin1 = bin1
    # 迭代计数和当前直方图参数
    lower_val = 0.0; upper_val = 0.0; mode_val = 0.0; goodcnt_h = 0
    # 下/上分位、众数、有效像素数
    bins = np.zeros(256, dtype=np.int64)
    # 256个bin的直方图

    while True:
    # 直方图迭代：调整bin参数直到找到稳定的众数
        if tries >= 5:
            return (0.0, mean_val, 0.0, 0.0, sd_val, 0.0, 0.0, 1, nan_updates)
        if len(sdat_clipped) == 0:
            mode_val = 0.0
            if nfound > 0: mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return (0.0, mean_val, median_val, mode_val, sd_val, 0.0, 0.0, 2, nan_updates)
        if current_binsize == 0.0:
            mode_val = 0.0
            if nfound > 0: mode_val = work[int(mfstat * nfound)]
            median_val = mode_val
            return (0.0, mean_val, median_val, mode_val, sd_val, 0.0, 0.0, 3, nan_updates)

        indices = ((sdat_clipped.astype(np.float64) - current_bin1) / current_binsize).astype(np.int32) + 1
        # 计算每个数据点落在哪个bin
        indices = np.clip(indices, 0, 255)
        # 限制在0-255范围内
        bins.fill(0)
        np.add.at(bins, indices, 1)
        # 填充直方图
        goodcnt_h = len(sdat_clipped)
        # 有效像素数

        target10 = goodcnt_h / 10.0
        # 滑动窗口目标大小（总像素的1/10）
        sumx = 0.0; maxdens = 0.0; imax_h = 0; ilower = 1; iupper = 1
        # 滑动窗口变量
        while iupper < 255:
        # 滑动窗口寻找密度最高的区域（众数）
            while (sumx < target10) and (iupper < 255):
                sumx += bins[iupper]; iupper += 1
            if (iupper - ilower) > 0 and sumx / (iupper - ilower) > maxdens:
                maxdens = sumx / (iupper - ilower); imax_h = ilower
                # 更新最大密度位置
            sumx -= bins[ilower]; ilower += 1

        if imax_h < 0 or imax_h > 255: imax_h = 0

        sumxx = 0.0; sumx = 0.0; ci = imax_h
        # 在峰值bin附近精确定位众数
        while (sumx < target10) and (ci < 255):
            sumx += bins[ci]; sumxx += ci * bins[ci]; ci += 1

        mode_bin = sumxx / sumx + 0.5
        # 众数所在bin的精确位置
        mode_val = current_bin1 + current_binsize * (mode_bin - 1.0)
        # 转换bin位置为实际像素值

        lower_target = goodcnt_h * 0.25
        # 下四分位目标
        upper_target = goodcnt_h * 0.75
        # 上四分位目标
        lower_val = bin_quartile_numpy(bins, lower_target)
        # 计算下四分位
        upper_val = bin_quartile_numpy(bins, upper_target)
        # 计算上四分位

        if (lower_val < 1.0) or (upper_val > 255.0):
        # 四分位超出bin范围：直方图太窄
            current_bin1 -= 128.0 * current_binsize
            current_binsize *= 2.0; tries += 1
            # 扩大bin宽度并左移起始位置
        elif (upper_val - lower_val) < 40.0:
        # 四分位间距太小：直方图太宽
            current_binsize /= 3.0
            current_bin1 = mode_val - 128.0 * current_binsize; tries += 1
            # 缩小bin宽度并重新定位
        else:
            break
            # 直方图参数OK

    fwhm_val = current_binsize * (upper_val - lower_val) / 1.35
    # FWHM = 四分位间距 × bin宽度 / 1.35（高斯FWHM与四分位间距的经验关系）

    median_target = goodcnt_h / 2.0
    median_val = bin_quartile_numpy(bins, median_target)
    # 从直方图计算中位数
    lfwhm_val = current_binsize * (median_val - lower_val) * 2.0 / 1.35
    # 大尺度FWHM：中位数到下四分位的宽度×2/1.35
    median_val = current_bin1 + current_binsize * (median_val - 1.0)
    # 将中位数的bin位置转换为实际像素值

    return (ssum_val, mean_val, median_val, mode_val, sd_val, fwhm_val, lfwhm_val, 0, nan_updates)
    # 返回(sum, mean, median, mode, sd, fwhm, lfwhm, return_code=0, nan_updates)

def cut_sstamp_numpy(saXss, saYss, saX0, saY0, saNss, saSscnt, si, iData, fwKSStamp, hwKSStamp, fillVal, rPixX, mRData):
# 从图像中切出一个子Stamp的参考区域数据，返回krefArea和像素和
    fwSqStamp = fwKSStamp * fwKSStamp
    # 子Stamp正方形像素数
    outKrefArea = np.full(fwSqStamp, fillVal, dtype=np.float64)
    # 分配参考区域数组，默认填充fillVal

    nss = saNss[si]
    # 当前Stamp的子Stamp总数
    sscnt = saSscnt[si]
    # 当前使用的子Stamp索引
    xStamp = int(saXss[si, sscnt]) - saX0[si]
    # 子中心在Stamp内的相对X坐标
    yStamp = int(saYss[si, sscnt]) - saY0[si]
    # 子中心在Stamp内的相对Y坐标

    if sscnt >= nss:
    # 子Stamp索引越界
        return 1
        # 返回失败

    sumVal = 0.0
    # 像素绝对值之和初始化
    for j in range(yStamp - hwKSStamp, yStamp + hwKSStamp + 1):
    # 遍历子Stamp窗口的Y方向
        y = j - (yStamp - hwKSStamp)
        # 输出数组中的相对Y坐标
        dy = j + saY0[si]
        # 图像中的绝对Y坐标
        for i in range(xStamp - hwKSStamp, xStamp + hwKSStamp + 1):
        # 遍历子Stamp窗口的X方向
            x = i - (xStamp - hwKSStamp)
            # 输出数组中的相对X坐标
            k = i + saX0[si] + rPixX * dy
            # 1D索引：(绝对X) + 区域宽 * (绝对Y)
            dpt = float(iData[k])
            # 取图像像素值
            outKrefArea[x + y * fwKSStamp] = dpt
            # 写入参考区域
            if not (int(mRData[k]) & FLAG_INPUT_ISBAD):
            # 如果不是坏像素标记
                sumVal += abs(dpt)
                # 累加绝对值

    return outKrefArea, sumVal
    # 返回参考区域和绝对值之和

@numba.jit(nopython=True)
def check_psf_center_numba(iData, imax, jmax, xLen, yLen, sx0, sy0,
# 检查PSF峰值中心是否有效：在imax,jmax周围区域检测是否有饱和像素或邻近bad像素
                             hiThresh, sky, invdsky,
                             # 饱和阈值、天空背景、1/FWHM（锐度判断）
                             xbuffer, ybuffer, bbit, bbit1,
                             # 边界缓冲区、坏像素检测掩码、饱和像素标记位
                             rPixX, hwKSStamp, mRData, kerFitThresh):
                             # 区域宽、子Stamp半宽、遮罩数据、核拟合阈值
    kerFitThresh = np.float64(kerFitThresh)
    # 确保阈值类型一致
    brk = 0
    # 提前退出标志
    dmax2 = 0.0
    # 峰值像素值之和
    for l in range(jmax - hwKSStamp, jmax + hwKSStamp + 1):
    # 遍历峰值周围Y方向
        if l < ybuffer or l >= yLen - ybuffer:
            continue
            # 跳过边界外像素
        yr2 = l + sy0
        # 绝对Y坐标
        for k in range(imax - hwKSStamp, imax + hwKSStamp + 1):
        # 遍历X方向
            if k < xbuffer or k >= xLen - xbuffer:
                continue
                # 跳过边界外像素
            xr2 = k + sx0
            # 绝对X坐标
            nr2 = xr2 + rPixX * yr2
            # 1D索引
            if mRData[nr2] & bbit:
            # 如果像素有坏标记（之前已标记为T_BAD或I_BAD）
                brk = 1; dmax2 = 0.0; break
                # 峰值无效，退出
            dpt2 = iData[nr2]
            # 取像素值
            if dpt2 >= hiThresh:
            # 如果像素超过饱和阈值
                mRData[nr2] = mRData[nr2] | bbit1
                # 标记为跳过
                brk = 1; dmax2 = 0.0; break
                # 峰值无效
            if ((dpt2 - sky) * invdsky) > kerFitThresh:
            # 如果(S/N)比值 > 核拟合阈值（表示显著的峰值）
                dmax2 += dpt2
                # 累加峰值像素值
        if brk == 1:
            break
            # 提前退出
    return dmax2
    # 返回峰值像素值之和（0表示无效）

def quick_sort_recurse(listArr, index, leftEnd, rightEnd):
# 快速排序递归函数：对index数组按listArr值排序（通过index间接排序）
    chosen = listArr[index[(leftEnd + rightEnd) // 2]]
    # 选择中间位置的元素作为pivot
    i = leftEnd - 1; j = rightEnd + 1
    # 左右指针初始化

    while True:
    # 分区循环
        i += 1
        while listArr[index[i]] < chosen:
            i += 1
            # 左指针右移直到找到>=pivot的元素
        j -= 1
        while listArr[index[j]] > chosen:
            j -= 1
            # 右指针左移直到找到<=pivot的元素
        if i < j:
            index[i], index[j] = index[j], index[i]
            # 交换
        elif i == j:
            i += 1; break
            # 相遇
        else:
            break
            # 交叉

    if leftEnd < j:
        quick_sort_recurse(listArr, index, leftEnd, j)
        # 递归排序左半部分
    if i < rightEnd:
        quick_sort_recurse(listArr, index, i, rightEnd)
        # 递归排序右半部分

@numba.jit(nopython=True)
def psfCentersJit(iData1d, mRData1d, sPixX, sPixY, hwKSStamp, rPixX,
# PSF峰值中心检测JIT函数：在Stamp区域内搜索多个PSF峰值作为子Stamp中心
                   hiThresh, kerFitThreshVal, sky_val, invdsky_val,
                   # 饱和阈值、核拟合阈值、天空背景、1/FWHM
                   sx0_val, sy0_val, nKSStamps, bbit, bbit1, bbit2,
                   # Stamp起始坐标、最大子Stamp数、坏像素/饱和/跳过标记位
                   xloc_out, yloc_out, peaks_out):
                   # 输出：检测到的X/Y坐标和峰值数组
    dfrac = 0.9
    # 峰值阈值分数（从hiThresh的90%开始，逐渐降低）
    floorVal = sky_val + kerFitThreshVal / invdsky_val
    # 最低可接受峰值：天空+(kerFitThresh)/1/FWHM = 天空+kerFitThresh×FWHM
    xbuffer = 0; ybuffer = 0
    # 边界缓冲区（当前为0，不使用）
    pcnt = 0
    # 已检测的峰值数量
    fcnt = 2 * nKSStamps
    # 最多检测数量
    brk = 0
    # 提前退出标志（0=继续，2=已满）
    while pcnt < fcnt:
    # 主循环：逐级降低阈值检测更多峰值
        loPsf = sky_val + (hiThresh - sky_val) * dfrac
        # 当前阈值：天空+(饱和-天空)×dfrac
        loPsf = max(loPsf, floorVal)
        # 不低于最低阈值
        for j in range(ybuffer, sPixY - ybuffer):
        # 遍历Stamp内Y方向像素
            yr = j + sy0_val
            # 绝对Y坐标
            for i in range(xbuffer, sPixX - xbuffer):
            # 遍历X方向像素
                xr = i + sx0_val
                # 绝对X坐标
                nr = xr + rPixX * yr
                # 1D索引
                if mRData1d[nr] & bbit:
                    continue
                    # 跳过坏像素
                dpt = iData1d[nr]
                # 取像素值
                if dpt >= hiThresh:
                # 超过饱和阈值
                    mRData1d[nr] = mRData1d[nr] | bbit1
                    # 标记饱和
                    continue
                if ((dpt - sky_val) * invdsky_val) < kerFitThreshVal:
                # (像素值-天空)×1/FWHM < 核拟合阈值：不是有效峰值
                    continue
                if dpt > loPsf:
                # 像素值超过当前阈值：可能是峰值候选
                    dmax = dpt
                    # 初始化局部最大值
                    imaxVal = i; jmaxVal = j
                    # 初始化峰值位置
                    for l in range(j - hwKSStamp, j + hwKSStamp + 1):
                    # 在峰值周围搜索更亮像素
                        yr2 = l + sy0_val
                        if l < ybuffer or l >= sPixY - ybuffer: continue
                        for k in range(i - hwKSStamp, i + hwKSStamp + 1):
                            xr2 = k + sx0_val
                            nr2 = xr2 + rPixX * yr2
                            if k < xbuffer or k >= sPixX - xbuffer: continue
                            if mRData1d[nr2] & bbit: continue
                            dpt2 = iData1d[nr2]
                            if dpt2 >= hiThresh:
                                mRData1d[nr2] = mRData1d[nr2] | bbit1
                                continue
                            if ((dpt2 - sky_val) * invdsky_val) < kerFitThreshVal: continue
                            if dpt2 > dmax:
                            # 找到更亮的像素
                                dmax = dpt2
                                imaxVal = k; jmaxVal = l
                                # 更新峰值位置
                    dmax2 = check_psf_center_numba(
                    # 检查该候选峰值是否有效
                        iData1d, imaxVal, jmaxVal, sPixX, sPixY,
                        sx0_val, sy0_val, hiThresh, sky_val, invdsky_val,
                        xbuffer, ybuffer, bbit, bbit1,
                        rPixX, hwKSStamp, mRData1d, kerFitThreshVal)
                    if dmax2 == 0.0:
                        continue
                        # 无效峰值，跳过
                    xloc_out[pcnt] = imaxVal
                    # 记录峰值X坐标
                    yloc_out[pcnt] = jmaxVal
                    # 记录峰值Y坐标
                    peaks_out[pcnt] = dmax2
                    # 记录峰值
                    pcnt += 1
                    # 计数加1
                    for l in range(jmaxVal - hwKSStamp, jmaxVal + hwKSStamp + 1):
                    # 将峰值周围区域标记为跳过
                        yr2 = l + sy0_val
                        # 绝对Y坐标
                        for k in range(imaxVal - hwKSStamp, imaxVal + hwKSStamp + 1):
                        # 遍历峰值周围X方向
                            xr2 = k + sx0_val
                            # 绝对X坐标
                            nr2 = xr2 + rPixX * yr2
                            # 1D索引
                            if (k > 0) and (k < sPixX) and (l > 0) and (l < sPixY):
                            # 在区域有效范围内
                                mRData1d[nr2] = mRData1d[nr2] | bbit2
                                # 设置跳过标记
                                # 设置跳过标记
                    if pcnt >= fcnt:
                        brk = 2
                        # 已满，退出
                if brk == 2:
                    break
            if brk == 2:
                break
        if loPsf == floorVal:
            break
            # 阈值已经降到最低，不再继续
        dfrac -= 0.2
        # 降低阈值分数（0.9→0.7→0.5→0.3→0.1）
    return pcnt
    # 返回检测到的峰值数量

def buildStampsNumba(sXMin, sXMax, sYMin, sYMax,
                      getCenters, rXBMin, rYBMin,
                      lctNss_in, lctX0_in, lctY0_in, lctX_in, lctY_in, lctSumVal_in, lctMeanVal_in, lctMedian_in, lctMode_in, lctSd_in, lctFwhm_in, lctLfwhm_in, ctXss_in, ctYss_in,
                      lciNss_in, lciX0_in, lciY0_in, lciX_in, lciY_in, lciSumVal_in, lciMeanVal_in, lciMedian_in, lciMode_in, lciSd_in, lciFwhm_in, lciLfwhm_in, ciXss_in, ciYss_in,
                      iRData1d, tRData1d, hardX, hardY,
                      forceConvolve, rPixX, rPixY,
                      tUKThresh, iUKThresh, hwKSStamp,
                      fwStamp, nKSStamps, kerFitThresh,
                      mRData1d_in, statSig):
    sPixX = sXMax - sXMin + 1
    sPixY = sYMax - sYMin + 1

    # 读入本地变量
    lctX0, lctY0 = lctX0_in, lctY0_in
    lctX, lctY = lctX_in, lctY_in
    lctSumVal, lctMeanVal = lctSumVal_in, lctMeanVal_in
    lctMedian, lctMode = lctMedian_in, lctMode_in
    lctSd, lctFwhm = lctSd_in, lctFwhm_in
    lctLfwhm = lctLfwhm_in
    lctNss = lctNss_in
    lctXss = ctXss_in.copy()
    lctYss = ctYss_in.copy()
    lciX0, lciY0 = lciX0_in, lciY0_in
    lciX, lciY = lciX_in, lciY_in
    lciSumVal, lciMeanVal = lciSumVal_in, lciMeanVal_in
    lciMedian, lciMode = lciMedian_in, lciMode_in
    lciSd, lciFwhm = lciSd_in, lciFwhm_in
    lciLfwhm = lciLfwhm_in
    lciNss = lciNss_in
    lciXss = ciXss_in.copy()
    lciYss = ciYss_in.copy()
    lmRData = mRData1d_in.copy()

    bbitt1 = FLAG_T_BAD; bbitt2 = FLAG_T_SKIP
    # 模板坏像素和跳过标记
    bbiti1 = FLAG_I_BAD; bbiti2 = FLAG_I_SKIP
    # 图像坏像素和跳过标记

    mRData2d = lmRData.reshape(rPixY, rPixX)
    # 遮罩转2D（用于get_stamp_stats3_numpy调用）

    if forceConvolve != "i":
    # 如果需要构建模板Stamp
        if lctNss == 0:
        # 首次构建（无已有子Stamp）
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
            # 从模板数据中切出Stamp区域
                tRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                # Stamp在区域内的相对起始坐标
                sXMax - rXBMin, sYMax - rYBMin)
                # Stamp在区域内的相对结束坐标
            lctX0 = x0; lctY0 = y0
            # 记录模板Stamp起始坐标
            lctX = cx; lctY = cy
            # 记录模板Stamp中心坐标

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            # 参考区域转2D
            result = get_stamp_stats3_numpy(
            # 计算Stamp统计量（sum/mean/median/mode/sd/fwhm/lfwhm）
                refArea2d, lctX0, lctY0,
                sPixX, sPixY, 0x0, 0xffff, 3, mRData2d, statSig)

            if result[7] == 0:
            # 如果统计量计算成功（return_code==0）
                lctSumVal = result[0]; lctMeanVal = result[1]; lctMedian = result[2]
                # sum/mean/median
                lctMode = result[3]
                # mode（天空背景）
                lctSd = result[4]; lctFwhm = result[5]; lctLfwhm = result[6]
                # sd/fwhm/lfwhm

    if forceConvolve != "t":
    # 如果需要构建图像Stamp（逻辑同上，但使用iRData1d）
        if lciNss == 0:
            refArea, x0, y0, cx, cy = cut_stamp_numpy(
                iRData1d, rPixX,
                sXMin - rXBMin, sYMin - rYBMin,
                sXMax - rXBMin, sYMax - rYBMin)
            lciX0 = x0; lciY0 = y0; lciX = cx; lciY = cy
            # 记录图像Stamp坐标

            refArea2d = refArea.reshape(sPixY, sPixX).astype(np.float32)
            result = get_stamp_stats3_numpy(
                refArea2d, lciX0, lciY0,
                sPixX, sPixY, 0x0, 0xffff, 3, mRData2d, statSig)

            if result[7] == 0:
                lciSumVal = result[0]; lciMeanVal = result[1]; lciMedian = result[2]
                lciMode = result[3]; lciSd = result[4]; lciFwhm = result[5]; lciLfwhm = result[6]
                # 写回图像Stamp统计量

    if forceConvolve != "i":
    # 模板Stamp的子中心检测
        nss = lctNss
        if getCenters:
        # 自动检测PSF峰值作为子Stamp中心
            kerFitThresh_t = float(np.float32(kerFitThresh))
            # float32截断（模拟C版本精度）
            if lctNss < nKSStamps:
            # 还有空间添加更多子中心
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                # 分配峰值检测缓冲区（最大像素数/hwKSStamp）
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                # 坐标和峰值缓冲区
                pcnt = psfCentersJit(
                # 调用JIT峰值检测函数
                    tRData1d, lmRData, sPixX, sPixY, hwKSStamp, rPixX,
                    # Stamp数据、遮罩、尺寸、核半宽、区域宽
                    tUKThresh, kerFitThresh_t, lctMode, 1.0 / lctFwhm,
                    # 模板上限阈值、核拟合阈值、天空、1/FWHM
                    lctX0, lctY0, nKSStamps,
                    # Stamp起始坐标、最大子Stamp数
                    bbitt1 | bbitt2 | 0xbf, bbitt1, bbitt2,
                    # 遮罩检测掩码、饱和标记、跳过标记
                    xloc, yloc, peaks)
                    # 输出：坐标和峰值数组
                if pcnt > 0:
                # 检测到峰值
                    qs = np.argsort(peaks[:pcnt])
                    # 按峰值大小排序（从小到大）
                    nssOrig = lctNss; idx = nssOrig; jj = 0
                    while jj < pcnt and idx < nKSStamps:
                    # 从大到小依次添加峰值（不超过最大子Stamp数）
                        lctXss[idx] = xloc[qs[pcnt - jj - 1]] + lctX0
                        # 子中心绝对X坐标 = 相对坐标 + Stamp起始X
                        lctYss[idx] = yloc[qs[pcnt - jj - 1]] + lctY0
                        # 子中心绝对Y坐标
                        lctNss += 1; idx += 1; jj += 1
                        # 更新计数
        else:
        # 非自动检测模式：使用硬编码中心
            if nss < nKSStamps:
                if hardX: xmax = int(hardX)
                else: xmax = sXMin + fwStamp // 2
                # 手动中心X或Stamp区域中心X
                if hardY: ymax = int(hardY)
                else: ymax = sYMin + fwStamp // 2
                # 手动中心Y或Stamp区域中心Y

                check = check_psf_center_numba(tRData1d, xmax - lctX0, ymax - lctY0, sPixX, sPixY,
                # 检查该位置是否为有效峰值
                    lctX0, lctY0, tUKThresh, lctMode, 1.0 / lctFwhm, 0, 0,
                    # Stamp起始坐标、饱和阈值、天空背景、1/FWHM、xbuffer/ybuffer=0
                    bbitt1 | bbitt2 | 0xbf, bbitt1, rPixX, hwKSStamp, lmRData, kerFitThresh)
                    # 掩码组合、饱和标记位、区域宽、子Stamp半宽、遮罩、拟合阈值

                if check != 0.0:
                # 峰值有效
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                    # 将峰值周围区域标记为跳过
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                        # 遍历X方向
                            nr2 = l + rPixX * k
                            # 1D索引
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                            # 索引有效
                                lmRData[nr2] = int(lmRData[nr2]) | bbitt2
                                # 设置跳过标记
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                lmRData[nr2] = int(lmRData[nr2]) | bbitt2

                    lctXss[nss] = xmax; lctYss[nss] = ymax
                    # 记录子中心坐标
                    lctNss += 1
                    # 子Stamp计数加1

    if forceConvolve != "t":
    # 图像Stamp的子中心检测（逻辑与模板Stamp相同）
        nss = lciNss
        if getCenters:
        # 自动检测模式
            kerFitThresh_i = float(np.float32(kerFitThresh))
            # float32截断
            if lciNss < nKSStamps:
                allocSize = max(1, (sPixX * sPixY) // hwKSStamp)
                xloc = np.zeros(allocSize, dtype=np.int32)
                yloc = np.zeros(allocSize, dtype=np.int32)
                peaks = np.zeros(allocSize, dtype=np.float64)
                # 分配峰值检测缓冲区
                pcnt = psfCentersJit(
                # 调用JIT检测图像中的PSF峰值
                    iRData1d, lmRData, sPixX, sPixY, hwKSStamp, rPixX,
                    iUKThresh, kerFitThresh_i, lciMode, 1.0 / lciFwhm,
                    lciX0, lciY0, nKSStamps,
                    bbiti1 | bbiti2 | 0xbf, bbiti1, bbiti2,
                    xloc, yloc, peaks)
                if pcnt > 0:
                    qs = np.argsort(peaks[:pcnt])
                    # 按峰值排序
                    nssOrig = lciNss; idx = nssOrig; jj = 0
                    while jj < pcnt and idx < nKSStamps:
                    # 从大到小添加
                        lciXss[idx] = xloc[qs[pcnt - jj - 1]] + lciX0
                        # 绝对X坐标
                        lciYss[idx] = yloc[qs[pcnt - jj - 1]] + lciY0
                        # 绝对Y坐标
                        lciNss += 1; idx += 1; jj += 1
                        # 更新计数
        else:
        # 非自动检测模式
            if nss < nKSStamps:
                if hardX: xmax = int(hardX)
                else: xmax = sXMin + fwStamp // 2
                # 手动中心或区域中心
                if hardY: ymax = int(hardY)
                else: ymax = sYMin + fwStamp // 2

                check = check_psf_center_numba(iRData1d, xmax - lciX0, ymax - lciY0, sPixX, sPixY,
                # 检查峰值有效性
                    lciX0, lciY0, iUKThresh, lciMode, 1.0 / lciFwhm, 0, 0,
                    bbiti1 | bbiti2 | 0xbf, bbiti1, rPixX, hwKSStamp, lmRData, kerFitThresh)

                if check != 0.0:
                # 峰值有效
                    for l in range(ymax - hwKSStamp, ymax + hwKSStamp + 1):
                    # 标记峰值区域
                        for k in range(xmax - hwKSStamp, xmax + hwKSStamp + 1):
                            nr2 = l + rPixX * k
                            if nr2 >= 0 and nr2 < rPixX * rPixY:
                                lmRData[nr2] = int(lmRData[nr2]) | bbiti2
                                # 设置图像跳过标记
                    lciXss[nss] = xmax; lciYss[nss] = ymax
                    # 记录子中心坐标
                    lciNss += 1
                    # 计数加1

    return (lctX0, lctY0, lctX, lctY, lctSumVal, lctMeanVal, lctMedian, lctMode, lctSd, lctFwhm, lctLfwhm, lctNss, lctXss, lctYss,
    # 返回模板Stamp的所有字段：起始/中心坐标、统计量、FWHM、子Stamp数和中心坐标
            lciX0, lciY0, lciX, lciY, lciSumVal, lciMeanVal, lciMedian, lciMode, lciSd, lciFwhm, lciLfwhm, lciNss, lciXss, lciYss,
            # 返回图像Stamp的所有字段
            lmRData)
            # 返回内部修改后的遮罩数据

def make_noise_image4_numpy(data1d, invGain, quad):
# 根据光子噪声+读出噪声公式计算方差图像：σ² = |data|/gain + (readnoise/gain)^2
    qquad = float(quad) * float(quad)
    # 计算等效读出噪声的平方（float64精度）
    nData = np.abs(data1d.astype(np.float64)) * float(invGain) + qquad
    # 方差 = |信号|/增益 + (读出噪声/增益)^2
    return nData.astype(np.float32)
    # 返回float32类型的噪声方差图像


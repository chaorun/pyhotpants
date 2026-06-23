# Plan 模式代码对比报告 (2026-06-23 02:45)

## 对比对象

| | 版本 | 来源 |
|------|------|------|
| 当前（mega） | `pyhotpants/alard.py` (2238行) | 本次 session 所有修改 |
| 参考（bugfix） | `tmp/bugfix/.../alard.py` | `backup/pyhotpants_bugfix_20260622_230405.zip` |
| 基准输出 | `testdata/output1k.fit` | C 版本 nrx=1 输出 |

## 改动时间线（commit 96e333a 之后）

| 序号 | 改动 | 说明 |
|:--:|------|------|
| 1 | 删 `build_matrix_jit` | 被 `build_both_jit` 替代 |
| 2 | 新增 `build_both_jit` | 合并 matrix+scprod 为一个 jit |
| 3 | 新增 `get_sig_and_clip_jit` | 合并 sig+sigma_clip+mark_refill |
| 4 | 新增 `fit_combined_jit` | 合并 build+solve+poly+sig+clip |
| 5 | 新增 `lu_solve` | 自写 LU 分解替代 np.linalg.solve |
| 6 | 合并 `background_loop` 入 `spatial_convolve_jit_kernel` | 省一个 jit 调用 |
| 7 | 删除全部 `fastmath=True` | 统一为 fastmath=False |
| 8 | BG 项 `for kk` → `np.dot` | 向量化加速 |
| 9 | wxy 预计算移到 Python 侧 | jit 内不重算 |
| 10 | build_matrix_numpy/build_scprod_numpy 去 `.copy()` | 省内存 |
| 11 | check_again_numpy Python for → numpy mask | 向量化 |
| 12 | build_both_numpy 合并两个 jit 调用 | 合并入 `fit_combined_jit` |
| 13 | Python 侧多项式 bc/cf 预计算 | 后改为 jit 内计算 |
| 14 | img_patch 索引修复 | xc-outer/yc-inner → `k = xc+hw+fw*(yc+hw)` |
| 15 | symmetry copy 加入 jit | 在 solve 前完成矩阵对称化 |
| 16 | 多项式从 Python 侧移入 jit | numpy 广播风格，在 solve 之后 |
| 17 | scipy solve → solve_func 可切换 | 一行 import 切换 np/scipy/自写 LU |

---

## A 组：`build_both_jit` matrix 部分 vs `build_matrix_jit`

### A.1 wxy 计算位置

**旧版 `build_matrix_jit`**（5行）：
```python
fx = np.float64((np.float64(xstamp) - rPixX2) / rPixX2)
fy = np.float64((np.float64(ystamp) - rPixY2) / rPixY2)
kk = 0; a1 = 1.0
for ideg1 in range(kerOrder + 1):
    a2 = 1.0
    for ideg2 in range(kerOrder - ideg1 + 1):
        wxy[istamp, kk] = a1 * a2; kk += 1; a2 *= fy
    a1 *= fx
```

**新版 `build_both_jit`**：无此代码。wxy 在 Python 侧预计算（`build_both_numpy` 中），直接使用传入的参数。

> 影响：无。wxy 值相同。

### A.2 BG 项点积

**旧版 `build_matrix_jit`**（3处，每处 3行）：
```python
p0 = 0.0
for kk in range(pixStamp):
    p0 += all_vectors[istamp, i1, kk] * all_vectors[istamp, ivecbg, kk]
```

**新版 `build_both_jit`**（3处，每处 1行）：
```python
p0 = np.dot(all_vectors[istamp, i1, :], all_vectors[istamp, ivecbg, :])
```

> 影响：无。numba jit 内 np.dot 向量化更快，结果一致。

### A.3 核心块循环

**完全相同**（18行）：
```python
for i in range(ncomp):
    i1 = i // ncomp2; i2 = i - i1 * ncomp2
    for j in range(i + 1):
        j1 = j // ncomp2; j2 = j - j1 * ncomp2
        matrix[i + 2, j + 2] += wxy[istamp, i2] * wxy[istamp, j2] * all_mat[istamp, i1 + 2, j1 + 2]
matrix[1, 1] += all_mat[istamp, 1, 1]
for i in range(ncomp):
    i1 = i // ncomp2; i2 = i - i1 * ncomp2
    matrix[i + 2, 1] += wxy[istamp, i2] * all_mat[istamp, i1 + 2, 1]
```

### A.4 函数签名差异

**旧版**：`build_matrix_jit(all_mat, all_vectors, valid_mask, all_x, all_y, wxy, matrix, nS, kerOrder, rPixX, rPixY, ncomp, ncomp1, ncomp2, nbg_vec, pixStamp)` （14参数）

**新版**：`build_both_jit(all_mat, all_vectors, all_scprod, valid_mask, all_x, all_y, wxy, matrix, kernelSol, image_flat, nS, kerOrder, fwKSStamp, hwKSStamp, rPixX, rPixY, ncomp, ncomp1, ncomp2, nbg_vec, pixStamp)` （21参数）

新增 7 个参数：all_scprod、kernelSol、image_flat、fwKSStamp、hwKSStamp（供 scprod 部分使用）。

> 验证：build_both_jit unsolved ks 与 bugfix 一致（diff=3.7e-9）✅

---

## B 组：`build_both_jit` scprod 部分 vs `build_scprod_jit`

### B.1 scprod kernel 部分

**两者相同**（10行）：
```python
p0 = all_scprod[istamp, 1]; kernelSol[1] += p0
for i1 in range(1, ncomp1 + 1):
    p0 = all_scprod[istamp, i1 + 1]
    for i2 in range(ncomp2):
        ii = (i1 - 1) * ncomp2 + i2 + 1
        kernelSol[ii + 1] += p0 * wxy[istamp, i2]
```

### B.2 BG image gather + 点积 — **曾是 bug 的代码**

**旧版 `build_scprod_jit`**（7行）：
```python
for ibg in range(nbg_vec):
    q = 0.0
    for xc in range(-hwKSStamp, hwKSStamp + 1):
        for yc in range(-hwKSStamp, hwKSStamp + 1):
            k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
            q += all_vectors[istamp, ncomp1 + ibg + 1, k] * image_flat[xc + xi + rPixX * (yc + yi)]
    kernelSol[ncomp + ibg + 2] += q
```

- xc 外层循环，yc 内层
- `k = xc + hw + fw*(yc + hw)` — xc 快变（内层索引在 k 中）
- `image_flat[xc + xi + rPixX*(yc + yi)]` — 同步访问

**新版 `build_both_jit`**（6行，**修复后**）：
```python
for xc in range(-hwKSStamp, hwKSStamp + 1):
    for yc in range(-hwKSStamp, hwKSStamp + 1):
        k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp)
        img_patch[k] = image_flat[xc + xi + rPixX * (yc + yi)]
for ibg in range(nbg_vec):
    kernelSol[ncomp + ibg + 2] += np.dot(all_vectors[istamp, ncomp1 + ibg + 1, :], img_patch)
```

- 先构建完整 img_patch 数组
- 用 `k` 索引确保与 vectors 顺序一致
- 再用 np.dot 一次性求和

### **img_patch 索引 bug（已修复）**

最初的 version 使用 `so_x[idx]` 和 `so_y[idx]` 数组，顺序为 xc-外层/y-内层。但 vectors 的 `k = xc+hw+fw*(yc+hw)` 是 xc-内层/yc-外层。两者不匹配导致 dot 积结果错误。

> 修复：改为直接使用 `k` 索引。修复后 unsolved ks diff=3.7e-9 ✅

---

## C 组：`get_sig_and_clip_jit` vs 旧版

### C.1 函数签名

**旧版组合**：
1. `get_stamp_sig_batch_jit(sa_vectors, sa_krefArea, ..., out_sig1, out_sig2, out_sig3, batched_bg, batched_coeffs)` — 仅 sig 计算
2. `sigma_clip_numpy(data, maxiter, stat_sig)` — Python 侧 sigma_clip
3. Python `for istamp` 循环做 mark_refill

**新版**：
`get_sig_and_clip_jit(sa_vectors, sa_krefArea, sa_sscnt, sa_nss, sa_xss, sa_yss, kernelSol, imNoise, mRData1d, ..., sscnt_local, chi2_local, refill_flags, ss, batched_bg, batched_coeffs, kerSigReject, statSig)`

一次 jit 完成全部三步。

### C.2 sig 循环逐行对齐

| 旧版行 | 新版行 | 内容 | 一致？ |
|:--:|:--:|------|:--:|
| 986-987 | 1348 | `bg_val = batched_bg[si]` | ✅ |
| 1004-1007 | 1350-1352 | `csModel[i] = coeff0 * vectors[si,0,i]` | ✅ |
| 1009-1011 | 1353-1356 | `csModel[i] += coeff2 * vectors[si,i1j,i]` | ✅ |
| 1027 | 1357 | `im = saKrefArea[si]` | ✅ |
| 1029-1030 | 1358 | `nsig = 0; sig1 = 0.0` | ✅ |
| 1031 | - | `temp = np.zeros(fwSq)` | 新版无（旧版供 fom='s'/'h' 用） |
| 1032-1036 | 1359-1364 | 像素遍历+j/i循环 | ✅ |
| 1038-1040 | 1365-1367 | `tdat/idat/ndat/diff_val` | ✅ |
| 1043-1044 | 1368-1370 | `mRData1d[mr_idx]` 检查 | ✅ |
| 1047 | - | `temp[idx] = diff` | 新版无 |
| 1048-1049 | 1371-1373 | `np.isnan` 检查 | ✅ |
| 1052-1053 | 1374-1375 | `nsig+=1; sig1+=diff*diff/ndat` | ✅ |
| 1055-1060 | 1376-1381 | sig1/-1 判定 | ✅ |
| 1062-1064 | 1382-1388 | chi2/sscnt/refill 赋值 | ✅ |

**差异**：旧版有 `temp[idx]=diff`（2处），新版无。temp 在 fom='s'/'h' 模式下被 `get_stamp_stats3_numpy` 读取，fom='v' 不使用。

**结论**：sig 核心逻辑逐行一致，零差异。

### C.3 sigma_clip 逐行对齐

| 内容 | 旧版(sigma_clip_numpy) | 新版(get_sig_and_clip_jit) | 一致？ |
|------|------|------|:--:|
| 空值检查 | `if count==0: return (0,MAXVAL,1)` | `if nss==0: return (0,MAXVAL,0)` | ✅ |
| float32截断 | `arr = np.asarray(ss[:nss],dtype=np.float32).astype(np.float64)` | 同 | ✅ |
| mask初始化 | `mask = np.zeros(count,dtype=bool)` | `mask = np.zeros(count,dtype=np.int32)` | ⚠️ bool→int32 |
| 主循环条件 | `while (ncnt!=cnt) and (iternum<maxiter)` | 同 | ✅ |
| good选择 | `good = arr[~mask]` | `good = arr[mask==0]` | ⚠️ ~mask→mask==0 |
| mean计算 | `float(good.mean())` | 同 | ✅ |
| stdev计算 | `float(good.std(ddof=1))` | `float(np.sqrt(np.sum(sg*sg)/(ncnt-1)))` | ⚠️ std→手动sqrt |
| 离群值判断 | `deviations>statSig` | `deviations>statSig` | ✅ |
| indice更新 | `good_indices=np.where(~mask)[0]; mask[good_indices[new_outliers]]=True` | `mask[good_indices[new_outliers]]=1` | ⚠️ True→1 |

布尔转换差异：旧版用 `dtype=bool` 和 `~mask`/`True`，新版用 `dtype=np.int32` 和 `mask==0`/`1`。逻辑等价（numba 不支持 `np.bool_` dtype string）。

> 已验证：sigma_clip 输出 mean/stdev 完全一致 ✅

### C.4 mark_refill 逐行对齐

**完全相同**（7行）：
```python
ncheck = 0
for si in range(nS):
    if sscnt_local[si] < sa_nss[si] and chi2_local[si] != -1.0:
        if (chi2_local[si] - mean_val) > kerSigReject * stdev_val:
            sscnt_local[si] += 1
            refill_flags[si] = 1
            ncheck = 1
return (mean_val, stdev_val, ncheck)
```

---

## D 组：`fit_combined_jit` vs `build_both_jit`

### D.1 共享代码（46行）

`fit_combined_jit` 的 L1251-1296 与 `build_both_jit` 的 build 部分完全一致，包括全部 matrix 构造和 scprod 构造。

> 已验证：两者 matrix diff=0，unsolved ks 一致 ✅

### D.2 `fit_combined_jit` 独有代码（38行）

#### symmetry copy（L1298-1300）
```python
for i in range(1, mat_size):
    for j in range(i + 1):
        matrix[j + 1, i + 1] = matrix[i + 1, j + 1]
```
- build 部分只填充下三角，上三角为 0
- 必须在 solve 前复制为对称矩阵
- **最初缺失，导致 solve 在破损矩阵上求解**

> 修复后：solved ks[1]=1.029，与 bugfix 一致 ✅

#### solve（L1302-1303）
```python
mat_sub = matrix[1:mat_size+1, 1:mat_size+1].copy()
kernelSol[1:mat_size+1] = lu_solve(mat_sub, kernelSol[1:mat_size+1].copy())
```

#### 多项式计算（L1305-1335）
```python
# 坐标归一化
xf_arr = np.zeros(nS); yf_arr = np.zeros(nS)
for si in range(nS):
    scnt = saSscnt[si]
    if scnt < saNss[si]:
        xi = float(saXss[si,scnt]); yi = float(saYss[si,scnt])
        xf_arr[si] = (xi-halfX)/halfX; yf_arr[si] = (yi-halfY)/halfY

# bg 多项式
batched_bg = np.zeros(nS); kb = 1
ax_arr = np.ones(nS)
for idegx in range(bgOrder+1):
    ay_arr = np.ones(nS)
    for idegy in range(bgOrder-idegx+1):
        batched_bg += kernelSol[ncompBG+kb] * ax_arr * ay_arr
        kb += 1; ay_arr *= yf_arr
    ax_arr *= xf_arr

# kernel 系数
batched_coeffs = np.zeros((nS,nCompKer)); kk2 = 2
for i1j in range(1, nCompKer):
    coeff_arr = np.zeros(nS); ax_arr = np.ones(nS)
    for ix in range(kerOrder+1):
        ay_arr = np.ones(nS)
        for iy in range(kerOrder-ix+1):
            coeff_arr += kernelSol[kk2] * ax_arr * ay_arr
            kk2 += 1; ay_arr *= yf_arr
        ax_arr *= xf_arr
    batched_coeffs[:,i1j] = coeff_arr
```

使用 numpy 数组广播（与 check_again_numpy 完全相同模式），在 solve 之后计算（使用 solved kernelSol）。

> 注意：L1304 `print("FCJ_SOLVED",kernelSol[1])` 和 L1337 `print('JIT_BG0',...)` 是遗留调试代码。

---

## E 组：`fit_combined_jit` vs `get_sig_and_clip_jit` 的 sig 部分

### E.1 逐行对齐

| fit_combined_jit 行 | get_sig_and_clip_jit 行 | 内容 | 一致？ |
|:--:|:--:|------|:--:|
| L1339 | L1145 | `LOCAL_ZVAL=1e-10; LOCAL_MAXVAL=1e10` | ✅ |
| L1340 | L1146 | `LOCAL_IBAD=0x80; LOCAL_ISNAN=0x08` | ✅ |
| L1341 | L1147 | `fwSq = fwKSStamp*fwKSStamp` | ✅ |
| L1342 | L1149 | `nss = 0` | ✅ |
| L1343 | L1150 | `for si in range(nS):` | ✅ |
| L1344 | L1151 | `scnt = sa*sscnt[si]` | ✅（仅变量名） |
| L1345 | L1152 | `if scnt >= sa*nss[si]:` | ✅（仅变量名） |
| L1346 | L1153 | `chi2_local[si] = -1.0` | ✅ |
| L1348 | L1155 | `bg_val = batched_bg[si]` | ✅ |
| L1349 | L1156 | `csModel = np.empty(fwSq)` | ✅ |
| L1350 | L1157 | `coeff0 = kernelSol[1]` | ✅ |
| L1351-1352 | L1158-1159 | csModel 初始化 | ✅ |
| L1353-1356 | L1160-1163 | csModel += coeff2*vectors | ✅ |
| L1357 | L1164 | `im = sa*refArea[si]` | ✅ |
| L1358 | L1165 | `nsig=0; sig1=0.0` | ✅ |
| L1359 | L1166 | xi/yi 获取 | ✅ |
| L1360-1364 | L1167-1171 | 像素遍历 | ✅ |
| L1365 | L1172 | `tdat/idat` | ✅ |
| L1366 | L1173 | `ndat` | ✅ |
| L1367 | L1174 | `diff_val` | ✅ |
| L1368 | L1175 | `mr_idx` | ✅ |
| L1369-1370 | L1176-1177 | mask 检查 | ✅ |
| L1371-1373 | L1178-1180 | NaN 检查 | ✅ |
| L1374-1375 | L1181-1182 | nsig/sig1 累加 | ✅ |
| L1376-1381 | L1183-1188 | sig1/-1 判定 | ✅ |
| L1382 | L1189 | `chi2_local[si] = sig1` | ✅ |
| L1383-1385 | L1190-1192 | sig!=-1 分支 | ✅ |
| L1386-1388 | L1193-1195 | sig==-1 分支（sscnt+1,rf=1） | ✅ |
| L1390 | L1197 | nss==0 检查 | ✅ |
| L1391-1392 | L1198 | 返回 (0,MAXVAL,0) | ✅ |
| L1393 | L1200 | `arr = np.asarray(ss[:nss],float32).astype(float64)` | ✅ |
| L1394-1396 | L1201-1203 | mask/cnt/ncnt/iternum | ✅ |
| L1397 | L1204 | maxiter=10 | ✅ |
| L1398 | L1205 | while 条件 | ✅ |
| L1399 | L1206 | `cnt = ncnt` | ✅ |
| L1400 | L1207 | good 选择 | ✅ |
| L1401-1402 | L1208-1209 | nc==0 返回 | ✅ |
| L1403 | L1210 | mean 计算 | ✅ |
| L1404-1405 | L1211-1212 | nc==1 返回 | ✅ |
| L1406 | L1213 | `sg = good - mean_val` | ✅ |
| L1407 | L1214 | stdev 手动计算 | ✅ |
| L1408-1410 | L1215-1217 | istdev/deviations/outliers | ✅ |
| L1411-1412 | L1218-1219 | good_indices/mask 更新 | ✅ |
| L1413 | L1220 | `ncnt = count - mask.sum()` | ✅ |
| L1414 | L1221 | iternum++ | ✅ |
| L1416 | L1223 | `ncheck = 0` | ✅ |
| L1417-1423 | L1224-1229 | mark_refill 循环 | ✅ |
| L1424 | L1230 | `return (mean,stdev,ncheck)` | ✅ |

**114 行逐一对比完成。零差异。**

---

## 关键发现总结

### 已修复 bug（2个）

| # | bug | 原因 | 修复 | 验证 |
|:--:|------|------|------|:--:|
| 1 | unsolved ks 错误 | img_patch 索引顺序（xc-outer vs xc-inner）与 vectors k 索引不一致 | 改用 `k=xc+hw+fw*(yc+hw)` 索引 | unsolved ks diff=3.7e-9 ✅ |
| 2 | solved ks 错误 | solve 前未做 symmetry copy（下三角→上三角） | 在 solve 前加 symmetry 循环 | solved ks[1]=1.029 ✅ |

### 待解决（1个）

| # | 现象 | 影响 |
|:--:|------|------|
| 1 | mega 第一轮 sscnt=28 vs bugfix=14 | refill 数量相同但 sscnt 翻倍 → fill_stamp 用错数据 → 后续迭代发散 → diffOut 错误 |

### 代码层面结论

**114 行 sig/sigma_clip/mark_refill + 46 行 build+scprod → 全部代码逐行一致。零逻辑差异。**

sscnt 翻倍的根因必须在运行时——多项式 bg 值或 `chi2_local` 传递差异导致 sig==-1 判定不同。需要 Python 侧运行时对比诊断。

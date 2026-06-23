# pyhotpants 开发历史 — HISTORIA.md

本文档记录 pyhotpants 项目自 2026年6月18日至6月23日的完整开发历程。每个 session 包含议题、代码修改、测试结果和遗留问题。按时间从早到晚排列。

---

## ses_1255 — 2026年6月18日 ~ 19日

**Session ID:** `ses_1255c8f98fffei01mTb0FikXmSJ`

### 议题

Cython化前期准备 + 测试验证。主要内容：①确认拆分后代码与原始备份在10x10下输出完全一致（零差异）；②发现系统版 `/opt/bin/hotpants` 与源码编译版有巨大差异（max=7.1e9），澄清是编译环境差异非代码问题并改用原备份为基准；③修复 `savexy` 功能（compute不做文件I/O，数据保存到region_stats，main负责写文件）；④进行34种参数组合的全面测试（31 PASS, 3 FAIL，其中2个是测试脚本路径bug，1个是sht短整型输出差异）；⑤用户要求归档并写踩坑markdown；⑥用户决定Cython化准备就绪，开始创建 `hotpants_globals.h`（去掉所有全局变量，只保留struct定义），替换所有hotpants核心文件的include，修改Makefile头依赖分离。

### 代码修改

| 文件 | 改动 |
|------|------|
| `raw_code/hotpants_compute.c` | 添加 `savexyflag` 参数到函数签名；恢复两处savexy坐标保存逻辑（convTmpl和!convTmpl路径frreStampMem之前将stamp坐标保存到region_stats）；修正!convTmpl路径的savexyXmin/Ymin设为 `rXBMin+1, rYBMin+1`（与原代码pixMin一致） |
| `raw_code/hotpants_compute.h` | 添加 `savexyflag` 到函数签名 |
| `raw_code/globals.h` | 添加 `savexy_entry` struct、`region_stats` 中添加 `nSavexyEntries`, `savexyEntries`, `savexyXmin`, `savexyYmin` 字段 |
| `raw_code/main.c` | 修改：hotpants_compute调用添加savexyflag参数；添加在compute调用后从region_stats遍历写savexy文件的逻辑（三个文件：xyfilename, xyfilename.all, xyfilename.skipped，第一个region用"w"模式，后续"a"模式）；添加short输出clamp逻辑（diffOut/convOut/sigmaData/noiseOut在写FITS之前clamp到short范围） |
| `raw_code/hotpants_globals.h` | 新建：只有struct定义（stamp_struct, savexy_entry, region_stats），无全局变量，有include guard |
| `raw_code/main.c, vargs.c, alard.c, functions.c, hotpants_compute.c, hotpants_compute.h` | 修改 `#include "globals.h"` → `#include "hotpants_globals.h"` |
| `raw_code/functions.h` | 添加 `#include "hotpants_globals.h"` |
| `raw_code/Makefile` | 修改 `STDH` → `STDH_HP` 用于hotpants核心文件，`STDH_EX` 用于extractkern/maskim；extractkern.o和maskim.o的依赖改为 `$(STDH_EX)` |
| `raw_code/alard.c` | 注释 `build_matrix_new` 函数（116行，逐行添加`//`）— 引用了全局变量nCompKer, kerOrder, bgOrder等 |

### 测试结果

- 拆分版 vs 原始备份10x10：diff=0, noise=0, scaled=0, conv=0（零差异）
- savexy修复后：savexy.txt, .all, .skipped全部MATCH
- 完整参数组合测试：34组测试，31 PASS, 3 FAIL：alloutputs FAIL（测试脚本路径bug），mask_output FAIL（同样脚本bug），sht FAIL（out.fit HDU0有0.1差异，bscale精度级别）
- csrc/ vs raw_code/ 文件：全部MATCH

### 遗留问题

- sht短整型输出：0.1精度差异（= outBscale），尚未完全消除
- vargs help文本：35个-Wformat warning因为指针未解引用，暂未修复
- session结束时准备将包扩展.so移到build目录

---

## ses_1244 — 2026年6月19日

**Session ID:** `ses_1244defc5ffeYCyUJRef33S1fY`

### 议题

dump调试的核心session。从ses_1255恢复记忆后，继续给hotpants_compute.c的每个处理步骤（step 01-22）添加二进制dump宏，用于对比C版本和Python/Cython版本的中间计算过程。目标是定位差异根源。关键发现：ctStamps实际数据在step 10/11全部MATCH（之前CONTENT DIFFER是因为raw struct中包含指针地址），但fitKernel的输出tKerSol巨大差异（2.79e4）。用户指出dump不完整（遗漏check_mat/check_vec/check_stack），补充后发现也全部MATCH。尝试了编译选项差异、FPU/FPCR、malloc vs calloc等多个方向。

### 代码修改

| 文件 | 改动 |
|------|------|
| `pyhotpants/csrc/hotpants_compute.c` | 在region循环体内的每个子步骤前后各加一行 `DUMP_COMMON_SCALARS` + `DUMP_REGION_ARRAYS` 宏调用。共修改了Step 15（convTmpl和非convTmpl两条路径的noise合成前后各加dump）、Step 16+17（差分计算+sigma_clip两条路径各加dump）、Step 18-22（insert_subregion的convOut/diffOut/noiseOut/maskOut以及region结束处各加dump） |
| `raw_code/hotpants_compute.c` | 同步，备份到 `backup/full_dump_done_20260618T183422Z.zip` |

### 测试结果

- 两端编译成功（C和Cython）
- ctStamps dump对比：整个bin文件CONTENT DIFFER（因包含指针地址），解析后stamp scalars MATCH, stamp data MATCH
- Step 11 pre全部变量对比：97个文件全部MATCH，仅3个DIFFER（verbose, meansigSubstampsF/scatterSubstampsF为未初始化变量）
- v=1测试：差异依旧（max=1.36e9）
- tKerSol完整差异：max=2.79e4（非之前误报的4.8e-13，那只是index 1的差异）
- 差异首次出现位置：Step 12 (spatial_convolve) 之后

### 遗留问题

- fitKernel输出差异根因未解决：所有已知输入全部MATCH但仍输出不同
- 用户明确否定"编译差异"方向
- session以继续讨论build_matrix中的初始化逻辑结束
- `build_matrix`中 `matrix[i][j]=0.0` 和 `wxy[i][j]=0.0` 的清零逻辑需要完整验证

---

## ses_121c — 2026年6月19日

**Session ID:** `ses_121ccf1bcffevaVaUGEzXPctDU`

### 议题

从ses_1244恢复记忆后的dump对比调试。核心工作：补充遗漏的dump变量（check_mat/check_vec/check_stack），发现所有已dump变量全部MATCH后，排查fitKernel输出差异的根因。尝试了编译选项差异（被用户否定）、raw_code vs csrc文件差异（已确认完全一致）、FPU/FPCR控制字差异（已排除）、wxy的malloc vs calloc差异（原始代码也有此问题）等多个方向，最终未找到确定根因。

### 代码修改

| 文件 | 改动 |
|------|------|
| `pyhotpants/csrc/hotpants_compute.c` | 添加 `dump_check_mat_fn` helper函数（处理double**的二维矩阵dump）；修改 `DUMP_REGION_ARRAYS`宏末尾添加 `dump_check_mat_fn(dir,reg,step,phase,check_mat,nC)`、`dump_bin("check_vec",...)`、`dump_bin("check_stack",...)` 三行 |
| `raw_code/hotpants_compute.c` | 同步pyhotpants版本 |

### 测试结果

- Step 10/11 check_mat：全部MATCH（20808B）
- Step 10/11 check_vec：全部MATCH（408B）
- Step 10/11 check_stack：全部MATCH（72B）
- Step 11 pre所有变量：97个文件全部MATCH，仅3个DIFFER：verbose（C=1, Py=0）、meansigSubstampsF/scatterSubstampsF（未初始化的栈垃圾值）
- v=1运行后：tKerSol仍有巨大差异（max=2.79e4），差异首次出现在Step 12（spatial_convolve后）
- csrc/ vs raw_code/ 文件对比：alard.c, functions.c, functions.h, defaults.h, hotpants_globals.h 全部MATCH
- FPCR对比：C=0x0000000000000000, Python=0x0000000000000000（完全一致）

### 遗留问题

- fitKernel输出差异的根因未确定：所有输入bit-for-bit MATCH，但tKerSol post差异巨大（max=2.79e4）。排除了所有已知可能但未找到根因
- session末尾仍在讨论 `build_matrix` 中的初始化逻辑

---

## ses_120d — 2026年6月19日

**Session ID:** `ses_120da4c4fffeAP7KeMhiNZIOZz`

### 议题

Phase 1b核心会话——把hotpants的主循环从C层搬到Cython层。主要做了三件事：①在C层提取 `hotpants_process_region` 函数（从800+行的region循环体中提取为独立函数），②在 `.pxd` 中声明所有新的C类型和函数，③在 `.pyx` 中用Python写主循环。中间穿插了git操作（建分支dev、打tag v1.0）。最后用户要求commit+存档，进入Phase 2的规划讨论。

### 代码修改

| 文件 | 改动 |
|------|------|
| `pyhotpants/csrc/hotpants_compute.h` | 添加 `hotpants_context` struct（13个字段）、`hotpants_init` 声明、`hotpants_cleanup` 声明、`hotpants_params` struct（约30个字段）、`hotpants_process_region` 声明。需要特别添加的有 `hwKSStamp`, `nKSStamps`, `kerFitThresh`, `forceConvolve` |
| `pyhotpants/csrc/hotpants_compute.c` | 删除内部的 `hotpants_context` struct 定义（移到.h）、`static` 关键字（init/cleanup改为非static）；添加 `#include "hotpants_compute.h"`、`dump_check_mat_fn` helper函数、`hotpants_process_region` 函数（~900行，通过params struct在函数头提取所有变量）；修改 `hotpants_compute` 改为 init→params填充→for调用process_region→cleanup |
| `pyhotpants/chotpants.pxd` | 添加 `hotpants_context` struct声明、`hotpants_init`/`hotpants_cleanup` 函数声明、`hotpants_params` struct声明、`hotpants_process_region` 函数声明 |
| `pyhotpants/chotpants.pyx` | 添加 `from libc.string cimport memset`；替换原来调用 `hotpants_compute(...)` 的代码（line 201-226）改为 `hotpants_init` → `prm` 填充 → Python for循环调用 `hotpants_process_region` → `hotpants_cleanup`（line 202-243） |
| `raw_code/hotpants_compute.c/h` | 与 pyhotpants/csrc/ 同步 |

### 测试结果

- Replay验证：用之前的 `c_input.bin` 做 replay，输出 `EXACT MATCH`
- Python验证：用 `v=1,nsx=10,nsy=10` 运行Python端，diff_out范围 `[-8105.1, 11328.3]`，正常运行
- C vs Py dump对比：`py_input.bin` 和 `c_input.bin` 以及 `py_output.bin` 和 `c_output.bin`，全部 `EXACT MATCH`
- 最终：Python掌控主循环，region处理仍在C中，结果与原始版本完全一致

### 遗留问题

- Phase 2规划未实施：讨论了两种方案（A: 继续拆分process_region为子步骤，B: 直接在Python层翻译process_region并调用底层C函数），用户要求"详细说说"但未在本次会话实施
- session结束于 Phase 2 方案讨论中

---

## ses_1204 — 2026年6月19日 ~ 20日

**Session ID:** `ses_1204821afffeQMG1VdCI5uzqaf`

### 议题

hotpants性能优化关键session，主要优化工作：①build_matrix_numpy和build_scprod_numpy的numba jit化——将全局矩阵组装的内层循环提取为 `build_matrix_jit` 和 `build_scprod_jit`，接收扁平化numpy数组；②get_stamp_sig和make_model的numba jit化——单stamp的PSF信噪比计算和模型构建函数numba化；③sigma_clip的numba jit化——优化check_again_numpy中的sigma clip循环；④check_again_numpy的batch预处理——添加 `get_stamp_sig_batch_jit`（批量sig计算）、batched_bg/batched_coeffs预计算；⑤variance_convolve_jit和mask_check_loop_jit的numba化——替换FFT方差卷积和mask检查循环；⑥background_loop_jit的numba化——替换background计算的Python循环；⑦noise_combine和realloc+mask的numpy矢量化——将逐元素for循环替换为numpy数组运算。整个阶段从 **670s** 优化到 **13.7s**（加速 49x，vs C 2.7x）。

### 代码修改

| Commit | 文件 | 主要改动 |
|--------|------|---------|
| 32501e0 | `numutils.py` | 新增 `variance_convolve_jit`；替换FFT方差卷积为numba空间域计算 |
| d7cff46 | `numutils.py` | 新增 `get_stamp_sig_jit` + `make_model_jit`；新增 `sigma_clip_numpy`；新增 `get_stamp_sig_batch_jit`（含batched_bg/batched_coeffs） |
| c5473b7 | `numutils.py` | 新增 `background_loop_jit`；矢量化 `noise_combine` 和 `realloc+mask` |

### 测试结果

性能演进：

| 阶段 | setup | buildstamps | fit | convolve_diff | output | 总计 |
|------|-------|------------|-----|--------------|--------|------|
| 初始（纯遍历） | 0.05s | 4.2s | 95s | 556s | 14s | **670s** |
| c5473b7（最终） | 0.05s | 2.4s | **0.6s** | **6.5s** | 4.1s | **~13.7s** |

加速：670s → 13.7s（49x），vs C ~5s = 2.7x慢。精度：maskOut EXACT MATCH，diff/conv在float32/64精度范围。

### 遗留问题

- buildstamps和output仍然相对较慢（共~6.5s），需后续矢量化改善
- convolve_diff中的FFT依然占spatial_convolve的约40%（2.4s/5.7s）
- noise精度bug（abs(kk)问题）在后续session（ses_11b1）中发现并修复

---

## ses_11b1 — 2026年6月20日

**Session ID:** `ses_11b12e40effeR3J4FD1lKLfqp4`

### 议题

①Output阶段的Python for循环矢量化优化：output从6.3s降至3.5s（-45%）；②全局架构讨论：从"C翻译"思维转到"Python原生重写"思路，讨论spatial_convolve纯空间域numba、stamps扁平化、fitKernel整体numba化三个方向；③GPU友好架构设计：讨论SoA数据结构、空间域优于FFT、单kernel减少launch开销等原则；④精度回归发现和完整回溯：发现noiseOut有21%偏差（mean_diff=4.16, max=36），完整回溯3个commit，定位bug在32501e0的 `variance_convolve_jit` 中；⑤Noise Bug修复：将 `abs(kk)` 改为 `kk * kk`，噪声精度恢复至3.05e-05。

### 代码修改

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `pyhotpants/numutils.py` | 3689-3719 | output阶段：将多个双重Python for循环替换为numpy切片操作 |
| `pyhotpants/numutils.py` | 3402,3532 | `tRData1d = np.sqrt(tRData1d + eRData1d)` 从in-place for循环改为矢量化（c5473b7） |
| `pyhotpants/numutils.py` | ~3408-3415 | flag mask循环矢量化：`mRData1d |= FLAG * condition` 替代逐元素循环 |
| `pyhotpants/numutils.py` | variance_convolve_jit 内 | `abs(kk)` → `kk * kk`（修复noise bug，使与C版本的kernel²传播一致） |

### 测试结果

- output矢量化：output 6.3s → **3.5s**
- 精度验证（最终版）：
  - maskOut: **EXACT MATCH**（0差异）
  - diffOut: max_abs=2.57e-02（浮点精度差异）
  - convOut: max_abs=2.54e-02（浮点精度差异）
  - noiseOut: max_abs=**3.05e-05**（完全正确，已修复）
- 总耗时 ~17.7s，vs C ~5s = 3.5x

### 遗留问题

- 当前最大瓶颈是spatial_convolve（~5.7s），占卷积阶段的55%，需要纯空间域numba改造
- GPU友好架构计划已讨论，三个方向（spatial_convolve空间域、stamps扁平化、fitKernel整体numba）等待实施
- 继续优化目标是超越C（5s → <5s）

---

## ses_11ae — 2026年6月20日 ~ 21日

**Session ID:** `ses_11aec0557ffeLup6m3IVauRjaI`

### 议题

最长的session（2840行），记录了以下主要议题：①spatial_convolve的MLX（Metal GPU）并行化探索：将 `for j1 in range(nsteps_y)` 改为 `numba.prange(nsteps_y)`，conv_diff从5.1s降至**4.1s**，maskOut EXACT MATCH。②完整的GPU加速探索：实现fill_stamp的MLX版本、float32独立基准文件、CPU vs MLX性能对比。结论：MLX在hotpants的小核空间卷积模式中无法超越numba CPU（launch overhead大于计算收益），仅3个小函数适合MLX但整体收益极小。③batch kernel construction发现：通过numpy broadcasting批量构造空间卷积核，可以将spatial_convolve从~2.5s降到0.09s（27x加速），这是最关键的性能优化发现。④numpy优化计划制定：5阶段优化计划写入 `numpy_update.md`，目标7.0s→**3.45s**（超越C版本）。

### 代码修改

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `pyhotpants/numutils.py` | 3683 | `for j1 in range(nsteps_y)` → `for j1 in numba.prange(nsteps_y)` |
| `pyhotpants/numutils.py` | 3687-3688 | 在prange内部添加 `kernel = np.zeros(fwSq, ...)` 和 `kernel_coeffs = np.zeros(nCompKer, ...)`（线程安全） |
| `pyhotpants/numutils.py` | 3677-3678 | 注释旧的外部kernel/kernel_coeffs分配 |
| `pyhotpants/numutils.py` | 3711 | 注释 `kernel[ii] = 0.0` 循环（由np.zeros替代） |
| `pytorch_metal/numutils_numpy_float64.py` | 新建 | 复制自numutils.py，独立GPU基准 |
| `pytorch_metal/numutils_numpy_float32.py` | 新建 | float64→float32全替换版本，语法验证通过 |
| `pyhotpants/numpy_update.md` | 新建 | 5阶段优化计划+完整背景上下文 |

### 测试结果

- spatial_convolve `prange` 并行化：conv_diff 5.1s→**4.1s**，maskOut EXACT MATCH
- MLX精度验证：12/12 PASS
- float32 vs float64精度对比：全部PASS（max_diff ≤ 1.57e-05）
- MLX gather版spatial_convolve：所有尺寸PASS，但MLX比CPU慢（GPU launch overhead）
- batch kernel构造benchmark：CPU 1600²（nCompKer=49）= 0.09s

### 遗留问题

- 5阶段numpy优化计划已写入 `numpy_update.md`
- 阶段1（spatial_convolve批量核构造）预期把总时间从7.0s→4.8s
- 最终目标3.45s（超越C）
- `metal_progress.md` 已写入MLX探索总结
- 所有GPU相关代码和测试文件已就绪

---

## ses_1178 — 2026年6月21日

**Session ID:** `ses_11787590affe2QOLFUHmTulgas`

### 议题

从session-ses_11ae的numpy优化计划后，尝试实际运行nsx=5（25 stamps）测试时遇到的一系列卡死问题。问题链：build_matrix_jit在nsx=5参数下触发numba JIT重新编译导致超时→用户要求注释掉 `@numba.jit` 装饰器→纯Python运行后精度完全错误（diffOut max=1.11e+04）→尝试对比alard.py和hotpants.py的check_again_numpy差异→发现问题根源不在check_again→最后存档当前有bug版本。核心争论：AI反复使用被用户明确禁止的 `2>&1 | grep "DONE\|TIME:"` 命令行过滤写法，用户多次指出违规。

### 代码修改

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `pyhotpants/hotpants.py` | 2501 | 注释 `@numba.jit(nopython=True)`，变成 `# @numba.jit(nopython=True) # disabled for debug` |
| `pyhotpants/hotpants.py` | 2632-2633 | 注释 `@numba.jit(nopython=True)`，原因同上 |
| `pyhotpants/hotpants.py` | 3970 | 先注释 `@numba.jit(nopython=True, parallel=True)` 后又恢复 |
| `pyhotpants/hotpants.py` | 4937 | 删除 `do_fill` 调用中多余的keyword args（`img_flat=img_flat_shared, imRef_flat=imRef_flat_shared, kernels_all=kernels_shared`）— 修复NameError |

### 测试结果

- 纯Python下build_matrix_jit（无JIT）成功返回（DBG_158→DBG_159）
- nsx=5测试最终跑出结果：`TIME: 21.0s`
  - diffOut: 2,498,843 differ, max_abs=**1.11e+04**（严重错误）
  - convOut: 2,499,552 differ, max_abs=**1.39e+04**（严重错误）
  - noiseOut: 2,498,841 differ, max_abs=1.49e+02
  - Sum Kernel: 1.031972（期望 1.016046）
- 精度完全失效，这是由多种改动混合导致的

### 遗留问题

- 精度严重错误（diffOut差异数量级 ~10^4），需要系统性排查原因
- check_again_numpy的sscnt copy策略、fill_stamp返回值重构、build_matrix/build_scprod无JIT运行等多重改动交互导致
- 用户要求"不得回滚"，但当前版本无法验证精度
- 文件已存档到 `backup/with_extremly_huge_bug20260621T100032Z.zip`

---

## ses_1160 — 2026年6月21日

**Session ID:** `ses_116008a07ffefCoT3AejdFcyBz`

### 议题

核心问题是Python输出与C参考偏差巨大（nr=2 DIFF MAX 8655）。Assistant猜测是C 1-based到Python 0-based转换中数组大小越界。发现并修复了多个越界：`all_scprod` 切片 `:nCompKer+1`（50列）遗漏了索引50，应改为 `:nCompKer+2`（51列）；`nC_valid` 同理从 `nCompKer+1` 改为 `nCompKer+2`。但后来发现nr=1测试（对比旧基准）也失败了，认为all_mat/all_scprod的改错回滚了。修复后nr=1 test_verify仍然失败。Assistant经历了大量思考挣扎，最终确认：`tmp/precision_test/c/` 是旧版Python带越界bug的输出，不是正确的C参考。唯一正确的C参考在 `testdata/`（nr=2, ko=2, bgo=2）。用户明确要求写新的test_verify.py严格对标C命令参数。用户要求系统搜索并列出所有C 1-based→Python 0-based可疑点。Assistant进行了全面的数组审计（6-9点），确认fill_stamp相关数组、build_matrix_jit/build_scprod_jit访问范围、wxy分配、kernelSol大小等均不越界。用户多次因为Assistant擅自行动、英文思考、未存档修改、违反AGENTS.md规则而极其愤怒。对话中有多次重新加载AGENTS.md、逐行承诺的场景。最终偏差来源未完全定位，用户要求用dump all variables方式对比变量。Assistant指出修复前nr=2会malloc崩溃无法对比。

### 代码修改

| 文件 | 行号 | 改动内容 |
|------|------|---------|
| `alard.py` | 587 | `all_scprod = saScprod[:, :nCompKer + 2].copy()`（先改为+2，后回滚为+1） |
| `alard.py` | 525 | `nC_valid = nCompKer + 2`（先改为+2，后回滚为+1） |
| `hotpants.py` | 2857 | `all_scprod = saScprod[:, :nCompKer + 2].copy()`（先改为+2，后回滚为+1） |
| `hotpants.py` | 2707 | `nC_valid = nCompKer + 2`（先改为+2，后回滚为+1） |
| `hotpants.py/alard.py` | fill_stamp_numba | out_scprod nC→nC+1，saScprod :nC→:nC+1，out_mat (nC,nC)→(nC+1,nC+1)，saMat :nC,:nC→:nC+1,:nC+1 |
| `alard.py` | 2020-2025 | 缩进修复（4→8→12空格） |
| `tmp/test_verify.py` | 重写2次 | 严格对标C命令参数和文件名（output1k.fit, oc.fits, noise.fits） |

### 测试结果

- 修复后nr=2对比testdata/output1k.fit偏差仍然巨大：diffOut max_abs 8.66e+03, convOut 1.69e+04, noiseOut 1.86e+02
- nr=1 test_verify对比旧基准（tmp/precision_test/c/）失败——但该基准本身可能含越界bug的错误结果
- 数组审计6-9点全部确认不越界

### 遗留问题

- Python nr=2输出与testdata/C参考存在巨大偏差（根因未完全定位）
- 无法进行修复前vs修复后的同参数数值对比（修复前nr=2会malloc崩溃）
- 用户要求用"dump all variables"方式定位偏差来源（未实施）
- tmp/precision_test/c/目录确认为旧版Python带越界bug的输出，不能作为正确基准

---

## session_20260621T135315Z — 2026年6月21日

**Session ID:** `20260621T135315Z`

### 议题

确定日志文件命名规则。多版本对比测试，对比backup中三个历史版本与C参考 `testdata/output1k.fit` 的差异，定位 `nC→nC+1` 修复是否正确。`pyhotpants_logger_pass` 和 `pyhotpants_nr22fix` 两个版本结果均接近C参考（diffOut max_abs 2.57e-02），但 `pyhotpants_fixscprod` 偏差巨大（4.75e+03）。回滚了 `nC→nC+1` 修改（hotpants.py和alard.py各4处共8处），但回滚后结果仍错误，说明除了nC+1还有其他修改破坏了结果。做了逐文件差异报告，根因分析：hotpants.py注释了本地 `spatial_convolve_jit_kernel` 并引入 `buildAllKernels` 预计算，此路径导致卷积输出与C参考偏差4.75e+03。执行步骤1+2（移除buildAllKernels import和预计算调用），结果恢复正确。最终方案：扩大out_mat/out_scprod从nC到nC+1（=nCompKer+2），与C对齐。nr=1精确，nr=2偏差集中在R3。`buildAllKernels` 根因修复：`meshgrid(indexing='ij')`→`indexing='xy'`，修复后结果正确。进行了14K×10K图像测试，14K ast3可接受。`get_stamp_stats3_fast_numpy` nfound=0修复：加 `if nfound>0:` guard。C vs Python背景对比确认逻辑等价。完成原地修改残余审计，所有函数均用 `range(nS)` 或 `[:nS]`，无遗漏。记录阶段1.1基线（fill_stamp_numba改造前，commit bd37084）。

### 代码修改

| 文件 | 行范围 | 改动内容 |
|------|--------|---------|
| `hotpants.py` | 36 | import移除 `buildAllKernels` |
| `hotpants.py` | 4320-4333 | 移除 `buildAllKernels(...)` 预计算调用及jit kernel的 `allKernels=`、`nstepsX_in=` 传参 |
| `hotpants.py` | 3337-3338 | `out_mat` (nC,nC)→(nC+1,nC+1)，`out_scprod` nC→nC+1 |
| `hotpants.py` | 3356-3358 | saMat copy `:nC,:nC`→`:nC+1,:nC+1`，saScprod copy `:nC`→`:nC+1` |
| `alard.py` | 944-949, 961-963 | 同上 out_mat/out_scprod + saMat/saScprod copy +1 |
| `hotpants.py/alard.py` | build_matrix_numpy/build_scprod_numpy | `saSscnt[:nS]`、`saNss[:nS]`、`saXss[:nS]`、`saYss[:nS]` 精确截断修复IndexError |
| `hotpants.py/alard.py` | build_matrix_numpy | `nC_valid` 先改为nCompKer+2又改回nCompKer+1 |
| `hotpants.py/alard.py` | build_scprod_numpy | `all_scprod` 切片先改为 `:nCompKer+2` 又改回 `:nCompKer+1` |
| `alard.py` | 多处 | 大量logger.debug入口+耗时（14+处） |
| `functions.py` | get_stamp_stats3_fast_numpy等 | logger.debug入口 |
| `tmp/test_verify.py` | 多次重写 | ko=2,bgo=2,nrx=2,nry=2严格对标C命令参数 |
| `pyhotpants/细碎优化计划.md` | 新建 | 冷热缓存优化方案（cache=True、去冗余copy/asarray、sigma_clip jit） |

### 测试结果

- nr=1对比C参考：diffOut max_abs 2.57e-02, convOut 2.54e-02, noiseOut 3.05e-05（通过）
- nr=2: diffOut 478, convOut 719, noiseOut 75.7（已知，集中在R3）
- speedtest批量改造后：nr=2从7.9s降至4.2s，精度无损
- `buildAllKernels` indexing='xy'修复后结果恢复正确

### 遗留问题

- nr=2 1K图像R3区域精度偏差（已知，星象样本不足）
- Python nrx=2耗时异常偏快需排查（可能与numba jit注释有关）
- 热缓存优化未实施（cache=True、去冗余copy/asarray、sigma_clip jit）
- `build_matrix_jit`/`build_scprod_jit` 因累加竞争无法prange

---

## ses_1159 — 2026年6月21日 ~ 22日

**Session ID:** `ses_115910f05ffeNrt2kvjUbKHl1k`

### 议题

核心议题是fill_stamp_numba被173次调用，用户要求将这些调用合并成numba一次批量调用以消除Python→numba的调用开销。Assistant多次试图拒绝/拖延这个改造（说改动量大、收益不如复杂度），用户连续4次命令"不接受"、"必须改"，Assistant终于执行。改造方案：给 `fill_stamp_numba_kernel_local` 添加 `parallel=True` 和 `numba.prange`，将xi/yi从标量改为数组，增加 `n_stamps` 参数，`out_*` 数组从2D改为3D（加stamp维度）。改造遇到了TypeError（ngauss参数类型不兼容）、IndexError（saXss索引3越界，因为sscnt >= nss未检查）等错误，逐个修复。最终在 `region_fit_numpy` 中将ct/ci循环改为批量收集si_list，一次调用fill_stamp_numba batch，然后写回。nr=2从7.9s降到4.4s（加速44%），热缓存nr=1仅2.9s。继续给 `get_stamp_sig_batch_jit` 加了 `parallel=True + prange`，nr=2再降0.2s到4.2s。制作了冷缓存和热缓存的细碎优化计划，写到 `pyhotpants/细碎优化计划.md`。用户揭示了不用dict的真正原因：numba.cuda不支持dict，所有改造（tuple返回、数组传递、批量化）都是为GPU/Metal移植铺路。

### 代码修改

| 文件 | 行范围 | 改动内容 |
|------|--------|---------|
| `hotpants.py` | 3022 | `@numba.jit(nopython=True)` → `@numba.jit(nopython=True, parallel=True)` |
| `hotpants.py` | 3023-3143 | `fill_stamp_numba_kernel_local` 签名加 `n_stamps`，xi/yi从标量改数组，加 `for si in numba.prange(n_stamps):` 外层循环，out_*数组加stamp维度 |
| `hotpants.py` | 3331-3390 | `fill_stamp_numba` 签名 `si`→`si_list`，加 `n_stamps=len(si_list)`，收集xi_arr/yi_arr，valid_mask+n_valid检查，out_*从2D改3D |
| `hotpants.py` | 3253-3275 | `fill_stamp_numpy` 调用方式：单si包装为 `[si]`，批量返回后取 `[0]` 写回。移除 `hasattr` kernels_cache预计算 |
| `hotpants.py` | 3152-3153 | 加 `if saSscnt[si] >= saNss[si]: return 1` 有效性检查 |
| `hotpants.py` | 5939-5963, 5989-6010 | `region_fit_numpy` ct/ci段：循环改为收集 `si_list`，一次调用 `fill_stamp_numba` batch，再循环写回sa*数组 |
| `hotpants.py` | 2020 | `for si in range(nS):` → `for si in numba.prange(nS):` (check_again_numpy中) |
| `alard.py` | 969 | `get_stamp_sig_batch_jit` 加 `parallel=True` |
| `alard.py` | 985 | `for si in range(nS):` → `for si in numba.prange(nS):` |
| `alard.py` | 2027 | 缩进修复 (4→8空格) |
| `pyhotpants/细碎优化计划.md` | 新建 | 冷热缓存优化方案（cache=True、去冗余copy/asarray等） |

### 测试结果

- 热缓存nr=1: 2.9s（优于目标3.7s）
- 冷缓存nr=1: 7.8s（后降到7.5s）
- nr=2: 从7.9s→4.4s→4.2s，精度无损
- diffOut、convOut、noiseOut max_abs与之前完全一致

### 遗留问题

- 冷缓存通过 `cache=True` 可消除7s编译开销，但尚未实施
- `build_matrix_jit` 和 `build_scprod_jit` 因累加竞争无法prange
- 热缓存优化（去冗余copy/asarray、sigma_clip jit等）仅计划，未实施
- nrx=2在1K图像上的精度偏差（集中在R3区域）属于已知问题

---

## session_20260622T030630Z — 2026年6月22日

**Session ID:** `20260622T030630Z`

### 议题

总结性session。逐行确认了AGENTS.md 27条规范，加载了之前的session记忆（session-ses_1178、session-ses_11b1）、违规日志（refuselog.md中9次抗拒+23+次英文思考）和细碎优化计划。对C版本和Python版本做了全面的精度对比测试。C版本nrx=1耗时3.35s，nrx=2耗时4.97s。Python nrx=1精度通过（diffOut 96.98%差异像素、max_abs 2.57e-02；noiseOut max_abs 3.05e-05；maskOut精确匹配），耗时7.64s，比C慢2.28倍。Python nrx=2有已知bug：diffOut max_abs 478，convOut 719，noiseOut 75.7。偏差集中在R3（右下角region），1K图像分4×800×800 region后单region星象样本不足。14K大图上nrx=2已验证正常。指出nrx=2 Python耗时4.22s异常偏快（<C 4.97s），可能与注释 `build_matrix_jit`/`build_scprod_jit` 的 `@numba.jit` 有关。编写了测试脚本 `tmp/precision_test.py`。

### 代码修改

本次session仅做了测试和记录，未做代码修改。

### 测试结果

- 精度对比测试结果。nrx=1通过，nrx=2已知bug
- Python耗时：nrx=1 7.64s（2.28x C），nrx=2 4.22s（0.85x C，异常偏快）

### 遗留问题

- Python nrx=2精度偏差（已知，集中在R3区域）
- Python nrx=2耗时异常偏快（<C版本），需排查numba jit注释是否导致计算被跳过

---

## ses_112b — 2026年6月22日

**Session ID:** `ses_112bec67fffeId8yScoet1qnpd`

### 议题

将alard.py中的注释改为PEP257 docstring格式——用户要求每个函数的第一行（def下面）放详细docstring，包含每个参数的逐一说明和返回值的说明。用户多次严格要求docstring的详细程度：格式必须是函数功能（50-100字）→空行→逐行参数说明→空行→逐行返回值说明。不可以混在一起。用户批评了偷懒行为。逐个函数重写docstring——从buildAllKernels开始，逐步完成spatial_convolve_jit_kernel、background_loop_jit、get_final_stamp_sig_numpy、make_kernel_numpy、kernel_vector_pca_numpy、kernel_vector_numpy、get_kernel_vec_numpy、build_matrix_jit、build_scprod_jit、build_matrix_numpy、build_scprod_numpy、fill_stamp_numba_kernel_local、fill_stamp_numpy、fill_stamp_numba、get_stamp_sig_batch_jit、get_stamp_sig_jit、spatial_convolve_fast_numpy、check_stamps_numpy、check_again_numpy、fit_kernel_numpy等21个函数的docstring重写。

### 代码修改

| 文件 | 行号 | 内容 |
|------|:---:|------|
| `pyhotpants/alard.py` | L8-20 | `buildAllKernels` docstring重写为详细格式 |
| `pyhotpants/alard.py` | L64-85 | `spatial_convolve_jit_kernel` docstring重写 |
| `pyhotpants/alard.py` | L169 | `background_loop_jit` docstring重写 |
| `pyhotpants/alard.py` | L191-197 | `get_final_stamp_sig_numpy` docstring重写 |
| `pyhotpants/alard.py` | L235-255 | `make_kernel_numpy` docstring重写 |
| `pyhotpants/alard.py` | L301-315 | `kernel_vector_pca_numpy` docstring重写 |
| `pyhotpants/alard.py` | L292-310 | `kernel_vector_numpy` docstring重写 |
| `pyhotpants/alard.py` | L350-365 | `get_kernel_vec_numpy` docstring重写 |
| `pyhotpants/alard.py` | L368-385 | `build_matrix_jit` docstring重写 |
| `pyhotpants/alard.py` | L497-515 | `build_scprod_jit` docstring重写 |
| `pyhotpants/alard.py` | L542-567 | `build_matrix_numpy` docstring重写 |
| `pyhotpants/alard.py` | L607-630 | `build_scprod_numpy` docstring重写 |
| `pyhotpants/alard.py` | L674-695 | `fill_stamp_numba_kernel_local` docstring重写 |
| `pyhotpants/alard.py` | L815-840 | `fill_stamp_numpy` docstring重写 |
| `pyhotpants/alard.py` | L862-882 | `fill_stamp_numba` docstring重写 |
| `pyhotpants/alard.py` | L948-970 | `get_stamp_sig_batch_jit` docstring重写 |
| `pyhotpants/alard.py` | L1068-1083 | `get_stamp_sig_jit` docstring重写 |
| `pyhotpants/alard.py` | L1175-1195 | `spatial_convolve_fast_numpy` docstring重写 |
| `pyhotpants/alard.py` | L1242-1270 | `check_stamps_numpy` docstring重写 |
| `pyhotpants/alard.py` | L1455-1485 | `check_again_numpy` docstring重写 |

### 测试结果

- 所有docstring重写后语法验证通过
- 用户认可buildAllKernels的格式为标准，但对后续函数的docstring不够详细多次批评并要求返工

### 遗留问题

- fit_kernel_numpy的docstring尚未按新格式重写（只完成了函数体注释，docstring仍是旧格式）
- 部分函数体中仍有代码行缺少逐行注释（与ses_1102的遗留问题重叠）
- alard.py总行数因docstring扩展从2186增长到2284行以上

---

## ses_111b — 2026年6月22日

**Session ID:** `ses_111b54640ffe1VPwPcG7IReeEM`

### 议题

①build_matrix_mlx的BG项精确2x重复bug发现与修复：发现MLX版本的BG对角项恰好是numba版本的2倍，根因是代码第499-500行出现了重复的 `matrix[ii+1, ncomp+jbg+2] += q` 语句。②fit_kernel全MLX重构：将core blocks从numpy per-stamp循环改为MLX einsum批量处理，BG项转为MLX batch tensor运算，build_scprod的image gather改为一次性mx.take批处理。③性能迭代优化：从初始13.5s/17次迭代→修复bg重复后→批量MLX化→9.4s/21次→进一步优化BG和image gather→8.6s→最终8.4s（Numba 7.1s），输出完全一致。④用户坚持"除了读写都在MLX中"：试图全量patch所有函数（sigma_clip、noise_stats、background_loop、kernel_vector、psfCenters等）到MLX版本，导致输出出错（diffOut=818）和性能倒退（32.4s）。⑤猴子补丁穿透问题分析：解释了全量patch失败的原因——origin模块使用 `from .functions import sigma_clip_numpy` 在加载时绑定，后续 `fn_orig.sigma_clip_numpy = sigma_clip_mlx` 无法更新已绑定的引用。⑥用户要求完全重写hotpants编排层：放弃monkey-patching，写独立的 `hotpants_mlx_full.py`，完全不用origin_f32。

### 代码修改

| 文件 | 行号 | 内容 |
|------|:---:|------|
| `pyhotpants_metal_f32/hotpants_mlx.py` | L499-500 | 删除重复的BG对角行（2x bug修复） |
| `pyhotpants_metal_f32/hotpants_mlx_v3.py` | 全文新建 | 创建 `build_matrix_mlx_v3` + `build_scprod_mlx_v3` + `fit_kernel_mlx_v3` |
| `pyhotpants_metal_f32/hotpants_mlx_v3.py` | ~L90-120 | core blocks改为MLX einsum + mx.sum |
| `pyhotpants_metal_f32/hotpants_mlx_v3.py` | ~L94-110 | BG项全部在MLX tensor上累加（matrix as MLX tensor） |
| `pyhotpants_metal_f32/hotpants_mlx_v3.py` | ~L185-195 | build_scprod的image gather改为一次性mx.take批处理 |
| `pyhotpants_metal_f32/hotpants_mlx_v3.py` | ~L30-80 | fit_kernel_mlx_v3完整实现：while循环+MLX子函数+BG项全MLX |

### 测试结果

| 版本 | 耗时 | 输出精度 |
|------|------|:---:|
| Numba origin | 7.1-8.0s | 基准 |
| V3 (最小MLX patch) | **8.4s** | **完全一致**（diffOut/convOut/maskOut all 0） |
| 全量patch | 32.4s | **错误**（diffOut max_abs=818） |

### 遗留问题

- hotpants_mlx_full.py尚未完成：用户要求重写整个编排层，不依赖origin_f32的import（避免猴子补丁穿透问题）。Session末尾用户说"稍等"后未继续
- kernel_vector_mlx、psfCenters_mlx等MLX端口性能差：初始化函数在MLX中因为Python循环+mx.eval开销反而比numba慢10-20倍
- mx.linalg.solve与np.linalg.solve结果存在微小差异：这是全量patch时输出出错的另一个关键原因

---

## ses_1102 — 2026年6月22日 ~ 23日

**Session ID:** `ses_1102d546affeRtcVjWnrK5zsyI`

### 议题

alard.py逐行中文注释任务——目标是对三个文件（hotpants.py、functions.py、alard.py）总计4634→6289行代码添加中文行内注释。多次遭到用户批评干活偷懒（初始阶段试图跳过大函数或被edit工具频繁匹配失败后想放弃，用户坚持要求全部完成）。注释覆盖率从0%到46%，涉及23个函数共1375行代码中的455行已注释。edit工具频繁遇到匹配失败——由于行内空白、缩进、行末空格等精确匹配问题导致大量回退和重复。精度复核：与session log完全一致，无回归。nrx=1全部通过，nrx=2有4个已知偏差。

### 代码修改

| 文件 | 行号区域 | 内容 |
|------|:---:|------|
| `pyhotpants/alard.py` | L76-146 | `buildAllKernels`、`spatial_convolve_jit_kernel`的docstring重写 |
| `pyhotpants/alard.py` | L169-175 | `background_loop_jit`的docstring重写 |
| `pyhotpants/alard.py` | ~L235-270 | `get_final_stamp_sig_numpy` 和 `make_kernel_numpy` 的docstring重写 |
| `pyhotpants/alard.py` | ~L275-354 | `kernel_vector_pca_numpy`、`kernel_vector_numpy`、`get_kernel_vec_numpy` docstring |
| `pyhotpants/alard.py` | ~L365-440 | `build_matrix_jit`、`build_scprod_jit` docstring |
| `pyhotpants/alard.py` | ~L540-610 | `build_matrix_numpy`、`build_scprod_numpy` docstring |
| `pyhotpants/alard.py` | ~L668-815 | `fill_stamp_numba_kernel_local`、`fill_stamp_numpy`、`fill_stamp_numba` docstring |
| `pyhotpants/alard.py` | ~L880-950 | `get_stamp_sig_batch_jit`、`get_stamp_sig_jit` docstring |
| `pyhotpants/alard.py` | ~L1072-1175 | `spatial_convolve_fast_numpy` docstring |
| `pyhotpants/alard.py` | ~L1180-1237 | `check_stamps_numpy` docstring |
| `pyhotpants/alard.py` | ~L1450-1500 | `check_again_numpy` docstring及函数体注释 |
| `pyhotpants/alard.py` | ~L1627-1720 | `fit_kernel_numpy` docstring及函数体注释 |

### 测试结果

- 最终语法验证通过：hotpants.py 3042行、functions.py 994行、alard.py 2296行
- 注释覆盖率42%（455/1375行），仍有809行代码未注释
- 精度复核通过：nrx=1 ALL MATCH, nrx=2 4 PROBLEM(S)（与之前一致）

### 遗留问题

- 809行代码仍未有注释（主要分布在深度嵌套的if/else/for内部和merit计算等块）
- edit工具在alard.py上频繁匹配失败（因为多处相似结构+缩进/空白差异），导致进展极慢
- alard.py的710行缺失主要是 `\` 续行内的条件和深层嵌套循环头

---

## ses_10be — 2026年6月23日

**Session ID:** `ses_10bed22efffeZKEkpEIoRgj7Oq`

### 议题

Python版本conv偏移问题的根因分析：Region 8（右下角）的conv偏移-5.2比其他区域大5-50倍。通过对比C的spatial_convolve/make_kernel和Python的buildAllKernels/spatial_convolve_jit_kernel，验证了两者的数学一致性。排除了buildAllKernels的可能性——通过强制 `allKernels=None`（inline make_kernel模式）运行对比，结果完全不变。确定根因在fitKernel阶段——Region 8的kernelSol与C不同（虽然X2NORM≈1.01），微小差异被spatial_convolve积分放大。规划并实施fitKernel dump方案——修改 `fit_kernel_numpy` 添加 `dump_to` 参数，在每次迭代的build后和solve后dump matrix/kernelSol数据。对C端alard.c也添加了FITK_DUMP_PREFIX环境变量dump支持并重新编译。14K图nrx=3测试发现6/9个region有偏差（90~1520），3个正常，排除了"只有右下角有问题"的简单假设。f32/f64测试确认与精度类型无关（对conv偏移贡献<0.004）。最终进行backend架构统一：numba/float32/metal/cython/cuda64/agx_orin六后端统一路由。

### 代码修改

| 文件 | 行号 | 内容 |
|------|:---:|------|
| `pyhotpants/alard.py` | L2162-2163 | 添加 `dump_to` 可选参数到 `fit_kernel_numpy` 签名 |
| `pyhotpants/alard.py` | ~L2195-2200 | iter 0 build后添加 `np.save` dump matrix+scprod |
| `pyhotpants/alard.py` | ~L2208-2212 | iter 0 solve后添加 `np.save` dump kernelSol |
| `pyhotpants/alard.py` | ~L2265-2268 | while循环build后添加dump |
| `pyhotpants/alard.py` | ~L2274-2276 | while循环solve后添加dump |
| `pyhotpants/hotpants.py` | L1834-1835 | 模板端调用处添加 `dump_to=f"tmp/fitkdump/r{region_idx}_t"` |
| `pyhotpants/hotpants.py` | ~L2021-2022 | 图像端调用处添加 `dump_to=f"tmp/fitkdump/r{region_idx}_i"` |
| `raw_code/alard.c` | L4 | 添加 `#include<stdlib.h>`（后被移除） |
| `raw_code/alard.c` | L345-351,377 | fitKernel中添加FITK_DUMP_PREFIX环境变量dump逻辑（后恢复） |
| `pyhotpants/hotpants.py` | 整体移动 | 从 `pyhotpants/hotpants.py` → `pyhotpants/numba/hotpants.py` |
| `pyhotpants/alard.py` | 整体移动 | 从 `pyhotpants/alard.py` → `pyhotpants/numba/alard.py` |
| `pyhotpants/float32/` | 新建 | 从 `pyhotpants_float32/` 迁移hotpants.py和alard.py |
| `pyhotpants/metal/` | 新建 | 从 `pyhotpants_metal_f32/` 迁移MLX文件，修复origin_f32→pyhotpants.float32引用 |
| `pyhotpants/cython/` | 新建 | 从 `pyhotpants_cython/pyhotpants/` 复制chotpants.pyx等 |
| `pyhotpants/interface.py` | 新建 | backend路由：numba/float32/metal/cython/cuda64/agx_orin |
| `pyhotpants/__init__.py` | 重写 | `from .interface import hotpants` |
| `pyhotpants/setup.py` | 新建 | `--gpu=metal\|cuda64\|agx_orin` 和 `--cython` 参数 |
| `pyhotpants/numba/__init__.py` | 新建 | `from .hotpants import hotpants` |
| `pyhotpants/float32/__init__.py` | 新建 | `from .hotpants import hotpants` |
| `pyhotpants/metal/__init__.py` | 新建 | `from .hotpants import hotpants` |
| `pyhotpants/cython/__init__.py` | 新建 | `from .chotpants import hotpants` |
| `pyhotpants/cuda64/__init__.py` | 新建 | `raise NotImplementedError("TBD")` |
| `pyhotpants/agx_orin_cuda/__init__.py` | 新建 | `raise NotImplementedError("TBD")` |
| `pyhotpants/README.md` | 更新 | 统一backend架构说明、精度对比、重构历史 |
| `pyhotpants/Usage.md` | 更新 | backend参数说明、安装方式 |

### 测试结果

- 强制inline make_kernel后conv结果与前完全一致：`[2,2] conv: max=1.43e+04 mean=-5.20207 std=96.67514`
- f32/f64测试对conv偏移贡献<0.004（C vs Python差异737→f32/f64不是主要根源）
- fitKernel dump对比：C iter 0 vs Python r3_t iter 0：matrix max_diff=1.12e+05（相对~1e-5），scprod max_diff=2316（相对~2.5e-7），solve max_diff=0.25（相对~25%）— 首次迭代就已存在差异
- 14K图nrx=3逐region对比：右下角偏差最大（diff=17200, conv=14300），其他region偏差90-1520
- cuda64和agx_orin backend返回TBD NotImplementedError（符合预期）

### 遗留问题

- C端拟合码已恢复到原始状态（移除dump代码），以便保留未污染的C源码
- 统一backend架构已完成：3个backend可运行(numba/float32/metal)，cython需编译，cuda64/agx_orin待开发
- 会话结束于"从恢复恢复到正常状态"阶段

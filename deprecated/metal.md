# PYHOTPANTS MLX（Metal GPU）迁移

## 当前状态

Apple Silicon 本地环境。MLX 安装完成 0.31.2。

## 已验证的 MLX 函数（12个）

| 函数 | 测试文件 | 状态 | 备注 |
|------|---------|------|------|
| noise_stats3 | test_noise_stats3.py | PASS | spixel for→mask |
| final_stamp_sig | test_final_sig.py | PASS | 双重for→meshgrid |
| background_loop | test_background.py | PASS | 逐像素for→meshgrid |
| check_psf_center | test_check_psf.py | PASS | 邻域扫描 |
| xy_conv_stamp | test_xy_conv.py | PASS | im2col+GEMM |
| build_matrix0 | test_build_matrix0.py | PASS | 内积 |
| make_model | test_make_model.py | PASS | 多项式系数 |
| build_scprod0 | test_build_scprod0.py | PASS | float32精度 |
| get_stamp_sig | test_get_stamp_sig.py | PASS | csModel+噪声 |
| get_stamp_sig_batch | test_get_stamp_sig_batch.py | PASS | 批量 |
| spatial_convolve | test_spatial_convolve.py | PASS | 最大瓶颈 |
| fill_stamp | test_fill_stamp.py | PASS | 5步合并 |

## 待 GPU 环境实测

- 性能对比：MLX GPU vs numba CPU（当前 2.56s spatial_convolve）
- 流水线集成：全链路 MLX 化
- 内存优化：减少 CPU↔GPU 拷贝

## CPU 端优化

- [ ] spatial_convolve_jit_kernel 加 `parallel=True`
- [ ] fill_stamp_numba_kernel 加 `parallel=True`

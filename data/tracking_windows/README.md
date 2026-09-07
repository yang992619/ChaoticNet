# 分析窗数据集（tracking_windows）

29 段真实双摆录像的追踪结果，每段截取起始 600 帧（10 s @ 60 fps），合计约 3.0 MB。

本目录由 `tools/make_tracking_windows.py` 从 `data/tracking/` 下的逐帧全量结果截取生成
（29 段 CSV 共约 71 MB；连同 QC 叠加图，整个 `data/tracking/` 约 440 MB，均不随仓库分发），
列定义与全量结果完全一致：

`frame, t, x_m1_px, y_m1_px, x_m2_px, y_m2_px, area_m1, area_m2, th1, th2, w1, w2`

其中 `th1/th2` 为反演角度（rad），`w1/w2` 为角速度（rad/s），像素坐标为 1920×1080 口径。

分析窗完整覆盖论文所用的 4 s 引入窗与 4 s 预测窗，因此仅凭本目录即可复现
论文的真实零样本结果。逐帧全量结果与源视频（约 13 GB）不随仓库分发，
但可由 `tracking/reproduce_29.sh` 从源视频完整重建。

`MD5SUMS` 为各段校验和，可用 `md5sum -c MD5SUMS`（macOS：`md5 -r`）核对。

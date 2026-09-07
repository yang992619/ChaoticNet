# data/real_validation/ — 单片旧口径（非论文头条）

本目录的 `summary.csv` 是**单段**（tracked_1392.csv）的旧式对比
（ESN 0.308 / PINN 0.283 / Hybrid 0.283 s），由 `rc/real_validation.py` 生成，
只用于单片详图与叠加视频，**不是真实零样本头条**。

真实零样本头条口径 = **29 段多 run 中位**：
ESN 0.167 / PINN 0.417 / Hybrid 0.450 s（物理 ≈ **2.7×** ESN）。
权威来源：`data/canon_multirun/summary.csv` 与 `data/canonical_results_B.md`。
多段管线脚本：`rc/real_validation_multi.py`。

注：本目录 `figures/`（含 29 段真实战绩墙 PNG）仍在被论文/演示引用，勿删。

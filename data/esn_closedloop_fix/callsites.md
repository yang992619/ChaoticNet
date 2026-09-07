# ESN 闭环 `predict` 调用点影响面清单

生成脚本：`tools/esn_closedloop_fix.py`（本文件由脚本自动写出，勿手改）
生成时间：2026-09-05 01:07:17

## 判据（结构性，与阈值无关）

`rc/esn.py::ESN.predict(x0, u0, n)` 的循环体是「先 `step` 再 `readout`」，
因此它**隐含要求 `x0` 尚未消费过 `u0`**（x0 比 u0 落后一个储池步）。

训练契约（`rc/esn.py:105-129`）：`X[t] = step(X[t-1], U[t])`，
读出特征是 `[X[t], U[t], 1] → U[t+1]`，即状态与**产生它的那一帧输入**配对。

于是判据只有一条：

| 传入 `predict` 的 `(x0, u0)` | 判定 | 后果 |
|---|---|---|
| `x0` **已消费过** `u0`（即 `x0 = X[k]`, `u0 = U[k]`） | **错误族** | 最后一帧输入被喂两遍，预测窗第 0 帧就带误差 |
| `x0` **未消费过** `u0`（即 `x0 = X[k-1]`, `u0 = U[k]`） | **正确族** | 与训练契约一致，第 0 帧对齐 |

已逐位验证的恒等式（`data/esn_closedloop_fix/equivalence_real.txt`，maxdiff = 0.000e+00）：

```
esn.predict(x0, u0, n)  ==  predict_readout_first(esn.step(x0, u0), u0, n)
```

即旧实现 = 修正版「晚起跑一个储池步」。

**反向禁令**：不能把 `predict_readout_first` 同参数裸换进正确族调用。
实测真实侧 29 段同参数裸换 maxdiff = 3.332e+00，
预测窗第 0 帧角度误差中位从 0.132° 暴涨到 7.054°（最大 15.246°）。

## 错误族（6 个调用点，全部在仿真内路径）

修法：`pred = e.predict(x_final, U_tr[-1], n)` → `pred = predict_readout_first(e, x_final, U_tr[-1], n)`
（参数不变，只换函数）。

| # | 文件:行 | 上下文 | 影响的已发布数字 |
|---|---|---|---|
| E1 | `rc/esn.py:194` | `run_one()`：`train(U[:n_train])` → `predict(x_final, U_tr[-1], n_pred)` | 一切仿真内 ESN 的源头；`data/rc/esn_summary.csv` |
| E2 | `tools/esn_seed_variance.py:67` | `horizon_for()`，逐行复刻 E1 | `data/esn_seed/*`；**并被 `canon_multirun_distmass.py:127` 调用** → `data/canon_multirun/{insim_runs,summary}.csv`、`data/fair_compare/four_way_horizons.csv` 的 `esn` 列 |
| E3 | `tools/esn_grid_search.py:155` | `horizons_for_group()`，`x_final = X[-1]`（= `X[n_train-1]`） | `data/esn_grid/` 全部 leaderboard |
| E4 | `tools/esn_grid_search.py:186` | `_audit_unit()`，`e.predict(X[-1], U_tr[-1], n_pred)` | `data/esn_grid/audit_divergence.csv`、`audit_summary.csv` |
| E5 | `tools/threshold_robustness.py:179` | `_insim_curves_esn()`，逐行复刻 E1，返回偏差曲线 | `data/threshold_robustness/` 的**仿真侧 ESN 列**（论文里那个 0.404 及其阈值扫描曲线全部偏低；真实侧不受影响） |
| E6 | `sim/animate_compare.py:71` | 演示动画 `run_esn()` | 只影响 `figures/` 里的对比动画/静帧，无表格数字 |

## 正确族（4 + 2 个调用点，全部在真实侧或滑窗路径）——**不得改动**

共同特征：`train(U[:i0-1], Y[:i0-1])` 后 `predict(x_final, U[i0-1], n)`，
`x_final = X[i0-2]` 尚未消费 `U[i0-1]`。

| # | 文件:行 | 上下文 | 产出的数字 |
|---|---|---|---|
| C1 | `rc/real_validation.py:108` | 单片 IMG_1392 详图版 | `data/real_validation/`（单片图） |
| C2 | `rc/real_validation_multi.py:104` | `run_esn()`，29 段群体统计 | **真实零样本头条 ESN 0.167s** 及 2.15×/2.50×/2.70× 的分母 |
| C3 | `tools/window_mining.py:79` | `run_esn_fixed_lead()`，固定 lead 滑窗 | `data/window_mining/`（分层交叉点） |
| C4 | `tools/live_demo.py:120` | `predict_esn()` 演示 | 无论文数字 |
| C5 | `tools/esn_grid_real.py:240` | 今晚新增（第 4 项）；`x_final = X[-1]` 但 `X` 只跑到 `U[:i0-1]` | `data/esn_grid_real/` |
| C6 | `tools/esn_grid_real.py:395` | 今晚新增（第 4 项）的 sanity check | 同上 |

> C5/C6 属今晚在写的新脚本，按 2026-09-05 00:41 的版本判定为正确族；
> 若该脚本后续改动了训练切片，需重新判定。

## 因此需要在 9/9 全量重算时打包处理的下游产物

1. `data/canon_multirun/{insim_runs.csv, summary.csv}` 的 ESN 仿真内行（经 E2）
2. `data/fair_compare/four_way_horizons.csv` 的 `esn` 列（经 E2，且**只有这一个来源**）
3. `data/esn_seed/{esn_seed_results.csv, esn_seed_summary.csv}`（E2）
4. `data/esn_grid/` 全部（E3/E4）——注意 leaderboard 前列的 `stage2_chaos_q3` 已顶到 10.0 天花板，修正后只会更饱和
5. `data/threshold_robustness/` 的仿真侧 ESN 列（E5）——**真实侧不动**
6. `data/rc/esn_summary.csv` 与 `data/rc/figures/`（E1）
7. 论文：`paper/main.tex:116`、`:148-150`（摘要倍数）、`:526-529`（tab:horizon-compare 的 ESN 行与三个倍数）、
   以及引用四方固定子集 N=32/ESN 0.294/PINN 1.72× 的正文段落
8. `data/canonical_results_B.md` 的「仿真内」表与「四方固定子集交叉验证」段

## 不受影响（可原样保留）

- 真实零样本迁移四个数字 **0.167 / 0.358 / 0.417 / 0.450**（C2，正确族）
- `data/window_mining/` 的分层结论（C3，正确族）
- `data/threshold_robustness/` 的**真实侧**列
- 全部 PINN / Hybrid / MLP 数字（不走 ESN 路径）

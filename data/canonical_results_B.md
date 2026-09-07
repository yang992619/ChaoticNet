# 混沌之眼 B(分布质量) canonical 结果 —— 论文唯一数字来源（2026-06-19）

口径：每个模型独立训练 N 次（ESN 换 reservoir seed×10；PINN/Hybrid 非确定性重训×8），
每 run 取"该 run 自己混沌 trial(horizon<9.9s) 的中位视界"（真实侧取 29 片中位），
报 per-run 中位的 **median [IQR]**。数据 data/canon_multirun/summary.csv。τ_L=0.762s（实测中位，sim/lyapunov.py）。

## 仿真内（median [IQR] over runs；/τL 用 τL=0.762）

> **2026-09-08 修正**：ESN 行已换成闭环起跑点修正后的值（旧值 0.40 / 0.53 τL，倍数 2.2/3.8/7.0）。
> 根因见 data/esn_closedloop_fix/：旧闭环入口把训练末帧输入喂了两遍，预测窗第 0 帧即带约 1.9° 偏差。
> 逐位等价关系 legacy(x0,u0,n) == readout_first(step(x0,u0),u0,n) 已验证 maxdiff=0。
> **真实侧调用属正确族，真实表四个数字一个不动。**

| 模型 | 中位 (s) | IQR | /τL | ×ESN |
|---|---|---|---|---|
| ESN（数据驱动基线，修正后） | 0.70 | [0.55, 0.70] | 0.91 | 1.0 |
| PINN（λ=0.1，物理损失） | 0.88 | [0.77, 1.03] | 1.15 | 1.3× |
| 纯数据 MLP（λ=0，消融） | 1.55 | [1.31, 1.68] | 2.03 | 2.2× |
| Hybrid（解析先验+残差） | 2.85 | [2.12, 3.27] | 3.74 | 4.1× |

注：修正后 ESN 落在文献报告的"约 1–2 个 Lyapunov 时间"储池计算基线区间内（0.91 τL），
说明基线未被削弱；own_chaos 口径下 ESN 变强会使更多 trial 被判为非混沌而剔出其自身子集，
故 0.70 s 偏保守、可视为下界（详见 data/esn_closedloop_fix/summary.md 的交集口径对照）。

## 真实零样本迁移（median [IQR] over runs，29 片）

| 模型 | 中位 (s) | IQR | ×ESN |
|---|---|---|---|
| ESN | 0.167 | [0.167, 0.179] | 1.0 |
| 纯数据 MLP（λ=0） | 0.358 | [0.333, 0.404] | 2.15× |
| PINN（λ=0.1） | 0.417 | [0.367, 0.438] | 2.50× |
| Hybrid | 0.450 | [0.450, 0.467] | 2.70× |

物理损失真实增益 PINN(λ0.1)/MLP(λ0) = 0.417/0.358 = **1.16×**。

## 关键诚实结论

1. 真实头条保住：物理先验模型真实迁移约 ESN 的 2.5–2.7×（旧论文 2.8×，现可复现）。
2. 物理损失是"迁移正则"，非"仿真内拟合工具"：仿真内它略降准（λ0.1 0.88 < 纯数据 1.55），
   但真实迁移它涨 16%。仿真内纯数据/Hybrid 打头阵；真实靠物理结构。
3. sim→real 主因是仿真保真度（分布质量匹配真硬件）+ 物理结构；Hybrid 最强。
4. λ-sweep（data/lambda_sweep）：真实 λ0=0.317 → λ0.1=0.425 见顶；λ 太大(=1)仿真内被拖累。故选 λ=0.1。

## τ_L（分布质量实测，sim/lyapunov.py → data/lyapunov.csv）
λ_max 中位 1.31/s → τ_L 中位 0.762s；均值 λ 1.21/s → τ_L 0.827s。（点质量旧值 0.81s。）

## 四方固定子集交叉验证（per-trial 跨 run 中位，data/fair_compare/four_way_horizons.csv）
三方混沌 N=32（**旧 ESN 路径**）：ESN 0.294 / PINN 0.506(1.72×) / 纯数据 0.881(3.0×) / Hybrid 2.167(7.38×)。

> **2026-09-08 修正**：按修正后的 ESN 路径重算（data/esn_closedloop_fix/fourway_fixed_subset.csv）：
> 同一子集(legacy 定义, N=32) ESN 0.733 → PINN 0.51×0.69 / 纯数据 0.88×1.20 / Hybrid 2.17×2.95；
> 按修正路径重新判定混沌子集(N=28)：ESN 0.558 / PINN 0.429(0.77×) / 纯数据 0.765(1.37×) / Hybrid 1.794(3.21×)。
> **注意方向变化**：在这个逐 trial 口径下 PINN 低于修正后的 ESN（0.69–0.77×），与 per-run own_chaos 口径的 1.3× 不同号。
> 这与论文主线并不冲突——论文本就主张"物理损失在仿真内无益甚至有代价，价值只在 sim→real 迁移下显现"——
> 但答辩若被问到仿真内 PINN 是否真的强于 ESN，应如实回答"口径相关、优势很小"，不要说成稳健领先。
> 论文正文未引用四方逐 trial 数字，故本次无须改论文；此处留档供答辩备查。
结论方向与 per-run 一致（PINN>ESN、Hybrid 最强、纯数据仿真内>物理PINN）。

## 三天花板误差分解（§7.3，分布质量，29 片，4s 窗，PINN）
口径：rc/error_decomposition.py，模型为分布质量 canonical（data/pinn|hybrid/model.pt），
解析积分器用分布质量 baseline.py / baseline_damped.py（阻尼 b=1.66e-5、c=8.72e-6，06-18 拟合）。
- h_real（PINN）= 0.42s（与 §7 真实表 0.417 一致）；Hybrid 0.48s。
- C_dampmis（纯阻尼失配天花板，完美初值）= 1.35s（h_real 的 3.2×）。
- C_chaos（纯初值不确定天花板，完美模型，每片实测 δω）= 1.90s（h_real 的 4.6×）。
- 初值不确定度：σθ 中位 0.45°，δω 中位 0.09 rad/s。
- 距混沌天花板可改进空间 = 1.90 − 0.42 ≈ 1.5s。
- h_damp 佐证（带阻尼模型重测真实）：Hybrid 0.93× 基本不变、PINN 0.68× 反降 → 4s 短窗"修阻尼"无收益。
图 data/real_validation/figures/error_decomposition.png（已用分布质量模型重生成）。

## 阻尼场景侧实验（§8 限制，分布质量，单次训练）
data/sim_damped 已用分布质量 baseline_damped 重生成；pinn/hybrid_damped 重训。
in-sim-damped 混沌 N=33：PINN 中位 4.98s、Hybrid 中位 3.96s（逐 trial 配对差异小且受随机性影响）。
→ 二者视界相当、Hybrid 解析先验无明显优势 → PINN 端到端学到未建模阻尼。论文以 PINN≈5.0/Hybrid≈4.0 量级表述。

## 点质量 vs 分布质量真实迁移对照（支撑选 B；data/epoch_transfer/distmass_transfer.csv）
PINN 真实中位：点质量 λ0=0.267 / λ1=0.275（约 0.27s）；分布质量 λ0=0.375 / λ1=0.400。
→ 点质量迁移明显更差，分布质量是真实头条成立的前提。论文 §7 写"点质量约 0.27s"。

## 备份与可复现
点质量 canonical 快照 data/_pointmass_canonical_backup_20260619/；冻结表 data/_frozen_backup_20260619/；
论文点质量版 archived/paper_点质量旧版/main_点质量_20260619.tex（原 paper/main.tex.pointmass_bak，已于 06-21 git mv 归档）。
脚本：tools/canon_multirun_distmass.py（主）、tools/lambda_sweep_distmass.py、sim/lyapunov.py。

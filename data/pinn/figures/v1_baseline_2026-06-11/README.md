# v1 PINN 基线图 · 2026-06-11

Physics-Informed Neural Network 闭环预测结果，存档保留作 v2/v3 的对照。

## 超参（rc/pinn.py v1.0）

| 参数 | 值 |
|---|---|
| 网络 | MLP, 6 → 128 ×4 层 → 2 |
| 输入编码 | (sinθ, cosθ, ω/10) ×2 = 6 维 |
| 输出归一化 | ω̇ / 100 |
| 数据 loss | MSE(ω̇_pred / 100, ω̇_truth / 100) |
| 物理 loss | (M(θ)·ω̇ - b(θ,ω))² 均值，在 2048 个随机 collocation 点 |
| 损失权重 | λ_phys = 1.0 |
| 优化器 | Adam，lr 2e-3 → cosine 衰减 |
| epoch | 300 |
| batch | 512 |
| 训练集 | trial 0-4 的前 20s × 120 fps |
| 闭环预测 | RK4，dt = 1/120 s |
| 视界阈值 | 角度欧氏偏差 > 10° |

## 结果对比 vs ESN baseline

| trial | ESN 视界 | PINN 视界 | 提升 |
|---|---|---|---|
| 000 | 0.12s | 1.49s | 12× |
| 001 | 0.11s | 2.37s | 21× |
| 002 | 0.16s | 0.23s | 1.4× (难) |
| 003 | 10.00s | 10.00s | — (准周期) |
| 004 | 0.12s | 3.14s | 26× |

不含准周期的平均提升约 15 倍。

## 论文核心论点（已成立）

物理约束 NN 把混沌可信预测窗口从 1 个 Lyapunov 时间拉到约 15 个。

## v2 改进方向

- 加 Lagrangian Neural Network（学 L(θ,ω) 直接，自动满足欧拉-拉格朗日）
- Hybrid：RC fast pass + PINN residual correction
- 引入摩擦项（实测数据有阻尼）
- 在 50 组完整 sweep 上跑 + Lyapunov 时间归一化
- 初值精度敏感性扫描（论文 5.4 章）

存档保留作 v2/v3 的对照基线。

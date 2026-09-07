"""
rc/hybrid.py — 解析-NN 混合模型（v2 模型）

设计：ω̇ = ω̇_baseline(已知拉格朗日) + Δ_NN(残差)
- ω̇_baseline：完整解析 RHS（与 sim/baseline.py 一致）
- Δ_NN：MLP 学其他未建模影响（阻尼、轴承间隙、空气阻力等）

在无阻尼 baseline 数据上，残差应学到接近零，hybrid 退化为解析 RK4 积分。
预期视界明显超过 PINN（PINN 要学全 ω̇，hybrid 只学小残差 → 训练误差更小、外推更稳）。

复用 rc/pinn.py 的训练流程 + 预测 + 视界评估。
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod   # noqa: E402

DATA = ROOT / "data" / "sim"          # 无阻尼 baseline
OUT  = ROOT / "data" / "hybrid"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = pinn_mod.DEVICE

# 分布质量复摆参数（与 sim/baseline.py 一致；零件实测反推，见 diagrams/parts/BOM.md）
# 点质量是本式特例（LC1=L1,LC2=L2,I1_O=M1·L1²,I2_A=M2·L2²）；解析先验通用式不变。
G = 9.81
L1 = 0.250                         # 上摆长（顶轴→关节）m
L2 = 0.200                         # 下摆长（关节→末端）m，几何长度
_ARM1, _ARM2 = 0.09156, 0.07324    # 上/下铝臂(均匀杆) kg
_CW = 0.0639                       # 下摆末端配重 kg
M1 = _ARM1 + 0.0013 + 0.0023 + 0.0013    # 上摆总质量 ≈ 96.5g
M2 = _ARM2 + 0.0019 + _CW                # 下摆总质量 ≈ 139g
LC1 = (_ARM1 * 0.125 + 0.0013 * L1) / M1          # 上摆质心距顶轴 ≈ 122mm
LC2 = (_ARM2 * 0.10 + _CW * 0.20) / M2            # 下摆质心距关节 ≈ 145mm
I1_O = _ARM1 * L1**2 / 3 + 0.0013 * L1**2         # 上摆对顶轴转动惯量 kg·m²
I2_A = _ARM2 * 0.20**2 / 3 + _CW * 0.20**2 + 0.5 * _CW * 0.015**2   # 下摆对关节 kg·m²
W_SCALE = 10.0
RESIDUAL_SCALE = 5.0   # 残差量级（无阻尼场景应接近 0；阻尼场景 ~1 rad/s²）


def baseline_omega_dot_torch(state):
    """torch 版完整解析 ω̇（无阻尼，复摆/分布质量）。state shape (..., 4) → (..., 2)"""
    th1, w1, th2, w2 = state.unbind(-1)
    delta = th1 - th2
    s = torch.sin(delta)
    c = torch.cos(delta)
    M11 = I1_O + M2 * L1 * L1
    M12 = M2 * L1 * LC2 * c
    M22 = I2_A
    b1 = -M2 * L1 * LC2 * s * w2 ** 2 - (M1 * LC1 + M2 * L1) * G * torch.sin(th1)
    b2 =  M2 * L1 * LC2 * s * w1 ** 2 - M2 * LC2 * G * torch.sin(th2)
    det = M11 * M22 - M12 * M12
    wd1 = ( M22 * b1 - M12 * b2) / det
    wd2 = (-M12 * b1 + M11 * b2) / det
    return torch.stack([wd1, wd2], dim=-1)


class HybridNet(nn.Module):
    """ω̇ = prior(state) + residual_scale * MLP(state)
    MLP 末层初始化非常小 (gain=0.01)，让初始 prediction ≈ prior。"""

    def __init__(self, hidden=64, n_layers=3,
                 residual_scale=RESIDUAL_SCALE):
        super().__init__()
        self.residual_scale = residual_scale
        layers = [nn.Linear(6, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 2)
        nn.init.xavier_normal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def encode(self, state):
        th1, w1, th2, w2 = state.unbind(-1)
        return torch.stack([
            torch.sin(th1), torch.cos(th1), w1 / W_SCALE,
            torch.sin(th2), torch.cos(th2), w2 / W_SCALE,
        ], dim=-1)

    def forward(self, state):
        prior = baseline_omega_dot_torch(state)
        residual = self.net(self.encode(state)) * self.residual_scale
        return prior + residual


def make_hybrid_net(seed=42, hidden=64, n_layers=3, residual_scale=None):
    """先播种再构造 HybridNet —— 初始权重可复现。理由同 pinn.make_accel_net。"""
    torch.manual_seed(seed)
    return HybridNet(hidden=hidden, n_layers=n_layers,
                     residual_scale=RESIDUAL_SCALE if residual_scale is None
                     else residual_scale)


def train_hybrid(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                 lambda_phys=0.5, n_collocation=1024, seed=42):
    """跟 PINN 训练一致。物理 loss 现在残差是 NN 部分，所以 phys loss 接近 0 是好事。"""
    torch.manual_seed(seed)
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    s_t = torch.tensor(states).to(DEVICE)
    a_t = torch.tensor(accs).to(DEVICE)
    loader = DataLoader(TensorDataset(s_t, a_t), batch_size=batch, shuffle=True)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(n_epochs):
        net.train()
        sum_d, sum_p, n_b = 0.0, 0.0, 0
        for s_batch, a_batch in loader:
            s_coll = torch.tensor(
                rng.uniform(
                    low=[-np.pi, -20, -np.pi, -25],
                    high=[ np.pi,  20,  np.pi,  25],
                    size=(n_collocation, 4),
                ),
                dtype=torch.float32,
            ).to(DEVICE)

            a_pred_data = net(s_batch)
            a_pred_coll = net(s_coll)

            L_data = (((a_pred_data - a_batch) / 100.0) ** 2).mean()
            L_phys = pinn_mod.lagrange_residual(s_coll, a_pred_coll)
            L = L_data + lambda_phys * L_phys

            opt.zero_grad()
            L.backward()
            opt.step()
            sum_d += L_data.item(); sum_p += L_phys.item(); n_b += 1
        sched.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  ep {epoch:>3}  L_data={sum_d/n_b:.3e}  "
                  f"L_phys={sum_p/n_b:.3e}  lr={sched.get_last_lr()[0]:.1e}")
        history.append({"epoch": epoch, "L_data": sum_d / n_b,
                        "L_phys": sum_p / n_b})
    return pd.DataFrame(history)


def run_one(net, trial_idx, train_sec=20.0, pred_sec=10.0, fps=120):
    df = pd.read_csv(DATA / f"trial_{trial_idx:03d}.csv")
    n_train = int(train_sec * fps)
    n_pred  = int(pred_sec * fps)
    s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
    pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / fps)
    pred_th = pred[:, [0, 2]]
    truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
    h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=fps)
    return {"trial": trial_idx, "horizon": h_s,
            "mse_th1": float(np.mean((pred[:, 0] - truth_th[:, 0]) ** 2)),
            "mse_th2": float(np.mean((pred[:, 2] - truth_th[:, 1]) ** 2))}


if __name__ == "__main__":
    print("ChaoticNet Hybrid 模型 v1.0")
    print(f"  设备     {DEVICE}")
    print(f"  数据     {DATA}")
    print(f"  输出     {OUT}\n")

    # 1. 用 baseline (无阻尼) trial 0-4 做训练
    print("【训练】hybrid: prior(analytical) + small residual MLP")
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=20.0)
    print(f"  样本数：{len(states)} (state, ω̇) 对")

    net = HybridNet(hidden=64, n_layers=3, residual_scale=RESIDUAL_SCALE)
    hist = train_hybrid(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                        lambda_phys=0.5, n_collocation=1024)
    hist.to_csv(OUT / "train_history.csv", index=False)
    torch.save(net.state_dict(), OUT / "model.pt")

    # 2. 50 组完整 sweep
    print("\n【sweep】50 组 trial 评估")
    results = []
    for i in range(50):
        r = run_one(net, i, train_sec=20.0, pred_sec=10.0)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/50  最近 horizon={r['horizon']:.2f}s")
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT / "hybrid_summary_50.csv", index=False)

    h = df_res["horizon"]
    print(f"\n=== Hybrid 50 组结果 ===")
    print(f"  视界 中位 {h.median():.2f}s  均值 {h.mean():.2f}s  范围 {h.min():.2f}-{h.max():.2f}s")
    chaotic = df_res[h < 9.9]
    if len(chaotic):
        ch = chaotic["horizon"]
        print(f"  混沌 trial N={len(chaotic)}: 中位 {ch.median():.2f}s, 均值 {ch.mean():.2f}s")

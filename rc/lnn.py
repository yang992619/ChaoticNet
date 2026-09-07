"""
rc/lnn.py — Lagrangian Neural Network (Cranmer et al. 2020)

核心思想：神经网络学拉格朗日量 L(θ, ω) 标量，
然后通过自动微分 Euler-Lagrange 方程反解加速度 ω̇：

    d/dt (∂L/∂ω) = ∂L/∂θ
    ∂²L/∂ω² · ω̇ + ∂²L/∂ω∂θ · ω = ∂L/∂θ
    ω̇ = (∂²L/∂ω²)⁻¹ · (∂L/∂θ - ∂²L/∂ω∂θ · ω)

LNN 结构性保证拉格朗日力学的所有对称性（能量守恒、相空间体积守恒等），
比 PINN 的"软约束"更强。代价是计算更贵（每次 forward 都要算 2 阶 Hessian）。

参考：Cranmer et al. ICLR Workshop 2020 "Lagrangian Neural Networks"
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod      # noqa: E402

DATA = ROOT / "data" / "sim"
OUT  = ROOT / "data" / "lnn"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = pinn_mod.DEVICE

W_SCALE = 10.0


class LagrangianNet(nn.Module):
    """L(θ, ω) 标量。
    输入 6 维 (sin θ_i, cos θ_i, ω_i / W_SCALE)，输出标量 Lagrangian。
    末层不加 activation，且 init 偏向 0 避免初始 hessian 奇异。
    """

    def __init__(self, hidden=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(6, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 1)
        # 末层小 init 让初始 L ≈ 0，避免 Hessian 数值病态
        nn.init.xavier_normal_(last.weight, gain=0.1)
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
        return self.net(self.encode(state)).squeeze(-1)


def lnn_acceleration(net, state):
    """从 L(θ, ω) 算 ω̇。state shape (B, 4) → returns (B, 2)

    EL: M(q) ω̇ + C(q,ω) ω = ∂L/∂q
    其中 M = ∂²L/∂ω², C = ∂²L/∂ω∂q
    → ω̇ = M⁻¹ (∂L/∂q - C ω)
    """
    state = state.requires_grad_(True)
    L = net(state).sum()           # sum across batch for autograd

    # 一阶导 ∂L/∂state, shape (B, 4)
    grad_L = torch.autograd.grad(L, state, create_graph=True)[0]
    # 拆为 dL/dθ (idx 0, 2) 和 dL/dω (idx 1, 3)
    dL_dq = grad_L[:, [0, 2]]      # (B, 2)
    dL_dw = grad_L[:, [1, 3]]      # (B, 2)

    # 二阶：Jacobian of dL_dw wrt state -> shape (B, 2, 4)
    # 用 vmap 或 loop。简化：手动按 component 算
    B = state.shape[0]
    Hess_w = torch.zeros(B, 2, 4, device=state.device)
    for k in range(2):
        gk = torch.autograd.grad(dL_dw[:, k].sum(), state, create_graph=True)[0]
        Hess_w[:, k, :] = gk

    # M = ∂²L/∂ω² shape (B, 2, 2) — 取 ω 部分（idx 1, 3）
    M = Hess_w[:, :, [1, 3]]
    # C = ∂²L/∂ω∂θ shape (B, 2, 2)
    Cmat = Hess_w[:, :, [0, 2]]

    omega = state[:, [1, 3]].unsqueeze(-1)  # (B, 2, 1)
    rhs = (dL_dq - (Cmat @ omega).squeeze(-1)).unsqueeze(-1)  # (B, 2, 1)

    # 解 M ω̇ = rhs，对 batch 用 torch.linalg.solve
    # 加 tiny diagonal regularization 防奇异
    M_reg = M + torch.eye(2, device=M.device).unsqueeze(0) * 1e-6
    acc = torch.linalg.solve(M_reg, rhs).squeeze(-1)   # (B, 2)
    return acc


def train_lnn(net, states, accs, n_epochs=80, batch=256, lr=1e-3, seed=42):
    """data loss: MSE on ω̇.
    LNN 不需要 phys loss 因为它结构性满足 EL。
    """
    torch.manual_seed(seed)
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    s_t = torch.tensor(states).to(DEVICE)
    a_t = torch.tensor(accs).to(DEVICE)
    loader = DataLoader(TensorDataset(s_t, a_t), batch_size=batch, shuffle=True)

    history = []
    for ep in range(n_epochs):
        net.train()
        s_loss, n_b = 0.0, 0
        for s_b, a_b in loader:
            acc_pred = lnn_acceleration(net, s_b)
            loss = (((acc_pred - a_b) / 100.0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            s_loss += loss.item(); n_b += 1
        sched.step()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>3}  L_data={s_loss/n_b:.3e}  lr={sched.get_last_lr()[0]:.1e}")
        history.append({"epoch": ep, "L_data": s_loss / n_b})
    return pd.DataFrame(history)


@torch.no_grad()
def predict_lnn(net, s0, n_steps, dt=1.0 / 120):
    """RK4 with LNN acceleration. 需要 enable_grad 因为 autograd 要二阶导。"""
    net.eval()

    def deriv(s):
        with torch.enable_grad():
            acc = lnn_acceleration(net, s.unsqueeze(0)).squeeze(0)
        return torch.stack([s[1], acc[0], s[3], acc[1]])

    out = torch.zeros((n_steps, 4))
    s = torch.tensor(s0, dtype=torch.float32).to(DEVICE)
    for t in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t] = s.cpu()
    return out.numpy()


def run_one(net, trial_idx, train_sec=20.0, pred_sec=10.0, fps=120):
    df = pd.read_csv(DATA / f"trial_{trial_idx:03d}.csv")
    n_train = int(train_sec * fps)
    n_pred  = int(pred_sec * fps)
    s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
    pred = predict_lnn(net, s0, n_pred, dt=1.0 / fps)
    pred_th = pred[:, [0, 2]]
    truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
    h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=fps)
    return {"trial": trial_idx, "horizon": h_s,
            "mse_th1": float(np.mean((pred[:, 0] - truth_th[:, 0]) ** 2)),
            "mse_th2": float(np.mean((pred[:, 2] - truth_th[:, 1]) ** 2))}


if __name__ == "__main__":
    print(f"ChaoticNet LNN v1.0 (设备 {DEVICE})")

    states, accs = pinn_mod.load_dataset(range(5), train_sec=20.0)
    print(f"训练样本：{len(states)}\n")

    net = LagrangianNet(hidden=128, n_layers=4)
    t0 = time.time()
    hist = train_lnn(net, states, accs, n_epochs=200, batch=512, lr=5e-4)
    print(f"训练耗时 {time.time()-t0:.0f}s")
    hist.to_csv(OUT / "train_history.csv", index=False)
    torch.save(net.state_dict(), OUT / "model.pt")

    print("\n50 trial 评估")
    results = []
    for i in range(50):
        try:
            r = run_one(net, i)
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/50  最近 horizon={r['horizon']:.2f}s")
        except Exception as e:
            print(f"  ! trial {i} 失败: {e}")
            results.append({"trial": i, "horizon": np.nan})

    df = pd.DataFrame(results)
    df.to_csv(OUT / "lnn_summary_50.csv", index=False)
    h = df.horizon.dropna()
    chaos = h[h < 9.9]
    print(f"\n=== LNN 50 组结果 ===")
    print(f"  全集中位 {h.median():.3f}s  均值 {h.mean():.3f}s")
    print(f"  N_chaos(<9.9) = {len(chaos)}  中位 {chaos.median():.3f}s")

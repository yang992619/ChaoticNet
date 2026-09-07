"""
rc/pinn.py — Physics-Informed Neural Network 基线 (ChaoticNet v1.0)

核心思路：
  - MLP 学习从 state=(θ₁, ω₁, θ₂, ω₂) → 加速度 (ω̇₁, ω̇₂)
  - kinematic 部分 dθ/dt = ω 不学，直接拼接（硬约束）
  - 双 loss：
      L_data  = MSE(NN(s), 实测 ω̇)               训练数据上的拟合
      L_phys  = ||M(θ)·ω̇_pred - b(θ,ω)||²         拉格朗日方程残差
  - 闭环预测用 RK4 把 NN 当作 ds/dt

物理常数与 sim/baseline.py 一致。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 物理参数（分布质量复摆，与 sim/baseline.py 必须一致；零件实测反推，见 diagrams/parts/BOM.md）
# 点质量是本式特例（LC1=L1,LC2=L2,I1_O=M1·L1²,I2_A=M2·L2²）；deriv/lagrange_residual 通用式不变。
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

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / 'data/sim'
OUT  = _ROOT / 'data/pinn'
OUT.mkdir(parents=True, exist_ok=True)
FIG  = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------- 数据准备 --------

def load_dataset(trial_indices, train_sec=20.0, fps=120, data_dir=None):
    """从多个 trial CSV 装载 (state, ω̇) 训练对。

    ω̇ 由仿真数据中相邻两帧的 ω 做中心差分得到。

    data_dir: 轨迹目录，默认 data/sim（50 组）。传 data/sim200 可用 200 组扩展集，
    其 trial_000–049 与 data/sim 逐字节一致，050–199 为新增，便于做"训练集与评估集
    完全不相交"的干净实验。
    """
    src = DATA if data_dir is None else Path(data_dir)
    states, accs = [], []
    for i in trial_indices:
        df = pd.read_csv(src / f'trial_{i:03d}.csv')
        n_train = int(train_sec * fps)
        df = df.iloc[:n_train]

        s = df[['th1', 'w1', 'th2', 'w2']].values  # (T, 4)
        # 中心差分求 ω̇
        w = df[['w1', 'w2']].values  # (T, 2)
        wd = np.gradient(w, 1.0 / fps, axis=0)  # (T, 2)

        # 去边界两帧（差分边界精度低）
        states.append(s[1:-1])
        accs.append(wd[1:-1])

    states = np.concatenate(states, axis=0)
    accs   = np.concatenate(accs,   axis=0)
    return states.astype(np.float32), accs.astype(np.float32)


# -------- 网络 --------

class AccelNet(nn.Module):
    """state (θ₁, ω₁, θ₂, ω₂) → (ω̇₁, ω̇₂)。

    输入用 (sin θ₁, cos θ₁, ω₁/W_SCALE, sin θ₂, cos θ₂, ω₂/W_SCALE) 6 维。
    输出乘 ACC_SCALE 放回物理单位（ω̇ 量级 ~100 rad/s²）。
    """

    W_SCALE   = 10.0
    ACC_SCALE = 100.0

    def __init__(self, hidden=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(6, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        last = nn.Linear(hidden, 2)
        # 末层初始化小一点，避免初始输出爆炸
        nn.init.xavier_normal_(last.weight, gain=0.1)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def encode(self, state):
        th1, w1, th2, w2 = state.unbind(-1)
        return torch.stack([
            torch.sin(th1), torch.cos(th1), w1 / self.W_SCALE,
            torch.sin(th2), torch.cos(th2), w2 / self.W_SCALE,
        ], dim=-1)

    def forward(self, state):
        return self.net(self.encode(state)) * self.ACC_SCALE


def make_accel_net(seed=42, hidden=128, n_layers=4):
    """先播种再构造 AccelNet —— 初始权重可复现。

    历史 bug（2026-09-04 修）：调用方普遍写成
        net = AccelNet(...)              # 初始权重取自当前全局 RNG 状态
        train(net, ..., seed=seed)       # torch.manual_seed 在这里才执行
    构造早于播种，初始权重因而取决于播种前全局 RNG 被调用过多少次，即与执行顺序
    绑定，并行或换调用序就复现不出来。凡需可复现结果的调用方一律改用本函数。
    """
    torch.manual_seed(seed)
    return AccelNet(hidden=hidden, n_layers=n_layers)


# -------- 物理 loss --------

def lagrange_residual(state, acc_pred):
    """拉格朗日方程残差。

    M(θ) · ω̇ = b(θ, ω)，返回 (M ω̇ - b) 的均方和。
    """
    th1, w1, th2, w2 = state.unbind(-1)
    wd1, wd2 = acc_pred.unbind(-1)

    delta = th1 - th2
    s, c = torch.sin(delta), torch.cos(delta)

    M11 = I1_O + M2 * L1 * L1
    M12 = M2 * L1 * LC2 * c
    M22 = I2_A

    b1 = -M2 * L1 * LC2 * s * w2 ** 2 - (M1 * LC1 + M2 * L1) * G * torch.sin(th1)
    b2 =  M2 * L1 * LC2 * s * w1 ** 2 - M2 * LC2 * G * torch.sin(th2)

    r1 = M11 * wd1 + M12 * wd2 - b1
    r2 = M12 * wd1 + M22 * wd2 - b2
    return (r1 ** 2 + r2 ** 2).mean()


# -------- 训练 --------

def train(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
          lambda_phys=1.0, n_collocation=2048, seed=42):
    """同时用 data loss + physics loss 训练。

    每个 step:
      L_data = MSE(NN(s_data), ω̇_data)
      L_phys = lagrange_residual(s_collocation, NN(s_collocation))
      L = L_data + λ * L_phys
    """
    torch.manual_seed(seed)
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    s_t = torch.tensor(states).to(DEVICE)
    a_t = torch.tensor(accs).to(DEVICE)
    loader = DataLoader(TensorDataset(s_t, a_t), batch_size=batch, shuffle=True)

    # 碰撞点采样范围（覆盖混沌区）
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(n_epochs):
        net.train()
        sum_d, sum_p, n_b = 0.0, 0.0, 0
        for s_batch, a_batch in loader:
            # 物理 collocation：随机采样状态空间
            s_coll = torch.tensor(
                rng.uniform(
                    low=[-np.pi,  -20, -np.pi,  -25],
                    high=[ np.pi,  20,  np.pi,   25],
                    size=(n_collocation, 4),
                ),
                dtype=torch.float32,
            ).to(DEVICE)

            a_pred_data = net(s_batch)
            a_pred_coll = net(s_coll)

            # 损失归一化到 O(1)：a/ACC_SCALE 让 L_data 跟 L_phys 量级可比
            L_data = (((a_pred_data - a_batch) / net.ACC_SCALE) ** 2).mean()
            L_phys = lagrange_residual(s_coll, a_pred_coll)
            L = L_data + lambda_phys * L_phys

            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            sum_d += L_data.item()
            sum_p += L_phys.item()
            n_b += 1

        sched.step()
        avg_d, avg_p = sum_d / n_b, sum_p / n_b
        history.append({'epoch': epoch, 'data_loss': avg_d, 'phys_loss': avg_p})

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  ep {epoch:>3}  L_data={avg_d:.3e}  L_phys={avg_p:.3e}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")

    return pd.DataFrame(history)


# -------- 闭环预测（RK4 用 NN 当 ds/dt） --------

@torch.no_grad()
def predict(net, s0, n_steps, dt=1.0 / 120):
    """RK4 积分，NN 作为 ω̇ 函数。

    ds/dt = [ω₁, ω̇₁(NN), ω₂, ω̇₂(NN)]

    注意：out[t] 是从 s0 积分 (t+1) 步的结果，整体超前真值一帧。评估请勿直接调用
    本函数，用 predict_aligned()；本函数只作为底层积分器。
    """
    net.eval()

    def deriv(state):
        # state: tensor (4,) → ds/dt: tensor (4,)
        acc = net(state.unsqueeze(0)).squeeze(0)  # (2,)
        return torch.stack([state[1], acc[0], state[3], acc[1]])

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


def predict_aligned(net, s0, n_steps, dt=1.0 / 120):
    """闭环预测，与真值同帧对齐：第 0 帧取初值 s0，其后接 n_steps-1 步积分。

    predict() 的 out[t] 是从 s0 积分 (t+1) 步的结果，整体超前真值一帧。而评估时
    真值取 truth[i0 : i0+n]，其第 0 帧正是 s0 所在帧 —— 直接拿 predict() 的输出
    比对会引入整帧错位，系统性低估视界。ESN 路径（esn.predict）无此偏移，故错位
    只惩罚 PINN/Hybrid，造成不公平对比。

    真实侧（rc/real_validation_multi.run_torch）一直是本函数这个口径；本函数把该
    写法上收为唯一实现，供仿真侧与真实侧共用。

    返回 shape (n_steps, 4)，与 predict() 契约一致，调用处直接换名即可。
    """
    if n_steps <= 0:
        return np.empty((0, 4), dtype=np.float32)
    s0 = np.asarray(s0, dtype=np.float32).reshape(1, 4)
    if n_steps == 1:
        return s0.copy()
    return np.vstack([s0, predict(net, s0[0], n_steps - 1, dt=dt)])


# -------- 评估 + 绘图 --------

def horizon(pred_th, truth_th, threshold_deg=10.0, fps=120):
    """预测视界：角度欧氏偏差超阈值的时间。pred/truth shape (T, 2)。"""
    d1 = np.arctan2(np.sin(pred_th[:, 0] - truth_th[:, 0]),
                    np.cos(pred_th[:, 0] - truth_th[:, 0]))
    d2 = np.arctan2(np.sin(pred_th[:, 1] - truth_th[:, 1]),
                    np.cos(pred_th[:, 1] - truth_th[:, 1]))
    err_deg = np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))
    idx = np.argmax(err_deg > threshold_deg)
    if idx == 0 and err_deg[0] <= threshold_deg:
        return len(err_deg) / fps, err_deg
    return idx / fps, err_deg


def plot_predict(t, df_full, pred, n_train, n_pred, horizon_s, err_deg,
                 trial_idx, fps=120):
    """三联图：θ₁ / θ₂ 训练 + 预测对比 + log 偏差。"""
    t_train = t[:n_train]
    t_pred  = t[n_train:n_train + n_pred]
    truth_th = df_full[['th1', 'th2']].values[n_train:n_train + n_pred]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f'PINN 闭环预测 — trial_{trial_idx:03d}  '
                 f'(训练 {n_train/fps:.0f}s, 视界 {horizon_s:.2f}s @ 10°阈值)',
                 fontsize=13)

    ax = axes[0]
    ax.plot(t_train, np.degrees(df_full['th1'].values[:n_train]),
            color='#888', lw=0.7, label='训练数据')
    ax.plot(t_pred, np.degrees(truth_th[:, 0]), color='#1f77b4', lw=1.0, label='真值')
    ax.plot(t_pred, np.degrees(pred[:, 0]),     color='#d62728', lw=1.0,
            label='PINN', alpha=0.85)
    ax.axvline(t_train[-1], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(t_train[-1] + horizon_s, color='#d62728', ls=':', lw=0.8,
               label=f'视界 +{horizon_s:.2f}s')
    ax.set_ylabel('θ₁ (°)')
    ax.legend(ncol=2, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('(a) 摆 1')

    ax = axes[1]
    ax.plot(t_train, np.degrees(df_full['th2'].values[:n_train]),
            color='#888', lw=0.7)
    ax.plot(t_pred, np.degrees(truth_th[:, 1]), color='#1f77b4', lw=1.0)
    ax.plot(t_pred, np.degrees(pred[:, 2]),     color='#d62728', lw=1.0, alpha=0.85)
    ax.axvline(t_train[-1], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(t_train[-1] + horizon_s, color='#d62728', ls=':', lw=0.8)
    ax.set_ylabel('θ₂ (°)')
    ax.grid(alpha=0.3)
    ax.set_title('(b) 摆 2')

    ax = axes[2]
    ax.semilogy(t_pred, err_deg, color='#2ca02c', lw=1.0)
    ax.axhline(10, color='k', ls='--', lw=0.6, label='10° 阈值')
    ax.axvline(t_train[-1] + horizon_s, color='#d62728', ls=':', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('角度偏差 (°), log')
    ax.set_title('(c) 预测偏差')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    fn = FIG / f'pinn_predict_trial_{trial_idx:03d}.png'
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {fn}")


def run_one(net, trial_idx, train_sec=20.0, pred_sec=10.0, fps=120):
    """对单个 trial 做闭环预测 + 评估 + 画图。"""
    df = pd.read_csv(DATA / f'trial_{trial_idx:03d}.csv')
    t = df['t'].values

    n_train = int(train_sec * fps)
    n_pred  = int(pred_sec * fps)
    s0 = df[['th1', 'w1', 'th2', 'w2']].values[n_train].astype(np.float32)

    pred = predict_aligned(net, s0, n_pred, dt=1.0 / fps)
    pred_th = pred[:, [0, 2]]
    truth_th = df[['th1', 'th2']].values[n_train:n_train + n_pred]

    h_s, err_deg = horizon(pred_th, truth_th, threshold_deg=10.0, fps=fps)
    plot_predict(t, df, pred, n_train, n_pred, h_s, err_deg, trial_idx, fps)

    return {'trial': trial_idx, 'horizon': h_s,
            'mse_th1': float(np.mean((pred[:, 0] - truth_th[:, 0]) ** 2)),
            'mse_th2': float(np.mean((pred[:, 2] - truth_th[:, 1]) ** 2))}


# -------- 入口 --------

if __name__ == '__main__':
    print("ChaoticNet PINN 基线 v1.0")
    print(f"  设备     {DEVICE}")
    print(f"  数据源   {DATA.resolve()}")
    print(f"  输出     {OUT.resolve()}\n")

    # 1. 训练（用前 5 个 trial 的训练段）
    print("【训练】用 trial 0-4 的前 20s 数据，200 epoch")
    states, accs = load_dataset(trial_indices=range(5), train_sec=20.0)
    print(f"  样本数：{len(states)} (state, ω̇) 对")

    net = AccelNet(hidden=128, n_layers=4)
    hist = train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                 lambda_phys=1.0, n_collocation=2048)
    hist.to_csv(OUT / 'train_history.csv', index=False)
    torch.save(net.state_dict(), OUT / 'model.pt')   # 复用：动画 + sweep 加载

    # 2. 评估（同一批 trial 的预测段）
    print("\n【评估】对 trial 0-4 各 trial 做闭环预测 10s")
    results = [run_one(net, i, train_sec=20.0, pred_sec=10.0) for i in range(5)]
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT / 'pinn_summary.csv', index=False)

    print("\n  结果汇总：")
    print(df_res.to_string(index=False))
    print(f"\n  平均视界：{df_res['horizon'].mean():.2f}s "
          f"(范围 {df_res['horizon'].min():.2f}–{df_res['horizon'].max():.2f}s)")

    # 3. 训练 loss 曲线
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(hist['epoch'], hist['data_loss'], label='data loss', lw=1)
    ax.semilogy(hist['epoch'], hist['phys_loss'], label='physics loss', lw=1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (log)')
    ax.set_title('PINN 训练曲线')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIG / 'pinn_train_loss.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  训练曲线 → {FIG / 'pinn_train_loss.png'}")

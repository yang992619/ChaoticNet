"""
rc/esn.py — Echo State Network 基线 (ChaoticNet v1.0)

Reservoir Computing baseline，用稀疏随机循环网络预测双摆轨迹。

Pipeline:
  1. 读 data/sim/trial_XXX.csv
  2. 把 (θ₁, ω₁, θ₂, ω₂) 编码成 (sinθ, cosθ, ω) ×2 = 6 维输入
  3. 训练 ESN 自回归预测下一帧
  4. Closed-loop 预测：把自己输出回灌当输入，看多久跑歪
  5. 计算预测视界、MSE，画对比图

不依赖 PyTorch/JAX，纯 numpy。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 字体（macOS）
plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / 'data/sim'
OUT  = _ROOT / 'data/rc'
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = OUT / 'figures'
FIG_OUT.mkdir(parents=True, exist_ok=True)


# -------- I/O 编码 --------

def encode(df):
    """轨迹 → 6 维特征 (sinθ₁, cosθ₁, ω₁, sinθ₂, cosθ₂, ω₂)。

    用 sin/cos 解决角度 ±π 跳变问题。ω 归一化避免数值过大。
    """
    th1, w1, th2, w2 = df['th1'].values, df['w1'].values, df['th2'].values, df['w2'].values
    W_SCALE = 10.0  # ω 典型量级 ~10 rad/s
    return np.column_stack([
        np.sin(th1), np.cos(th1), w1 / W_SCALE,
        np.sin(th2), np.cos(th2), w2 / W_SCALE,
    ])


def decode(u):
    """6 维特征 → (θ₁, ω₁, θ₂, ω₂)。"""
    W_SCALE = 10.0
    th1 = np.arctan2(u[..., 0], u[..., 1])
    w1  = u[..., 2] * W_SCALE
    th2 = np.arctan2(u[..., 3], u[..., 4])
    w2  = u[..., 5] * W_SCALE
    return th1, w1, th2, w2


# -------- ESN 实现 --------

class ESN:
    """Echo State Network。

    动力学：x(t+1) = (1-α) x(t) + α tanh(W·x(t) + W_in·u(t))
    读出：  y(t)   = W_out · [x(t); u(t); 1]

    W_in / W 固定（不训），仅训 W_out（岭回归）。
    """

    def __init__(self, n_in, n_out, n_res=1000,
                 spectral_radius=0.9, input_scale=1.0,
                 leak=0.3, sparsity=0.1, ridge=1e-6,
                 seed=42):
        self.n_in, self.n_out, self.n_res = n_in, n_out, n_res
        self.leak, self.ridge = leak, ridge
        rng = np.random.default_rng(seed)

        # 输入权重：稠密随机
        self.W_in = rng.uniform(-input_scale, input_scale, (n_res, n_in))

        # 循环权重：稀疏随机 + 谱半径缩放
        W = rng.uniform(-1, 1, (n_res, n_res))
        mask = rng.random((n_res, n_res)) < sparsity
        W = W * mask
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / eig_max)

        # 读出（训练时填）
        self.W_out = None

    def step(self, x, u):
        """单步更新储池状态。"""
        pre = self.W @ x + self.W_in @ u
        return (1 - self.leak) * x + self.leak * np.tanh(pre)

    def collect_states(self, U, x0=None):
        """跑 reservoir，收集每一步的状态。U shape=(T, n_in)。"""
        T = len(U)
        X = np.zeros((T, self.n_res))
        x = np.zeros(self.n_res) if x0 is None else x0.copy()
        for t in range(T):
            x = self.step(x, U[t])
            X[t] = x
        return X

    def train(self, U, Y, washout=100):
        """岭回归训 W_out。

        Args:
            U: 输入序列 (T, n_in)
            Y: 目标序列 (T, n_out)
            washout: 前几步丢掉（reservoir 还没稳定）
        """
        X = self.collect_states(U)
        # 拼接 [reservoir state, input, bias 1] 作为读出特征
        H = np.column_stack([X, U, np.ones(len(X))])
        H_train = H[washout:]
        Y_train = Y[washout:]

        # Ridge regression：W_out = Y Hᵀ (H Hᵀ + λI)⁻¹
        I = np.eye(H_train.shape[1])
        self.W_out = np.linalg.solve(
            H_train.T @ H_train + self.ridge * I,
            H_train.T @ Y_train
        ).T  # shape (n_out, n_features)

        # 训练误差
        Y_pred = H_train @ self.W_out.T
        train_mse = np.mean((Y_train - Y_pred) ** 2)
        return train_mse, X[-1]

    def predict(self, x0, u0, n_steps):
        """Closed-loop 预测：用自己输出回灌。

        Args:
            x0: reservoir 状态初值（训练结束的状态）
            u0: 输入初值（训练数据最后一帧）
            n_steps: 预测多少步
        """
        out = np.zeros((n_steps, self.n_out))
        x, u = x0.copy(), u0.copy()
        for t in range(n_steps):
            x = self.step(x, u)
            h = np.concatenate([x, u, [1.0]])
            y = self.W_out @ h
            out[t] = y
            u = y  # 关键：把预测当下一步输入
        return out


# -------- 评估 --------

def prediction_horizon(pred, truth, threshold_deg=10.0, fps=120):
    """预测视界：偏差首次超 threshold 的时间（秒）。

    用角度欧氏距离 √[(θ₁_pred-θ₁_truth)² + (θ₂_pred-θ₂_truth)²]。
    """
    th1_p, _, th2_p, _ = decode(pred)
    th1_t, _, th2_t, _ = decode(truth)
    # 角度差用 atan2 处理周期
    d1 = np.arctan2(np.sin(th1_p - th1_t), np.cos(th1_p - th1_t))
    d2 = np.arctan2(np.sin(th2_p - th2_t), np.cos(th2_p - th2_t))
    err_deg = np.degrees(np.sqrt(d1**2 + d2**2))

    idx = np.argmax(err_deg > threshold_deg)
    if idx == 0 and err_deg[0] <= threshold_deg:
        return len(err_deg) / fps  # 一直没超阈值
    return idx / fps


# -------- 主流程 --------

def run_one(trial_idx=0, train_sec=20.0, pred_sec=10.0, fps=120, plot=True):
    """加载一个 trial，训练 ESN，做 closed-loop 预测，画图。"""
    df = pd.read_csv(DATA / f'trial_{trial_idx:03d}.csv')

    # 编码
    U = encode(df)               # shape (T, 6)
    Y = np.roll(U, -1, axis=0)   # 下一帧目标
    U, Y = U[:-1], Y[:-1]        # 去最后一行
    t  = df['t'].values[:-1]

    # 切训练 / 测试
    n_train = int(train_sec * fps)
    n_pred  = int(pred_sec * fps)
    U_tr, Y_tr = U[:n_train], Y[:n_train]
    U_te       = U[n_train:n_train+n_pred]  # ground truth for prediction

    # 训练
    esn = ESN(n_in=6, n_out=6, n_res=800,
              spectral_radius=0.95, leak=0.25, ridge=1e-5)
    train_mse, x_final = esn.train(U_tr, Y_tr, washout=200)

    # 闭环预测
    pred = esn.predict(x_final, U_tr[-1], n_pred)

    # 评估
    horizon = prediction_horizon(pred, U_te, threshold_deg=10.0, fps=fps)
    th1_p, _, th2_p, _ = decode(pred)
    th1_t, _, th2_t, _ = decode(U_te)
    mse_th1 = np.mean((th1_p - th1_t) ** 2)
    mse_th2 = np.mean((th2_p - th2_t) ** 2)

    print(f"  trial_{trial_idx:03d}: train_MSE={train_mse:.2e}, "
          f"horizon={horizon:.2f}s, "
          f"MSE θ₁={mse_th1:.3f} rad², θ₂={mse_th2:.3f} rad²")

    if plot:
        _plot_compare(t, df, n_train, n_pred, U_te, pred, horizon, trial_idx, fps)

    return {'trial': trial_idx, 'train_mse': train_mse, 'horizon': horizon,
            'mse_th1': mse_th1, 'mse_th2': mse_th2}


def _plot_compare(t, df, n_train, n_pred, truth, pred, horizon, trial_idx, fps):
    """画对比图：训练段 + 预测段 + 偏差。"""
    t_train = t[:n_train]
    t_pred  = t[n_train:n_train + n_pred]
    th1_p, _, th2_p, _ = decode(pred)
    th1_t, _, th2_t, _ = decode(truth)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f'ESN 闭环预测 — trial_{trial_idx:03d}  '
                 f'(N_res=800, 训练 {len(t_train)/fps:.0f}s, '
                 f'预测视界 {horizon:.2f}s @ 10°阈值)', fontsize=13)

    # (a) θ₁
    ax = axes[0]
    ax.plot(t_train, np.degrees(df['th1'].values[:n_train]),
            color='#888', lw=0.7, label='训练数据')
    ax.plot(t_pred, np.degrees(th1_t), color='#1f77b4', lw=1.0, label='真值 (仿真)')
    ax.plot(t_pred, np.degrees(th1_p), color='#d62728', lw=1.0, label='ESN 预测', alpha=0.85)
    ax.axvline(t_train[-1], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(t_train[-1] + horizon, color='#d62728', ls=':', lw=0.8, alpha=0.7,
               label=f'视界 +{horizon:.2f}s')
    ax.set_ylabel('θ₁ (°)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title('(a) 摆 1')

    # (b) θ₂
    ax = axes[1]
    ax.plot(t_train, np.degrees(df['th2'].values[:n_train]),
            color='#888', lw=0.7)
    ax.plot(t_pred, np.degrees(th2_t), color='#1f77b4', lw=1.0)
    ax.plot(t_pred, np.degrees(th2_p), color='#d62728', lw=1.0, alpha=0.85)
    ax.axvline(t_train[-1], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(t_train[-1] + horizon, color='#d62728', ls=':', lw=0.8, alpha=0.7)
    ax.set_ylabel('θ₂ (°)')
    ax.grid(alpha=0.3)
    ax.set_title('(b) 摆 2')

    # (c) 误差 log
    ax = axes[2]
    d1 = np.arctan2(np.sin(th1_p - th1_t), np.cos(th1_p - th1_t))
    d2 = np.arctan2(np.sin(th2_p - th2_t), np.cos(th2_p - th2_t))
    err_deg = np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))
    ax.semilogy(t_pred, err_deg, color='#2ca02c', lw=1.0)
    ax.axhline(10, color='k', ls='--', lw=0.6, label='10° 阈值')
    ax.axvline(t_train[-1] + horizon, color='#d62728', ls=':', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('角度偏差 (°), log')
    ax.set_title('(c) 预测偏差')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    fn = FIG_OUT / f'esn_predict_trial_{trial_idx:03d}.png'
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'    ✓ {fn}')


def sweep(trials=range(5), out_csv='esn_summary.csv'):
    """跑前 N 个 trial 评估 horizon 分布。"""
    results = [run_one(i, plot=True) for i in trials]
    df = pd.DataFrame(results)
    df.to_csv(OUT / out_csv, index=False)

    print(f"\n[sweep] 完成 {len(results)} 组")
    print(f"  预测视界：{df['horizon'].mean():.2f} ± {df['horizon'].std():.2f}s "
          f"(范围 {df['horizon'].min():.2f}–{df['horizon'].max():.2f}s)")
    return df


if __name__ == '__main__':
    print("ChaoticNet Echo State Network 基线 v1.0")
    print(f"  数据源 {DATA.resolve()}")
    print(f"  输出   {OUT.resolve()}\n")

    print("【预热】单次跑通 trial_000")
    run_one(0, train_sec=20.0, pred_sec=10.0)

    print("\n【sweep】前 5 个 trial 评估视界分布")
    summary = sweep(trials=range(5))
    print(f"\n详细结果：{OUT / 'esn_summary.csv'}")

"""
rc/real_validation.py — 实拍真实数据验证（单片详图，ChaoticNet）

⚠ 非论文头条来源：本脚本只跑【单段】(tracked_1392.csv) 做详细对比图，输出的
data/real_validation/summary.csv 是【单片旧口径】(ESN 0.308/PINN 0.283/Hybrid 0.283)，
不是真实零样本头条。真实零样本头条 = 29 段多 run 中位，见 data/canon_multirun/ 与
data/canonical_results_B.md（多段管线在 rc/real_validation_multi.py）。本脚本的叠加图仍可用。

把视觉追踪得到的真实双摆角度序列 (data/tracking/tracked_1392.csv) 喂给
三个模型，在同一测试窗口内公平对比闭环预测视界：

  - ESN     ：纯数据驱动。在引入段(lead-in)上自训 W_out + 暖机储池，再闭环预测。
  - PINN    ：加载仿真(无阻尼)训练好的 model.pt，从实测初值 RK4 前推 —— 检验 sim→real 迁移。
  - Hybrid  ：解析先验 + 小残差 NN，同样加载 model.pt 从实测初值前推。

预处理：对 θ 做轻度 Savitzky-Golay 平滑后求 ω，降低追踪像素抖动、给出干净初值。
角度约定与 sim/baseline.py 完全一致（竖直向下=0，θ1/θ2 均为对竖直的绝对角）。

输出：data/real_validation/ 下的对比图 + summary.csv。
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'rc'))

import esn as esn_mod        # noqa: E402  ESN, encode, decode
import pinn as pinn_mod      # noqa: E402  AccelNet, predict, DEVICE
import hybrid as hybrid_mod  # noqa: E402  HybridNet

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

TRACKED = _ROOT / 'data/tracking/tracked_1392.csv'
OUT = _ROOT / 'data/real_validation'
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# ---- 配置 ----
I0 = 350            # 引入段末帧（之后为预测对比窗口）
SAVGOL_WIN = 11     # θ 平滑窗口（奇数）
SAVGOL_POLY = 3
THRESH_DEG = 10.0   # 预测视界阈值


# -------- 数据预处理 --------

def load_real():
    """读实测 csv，平滑 θ 后求 ω，返回 dict。"""
    df = pd.read_csv(TRACKED)
    t = df['t'].values
    fps = 1.0 / np.median(np.diff(t))
    dt = 1.0 / fps

    th1 = np.unwrap(df['th1'].values)
    th2 = np.unwrap(df['th2'].values)
    th1_s = savgol_filter(th1, SAVGOL_WIN, SAVGOL_POLY)
    th2_s = savgol_filter(th2, SAVGOL_WIN, SAVGOL_POLY)
    w1_s = np.gradient(th1_s, dt)
    w2_s = np.gradient(th2_s, dt)

    return {
        't': t, 'fps': fps, 'dt': dt,
        'th1': th1_s, 'th2': th2_s, 'w1': w1_s, 'w2': w2_s,
        'df_state': np.column_stack([th1_s, w1_s, th2_s, w2_s]),
        'N': len(t),
    }


def angle_err_deg(pred_th1, pred_th2, true_th1, true_th2):
    """两摆角度欧氏偏差（度），用 atan2 处理周期。"""
    d1 = np.arctan2(np.sin(pred_th1 - true_th1), np.cos(pred_th1 - true_th1))
    d2 = np.arctan2(np.sin(pred_th2 - true_th2), np.cos(pred_th2 - true_th2))
    return np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))


def horizon_s(err_deg, fps, thresh=THRESH_DEG):
    idx = np.argmax(err_deg > thresh)
    if idx == 0 and err_deg[0] <= thresh:
        return len(err_deg) / fps
    return idx / fps


# -------- ESN（数据驱动，自训于实测引入段） --------

def run_esn(R, i0, n_pred, n_res=300, spectral_radius=0.95,
            leak=0.3, ridge=1e-3, washout=50, seed=42):
    """在实测 [0,i0) 上训 W_out，暖机后闭环预测 n_pred 步。"""
    df = pd.DataFrame({'th1': R['th1'], 'w1': R['w1'],
                       'th2': R['th2'], 'w2': R['w2']})
    U = esn_mod.encode(df)              # (N,6)
    Y = np.roll(U, -1, axis=0)
    U_tr, Y_tr = U[:i0 - 1], Y[:i0 - 1]   # 训练对（去最后一行）

    net = esn_mod.ESN(n_in=6, n_out=6, n_res=n_res,
                      spectral_radius=spectral_radius, leak=leak,
                      ridge=ridge, seed=seed)
    train_mse, x_final = net.train(U_tr, Y_tr, washout=washout)

    # 从引入段末状态闭环预测
    pred = net.predict(x_final, U[i0 - 1], n_pred)   # (n_pred,6)
    th1_p, _, th2_p, _ = esn_mod.decode(pred)
    return th1_p, th2_p, train_mse


# -------- PINN / Hybrid（sim 训练，迁移到实测初值） --------

def run_torch_model(net, s0, n_pred, dt):
    pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=dt)   # (n_pred,4) [th1,w1,th2,w2]
    return pred[:, 0], pred[:, 2]


def load_pinn():
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    net.load_state_dict(torch.load(_ROOT / 'data/pinn/model.pt',
                                   map_location=pinn_mod.DEVICE))
    net.to(pinn_mod.DEVICE).eval()
    return net


def load_hybrid():
    net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                               residual_scale=hybrid_mod.RESIDUAL_SCALE)
    net.load_state_dict(torch.load(_ROOT / 'data/hybrid/model.pt',
                                   map_location=hybrid_mod.DEVICE))
    net.to(hybrid_mod.DEVICE).eval()
    return net


# -------- 主流程 --------

def main():
    R = load_real()
    fps, dt, N = R['fps'], R['dt'], R['N']
    i0 = I0
    n_pred = N - i0
    t = R['t']
    t_pred = t[i0:i0 + n_pred]

    print(f"实测序列 {N} 帧 @ {fps:.2f}fps，引入段 0..{i0}({i0/fps:.2f}s)，"
          f"预测窗 {n_pred} 帧({n_pred/fps:.2f}s)")

    # 真值（平滑后）
    true_th1 = R['th1'][i0:i0 + n_pred]
    true_th2 = R['th2'][i0:i0 + n_pred]

    # 实测初值（给 PINN/Hybrid）
    s0 = R['df_state'][i0].astype(np.float32)
    print(f"实测初值 θ1={np.degrees(s0[0]):+.1f}° ω1={s0[1]:+.2f} "
          f"θ2={np.degrees(s0[2]):+.1f}° ω2={s0[3]:+.2f}")

    results = {}

    # ESN
    print("\n[ESN] 数据驱动，自训于实测引入段 ...")
    esn_th1, esn_th2, esn_mse = run_esn(R, i0, n_pred)
    e = angle_err_deg(esn_th1, esn_th2, true_th1, true_th2)
    results['ESN'] = {'th1': esn_th1, 'th2': esn_th2, 'err': e,
                      'horizon': horizon_s(e, fps), 'color': '#d62728'}
    print(f"      训练MSE={esn_mse:.2e}  视界={results['ESN']['horizon']:.3f}s")

    # PINN
    print("[PINN] 加载 sim 训练权重，从实测初值前推 ...")
    pinn_net = load_pinn()
    p_th1, p_th2 = run_torch_model(pinn_net, s0, n_pred, dt)
    e = angle_err_deg(p_th1, p_th2, true_th1, true_th2)
    results['PINN'] = {'th1': p_th1, 'th2': p_th2, 'err': e,
                       'horizon': horizon_s(e, fps), 'color': '#2ca02c'}
    print(f"      视界={results['PINN']['horizon']:.3f}s")

    # Hybrid
    print("[Hybrid] 加载 sim 训练权重，从实测初值前推 ...")
    hyb_net = load_hybrid()
    h_th1, h_th2 = run_torch_model(hyb_net, s0, n_pred, dt)
    e = angle_err_deg(h_th1, h_th2, true_th1, true_th2)
    results['Hybrid'] = {'th1': h_th1, 'th2': h_th2, 'err': e,
                         'horizon': horizon_s(e, fps), 'color': '#9467bd'}
    print(f"      视界={results['Hybrid']['horizon']:.3f}s")

    # ---- 汇总表 ----
    rows = []
    for name in ['ESN', 'PINN', 'Hybrid']:
        r = results[name]
        rms_deg = float(np.sqrt(np.mean(r['err'] ** 2)))  # 全窗口角度偏差 RMS（度）
        rows.append({
            'model': name,
            'horizon_s': round(r['horizon'], 3),
            'rms_err_deg': round(rms_deg, 2),
            'mse_th1_rad2': round(float(np.mean((r['th1'] - true_th1) ** 2)), 4),
            'mse_th2_rad2': round(float(np.mean((r['th2'] - true_th2) ** 2)), 4),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    print("\n=== 实测验证汇总 ===")
    print(summary.to_string(index=False))

    # ---- 绘图 ----
    _plot(t, R, i0, n_pred, t_pred, true_th1, true_th2, results, fps)
    _plot_zoom(t, R, i0, n_pred, t_pred, true_th1, true_th2, results, fps)
    return summary


def _plot(t, R, i0, n_pred, t_pred, true_th1, true_th2, results, fps):
    t_lead = t[:i0]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f'实拍双摆真实数据验证 (IMG_1392, {R["N"]}帧@{fps:.1f}fps) — '
                 f'三模型同窗口闭环预测对比', fontsize=13)

    # (a) θ1
    ax = axes[0]
    ax.plot(t_lead, np.degrees(R['th1'][:i0]), color='#888', lw=0.7, label='实测引入段')
    ax.plot(t_pred, np.degrees(true_th1), color='#1f77b4', lw=1.4, label='实测真值')
    for name in ['ESN', 'PINN', 'Hybrid']:
        r = results[name]
        ax.plot(t_pred, np.degrees(r['th1']), color=r['color'], lw=1.0, alpha=0.85,
                label=f'{name} (视界 {r["horizon"]:.2f}s)')
    ax.axvline(t[i0], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.set_ylabel('θ1 (°)'); ax.set_title('(a) 上摆'); ax.grid(alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=8)

    # (b) θ2
    ax = axes[1]
    ax.plot(t_lead, np.degrees(R['th2'][:i0]), color='#888', lw=0.7)
    ax.plot(t_pred, np.degrees(true_th2), color='#1f77b4', lw=1.4)
    for name in ['ESN', 'PINN', 'Hybrid']:
        r = results[name]
        ax.plot(t_pred, np.degrees(r['th2']), color=r['color'], lw=1.0, alpha=0.85)
    ax.axvline(t[i0], color='k', ls='--', lw=0.6, alpha=0.5)
    ax.set_ylabel('θ2 (°)'); ax.set_title('(b) 下摆'); ax.grid(alpha=0.3)

    # (c) 角度偏差 log
    ax = axes[2]
    for name in ['ESN', 'PINN', 'Hybrid']:
        r = results[name]
        ax.semilogy(t_pred, np.maximum(r['err'], 1e-2), color=r['color'], lw=1.0,
                    label=name)
    ax.axhline(THRESH_DEG, color='k', ls='--', lw=0.6, label=f'{THRESH_DEG:.0f}° 阈值')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('角度偏差 (°), log')
    ax.set_title('(c) 预测偏差'); ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    fn = FIG / 'real_validation_1392.png'
    plt.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n图 → {fn}")


def _plot_zoom(t, R, i0, n_pred, t_pred, true_th1, true_th2, results, fps):
    """放大预测窗口（含少量引入段尾巴），看清各模型贴合度。"""
    lead = int(0.4 * fps)
    j = max(0, i0 - lead)
    t_tail = t[j:i0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('预测窗口放大 — 物理模型(PINN/Hybrid)对真实双摆的 sim→real 迁移', fontsize=12)
    for ax, key, true_v, ttl in [
        (axes[0], 'th1', true_th1, '(a) 上摆 θ1'),
        (axes[1], 'th2', true_th2, '(b) 下摆 θ2'),
    ]:
        ax.plot(t_tail, np.degrees(R[key][j:i0]), color='#888', lw=1.0, label='实测引入段')
        ax.plot(t_pred, np.degrees(true_v), color='#1f77b4', lw=2.2, label='实测真值', zorder=5)
        for name in ['ESN', 'PINN', 'Hybrid']:
            r = results[name]
            ax.plot(t_pred, np.degrees(r[key]), color=r['color'], lw=1.3, alpha=0.9,
                    label=f'{name}')
        ax.axvline(t[i0], color='k', ls='--', lw=0.6, alpha=0.5)
        ax.set_ylabel(f'{key} (°)'); ax.set_title(ttl); ax.grid(alpha=0.3)
    axes[0].legend(loc='best', ncol=2, fontsize=9)
    axes[1].set_xlabel('时间 (s)')
    plt.tight_layout()
    fn = FIG / 'real_validation_1392_zoom.png'
    plt.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"图(放大) → {fn}")


if __name__ == '__main__':
    main()

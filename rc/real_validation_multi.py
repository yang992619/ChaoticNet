"""
rc/real_validation_multi.py — 多条实拍真实数据的 sim→real 统计验证

把视觉追踪得到的多条真实双摆轨迹批量喂三模型，按秒切窗（帧率无关），
统计各模型在多条独立轨迹上的有效预测视界与角度 RMS 分布。
单条(IMG_1392)详图版见 real_validation.py，本脚本不改它，做群体统计。

每条片：
  引入段 lead_s 秒 —— ESN 在此段自训 W_out（数据驱动）；末点状态作 PINN/Hybrid 初值
  预测窗 pred_s 秒 —— 三模型闭环/前推，与实测真值比
角度约定与 sim/baseline.py 一致（竖直向下=0），θ 先轻度 Savitzky-Golay 平滑再求 ω。
视界 = 两摆角度欧氏偏差首次超 THRESH_DEG 的时间。

注意：PINN/Hybrid 用的是无阻尼 sim 训练权重，真实摆有阻尼，故长程会系统性偏离——
这正是后续上「带阻尼模型」的动机。本脚本给出无阻尼基线的多片统计。
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'rc'))
import esn as esn_mod        # noqa: E402
import pinn as pinn_mod      # noqa: E402
import hybrid as hybrid_mod  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 逐帧全量追踪结果（data/tracking/，约 71 MB）不随仓库分发；随仓库发布的是
# 每段起始 600 帧的分析窗（data/tracking_windows/，2.95 MB）。分析窗完整覆盖
# 本文所用的 4 s 引入窗 + 4 s 预测窗（478 帧），在其上重跑与全量结果一致
# （角度逐位相同，角速度最大差 6.6e-12 rad/s；见 tools/make_tracking_windows.py）。
# 因此全量目录缺失时自动回退到分析窗，保证 fresh clone 可直接复现。
_FULL = _ROOT / 'data/tracking'
_WIN = _ROOT / 'data/tracking_windows'
if list(_FULL.glob('tracked_14*.csv')):
    TRACK = _FULL
elif list(_WIN.glob('tracked_14*.csv')):
    TRACK = _WIN
    print(f'[real_validation_multi] 未找到逐帧全量结果，改用分析窗数据集 {_WIN.name}/')
else:
    raise SystemExit(
        f'找不到追踪数据：{_FULL} 与 {_WIN} 都没有 tracked_14*.csv。\n'
        '随仓库发布的分析窗应位于 data/tracking_windows/；'
        '若要从源视频重建逐帧全量结果，见 tracking/reproduce_29.sh。')
OUT = _ROOT / 'data/real_validation'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SAVGOL_WIN = 11
SAVGOL_POLY = 3
THRESH_DEG = 10.0

DEFAULT_TAGS = ['1392', '1430', '1431', '1432', '1434', '1435', '1436', '1437',
                '1438', '1439', '1440', '1441', '1442', '1443', '1444', '1445',
                '1446', '1447', '1448', '1449', '1450', '1451', '1452', '1453',
                '1454', '1455', '1456', '1458', '1459', '1460']


# -------- 数据 --------

def load_clip(tag, lead_s, pred_s):
    df = pd.read_csv(TRACK / f'tracked_{tag}.csv')
    t = df['t'].values
    t = t - t[0]
    fps = 1.0 / np.median(np.diff(t))
    dt = 1.0 / fps
    N = len(t)
    th1 = np.unwrap(df['th1'].values)
    th2 = np.unwrap(df['th2'].values)
    win = SAVGOL_WIN if N > SAVGOL_WIN else (N // 2) * 2 - 1
    th1s = savgol_filter(th1, win, SAVGOL_POLY)
    th2s = savgol_filter(th2, win, SAVGOL_POLY)
    w1 = np.gradient(th1s, dt)
    w2 = np.gradient(th2s, dt)
    lead_n = min(int(lead_s * fps), N // 2)
    pred_n = min(int(pred_s * fps), N - lead_n - 1)
    state = np.column_stack([th1s, w1, th2s, w2])
    rel_deg = float(np.degrees(np.max(np.abs(th1s[:max(lead_n + pred_n, 1)]))))
    return dict(tag=tag, t=t, fps=fps, dt=dt, N=N, th1=th1s, th2=th2s,
                w1=w1, w2=w2, state=state, lead_n=lead_n, pred_n=pred_n,
                rel_deg=rel_deg)


def angle_err_deg(p1, p2, t1, t2):
    d1 = np.arctan2(np.sin(p1 - t1), np.cos(p1 - t1))
    d2 = np.arctan2(np.sin(p2 - t2), np.cos(p2 - t2))
    return np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))


def horizon_s(err, fps, thr=THRESH_DEG):
    """返回 (视界秒, 是否右删失)。整窗都没超阈 → 删失=True，值为窗长(下界)。"""
    over = err > thr
    if not over.any():
        return len(err) / fps, True
    return int(np.argmax(over)) / fps, False


# -------- 模型 --------

def run_esn(C, n_res=300, sr=0.95, leak=0.3, ridge=1e-3, washout=50, seed=42):
    i0, n = C['lead_n'], C['pred_n']
    df = pd.DataFrame({'th1': C['th1'], 'w1': C['w1'],
                       'th2': C['th2'], 'w2': C['w2']})
    U = esn_mod.encode(df)
    Y = np.roll(U, -1, axis=0)
    net = esn_mod.ESN(n_in=6, n_out=6, n_res=n_res, spectral_radius=sr,
                      leak=leak, ridge=ridge, seed=seed)
    _, xf = net.train(U[:i0 - 1], Y[:i0 - 1], washout=min(washout, max(i0 // 3, 1)))
    pred = net.predict(xf, U[i0 - 1], n)
    t1, _, t2, _ = esn_mod.decode(pred)
    return t1, t2


def run_torch(net, s0, n, dt):
    # 与 ESN 对齐：预测窗第 0 帧取实测初值 s0(=真值 i0)，其后接 n-1 步积分。
    # 该口径已上收为 pinn_mod.predict_aligned（唯一实现，仿真侧与真实侧共用）；
    # 此处原有本地写法与之逐位等价，2026-09-04 重构为直接调用。
    seq = pinn_mod.predict_aligned(net, s0, n, dt=dt)
    th1 = seq[:, 0]
    th2 = seq[:, 2]
    return th1, th2


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


def process(C, pinn_net, hyb_net):
    i0, n, fps, dt = C['lead_n'], C['pred_n'], C['fps'], C['dt']
    tt1 = C['th1'][i0:i0 + n]
    tt2 = C['th2'][i0:i0 + n]
    s0 = C['state'][i0].astype(np.float32)
    out = {}
    runners = [('ESN', lambda: run_esn(C)),
               ('PINN', lambda: run_torch(pinn_net, s0, n, dt)),
               ('Hybrid', lambda: run_torch(hyb_net, s0, n, dt))]
    for name, fn in runners:
        p1, p2 = fn()
        e = angle_err_deg(p1, p2, tt1, tt2)
        hz, cens = horizon_s(e, fps)
        out[name] = dict(th1=p1, th2=p2, err=e, horizon=hz, censored=cens,
                         rms=float(np.sqrt(np.mean(e ** 2))))
    return out


# -------- 批量 + 出图 --------

def batch(tags, lead_s, pred_s):
    pinn_net, hyb_net = load_pinn(), load_hybrid()
    rows, store = [], {}
    for tg in tags:
        f = TRACK / f'tracked_{tg}.csv'
        if not f.exists():
            print(f'  跳过 {tg}(缺)')
            continue
        C = load_clip(tg, lead_s, pred_s)
        if C['pred_n'] < 10:
            print(f"  跳过 {tg}(片太短 pred_n={C['pred_n']})")
            continue
        r = process(C, pinn_net, hyb_net)
        store[tg] = (C, r)
        rows.append(dict(
            tag=tg, rel_deg=round(C['rel_deg'], 1), fps=round(C['fps'], 1),
            lead_s=round(C['lead_n'] / C['fps'], 2),
            pred_s=round(C['pred_n'] / C['fps'], 2),
            ESN_h=round(r['ESN']['horizon'], 3),
            PINN_h=round(r['PINN']['horizon'], 3),
            Hybrid_h=round(r['Hybrid']['horizon'], 3),
            ESN_cens=int(r['ESN']['censored']),
            PINN_cens=int(r['PINN']['censored']),
            Hybrid_cens=int(r['Hybrid']['censored']),
            ESN_rms=round(r['ESN']['rms'], 1),
            PINN_rms=round(r['PINN']['rms'], 1),
            Hybrid_rms=round(r['Hybrid']['rms'], 1)))
        print(f"  {tg} rel={C['rel_deg']:6.1f}° pred={C['pred_n']/C['fps']:.1f}s  "
              f"视界 ESN={r['ESN']['horizon']:.2f} PINN={r['PINN']['horizon']:.2f} "
              f"Hyb={r['Hybrid']['horizon']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'summary_multi.csv', index=False)
    print('\n=== 多片汇总 ===')
    print(df.to_string(index=False))
    print('\n=== 统计（%d 条）===' % len(df))
    for m in ['ESN', 'PINN', 'Hybrid']:
        h = df[f'{m}_h']
        cens = df[f'{m}_cens'].astype(bool)
        hd = h[~cens]  # 仅真发散（未删失）
        med_div = hd.median() if len(hd) else float('nan')
        print(f"  {m:>6}: 视界 中位={h.median():.2f}s 均值={h.mean():.2f}s "
              f"[{h.min():.2f},{h.max():.2f}]   RMS中位={df[f'{m}_rms'].median():.1f}°"
              f"  | 真发散{(~cens).sum()}条 中位={med_div:.2f}s 删失{cens.sum()}条")
    if len(df):
        plot_scatter(df)
        plot_box(df)
        plot_reps(store)
    return df


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for m, c, mk in [('ESN', '#d62728', 'o'), ('PINN', '#2ca02c', 's'),
                     ('Hybrid', '#9467bd', '^')]:
        cens = df[f'{m}_cens'].values.astype(bool)
        # 真发散：实心
        ax.scatter(df['rel_deg'][~cens], df[f'{m}_h'][~cens], c=c, marker=mk,
                   s=48, alpha=0.8, label=m, edgecolor='k', lw=0.4)
        # 右删失（整窗未达10°，视界为下界）：空心
        ax.scatter(df['rel_deg'][cens], df[f'{m}_h'][cens], facecolors='none',
                   edgecolors=c, marker=mk, s=48, lw=1.3)
    ax.set_xlabel('释放角 |θ1| 峰值 (°)')
    ax.set_ylabel('有效预测视界 (s, 角度偏差<10°)')
    ax.set_title('sim→real 迁移：预测视界 vs 释放角（%d 条真实轨迹）' % len(df))
    ax.text(0.98, 0.02, '空心=整窗未发散(视界下界)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='dimgray')
    ax.grid(alpha=0.3)
    ax.legend()
    fn = FIG / 'multi_horizon_vs_angle.png'
    plt.tight_layout()
    plt.savefig(fn, dpi=130)
    plt.close()
    print('图 →', fn)


def plot_box(df):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    data = [df['ESN_h'], df['PINN_h'], df['Hybrid_h']]
    bp = ax.boxplot(data, tick_labels=['ESN', 'PINN', 'Hybrid'],
                    patch_artist=True, showmeans=True)
    for p, c in zip(bp['boxes'], ['#d62728', '#2ca02c', '#9467bd']):
        p.set_facecolor(c)
        p.set_alpha(0.5)
    ax.set_ylabel('有效预测视界 (s)')
    ax.set_title('各模型视界分布（%d 条真实轨迹）' % len(df))
    ax.grid(alpha=0.3, axis='y')
    fn = FIG / 'multi_horizon_box.png'
    plt.tight_layout()
    plt.savefig(fn, dpi=130)
    plt.close()
    print('图 →', fn)


def plot_reps(store, want=('1453', '1460', '1447', '1455')):
    reps = [t for t in want if t in store]
    if not reps:
        return
    n = len(reps)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), squeeze=False)
    axes = axes[:, 0]
    for ax, tg in zip(axes, reps):
        C, r = store[tg]
        i0, npd, fps = C['lead_n'], C['pred_n'], C['fps']
        tp = C['t'][i0:i0 + npd] - C['t'][i0]
        ax.plot(tp, np.degrees(C['th2'][i0:i0 + npd]), 'b-', lw=2,
                label='实测真值', zorder=5)
        for m, c in [('ESN', '#d62728'), ('PINN', '#2ca02c'),
                     ('Hybrid', '#9467bd')]:
            ax.plot(tp, np.degrees(r[m]['th2']), color=c, lw=1.1, alpha=0.85,
                    label=f"{m}(视界{r[m]['horizon']:.2f}s)")
        ax.set_title(f"片{tg} 释放角{C['rel_deg']:.0f}° — 下摆 θ2 预测", fontsize=10)
        ax.set_ylabel('θ2 (°)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='best')
    axes[-1].set_xlabel('预测时间 (s)')
    fn = FIG / 'multi_reps_theta2.png'
    plt.tight_layout()
    plt.savefig(fn, dpi=130)
    plt.close()
    print('图 →', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=DEFAULT_TAGS)
    ap.add_argument('--lead', type=float, default=4.0)
    ap.add_argument('--pred', type=float, default=4.0)
    a = ap.parse_args()
    print(f'多片 sim→real 验证  lead={a.lead}s pred={a.pred}s  {len(a.tags)} 条')
    batch(a.tags, a.lead, a.pred)

"""
sim/visualize.py — 双摆仿真可视化 (ChaoticNet v1.0)

读 data/sim/ 的 CSV，画四张图：
  1. fig_single_trial.png  单次仿真四联图（时间序列 + xy 轨迹 + 相空间 + 能量）
  2. fig_butterfly.png      初值微扰对比（蝴蝶效应可视化）
  3. fig_energy_sweep.png   50 组的能量漂移分布
  4. fig_phase_grid.png     不同初值的相空间叠加

依赖 baseline.py 已经跑过，data/sim/ 下有 trial_*.csv 和 summary.csv。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import baseline as bl  # 复用 simulate() 跑微扰对比

# 中文字体（macOS 用 Songti SC，没有的话退到 Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

DATA = Path(__file__).resolve().parents[1] / 'data/sim'
OUT  = DATA / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def xy_from_angles(th1, th2, L1=bl.L1, L2=bl.L2):
    """把角度转换成摆 2 末端 (x, y) 坐标。悬挂点为原点，y 朝下为正。"""
    x1 = L1 * np.sin(th1)
    y1 = L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 + L2 * np.cos(th2)
    return x1, y1, x2, y2


def fig_single_trial(trial_idx=0):
    """单次仿真四联图。"""
    df = pd.read_csv(DATA / f'trial_{trial_idx:03d}.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'单次仿真展示 — trial_{trial_idx:03d}  '
                 f'(θ₁(0)={np.degrees(df["th1"].iloc[0]):+.1f}°, '
                 f'θ₂(0)={np.degrees(df["th2"].iloc[0]):+.1f}°)',
                 fontsize=14)

    # (a) 角度时间序列
    ax = axes[0, 0]
    ax.plot(df['t'], np.degrees(df['th1']), label='θ₁ (摆 1)', lw=1.0, color='#1f77b4')
    ax.plot(df['t'], np.degrees(df['th2']), label='θ₂ (摆 2)', lw=1.0, color='#d62728')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('角度 (°)')
    ax.set_title('(a) 角度时间序列')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # (b) 摆 2 末端 xy 轨迹
    ax = axes[0, 1]
    _, _, x2, y2 = xy_from_angles(df['th1'].values, df['th2'].values)
    # 上色按时间渐变
    pts = ax.scatter(x2 * 1000, y2 * 1000, c=df['t'], s=1, cmap='viridis')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('(b) 摆 2 末端 xy 轨迹（颜色 = 时间）')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # y 朝下为正
    ax.grid(alpha=0.3)
    plt.colorbar(pts, ax=ax, label='t (s)')

    # (c) 相空间 θ₂ vs ω₂
    ax = axes[1, 0]
    ax.plot(np.degrees(df['th2']), df['w2'], lw=0.4, color='#9467bd')
    ax.set_xlabel('θ₂ (°)')
    ax.set_ylabel('ω₂ (rad/s)')
    ax.set_title('(c) 相空间投影 θ₂-ω₂')
    ax.grid(alpha=0.3)

    # (d) 能量守恒
    ax = axes[1, 1]
    E0 = df['energy'].iloc[0]
    drift = (df['energy'] - E0) / abs(E0)
    ax.plot(df['t'], drift, lw=0.8, color='#2ca02c')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('(E - E₀) / |E₀|')
    ax.set_title(f'(d) 能量相对漂移（峰值 {drift.abs().max():.2e}）')
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    fn = OUT / f'fig_single_trial_{trial_idx:03d}.png'
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fn}')


def fig_butterfly(th1_0_deg=60, th2_0_deg=30, t_end=15.0, perturb=0.1):
    """初值微扰对比图：两条只差 perturb° 的轨迹，看多久分叉。"""
    th1_0 = np.radians(th1_0_deg)
    th2_0 = np.radians(th2_0_deg)
    th2_0_p = np.radians(th2_0_deg + perturb)

    df_a = bl.simulate(th1_0, th2_0,   t_end=t_end, fps=120)
    df_b = bl.simulate(th1_0, th2_0_p, t_end=t_end, fps=120)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'蝴蝶效应  —  初值差 {perturb}° 的两条轨迹  '
                 f'(θ₁(0)={th1_0_deg}°, θ₂(0)={th2_0_deg}° vs {th2_0_deg+perturb}°)',
                 fontsize=13)

    # (a) θ₁ 对比
    ax = axes[0]
    ax.plot(df_a['t'], np.degrees(df_a['th1']), lw=1.0, color='#1f77b4', label='基准轨迹')
    ax.plot(df_b['t'], np.degrees(df_b['th1']), lw=1.0, color='#d62728', label=f'扰动 +{perturb}°', alpha=0.8)
    ax.set_ylabel('θ₁ (°)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('(a) 摆 1 角度对比')

    # (b) 偏差 log scale
    ax = axes[1]
    th1_diff = np.abs(df_a['th1'].values - df_b['th1'].values)
    th2_diff = np.abs(df_a['th2'].values - df_b['th2'].values)
    ax.semilogy(df_a['t'], np.degrees(th1_diff), lw=1.0, color='#1f77b4', label='|Δθ₁|')
    ax.semilogy(df_a['t'], np.degrees(th2_diff), lw=1.0, color='#d62728', label='|Δθ₂|')
    ax.axhline(perturb, color='k', ls='--', lw=0.6, alpha=0.5, label=f'初值差 {perturb}°')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('|偏差| (°)，log 尺度')
    ax.set_title('(b) 偏差指数发散 — 混沌特征')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    fn = OUT / 'fig_butterfly.png'
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fn}')


def fig_energy_sweep():
    """50 组仿真的能量漂移分布。"""
    summary = pd.read_csv(DATA / 'summary.csv')

    # 两张子图改为纵向排布；A4 版面中每张都能以大字号印刷。
    fig, axes = plt.subplots(2, 1, figsize=(12, 10.5), dpi=240)
    fig.suptitle('50 组仿真能量守恒检验（RK45，rtol=1e-10）', fontsize=26, fontweight='bold')

    # (a) 直方图
    ax = axes[0]
    ax.hist(np.log10(summary['energy_drift']), bins=20, color='#2ca02c', edgecolor='k', alpha=0.7)
    ax.set_xlabel('log10（相对能量漂移）', fontsize=20)
    ax.set_ylabel('试验组数', fontsize=20)
    ax.set_title(f'(a) 能量漂移分布（中位数 {np.median(summary["energy_drift"]):.1e}）', fontsize=22, pad=12)
    ax.grid(alpha=0.3)

    # (b) 与初值能量的散点
    ax = axes[1]
    sc = ax.scatter(summary['th1_0_deg'], summary['th2_0_deg'],
                    c=np.log10(summary['energy_drift']),
                    s=105, cmap='plasma', edgecolor='k', linewidth=0.6)
    ax.set_xlabel('θ1(0)（°）', fontsize=20)
    ax.set_ylabel('θ2(0)（°）', fontsize=20)
    ax.set_title('(b) 初值与漂移（颜色 = log10 漂移）', fontsize=22, pad=12)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log10（漂移）', fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    for ax in axes:
        ax.tick_params(labelsize=17)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fn = OUT / 'fig_energy_sweep.png'
    plt.savefig(fn, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {fn}')


def fig_phase_grid(n_show=12):
    """前 n 组的相空间投影并排，直观对比混沌轨迹形态。"""
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9.5),
                              sharex=True, sharey=True)
    fig.suptitle(f'前 {n_show} 组不同初值的相空间 θ₂-ω₂', fontsize=13)

    summary = pd.read_csv(DATA / 'summary.csv').head(n_show)

    for ax, (_, row) in zip(axes.flat, summary.iterrows()):
        df = pd.read_csv(DATA / row['file'])
        ax.plot(np.degrees(df['th2']), df['w2'], lw=0.3, color='#9467bd', alpha=0.7)
        ax.set_title(f'#{int(row["trial"]):02d}  '
                     f'θ₁(0)={row["th1_0_deg"]:+.0f}°, θ₂(0)={row["th2_0_deg"]:+.0f}°',
                     fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('θ₂ (°)')
    for ax in axes[:, 0]:
        ax.set_ylabel('ω₂ (rad/s)')

    plt.tight_layout()
    fn = OUT / 'fig_phase_grid.png'
    plt.savefig(fn, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {fn}')


if __name__ == '__main__':
    print('ChaoticNet 仿真可视化 v1.0')
    print(f'  读自 {DATA.resolve()}')
    print(f'  存到 {OUT.resolve()}\n')

    print('生成四张图：')
    fig_single_trial(trial_idx=0)
    fig_butterfly(th1_0_deg=60, th2_0_deg=30, t_end=15.0, perturb=0.1)
    fig_energy_sweep()
    fig_phase_grid(n_show=12)

    print('\n完成。打开 data/sim/figures/ 看效果。')

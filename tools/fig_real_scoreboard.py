#!/usr/bin/env python3
"""
tools/fig_real_scoreboard.py — 29 段真实实拍·零样本预测视界逐段战绩墙

把 data/real_validation/summary_multi.csv 里 29 段真实双摆实拍的三模型预测视界，
按释放角排序逐段画出来。诚实呈现：
  · 低释放角(近规则、不太混沌)的段，纯数据 ESN 追平甚至反超物理模型 —— 如实标出，
    主动堵「是不是只挑物理赢的段」的樱桃挑选质疑；
  · 高角度混沌段，物理模型(PINN/Hybrid)才系统性拉开 —— 这正是项目核心论点。
  · 右删失段(整窗未超阈、视界=窗长下界)用斜纹 + ≥ 标出。
三条横线 = 各模型 29 段中位（与论文 canonical 一致：ESN 0.167 / PINN 0.417 / Hybrid 0.450）。

诚实口径：真值=逐帧视频追踪(非模型自仿)；sim→real 零样本；视界阈 10°。
输出 PNG 供答辩 PPT 用，不交互、任何 Mac 都能渲。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS',
                                   'PingFang SC', 'Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data/real_validation/summary_multi.csv'
OUT = ROOT / 'data/real_validation/figures/真实战绩墙_29段.png'

# 配色沿用 live_demo（橙=ESN / 青=Hybrid），PINN 用区分度高的柔绿
BG = '#0d0d12'
C_ESN = (1.00, 0.55, 0.16)
C_PINN = (0.40, 0.82, 0.45)
C_HYB = (0.22, 0.85, 0.93)
C_TXT = (0.90, 0.90, 0.93)
C_DIM = (0.62, 0.62, 0.70)

# 论文头条（canonical 多 run 中位统计，data/canonical_results_B.md）——口径不同于本图单 run，故只在副标题引用
CANON = {'ESN': 0.167, 'PINN': 0.417, 'Hybrid': 0.450, 'ratio': 2.7}


def main():
    df = pd.read_csv(CSV).sort_values('rel_deg').reset_index(drop=True)
    n = len(df)
    x = np.arange(n)
    w = 0.27
    # 本 run 自己的逐段中位（画线用，保证线与柱口径一致；论文头条 2.7× 是多 run 统计，副标题另注）
    MED = {'ESN': float(df['ESN_h'].median()),
           'PINN': float(df['PINN_h'].median()),
           'Hybrid': float(df['Hybrid_h'].median())}
    run_ratio = MED['Hybrid'] / MED['ESN'] if MED['ESN'] > 0 else float('inf')

    fig, ax = plt.subplots(figsize=(17, 8.2), facecolor=BG)
    ax.set_facecolor(BG)

    series = [('ESN_h', 'ESN_cens', C_ESN, '纯数据 ESN', -w),
              ('PINN_h', 'PINN_cens', C_PINN, '物理损失 PINN', 0.0),
              ('Hybrid_h', 'Hybrid_cens', C_HYB, '物理先验 Hybrid', w)]
    for hcol, ccol, col, lab, dx in series:
        h = df[hcol].values
        cens = df[ccol].values.astype(bool)
        bars = ax.bar(x + dx, h, width=w, color=col, label=lab,
                      edgecolor=BG, linewidth=0.5, zorder=3)
        # 右删失段加斜纹 + ≥（视界是下界，真实更长）
        for i, b in enumerate(bars):
            if cens[i]:
                b.set_hatch('///')
                ax.text(b.get_x() + b.get_width() / 2, h[i] + 0.05, '≥',
                        ha='center', va='bottom', color=col, fontsize=9,
                        fontweight='bold', zorder=5)

    # 各模型中位横线
    for name, col, hcol in [('ESN', C_ESN, 'ESN_h'), ('PINN', C_PINN, 'PINN_h'),
                            ('Hybrid', C_HYB, 'Hybrid_h')]:
        ax.axhline(MED[name], color=col, ls=(0, (5, 4)), lw=1.1, alpha=0.7, zorder=2)
    ax.text(n - 0.4, MED['Hybrid'] + 0.04, f"Hybrid 中位 {MED['Hybrid']:.2f}s",
            color=C_HYB, fontsize=9, ha='right', va='bottom')
    ax.text(n - 0.4, MED['ESN'] - 0.06, f"ESN 中位 {MED['ESN']:.2f}s",
            color=C_ESN, fontsize=9, ha='right', va='top')

    # 诚实标记：ESN 追平/反超物理的段（堵樱桃挑选）。再据「ESN 自己是否整窗存活」
    # 拆两类：真追上(ESN_h≥0.5s) vs 深混沌区三者均瞬崩、险胜噪声级(top<0.3s)。
    esn_win, genuine, noise = [], [], []
    for i in range(n):
        if df['ESN_h'][i] >= max(df['PINN_h'][i], df['Hybrid_h'][i]) - 1e-6:
            esn_win.append(i)
            top = max(df['ESN_h'][i], df['PINN_h'][i], df['Hybrid_h'][i])
            (genuine if df['ESN_h'][i] >= 0.5 else noise).append(i)
            ax.text(x[i], top + 0.18, '◆', ha='center', va='bottom',
                    color=C_DIM, fontsize=9, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}°" for d in df['rel_deg']], rotation=0,
                       color=C_TXT, fontsize=8.5)
    ax.set_xlabel('释放角（释放时上摆角度，越大越混沌）→', color=C_TXT, fontsize=12)
    ax.set_ylabel('零样本预测视界 (s)  · 偏差首超 10°', color=C_TXT, fontsize=12)
    ax.tick_params(colors=C_TXT)
    for sp in ax.spines.values():
        sp.set_color('#33333d')
    ax.set_ylim(0, max(df[['ESN_h', 'PINN_h', 'Hybrid_h']].values.max() + 0.5, 4.2))
    ax.grid(axis='y', color='#22222c', lw=0.7, zorder=0)

    # 区间底注（放在 0.80 高度，避开顶部图例）
    ax.text(0.6, ax.get_ylim()[1] * 0.80,
            '← 低释放角·近规则：纯数据也追得上',
            color=C_DIM, fontsize=10.5, ha='left', va='top')
    ax.text(n - 0.6, ax.get_ylim()[1] * 0.80,
            '大角度·深混沌：物理系统性领先 →',
            color=C_HYB, fontsize=10.5, ha='right', va='top')

    fig.suptitle('29 段真实实拍 · 零样本预测视界逐段战绩', color='white',
                 fontsize=20, fontweight='bold', x=0.5, y=0.985)
    fig.text(0.5, 0.93,
             f'仿真训练，直接迁到真实双摆视频，零微调。本 run 29 段中位 '
             f"ESN {MED['ESN']:.2f}s / PINN {MED['PINN']:.2f}s / Hybrid {MED['Hybrid']:.2f}s "
             f'≈ {run_ratio:.1f}×　（论文头条 2.7× 为多 run 中位统计 canonical，方向一致）',
             color=C_TXT, fontsize=12, ha='center')

    leg = ax.legend(loc='upper center', ncol=3, framealpha=0.0, fontsize=12,
                    bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(C_TXT)

    fig.text(0.5, 0.038,
             f'◆ = ESN 追平/反超 {len(esn_win)} 段：仅 {len(genuine)} 段（最低能、整窗未崩）'
             f'是真追上，其余 {len(noise)} 段为深混沌区三模型均 <0.3s 瞬崩、胜负噪声级。如实标出，未挑片。',
             color=C_DIM, fontsize=10, ha='center')
    fig.text(0.5, 0.014,
             '真值 = 逐帧视频追踪点（非模型自仿）；sim→real 零样本；视界阈 10°；'
             '斜纹 ≥ = 右删失（整窗未失准，视界为下界）。数据 summary_multi.csv，详见论文。',
             color=C_DIM, fontsize=9.5, ha='center')

    fig.subplots_adjust(left=0.055, right=0.985, top=0.875, bottom=0.135)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f'导出 → {OUT}')
    print(f'  本 run 29 段中位：ESN {MED["ESN"]:.3f}  PINN {MED["PINN"]:.3f}  '
          f'Hybrid {MED["Hybrid"]:.3f}  (Hybrid/ESN={run_ratio:.1f}×；canonical 多 run 2.7×)')
    print(f'  ESN 追平/反超 {len(esn_win)} 段：真追上 {[int(df["tag"][i]) for i in genuine]}；'
          f'噪声级瞬崩 {[int(df["tag"][i]) for i in noise]}')
    print(f'  右删失：ESN {int(df["ESN_cens"].sum())}  PINN {int(df["PINN_cens"].sum())}  '
          f'Hybrid {int(df["Hybrid_cens"].sum())}')


if __name__ == '__main__':
    main()

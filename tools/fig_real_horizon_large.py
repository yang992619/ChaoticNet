"""从已审计的 29 段真实验证汇总表重画图7，专门解决图例和字号问题。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real_validation" / "summary_multi.csv"
OUT = ROOT / "data" / "real_validation" / "figures" / "multi_horizon_vs_angle.png"


def main():
    plt.rcParams.update({
        "font.sans-serif": ["PingFang SC", "Songti SC", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 15,
    })
    df = pd.read_csv(SRC).sort_values("rel_deg")
    fig, ax = plt.subplots(figsize=(13.8, 8.4), dpi=220)
    styles = [("ESN", "#d62728", "o"), ("PINN", "#2ca02c", "s"),
              ("Hybrid", "#9467bd", "^")]
    for name, color, marker in styles:
        cens = df[f"{name}_cens"].astype(bool)
        ax.scatter(df.loc[~cens, "rel_deg"], df.loc[~cens, f"{name}_h"],
                   c=color, marker=marker, s=86, alpha=.88, edgecolor="black", linewidth=.45,
                   label=name, zorder=3)
        ax.scatter(df.loc[cens, "rel_deg"], df.loc[cens, f"{name}_h"],
                   facecolors="white", edgecolors=color, marker=marker, s=94, linewidth=2.0,
                   zorder=4)

    ax.set_title("sim-to-real：29 段真实轨迹的预测视界", fontsize=22, fontweight="bold", pad=16)
    ax.set_xlabel("释放角绝对峰值（°）", fontsize=18)
    ax.set_ylabel("可信预测视界（s，角度偏差首超 10°）", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=.28, zorder=0)
    model_handles = [Line2D([0], [0], color="none", marker=m, markerfacecolor=c,
                            markeredgecolor="black", markersize=9, label=n)
                     for n, c, m in styles]
    censor_handles = [
        Line2D([0], [0], color="none", marker="o", markerfacecolor="#333333",
               markeredgecolor="#333333", markeredgewidth=1.0, markersize=9,
               label="实心：在 4 s 窗内首次超 10°"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="white",
               markeredgecolor="#333333", markeredgewidth=1.8, markersize=9,
               label="空心：右删失（4 s 窗内未超 10°，视界为下界）"),
    ]
    # 模型颜色和右删失含义分开成两个图例，避免文字压到坐标区。
    leg1 = ax.legend(handles=model_handles, loc="upper right", fontsize=14,
                     title="模型", title_fontsize=14, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=censor_handles, loc="upper center", bbox_to_anchor=(.52, -.16),
              ncol=2, fontsize=13, frameon=False)
    fig.subplots_adjust(left=.11, right=.97, top=.90, bottom=.22)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT} from {SRC} (N={len(df)})")


if __name__ == "__main__":
    main()

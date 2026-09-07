"""生成报告与答辩共用的大字版双摆初值敏感性图。

仅改变上摆初角 θ1(0) 0.1°，采用项目当前的分布质量拉格朗日模型。
图中明确标注了对数纵轴的零值显示下限，避免将 Δθ2(0)=0° 误读为非零偏差。
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
import baseline_distmass as model  # noqa: E402


OUT = ROOT / "data" / "sim" / "figures" / "fig_butterfly_distmass_large.png"
FPS = 120
T_END = 12.0
THETA1_BASE = 60.0
THETA1_PERTURBED = 60.1
THETA2 = 0.0
THRESHOLD = 10.0
DISPLAY_FLOOR = 1e-4


def main():
    plt.rcParams.update({
        "font.sans-serif": ["PingFang SC", "Songti SC", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 18,
        "axes.labelcolor": "#40516b",
        "xtick.color": "#52647f",
        "ytick.color": "#52647f",
    })

    base = model.simulate(np.deg2rad(THETA1_BASE), np.deg2rad(THETA2),
                          t_end=T_END, fps=FPS)
    perturbed = model.simulate(np.deg2rad(THETA1_PERTURBED), np.deg2rad(THETA2),
                               t_end=T_END, fps=FPS)
    t = base["t"].to_numpy()
    dtheta1 = np.abs(np.rad2deg(base["th1"].to_numpy() - perturbed["th1"].to_numpy()))
    dtheta2 = np.abs(np.rad2deg(base["th2"].to_numpy() - perturbed["th2"].to_numpy()))
    first = np.flatnonzero(np.maximum(dtheta1, dtheta2) > THRESHOLD)[0]
    threshold_time = t[first]

    fig = plt.figure(figsize=(15, 10), dpi=240, facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.02],
                            left=0.08, right=0.96, top=0.79, bottom=0.17, hspace=0.35)
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1], sharex=ax_top)

    fig.text(0.5, 0.955, "双摆初值敏感性：上摆初角仅差 0.1°",
             ha="center", va="top", fontsize=31, fontweight="bold", color="#15233d")
    fig.text(0.5, 0.908,
             "分布质量拉格朗日模型 · θ1(0)=60.0° / 60.1°，θ2(0)=0°，ω1(0)=ω2(0)=0",
             ha="center", va="top", fontsize=19, color="#5d6d86")

    blue, red, green = "#2563eb", "#dc2626", "#15803d"
    ax_top.plot(t, np.rad2deg(base["th1"]), color=blue, lw=2.5,
                label="基准轨迹：θ1(0)=60.0°")
    ax_top.plot(t, np.rad2deg(perturbed["th1"]), color=red, lw=2.4, ls="--",
                label="扰动轨迹：θ1(0)=60.1°")
    ax_top.set_title("(a) 摆 1 角度轨迹", loc="left", pad=14,
                     fontsize=23, fontweight="bold", color="#17243d")
    ax_top.set_ylabel("摆 1 角度 θ1（°）", fontsize=18)
    ax_top.set_ylim(-80, 80)
    ax_top.set_yticks([-80, -40, 0, 40, 80])
    ax_top.grid(True, color="#dbe3ef", lw=0.8)
    ax_top.legend(loc="upper right", fontsize=16, frameon=False, handlelength=3)

    ax_bottom.semilogy(t, np.maximum(dtheta1, DISPLAY_FLOOR), color=blue, lw=2.4,
                       label="蓝色：|Δθ1|（摆 1 偏差）")
    ax_bottom.semilogy(t, np.maximum(dtheta2, DISPLAY_FLOOR), color=red, lw=2.4,
                       label="红色：|Δθ2|（摆 2 偏差）")
    ax_bottom.axhline(THRESHOLD, color=green, lw=1.7, ls=(0, (4, 3)),
                      label="10° 偏差阈值")
    ax_bottom.axvline(threshold_time, color=green, lw=1.5, ls=(0, (2, 3)))
    ax_bottom.annotate(
        f"首次 max(|Δθ1|, |Δθ2|) > 10°\n t = {threshold_time:.2f} s",
        xy=(threshold_time, THRESHOLD), xytext=(threshold_time + 0.35, 36),
        fontsize=15, color=green, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=green, lw=1.0),
        arrowprops=dict(arrowstyle="-", color=green, lw=1.1),
    )
    ax_bottom.set_title("(b) 两摆角度偏差的快速放大（对数纵轴）", loc="left", pad=14,
                        fontsize=23, fontweight="bold", color="#17243d")
    ax_bottom.set_xlabel("时间 t（s）", fontsize=18, labelpad=12)
    ax_bottom.set_ylabel("角度偏差（°）", fontsize=18)
    ax_bottom.set_xlim(0, T_END)
    ax_bottom.set_ylim(DISPLAY_FLOOR, 300)
    ax_bottom.grid(True, which="both", color="#dbe3ef", lw=0.8)
    ax_bottom.legend(loc="lower right", fontsize=15, frameon=False, handlelength=3)

    fig.text(0.5, 0.082,
             "说明：仅改变 θ1(0) 0.1°；Δθ1(0)=0.1°，Δθ2(0)=0°。",
             ha="center", va="bottom", fontsize=14, color="#53627a")
    fig.text(0.5, 0.049,
             "对数纵轴的零值按 10^-4° 显示下限绘制；数据由当前分布质量模型重新积分（120 fps）。",
             ha="center", va="bottom", fontsize=14, color="#53627a")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"first max angular error > {THRESHOLD:g} deg: {threshold_time:.6f} s")
    print(f"initial errors: dtheta1={dtheta1[0]:.6f} deg, dtheta2={dtheta2[0]:.6f} deg")


if __name__ == "__main__":
    main()

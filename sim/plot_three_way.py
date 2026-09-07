"""
三方对比：ESN / PINN / Hybrid 在同一批 50 组 trial 上的视界。

数据源（权威，与论文表 tab:horizon-compare、data/fair_compare/fair_summary.csv 同源）：
  data/fair_compare/four_way_horizons.csv  （逐 trial：trial, esn, pinn, hybrid, lam0）
此前误读 data/hybrid/hybrid_summary_50.csv（06-17 异常重跑，Hybrid 中位虚高至 3.0s），已弃用。

输出：
  data/figures/three_way_compare.png
  data/figures/three_way_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "figures"
TAU_L = 0.762  # 分布质量自测中位值（50 组扰动轨迹短期分叉率），与论文/README 统一

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
# 字号与论文其余插图统一（此前为答辩 PPT 投影放大到 24pt，放进正文显得过大、格式不一）
plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14,
    "figure.titlesize": 22,
})


def _decimal_log_axis(ax, axis="y"):
    """log 轴主刻度用普通十进制标签（0.1 / 1 / 10），避开字体缺失的上标负号字形。"""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10.0))
    target.set_major_formatter(FuncFormatter(lambda v, _pos: ("%g" % v)))


def main():
    f = pd.read_csv(ROOT / "data" / "fair_compare" / "four_way_horizons.csv")

    # 三方一致混沌：ESN / PINN / Hybrid 都未跑满（< 9.9s），公平同口径子集
    qm = (f.esn < 9.9) & (f.pinn < 9.9) & (f.hybrid < 9.9)
    chaos = f[qm].copy()
    print(f"全集 N={len(f)}, 三方一致混沌 N={len(chaos)}")

    np.random.seed(0)  # jitter 可复现
    # 2+1 版式：避免三联横排被压成三个难读的小格。
    fig = plt.figure(figsize=(13.6, 13.2), dpi=220)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=.36, wspace=.24)
    fig.suptitle(
        "ESN / PINN / Hybrid 三方对比 · 三方一致混沌子集 N=32（取自 50 组 sweep，无阻尼 baseline）",
        fontsize=22, y=.985,
    )
    colors = {"esn": "#5fb0ff", "pinn": "#ff5f7a", "hybrid": "#55cc77"}

    # (a) 箱线图 + 散点
    ax = fig.add_subplot(grid[0, 0])
    data = [chaos.esn, chaos.pinn, chaos.hybrid]
    bp = ax.boxplot(data, tick_labels=["ESN", "PINN", "Hybrid"],
                    patch_artist=True, widths=0.55)
    for patch, name in zip(bp["boxes"], ["esn", "pinn", "hybrid"]):
        patch.set_facecolor(colors[name]); patch.set_alpha(0.7)
    # 数据点 jitter
    for i, name in enumerate(["esn", "pinn", "hybrid"]):
        x = np.random.normal(i + 1, 0.06, size=len(chaos))
        ax.scatter(x, chaos[name], color=colors[name], edgecolor="white",
                   lw=0.5, alpha=0.7, s=34, zorder=3)
    ax.axhline(TAU_L, color="#888", ls="--", lw=1)
    ax.text(3.4, TAU_L, f"  τ_L ≈ {TAU_L}s", color="#444", fontsize=12, va="center")
    ax.set_ylabel("可信预测视界 (s)")
    ax.set_title(f"(a) 视界分布（三方一致混沌 N={len(chaos)}）")
    ax.set_yscale("log")
    _decimal_log_axis(ax, "y")
    ax.grid(alpha=0.25, which="both")

    # (b) CDF
    ax = fig.add_subplot(grid[0, 1])
    for name in ["esn", "pinn", "hybrid"]:
        x = np.sort(chaos[name].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=colors[name], lw=2,
                label=f"{name.upper()}  中位 {np.median(x):.2f}s")
    ax.axvline(TAU_L, color="#888", ls="--", lw=1)
    ax.set_xlabel("可信预测视界 (s)")
    ax.set_ylabel("经验 CDF")
    ax.set_title("(b) 累积分布")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 11)
    ax.set_ylim(0, 1.02)
    _decimal_log_axis(ax, "x")
    ax.legend(loc="lower right", fontsize=13)
    ax.grid(alpha=0.25, which="both")

    # (c) 散点矩阵：横轴 ESN，纵轴对应模型
    ax = fig.add_subplot(grid[1, :])
    ax.scatter(chaos.esn, chaos.pinn, color=colors["pinn"], s=60,
               alpha=0.7, edgecolor="white", label="PINN", marker="o")
    ax.scatter(chaos.esn, chaos.hybrid, color=colors["hybrid"], s=60,
               alpha=0.7, edgecolor="white", label="Hybrid", marker="^")
    lim = np.array([0.05, 11])
    ax.plot(lim, lim, "--", color="#888", lw=1)
    for k in (10, 30, 100):
        ax.plot(lim, lim * k, ":", color="#bbb", lw=0.8)
        ax.text(lim[1] / k * 1.1, lim[1] * 0.92, f"{k}×", color="#888",
                fontsize=12, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    _decimal_log_axis(ax, "x")
    _decimal_log_axis(ax, "y")
    ax.set_xlabel("ESN 视界 (s)")
    ax.set_ylabel("PINN / Hybrid 视界 (s)")
    ax.set_title("(c) 相对 ESN 的提升散点")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(alpha=0.25, which="both")
    # 让底部散点图充分使用页面宽度，避免对数同尺度造成大面积空白。
    ax.set_aspect("auto")

    fig.subplots_adjust(left=.10, right=.97, bottom=.08, top=.94)
    out = OUT / "three_way_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"写出 {out}")

    # 汇总
    print("\n=== 三方一致混沌 trial 上的统计 ===")
    for name in ["esn", "pinn", "hybrid"]:
        h = chaos[name]
        print(f"  {name.upper():>6}  中位 {h.median():.3f}s  均值 {h.mean():.3f}s  "
              f"= {h.median() / TAU_L:.2f} τ_L (中位)")
    print(f"  PINN  / ESN 中位提升 {(chaos.pinn / chaos.esn).median():.1f}×")
    print(f"  Hybrid/ ESN 中位提升 {(chaos.hybrid / chaos.esn).median():.1f}×")
    print(f"  Hybrid/PINN 中位提升 {(chaos.hybrid / chaos.pinn).median():.1f}×")

    # 写联合汇总 csv
    summary = pd.DataFrame({
        "model": ["ESN", "PINN", "Hybrid"],
        "n_chaos": [len(chaos)] * 3,
        "horizon_median_s": [chaos.esn.median(), chaos.pinn.median(), chaos.hybrid.median()],
        "horizon_mean_s":   [chaos.esn.mean(),   chaos.pinn.mean(),   chaos.hybrid.mean()],
        "horizon_tau_L":    [chaos.esn.median() / TAU_L,
                              chaos.pinn.median() / TAU_L,
                              chaos.hybrid.median() / TAU_L],
    })
    summary.to_csv(OUT / "three_way_summary.csv", index=False)
    print(f"\n写出 {OUT}/three_way_summary.csv")


if __name__ == "__main__":
    main()

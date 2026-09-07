"""
50 组完整 sweep 的视界分布可视化（论文最关键图）

四联图：
  (a) ESN / PINN 视界直方图（区分混沌 vs 准周期）
  (b) 经验 CDF
  (c) 每 trial 的 ESN-PINN 散点对比（含 y=x 参考线）
  (d) PINN/ESN 提升倍数直方图（log 刻度）

输出：data/figures/horizon_compare_50.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TAU_L = 0.12                 # Lyapunov 时间
QUASI_THRESH = 9.9           # 视界 >= 9.9s 视作准周期

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(ROOT / "data" / "horizon_compare_50.csv")
    quasi = (df.esn >= QUASI_THRESH) | (df.pinn >= QUASI_THRESH)
    chaos = df[~quasi].copy()
    print(f"N total={len(df)}  chaos={len(chaos)}  quasi={quasi.sum()}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=120)
    fig.suptitle(
        "ChaoticNet 50 组完整 sweep（trial 0-49）· 闭环预测视界对比",
        fontsize=14, color="#222", y=0.995,
    )

    # (a) 直方图：只画 chaos 段
    ax = axes[0, 0]
    bins = np.linspace(0, 10, 41)
    ax.hist(chaos.esn, bins=bins, color="#5fb0ff", alpha=0.6,
            label=f"ESN  均值 {chaos.esn.mean():.2f}s  中位 {chaos.esn.median():.2f}s")
    ax.hist(chaos.pinn, bins=bins, color="#ff5f7a", alpha=0.6,
            label=f"PINN 均值 {chaos.pinn.mean():.2f}s  中位 {chaos.pinn.median():.2f}s")
    ax.axvline(TAU_L, color="#888", ls="--", lw=1)
    ax.text(TAU_L, ax.get_ylim()[1] * 0.92, f"  τ_L ≈ {TAU_L}s",
            color="#444", fontsize=9)
    ax.set_xlabel("可信预测视界 (s)")
    ax.set_ylabel("trial 数")
    ax.set_title(f"(a) 视界直方图（剔除 {quasi.sum()} 个准周期 trial，N={len(chaos)}）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    # (b) 经验 CDF
    ax = axes[0, 1]
    for col, color, name in [("esn", "#5fb0ff", "ESN"),
                              ("pinn", "#ff5f7a", "PINN")]:
        x = np.sort(chaos[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, lw=2, label=name)
    ax.axvline(TAU_L, color="#888", ls="--", lw=1)
    ax.text(TAU_L, 0.05, f"  τ_L ≈ {TAU_L}s", color="#444", fontsize=9)
    ax.set_xlabel("可信预测视界 (s)")
    ax.set_ylabel("经验 CDF")
    ax.set_title("(b) 累积分布函数")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 11)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, which="both")

    # (c) 散点对比（log-log）
    ax = axes[1, 0]
    ax.scatter(chaos.esn, chaos.pinn, c="#cc3344", s=42, alpha=0.7,
               edgecolor="white", lw=0.6, label=f"混沌 trial N={len(chaos)}")
    if quasi.sum() > 0:
        ax.scatter(df.loc[quasi, "esn"], df.loc[quasi, "pinn"],
                   c="#888", s=42, alpha=0.5, marker="x",
                   label=f"准周期 trial N={quasi.sum()}")
    lim = np.array([0.05, 11])
    ax.plot(lim, lim, "--", color="#888", lw=1, label="y = x")
    for k in (3, 10, 30):
        ax.plot(lim, lim * k, ":", color="#bbb", lw=0.8)
        ax.text(lim[1] / k * 1.05, lim[1] * 0.92, f"{k}×",
                color="#888", fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ESN 视界 (s)")
    ax.set_ylabel("PINN 视界 (s)")
    ax.set_title("(c) 散点对比：每个 trial 一个点")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.set_aspect("equal", adjustable="box")

    # (d) 提升倍数直方图
    ax = axes[1, 1]
    ratios = chaos.ratio.replace([np.inf, -np.inf], np.nan).dropna()
    bins = np.logspace(-0.5, 2.5, 30)   # 0.3x ~ 300x
    ax.hist(ratios, bins=bins, color="#ff8855", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="#888", ls="--", lw=1)
    ax.axvline(ratios.median(), color="#cc3344", ls="-", lw=2,
               label=f"中位 {ratios.median():.1f}×")
    ax.axvline(ratios.mean(), color="#cc3344", ls=":", lw=2,
               label=f"均值 {ratios.mean():.1f}×")
    ax.set_xscale("log")
    ax.set_xlabel("提升倍数  PINN 视界 / ESN 视界")
    ax.set_ylabel("trial 数")
    ax.set_title("(d) 物理约束相对纯数据驱动的提升")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")

    plt.tight_layout()
    out = OUT / "horizon_compare_50.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"写出 {out}")

    # 一行汇总打印
    print("\n=== 论文该用的数字 ===")
    print(f"N 总 {len(df)}, 混沌 {len(chaos)}, 准周期 {quasi.sum()}")
    print(f"ESN  视界 中位 {chaos.esn.median():.3f}s  均值 {chaos.esn.mean():.3f}s")
    print(f"PINN 视界 中位 {chaos.pinn.median():.3f}s  均值 {chaos.pinn.mean():.3f}s")
    print(f"PINN/ESN 比 中位 {ratios.median():.2f}×  均值 {ratios.mean():.2f}×")
    print(f"用 τ_L={TAU_L}s 归一化：")
    print(f"  ESN  中位 = {chaos.esn.median() / TAU_L:.2f} τ_L")
    print(f"  PINN 中位 = {chaos.pinn.median() / TAU_L:.2f} τ_L")


if __name__ == "__main__":
    main()

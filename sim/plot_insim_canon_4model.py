"""
仿真内 · 四模型 canonical 视界图（答辩 PPT 第10页 / idx9, ppt media image9.png）。

口径与论文表 tab:horizon-compare / data/canon_multirun/summary.csv 完全一致：
  每个模型独立训练多次（ESN 换 reservoir seed×10；PINN/PINN0/Hybrid 非确定性重训×8），
  每 run 取"该 run 自己混沌 trial(horizon<9.9s) 的中位视界"，再看这批 per-run 中位的分布。
  中位落在 ESN 0.40 / PINN 0.88 / MLP 0.88... → 与右侧数字卡片、论文头条表逐一对齐。

此前 PPT 内嵌的是 three_way_compare.png（N=32 三方一致子集，单 run，中位 0.29/0.51/2.17），
与卡片(0.40/0.88/1.55/2.85)口径不同、同页并排数字对不上。本图换成 per-run 口径消除该分歧。
注意：不覆写 data/figures/three_way_compare.png（论文 fig:three-way 仍用它），另存新文件。

输出：data/figures/insim_canon_4model.png（比例对齐 PPT 图框 2.887，直接替换内嵌 image9.png）。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "canon_multirun" / "insim_runs.csv"
OUT = ROOT / "data" / "figures" / "insim_canon_4model.png"
TAU_L = 0.762
CHAOS_CAP = 9.9  # 视界 >= 9.9s 视为落入/被判长周期轨道，剔除

# 模型内部名 -> 展示名 / 颜色 / 相对 ESN 倍数（与论文表一致的整洁标签）
MODELS = [
    ("ESN",    "ESN",    "#5fb0ff", "基线"),
    ("PINN",   "PINN",   "#ff5f7a", "2.2×"),
    ("PINN0",  "MLP",    "#f0a03c", "3.8×"),
    ("Hybrid", "Hybrid", "#55cc77", "7.0×"),
]

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
# 本图在 PPT 里缩到约 6.8in 宽（约 0.42×），沿用大字号以保证投影清晰
plt.rcParams.update({
    "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 23,
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
})


def _decimal_log(ax, axis="y"):
    t = ax.yaxis if axis == "y" else ax.xaxis
    t.set_major_locator(LogLocator(base=10.0))
    t.set_major_formatter(FuncFormatter(lambda v, _p: ("%g" % v)))


def per_run_medians(df, model):
    """canonical 口径：每 run 取该 run 混沌子集(horizon<9.9)的中位视界。"""
    g = df[df.model == model]
    out = []
    for _run, r in g.groupby("run"):
        chaos = r.horizon[r.horizon < CHAOS_CAP]
        if len(chaos):
            out.append(float(np.median(chaos)))
    return np.array(sorted(out))


def main():
    df = pd.read_csv(SRC)
    runs = {k: per_run_medians(df, k) for k, *_ in MODELS}
    meds = {k: float(np.median(v)) for k, v in runs.items()}

    # 自检：必须复现 summary.csv / 论文表 0.40 / 0.88 / 1.55 / 2.85
    expect = {"ESN": 0.404, "PINN": 0.875, "PINN0": 1.55, "Hybrid": 2.85}
    print("per-run 中位复核：")
    for k in expect:
        got, exp = meds[k], expect[k]
        flag = "OK" if abs(got - exp) < 0.02 else "!!MISMATCH!!"
        print(f"  {k:>7}: n={len(runs[k]):>2}  中位 {got:.3f}  期望 {exp:.3f}  {flag}")
        assert abs(got - exp) < 0.02, f"{k} 中位 {got} 偏离期望 {exp}"

    np.random.seed(0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=120)
    fig.suptitle(
        "ESN / PINN / MLP / Hybrid 四方对比 · 50 组 sweep · 多次训练 per-run 中位",
        fontsize=26, y=1.03,
    )
    names = [m[0] for m in MODELS]
    disp = [m[1] for m in MODELS]
    cols = [m[2] for m in MODELS]
    ratio_lab = [m[3] for m in MODELS]

    # (a) 箱线 + 散点：per-run 中位的分布
    ax = axes[0]
    data = [runs[n] for n in names]
    bp = ax.boxplot(data, tick_labels=disp, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("#222"); med.set_linewidth(2)
    for i, n in enumerate(names):
        x = np.random.normal(i + 1, 0.06, size=len(runs[n]))
        ax.scatter(x, runs[n], color=cols[i], edgecolor="white",
                   lw=0.6, alpha=0.85, s=42, zorder=3)
    ax.axhline(TAU_L, color="#888", ls="--", lw=1)
    ax.text(4.35, TAU_L, "  τ_L ≈ 0.76s", color="#444", fontsize=18,
            va="center", ha="left")
    ax.set_ylabel("可信预测视界 (s)")
    ax.set_title("(a) per-run 中位分布")
    ax.set_yscale("log")
    _decimal_log(ax, "y")
    ax.grid(alpha=0.25, which="both")

    # (b) CDF
    ax = axes[1]
    for i, n in enumerate(names):
        x = runs[n]
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=cols[i], lw=2.4,
                label=f"{disp[i]}  中位 {meds[n]:.2f}s")
    ax.axvline(TAU_L, color="#888", ls="--", lw=1)
    ax.set_xlabel("可信预测视界 (s)")
    ax.set_ylabel("经验 CDF")
    ax.set_title("(b) 累积分布")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 6)
    ax.set_ylim(0, 1.02)
    _decimal_log(ax, "x")
    ax.legend(loc="lower right", fontsize=18)
    ax.grid(alpha=0.25, which="both")

    # (c) 中位 [IQR] 棒棒糖 + 相对 ESN 倍数
    ax = axes[2]
    ypos = np.arange(len(names))[::-1]  # ESN 在上、Hybrid 在下更贴阅读顺序
    for i, n in enumerate(names):
        v = runs[n]
        q25, q75 = np.percentile(v, [25, 75])
        y = ypos[i]
        ax.plot([q25, q75], [y, y], color=cols[i], lw=6, alpha=0.55,
                solid_capstyle="round")
        ax.scatter([meds[n]], [y], color=cols[i], edgecolor="#222",
                   lw=1.2, s=170, zorder=4)
        ax.text(meds[n], y + 0.28, f"{meds[n]:.2f}s  {ratio_lab[i]}",
                color="#222", fontsize=19, ha="center", va="bottom")
    ax.axvline(TAU_L, color="#888", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(disp)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 6)
    _decimal_log(ax, "x")
    ax.set_xlabel("中位视界 (s)")
    ax.set_title("(c) 中位 [IQR] · 相对 ESN")
    ax.grid(alpha=0.25, which="both", axis="x")

    plt.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    from PIL import Image
    im = Image.open(OUT)
    print(f"\n写出 {OUT}")
    print(f"尺寸 {im.size}  比例 {im.size[0]/im.size[1]:.3f}  (目标 2.887)")


if __name__ == "__main__":
    main()

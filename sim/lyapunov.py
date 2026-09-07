"""
sim/lyapunov.py — 最大 Lyapunov 指数实测

方法：固定参考初值，加 epsilon 扰动，跑两条轨迹，跟踪相对距离随时间增长。
线性区拟合 d(t) ≈ d0 · exp(λ·t)，对数斜率即 λ_max。

为提高鲁棒性：用 50 组不同初值（trial_000..049 的 (θ₁₀, θ₂₀)）各做一次，
取平均斜率。

输出：
  data/lyapunov.csv             每组 (trial, lambda_fit, tau_L)
  data/figures/lyapunov.png     50 组 log(distance) vs t 叠加 + 拟合直线 + 直方图
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "sim"))
import baseline as bl

EPS = 1e-6                    # 扰动幅度（rad）
T_END = 3.0                   # 跟踪时长（秒，短期初始分叉率）
FPS = 240                     # 高时间分辨率
FIT_START_FRAC = 0.05         # 拟合从 5% 开始（跳过初始瞬变）
FIT_END_FRAC   = 0.50         # 到 50% 截止（避免饱和）

# 说明：本脚本估算的是"扰动短期初始分叉率"，不是严格最大 Lyapunov 指数。
# 严格估算需 Wolf 算法 + renormalization 跟踪到 long-time limit。
# 当前默认（分布质量 50 组扰动轨迹）：λ_max 中位 ≈ 1.31/s → τ_L ≈ 0.762s，
# 即全项目权威标尺（见 data/canonical_results_B.md；本脚本现跑出，落 data/lyapunov.csv）。
# 拟合区间依赖（点质量期自测，定性说明该方法只捕获短期增长率、不收敛到 long-time limit）：
#   t_end=3s,fit5-50%: τ_L≈0.8s（短窗）  ·  t_end=6s: τ_L≈1.8s  ·  t_end=8s: τ_L≈2.5s
# λ 随拟合窗变长单调下降、未收敛 → 该绝对值仅作局部分叉率参考，物理增益结论不依赖它。

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def simulate_pair(th1_0, th2_0, eps=EPS, t_end=T_END, fps=FPS):
    """跑两条仅初值差 eps 的轨迹，返回 (t, d_array)"""
    y_a = [th1_0,             0.0, th2_0,        0.0]
    y_b = [th1_0 + eps,       0.0, th2_0,        0.0]
    t_eval = np.linspace(0, t_end, int(t_end * fps) + 1)
    sa = solve_ivp(bl.deriv, [0, t_end], y_a, t_eval=t_eval,
                   method="RK45", rtol=1e-10, atol=1e-12, max_step=1e-3)
    sb = solve_ivp(bl.deriv, [0, t_end], y_b, t_eval=t_eval,
                   method="RK45", rtol=1e-10, atol=1e-12, max_step=1e-3)
    # 4 维相空间欧氏距离
    d = np.linalg.norm(sa.y - sb.y, axis=0)
    return sa.t, d


def fit_lambda(t, d):
    """对数线性拟合 log(d) = log(d0) + λ·t  返回 λ"""
    log_d = np.log(d + 1e-30)
    n = len(t)
    i0 = int(n * FIT_START_FRAC)
    i1 = int(n * FIT_END_FRAC)
    slope, intercept = np.polyfit(t[i0:i1], log_d[i0:i1], 1)
    return slope, intercept


def main():
    summary = pd.read_csv(ROOT / "data" / "sim" / "summary.csv")
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    ax0, ax1 = axes

    print(f"Lyapunov 指数估算  eps={EPS:.0e},  t_end={T_END}s,  fps={FPS}")
    for i, row in summary.iterrows():
        th1_0 = np.deg2rad(row.th1_0_deg)
        th2_0 = np.deg2rad(row.th2_0_deg)
        t, d = simulate_pair(th1_0, th2_0)
        slope, intercept = fit_lambda(t, d)
        tau_L = 1.0 / slope if slope > 0 else np.nan
        records.append({"trial": int(row.trial),
                        "th1_0_deg": row.th1_0_deg, "th2_0_deg": row.th2_0_deg,
                        "lambda_max": slope, "tau_L": tau_L})
        ax0.plot(t, np.log(d + 1e-30), color="#888", alpha=0.25, lw=0.6)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50   λ={slope:+.3f}/s   τ_L={tau_L:.3f}s")

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "data" / "lyapunov.csv", index=False)

    # 在左图叠加均值拟合线
    valid = df[df.lambda_max > 0]
    lam_mean = valid.lambda_max.mean()
    lam_median = valid.lambda_max.median()
    tau_mean = 1.0 / lam_mean
    tau_median = 1.0 / lam_median
    t_ref = np.linspace(0, T_END, 200)
    # 视觉化用中间区间的对数距离均值
    ax0.set_xlabel("t (s)")
    ax0.set_ylabel(r"$\log\ d(t)$")
    ax0.set_title(f"50 组初值扰动后相空间距离  ({len(valid)} 组拟合有效)")
    ax0.text(0.02, 0.95,
             f"λ_max 均值 = {lam_mean:.2f}/s,  τ_L = {tau_mean:.3f}s\n"
             f"λ_max 中位 = {lam_median:.2f}/s,  τ_L = {tau_median:.3f}s",
             transform=ax0.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="#888"))
    ax0.grid(alpha=0.25)

    # 直方图
    ax1.hist(valid.lambda_max, bins=20, color="#5fb0ff", alpha=0.7,
             edgecolor="white")
    ax1.axvline(lam_mean, color="#cc3344", lw=2, label=f"均值 {lam_mean:.2f}/s")
    ax1.axvline(lam_median, color="#222", lw=2, ls="--", label=f"中位 {lam_median:.2f}/s")
    ax1.set_xlabel("最大 Lyapunov 指数 λ_max (1/s)")
    ax1.set_ylabel("trial 数")
    ax1.set_title("50 组 λ_max 分布")
    ax1.legend()
    ax1.grid(alpha=0.25)

    plt.tight_layout()
    out_png = ROOT / "data" / "figures" / "lyapunov.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n=== 50 组结果 ===")
    print(f"  λ_max 均值 = {lam_mean:.3f}/s,  τ_L 均值 = {tau_mean:.4f}s")
    print(f"  λ_max 中位 = {lam_median:.3f}/s,  τ_L 中位 = {tau_median:.4f}s")
    print(f"  论文先前引用文献 τ_L ≈ 0.12s，本研究实测 τ_L 中位 = {tau_median:.3f}s")
    print(f"  写出 data/lyapunov.csv + {out_png}")


if __name__ == "__main__":
    main()

"""从误差分解 CSV 重画图8；统一子图顺序、字号和阻尼口径。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real_validation" / "error_decomposition.csv"
OUT = ROOT / "data" / "real_validation" / "figures" / "error_decomposition.png"
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main():
    plt.rcParams.update({
        "font.sans-serif": ["PingFang SC", "Songti SC", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 14,
    })
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=220)

    # (a) 自然阅读顺序：左上。
    ax = axes[0, 0]
    ax.scatter(df.rel_deg, df.C_chaos_self, c="#1f77b4", marker="*", s=135,
               label="C_chaos：初值不确定天花板", zorder=3)
    ax.scatter(df.rel_deg, df.C_dampmis, c="#ff7f0e", marker="D", s=58, alpha=.85,
               label="C_dampmis：阻尼失配天花板")
    ax.scatter(df.rel_deg, df.h_real_PINN, c="#d62728", marker="o", s=65, alpha=.85,
               label="h_real：PINN 实际迁移")
    ax.set_title("(a) 实测视界与两条理论天花板", loc="left", fontsize=18, fontweight="bold")
    ax.set_xlabel("释放角绝对峰值（°）"); ax.set_ylabel("预测视界（s）")
    ax.grid(alpha=.28); ax.legend(fontsize=11)

    # (b) 右上，避免原图 a/c/b/d 的阅读次序错乱。
    ax = axes[0, 1]
    medc = [df[f"C_chaos_{d}"].median() for d in DELTAS]
    ax.plot(DELTAS, medc, "o-", color="#1f77b4", lw=2.2, label="C_chaos 中位")
    for x, y in zip(DELTAS, medc):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=12)
    dw = df.dw_self.median(); hr = df.h_real_PINN.median()
    ax.axvline(dw, color="#2ca02c", lw=1.8, label=f"实测 δω 中位 = {dw:.3f} rad/s")
    ax.axhline(hr, color="#d62728", ls=":", lw=1.8, label=f"h_real 中位 = {hr:.2f} s")
    ax.set_title("(b) 初值不确定度对混沌天花板的影响", loc="left", fontsize=18, fontweight="bold")
    ax.set_xlabel("初值角速度不确定度 δω（rad/s）"); ax.set_ylabel("C_chaos 中位（s）")
    ax.grid(alpha=.28); ax.legend(fontsize=11)

    # (c) 左下。阻尼参数写入标题，不能让读者猜“阻尼”具体是什么。
    ax = axes[1, 0]
    labels = ["h_real", "h_damp", "h_model", "C_dampmis", "C_chaos"]
    pinn = [df.h_real_PINN.median(), df.h_damp_PINN.median(), df.h_model_PINN.median(),
            df.C_dampmis.median(), df.C_chaos_self.median()]
    hybrid = [df.h_real_Hybrid.median(), df.h_damp_Hybrid.median(), df.h_model_Hybrid.median(),
              df.C_dampmis.median(), df.C_chaos_self.median()]
    x = np.arange(len(labels)); w = .36
    ax.bar(x - w/2, pinn, w, label="PINN", color="#2ca02c", alpha=.85)
    ax.bar(x + w/2, hybrid, w, label="Hybrid", color="#9467bd", alpha=.85)
    for i, (p, h) in enumerate(zip(pinn, hybrid)):
        ax.text(i-w/2, p+.05, f"{p:.2f}", ha="center", fontsize=11)
        ax.text(i+w/2, h+.05, f"{h:.2f}", ha="center", fontsize=11)
    ax.axvspan(2.5, 4.5, color="grey", alpha=.08)
    ax.set_xticks(x, labels); ax.set_ylabel("视界中位（s）")
    ax.set_title("(c) 视界中位对比；阻尼模型为拟合 b=1.66e-5、c=8.72e-6", loc="left", fontsize=15, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=.28)

    # (d) 右下。
    ax = axes[1, 1]
    upper = max(df.C_chaos_self.max(), df.h_real_Hybrid.max()) * 1.08
    ax.plot([0, upper], [0, upper], "k--", lw=1, label="h_real = C_chaos")
    ax.scatter(df.C_chaos_self, df.h_real_PINN, c="#2ca02c", marker="s", s=65, label="PINN")
    ax.scatter(df.C_chaos_self, df.h_real_Hybrid, c="#9467bd", marker="^", s=65, label="Hybrid")
    ax.set_title("(d) 实测视界低于自身混沌天花板", loc="left", fontsize=18, fontweight="bold")
    ax.set_xlabel("C_chaos（s）"); ax.set_ylabel("h_real（s）")
    ax.grid(alpha=.28); ax.legend()

    fig.suptitle("误差成因三天花板分解（29 段真实轨迹）", fontsize=23, fontweight="bold", y=.99)
    fig.subplots_adjust(left=.08, right=.98, top=.93, bottom=.08, wspace=.24, hspace=.34)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT} from {SRC} (N={len(df)})")


if __name__ == "__main__":
    main()

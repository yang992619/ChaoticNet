"""
rc/sensitivity.py — 初值精度敏感性扫描（论文 5.4 章）

模拟初值状态估计不确定度（视频反演噪声）：标称初值 (θ₁₀, θ₂₀) 加 ±X° 高斯扰动 → K 个独立 trial，
各跑解析 baseline 拿到"扰动后 GT"。
然后用统一的 PINN 模型（trial 0-4 训练）从扰动初值出发做闭环预测，
评估扰动幅度对预测视界的影响。
（±X° 代表视频反演初值的状态估计不确定度，非释放机构的机械重复性。）

扰动档位：±0.1° (理想), ±0.5° (本研究实测量级), ±1.0° (保守上限)
每档 K=10 次。

输出：
  data/sensitivity/sensitivity.csv     每次评估的 horizon
  data/sensitivity/sensitivity_box.png 三档 boxplot
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "sim"))
import pinn as pinn_mod                # noqa: E402
import baseline as bl                   # noqa: E402

OUT = ROOT / "data" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# 参考初值：trial_000 的标称值（49.3°, -11.0°）
REF_TH1_DEG, REF_TH2_DEG = 49.31, -11.00
K = 10
LEVELS_DEG = [0.1, 0.5, 1.0]
TRAIN_SEC = 20.0
PRED_SEC = 10.0
FPS = 120

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def load_pinn():
    pt = ROOT / "data" / "pinn" / "model.pt"
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    net.load_state_dict(torch.load(pt))
    net.eval()
    return net


def run_one(net, th1_deg, th2_deg, seed=0):
    th1_0 = np.deg2rad(th1_deg)
    th2_0 = np.deg2rad(th2_deg)
    df = bl.simulate(th1_0, th2_0, t_end=TRAIN_SEC + PRED_SEC, fps=FPS)
    n_train = int(TRAIN_SEC * FPS)
    n_pred = int(PRED_SEC * FPS)
    s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
    pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / FPS)
    pred_th = pred[:, [0, 2]]
    truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
    h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=FPS)
    return h_s


def main():
    net = load_pinn()
    print(f"参考初值：θ₁ = {REF_TH1_DEG}°, θ₂ = {REF_TH2_DEG}°")
    print(f"扰动档位：±{LEVELS_DEG}°,  每档 K = {K}")

    # baseline：无扰动 reference horizon
    ref_h = run_one(net, REF_TH1_DEG, REF_TH2_DEG)
    print(f"\n[参考] 无扰动 horizon = {ref_h:.2f}s\n")

    rng = np.random.default_rng(42)
    records = []
    for level in LEVELS_DEG:
        print(f"=== ±{level}° ===")
        for k in range(K):
            d1 = rng.normal(0, level)
            d2 = rng.normal(0, level)
            th1 = REF_TH1_DEG + d1
            th2 = REF_TH2_DEG + d2
            h = run_one(net, th1, th2)
            records.append({
                "level_deg": level, "rep": k,
                "delta_th1_deg": d1, "delta_th2_deg": d2,
                "horizon": h,
            })
            print(f"  k={k}  δθ₁={d1:+5.2f}°  δθ₂={d2:+5.2f}°  horizon={h:.2f}s")
    df = pd.DataFrame(records)
    df.to_csv(OUT / "sensitivity.csv", index=False)

    # boxplot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    data = [df[df.level_deg == lv].horizon.values for lv in LEVELS_DEG]
    bp = ax.boxplot(data, labels=[f"±{lv}°" for lv in LEVELS_DEG],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#5fb0ff", "#ffb555", "#ff5f7a"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(ref_h, color="#222", ls="--", lw=1, alpha=0.5)
    ax.text(0.5, ref_h, f"  无扰动参考 = {ref_h:.2f}s",
            color="#444", fontsize=9, transform=ax.get_yaxis_transform(),
            va="center")
    ax.set_ylabel("PINN 预测视界 (s)")
    ax.set_xlabel("初值状态估计扰动幅度（高斯标准差）")
    ax.set_title(f"PINN 视界对初值扰动的敏感性  ·  参考初值 ({REF_TH1_DEG}°, {REF_TH2_DEG}°)  ·  K={K}/档")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(OUT / "sensitivity_box.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # 汇总
    print("\n=== 汇总 ===")
    summary = df.groupby("level_deg").horizon.agg(["mean", "median", "std", "min", "max"])
    print(summary)
    summary.to_csv(OUT / "sensitivity_summary.csv")
    print(f"\n写出：{OUT}/sensitivity.csv, sensitivity_summary.csv, sensitivity_box.png")


if __name__ == "__main__":
    main()

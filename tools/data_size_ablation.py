"""
tools/data_size_ablation.py — 训练数据量 ablation

PINN λ=1.0 / 128×4 / 200 epoch 用不同训练 trial 数：{2, 5, 10, 20, 50}（含 50 即全集）
评估 50 trial（注：训练集越大可能跟测试集有 overlap，但本研究测试是闭环预测段，
不是同 trial 的预测段，所以 OK——而且实际中 paper baseline 也用了 trial 0-4 训练 + 评估 0-4）

每个 size 跑 3 个 seed 看 variance。

输出 data/data_size/results.csv + curve.png
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
import pinn as pinn_mod        # noqa: E402

DATA = ROOT / "data" / "sim"
OUT  = ROOT / "data" / "data_size"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [2, 5, 10, 20, 50]
SEEDS = [42, 7, 13]
N_EPOCHS = 200
QUASI = 9.9

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate(net):
    horizons = []
    for i in range(50):
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[2400].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, 1200, dt=1/120)
        truth = df[["th1", "th2"]].values[2400:3600]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=120)
        horizons.append(h)
    return np.array(horizons)


def main():
    rows = []
    for size in SIZES:
        for seed in SEEDS:
            print(f"\n[size={size} seed={seed}]")
            states, accs = pinn_mod.load_dataset(range(size), train_sec=20.0)
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=N_EPOCHS, batch=512, lr=2e-3,
                           lambda_phys=1.0, n_collocation=2048, seed=seed)
            net.eval()
            h = evaluate(net)
            chaos = h[h < QUASI]
            row = {
                "n_train_trials": size, "seed": seed,
                "n_samples": int(len(states)),
                "full_median": float(np.median(h)),
                "chaos_median": float(np.median(chaos)) if len(chaos) else np.nan,
                "n_chaos": int(len(chaos)),
            }
            print(f"  全集中位 {row['full_median']:.3f}s  N_chaos={row['n_chaos']}")
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== 按 size 聚合 ===")
    agg = df.groupby("n_train_trials").full_median.agg(["median", "mean", "std", "min", "max"])
    print(agg)
    agg.to_csv(OUT / "summary.csv")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for s in SEEDS:
        sub = df[df.seed == s]
        ax.plot(sub.n_train_trials, sub.full_median, "o-", alpha=0.5, label=f"seed {s}")
    mean_curve = df.groupby("n_train_trials").full_median.mean()
    ax.plot(mean_curve.index, mean_curve.values, "k-", lw=2.5, label="均值")
    ax.set_xscale("log")
    ax.set_xlabel("训练 trial 数")
    ax.set_ylabel("PINN 视界中位 (s) · 全集 50 trial")
    ax.set_title("训练数据量 ablation · PINN λ=1.0, 128×4, 200 epoch")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(OUT / "curve.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"写出 {OUT}/curve.png + results.csv + summary.csv")


if __name__ == "__main__":
    main()

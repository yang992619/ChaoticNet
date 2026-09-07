"""
tools/multi_seed.py — 多 seed 鲁棒性测试

验证 v2.1 反直觉发现（λ=0 PINN 在仿真数据上 > PINN）是不是单 seed 巧合。
每个 config 跑 5 个 seed，看 horizon median 在 seed 间的 variance。

config: λ_phys ∈ {0.0, 1.0, 2.0}, 固定 hidden=128 layers=4 epoch=200
seeds: {42, 7, 13, 21, 100}

输出 data/multi_seed/results.csv + summary.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod        # noqa: E402

DATA = ROOT / "data" / "sim"
OUT  = ROOT / "data" / "multi_seed"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.0, 1.0, 2.0]
SEEDS = [42, 7, 13, 21, 100]
N_EPOCHS = 200
QUASI = 9.9
N_TRIALS = 50


def evaluate(net, n=N_TRIALS):
    horizons = []
    for i in range(n):
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[2400].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, 1200, dt=1/120)
        truth = df[["th1", "th2"]].values[2400:3600]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=120)
        horizons.append(h)
    return np.array(horizons)


def main():
    states, accs = pinn_mod.load_dataset(range(5), train_sec=20.0)
    print(f"训练样本 {len(states)}")
    print(f"配置：λ ∈ {LAMBDAS} × seed ∈ {SEEDS} = {len(LAMBDAS) * len(SEEDS)} 组\n")

    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            print(f"[λ={lam}, seed={seed}]")
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=N_EPOCHS, batch=512, lr=2e-3,
                           lambda_phys=lam, n_collocation=2048, seed=seed)
            net.eval()
            horizons = evaluate(net)
            full_med  = float(np.median(horizons))
            full_mean = float(np.mean(horizons))
            chaos = horizons[horizons < QUASI]
            ch_med  = float(np.median(chaos)) if len(chaos) else np.nan
            print(f"  全集中位 {full_med:.3f}s  N_chaos={len(chaos)}  chaos中位 {ch_med:.3f}s")
            rows.append({
                "lambda_phys": lam, "seed": seed,
                "full_median": full_med, "full_mean": full_mean,
                "n_chaos": int(len(chaos)),
                "chaos_median": ch_med,
            })
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== 按 λ 聚合 ===")
    agg = df.groupby("lambda_phys").agg(
        median_of_full_median=("full_median", "median"),
        mean_of_full_median=("full_median", "mean"),
        std_of_full_median=("full_median", "std"),
        min_full_median=("full_median", "min"),
        max_full_median=("full_median", "max"),
    )
    print(agg)
    agg.to_csv(OUT / "summary.csv")

    print("\n=== v2.1 反直觉发现是否稳健 ===")
    lam0 = df[df.lambda_phys == 0.0].full_median.values
    lam1 = df[df.lambda_phys == 1.0].full_median.values
    print(f"  λ=0 视界中位 5 seed: {lam0.round(3).tolist()}  → 均 {lam0.mean():.3f} ± {lam0.std():.3f}")
    print(f"  λ=1 视界中位 5 seed: {lam1.round(3).tolist()}  → 均 {lam1.mean():.3f} ± {lam1.std():.3f}")
    wins = (lam0 > lam1).sum()
    print(f"  λ=0 在 {wins}/5 个 seed 上视界 > λ=1")
    if wins >= 4:
        print("  → 发现稳健，phys loss 在干净仿真数据上确实没帮助")
    elif wins >= 3:
        print("  → 发现弱稳健，可能受 seed 影响显著")
    else:
        print("  → 发现不稳健，原 paper 论点保留")


if __name__ == "__main__":
    main()

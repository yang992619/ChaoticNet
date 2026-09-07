"""
tools/lambda_ablation_distmass.py — tab:ablation-lambda 的分布质量(B)重跑

忠实对齐论文 tab:ablation-lambda 的实验设计，只把数据源换成分布质量：
  λ_phys ∈ {0.0, 1.0, 2.0}，每个 λ 跑 5 个 seed {42, 7, 13, 21, 100}
  PINN 128×4，200 epoch，batch 512，lr 2e-3，n_collocation 2048
  训练用 trial 0-4（与 baseline 一致），评估 50 trial 视界中位
  data/sim 已是分布质量 B，故本脚本产出即 B 口径

每格同时报 full_median（全 50 trial）与 chaos_median（own-chaos <9.9s 子集），
便于与主表的混沌子集口径对齐。

输出 data/ablation_distmass/lambda.csv（不覆盖点质量 data/multi_seed/）
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod        # noqa: E402

DATA = ROOT / "data" / "sim"
OUT  = ROOT / "data" / "ablation_distmass"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.0, 1.0, 2.0]
SEEDS = [42, 7, 13, 21, 100]
N_EPOCHS = 200
QUASI = 9.9


def evaluate(net):
    horizons = []
    for i in range(50):
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[2400].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, 1200, dt=1 / 120)
        truth = df[["th1", "th2"]].values[2400:3600]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=120)
        horizons.append(h)
    return np.array(horizons)


def main():
    states, accs = pinn_mod.load_dataset(range(5), train_sec=20.0)
    rows = []
    print(f"{'lam':>5} {'seed':>5} | {'full_med':>8} {'chaos_med':>9} {'n_chaos':>7}")
    print("-" * 46)
    for lam in LAMBDAS:
        for seed in SEEDS:
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=N_EPOCHS, batch=512, lr=2e-3,
                           lambda_phys=lam, n_collocation=2048, seed=seed)
            net.eval()
            h = evaluate(net)
            chaos = h[h < QUASI]
            row = {
                "lam": lam, "seed": seed,
                "full_median": float(np.median(h)),
                "chaos_median": float(np.median(chaos)) if len(chaos) else np.nan,
                "n_chaos": int(len(chaos)),
            }
            rows.append(row)
            print(f"{lam:>5} {seed:>5} | {row['full_median']:>8.4f} "
                  f"{row['chaos_median']:>9.4f} {row['n_chaos']:>7}")
            pd.DataFrame(rows).to_csv(OUT / "lambda.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== 按 λ 聚合（5 seed）===")
    agg = df.groupby("lam").agg(
        full_mean=("full_median", "mean"), full_std=("full_median", "std"),
        chaos_mean=("chaos_median", "mean"), chaos_std=("chaos_median", "std"))
    print(agg.to_string())
    agg.to_csv(OUT / "lambda_summary.csv")
    print(f"\n写出 {OUT}/lambda.csv + lambda_summary.csv")


if __name__ == "__main__":
    main()

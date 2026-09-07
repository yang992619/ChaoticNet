"""
tools/fair_compare.py — 四方公平对比

修复 hp_sweep 暴露的 metric bug：原 paper 表 1 用"ESN+PINN 二方都未跑满"筛 N=29，
hp_sweep 用全 50 trial。两者不可比。

本脚本统一口径：
  1. 训练 λ=0 PINN（"pure-data MLP"）300 epoch，跟原 baseline 同 epoch
  2. 在 50 trial 评估，得 lam0 50 个 horizon
  3. merge ESN + PINN + Hybrid + lam0 四列
  4. 共同筛选"四方都 < 9.9s"作为四方一致混沌 trial
  5. 在这个共同 N 上对比四方 horizon 中位

如果 λ=0 在共同 N 上仍超过 PINN 显著差距，paper 故事确需调整；
若差距小或反向，hp_sweep 的"赢"是统计假象（N 不一致造成），可保留 paper 论点。
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
OUT  = ROOT / "data" / "fair_compare"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SEC = 20.0
PRED_SEC  = 10.0
FPS = 120
QUASI = 9.9
N_TRIALS = 50


def train_lam0(n_epochs=300, seed=42):
    print(f"=== 训练 λ=0 PINN（pure-data MLP）{n_epochs} epoch ===")
    states, accs = pinn_mod.load_dataset(range(5), TRAIN_SEC)
    net = pinn_mod.make_accel_net(seed=seed, hidden=128, n_layers=4)
    pinn_mod.train(net, states, accs, n_epochs=n_epochs, batch=512, lr=2e-3,
                   lambda_phys=0.0, n_collocation=2048, seed=seed)
    torch.save(net.state_dict(), OUT / "lam0_model.pt")
    net.eval()
    return net


def evaluate_50(net, name):
    print(f"\n=== {name} 50 trial 评估 ===")
    rows = []
    for i in range(N_TRIALS):
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        n_train = int(TRAIN_SEC * FPS)
        n_pred = int(PRED_SEC * FPS)
        s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / FPS)
        pred_th = pred[:, [0, 2]]
        truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
        h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=FPS)
        rows.append({"trial": i, "horizon": h_s})
    df = pd.DataFrame(rows)
    print(f"  全集中位 {df.horizon.median():.3f}s  N_chaos(<9.9) = {(df.horizon < QUASI).sum()}")
    return df


def main():
    # 1. 训练 + 评估 λ=0
    net0 = train_lam0(n_epochs=300)
    lam0 = evaluate_50(net0, "λ=0 PINN")
    lam0.to_csv(OUT / "lam0_summary_50.csv", index=False)

    # 2. 加载已有结果
    base = pd.read_csv(ROOT / "data" / "horizon_compare_50.csv")
    hyb  = pd.read_csv(ROOT / "data" / "hybrid" / "hybrid_summary_50.csv")[["trial", "horizon"]]
    hyb  = hyb.rename(columns={"horizon": "hybrid"})

    # 3. merge 四列
    m = base.merge(hyb, on="trial").merge(
        lam0.rename(columns={"horizon": "lam0"}), on="trial",
    )[["trial", "esn", "pinn", "hybrid", "lam0"]]
    m.to_csv(OUT / "four_way_horizons.csv", index=False)

    # 4. 共同筛选：四方都 < 9.9s
    mask = (m.esn < QUASI) & (m.pinn < QUASI) & (m.hybrid < QUASI) & (m.lam0 < QUASI)
    common = m[mask]
    print(f"\n=== 四方一致混沌 N = {len(common)} / {len(m)} ===")
    for col in ["esn", "pinn", "hybrid", "lam0"]:
        h = common[col]
        print(f"  {col.upper():>7}  中位 {h.median():.3f}s   均值 {h.mean():.3f}s")
    print(f"\n  PINN  / ESN  中位 {(common.pinn / common.esn).median():.2f}×")
    print(f"  Hybrid / ESN 中位 {(common.hybrid / common.esn).median():.2f}×")
    print(f"  lam0   / ESN 中位 {(common.lam0 / common.esn).median():.2f}×")
    print(f"  PINN / lam0 中位提升 {(common.pinn / common.lam0).median():.2f}× (>1 → phys 有用)")
    print(f"  Hybrid / lam0 中位提升 {(common.hybrid / common.lam0).median():.2f}×")

    # 5. 对全集（不剔除准周期）的同样统计，供横向对比
    print(f"\n=== 全集 N = {len(m)} 中位（不剔除准周期）===")
    for col in ["esn", "pinn", "hybrid", "lam0"]:
        h = m[col]
        print(f"  {col.upper():>7}  中位 {h.median():.3f}s   均值 {h.mean():.3f}s   "
              f"N_chaos({h.lt(QUASI).sum()}/{len(h)})")

    # 6. 写出汇总
    summary = pd.DataFrame({
        "model": ["ESN", "PINN", "Hybrid", "lam0_PINN"],
        "n_common_chaos": [len(common)] * 4,
        "horizon_median_common":  [common.esn.median(), common.pinn.median(),
                                    common.hybrid.median(), common.lam0.median()],
        "horizon_mean_common":    [common.esn.mean(),   common.pinn.mean(),
                                    common.hybrid.mean(),  common.lam0.mean()],
        "horizon_median_full50":  [m.esn.median(), m.pinn.median(),
                                    m.hybrid.median(), m.lam0.median()],
    })
    summary.to_csv(OUT / "fair_summary.csv", index=False)
    print(f"\n写出 {OUT}/four_way_horizons.csv + fair_summary.csv + lam0_summary_50.csv")


if __name__ == "__main__":
    main()

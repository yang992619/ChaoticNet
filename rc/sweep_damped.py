"""
PINN vs Hybrid 在阻尼数据 baseline_damped 上的 50 组对比。

设计：
  - GT 数据 data/sim_damped/trial_xxx.csv（含粘性阻尼 b·ω）
  - 已知物理 prior：无阻尼解析 RHS（≈ 真实 ω̇ 主导项）
  - PINN: 学完整 ω̇（数据 + 拉格朗日 phys loss）
  - Hybrid: ω̇ = prior + residual_NN，残差应学到 ~ -b·ω/I

输出：
  data/hybrid_damped/{pinn_summary_damped.csv, hybrid_summary_damped.csv, compare.png}
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod      # noqa: E402
import hybrid as hyb_mod     # noqa: E402

DATA = ROOT / "data" / "sim_damped"
OUT  = ROOT / "data" / "hybrid_damped"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SEC = 20.0
PRED_SEC = 10.0
FPS = 120
N_TRIALS = 50


def load_damped_dataset(trial_indices, train_sec=20.0):
    states, accs = [], []
    for i in trial_indices:
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        n_train = int(train_sec * FPS)
        df = df.iloc[:n_train]
        s = df[["th1", "w1", "th2", "w2"]].values
        w = df[["w1", "w2"]].values
        wd = np.gradient(w, 1.0 / FPS, axis=0)
        states.append(s[1:-1])
        accs.append(wd[1:-1])
    return (np.concatenate(states, 0).astype(np.float32),
            np.concatenate(accs, 0).astype(np.float32))


def run_one(net, trial_idx, train_sec=20.0, pred_sec=10.0, fps=120):
    df = pd.read_csv(DATA / f"trial_{trial_idx:03d}.csv")
    n_train = int(train_sec * fps)
    n_pred = int(pred_sec * fps)
    s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
    pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / fps)
    pred_th = pred[:, [0, 2]]
    truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
    h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=fps)
    return {"trial": trial_idx, "horizon": h_s,
            "mse_th1": float(np.mean((pred[:, 0] - truth_th[:, 0]) ** 2)),
            "mse_th2": float(np.mean((pred[:, 2] - truth_th[:, 1]) ** 2))}


def main():
    print(f"加载阻尼数据集（trial 0-4 训练）...")
    states, accs = load_damped_dataset(range(5), TRAIN_SEC)
    print(f"  {len(states)} 个 (state, ω̇_with_damping) 训练对")

    # 1. PINN
    print("\n=== PINN 训练（学完整 ω̇ 含阻尼）===")
    pinn_net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    pinn_mod.train(pinn_net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                   lambda_phys=0.1, n_collocation=2048)
    torch.save(pinn_net.state_dict(), OUT / "pinn_damped_model.pt")

    print("\n=== PINN sweep 50 组（阻尼 GT）===")
    pinn_res = []
    for i in range(N_TRIALS):
        r = run_one(pinn_net, i)
        pinn_res.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TRIALS}  horizon={r['horizon']:.2f}s")
    pinn_df = pd.DataFrame(pinn_res)
    pinn_df.to_csv(OUT / "pinn_summary_damped.csv", index=False)

    # 2. Hybrid
    print("\n=== Hybrid 训练（学 ω̇ - prior 残差）===")
    hyb_net = hyb_mod.HybridNet(hidden=64, n_layers=3,
                                 residual_scale=hyb_mod.RESIDUAL_SCALE)
    hyb_mod.train_hybrid(hyb_net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                          lambda_phys=0.1, n_collocation=1024)
    torch.save(hyb_net.state_dict(), OUT / "hybrid_damped_model.pt")

    print("\n=== Hybrid sweep 50 组（阻尼 GT）===")
    hyb_res = []
    for i in range(N_TRIALS):
        r = run_one(hyb_net, i)
        hyb_res.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TRIALS}  horizon={r['horizon']:.2f}s")
    hyb_df = pd.DataFrame(hyb_res)
    hyb_df.to_csv(OUT / "hybrid_summary_damped.csv", index=False)

    # 联合汇总
    m = pinn_df[["trial", "horizon"]].rename(columns={"horizon": "pinn"}).merge(
        hyb_df[["trial", "horizon"]].rename(columns={"horizon": "hybrid"}),
        on="trial",
    )
    qm = (m.pinn < 9.9) & (m.hybrid < 9.9)
    chaos = m[qm]
    print(f"\n=== 阻尼场景 50 组结果 ===")
    print(f"两方一致混沌 N={len(chaos)}")
    print(f"  PINN  : 中位 {chaos.pinn.median():.3f}s  均值 {chaos.pinn.mean():.3f}s")
    print(f"  Hybrid: 中位 {chaos.hybrid.median():.3f}s  均值 {chaos.hybrid.mean():.3f}s")
    print(f"  Hybrid/PINN 中位提升 {(chaos.hybrid / chaos.pinn).median():.2f}×")
    m.to_csv(OUT / "compare_damped.csv", index=False)


if __name__ == "__main__":
    main()

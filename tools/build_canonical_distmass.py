"""
tools/build_canonical_distmass.py — 构建分布质量(B) canonical 仿真内对比

一次性、自洽地产出 B 的 canonical 模型 + 四方表（全部来自分布质量 data/sim）：
  - canonical PINN：λ_phys=0.1（λ-sweep 选定：真实迁移最佳、仿真内代价最小）→ data/pinn/model.pt
  - canonical PINN0：λ_phys=0（消融，纯数据）→ data/fair_compare/lam0_model.pt
  - canonical Hybrid：λ_phys=0.5（解析先验+残差）→ data/hybrid/model.pt + hybrid_summary_50.csv
  - ESN：seed=42 逐 trial（保持与原论文同口径，便于审计/bootstrap 管线不变）
写出：
  data/horizon_compare_50.csv         (trial, esn, pinn)
  data/fair_compare/four_way_horizons.csv  (trial, esn, pinn, hybrid, lam0)
  data/fair_compare/lam0_summary_50.csv
诚实口径修正（ESN 多 seed 中位等）走 multirun_distmass + bootstrap，另表呈现。
原冻结表已备份 data/_frozen_backup_20260619/ 与 data/_pointmass_canonical_backup_20260619/。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "tools"))

import pinn as pinn_mod              # noqa: E402
import hybrid as hybrid_mod         # noqa: E402
import esn_seed_variance as esv     # noqa: E402

SIM = ROOT / "data" / "sim"
FC = ROOT / "data" / "fair_compare"; FC.mkdir(parents=True, exist_ok=True)
HY = ROOT / "data" / "hybrid"; HY.mkdir(parents=True, exist_ok=True)
PI = ROOT / "data" / "pinn"; PI.mkdir(parents=True, exist_ok=True)

esv.DATA = SIM   # ESN 读分布质量数据

TRAIN_SEC, PRED_SEC, FPS = 20.0, 10.0, 120
N_TRAIN, N_PRED = 2400, 1200
N_TRIALS, QUASI = 50, 9.9
SEED = 42
LAM_CANON = 0.1
TAU_L = 0.762   # 分布质量实测 τ_L 中位（sim/lyapunov.py 重算）


def pinn_eval_50(net):
    net.eval()
    out = []
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
        truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=FPS)
        out.append(h)
    return np.array(out)


def train_pinn(lam, seed=SEED):
    states, accs = pinn_mod.load_dataset(range(5), TRAIN_SEC)
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                   lambda_phys=lam, n_collocation=2048, seed=seed)
    net.eval()
    return net


def main():
    print(f"=== canonical PINN  λ={LAM_CANON} ===")
    pinn_net = train_pinn(LAM_CANON)
    torch.save(pinn_net.state_dict(), PI / "model.pt")
    pinn_h = pinn_eval_50(pinn_net)

    print(f"=== canonical PINN0 λ=0 (消融) ===")
    pinn0_net = train_pinn(0.0)
    torch.save(pinn0_net.state_dict(), FC / "lam0_model.pt")
    lam0_h = pinn_eval_50(pinn0_net)

    print(f"=== canonical Hybrid λ=0.5 ===")
    states, accs = pinn_mod.load_dataset(range(5), TRAIN_SEC)
    hyb = hybrid_mod.HybridNet(hidden=64, n_layers=3, residual_scale=hybrid_mod.RESIDUAL_SCALE)
    hybrid_mod.train_hybrid(hyb, states, accs, n_epochs=200, batch=512, lr=1e-3,
                            lambda_phys=0.5, n_collocation=1024, seed=SEED)
    torch.save(hyb.state_dict(), HY / "model.pt")
    hyb_h = np.array([hybrid_mod.run_one(hyb, i, TRAIN_SEC, PRED_SEC, FPS)["horizon"]
                      for i in range(N_TRIALS)])
    pd.DataFrame({"trial": range(N_TRIALS), "horizon": hyb_h}).to_csv(
        HY / "hybrid_summary_50.csv", index=False)

    print(f"=== ESN seed={SEED} 逐 trial ===")
    esn_h = np.array([esv.horizon_for(i, seed=SEED) for i in range(N_TRIALS)])

    # 写表
    hc = pd.DataFrame({"trial": range(N_TRIALS), "esn": esn_h, "pinn": pinn_h})
    hc.to_csv(ROOT / "data" / "horizon_compare_50.csv", index=False)
    fw = pd.DataFrame({"trial": range(N_TRIALS), "esn": esn_h, "pinn": pinn_h,
                       "hybrid": hyb_h, "lam0": lam0_h})
    fw.to_csv(FC / "four_way_horizons.csv", index=False)
    pd.DataFrame({"trial": range(N_TRIALS), "horizon": lam0_h}).to_csv(
        FC / "lam0_summary_50.csv", index=False)

    # 子集 + 统计
    two = fw[(fw.esn < QUASI) & (fw.pinn < QUASI)]
    three = fw[(fw.esn < QUASI) & (fw.pinn < QUASI) & (fw.hybrid < QUASI)]
    four = fw[(fw.esn < QUASI) & (fw.pinn < QUASI) & (fw.hybrid < QUASI) & (fw.lam0 < QUASI)]
    print("\n" + "=" * 60)
    print(f"两方混沌 N={len(two)} | 三方 N={len(three)} | 四方 N={len(four)}")
    print("=" * 60)
    for col in ["esn", "pinn", "hybrid", "lam0"]:
        sub = three if col == "hybrid" else two
        h = sub[col]
        print(f"  {col.upper():>7}  中位 {h.median():.3f}s  均值 {h.mean():.3f}s  /τL {h.median()/TAU_L:.2f}")
    print(f"\n  PINN/ESN   中位 {(two.pinn/two.esn).median():.2f}×  (medians {two.pinn.median():.3f}/{two.esn.median():.3f}={two.pinn.median()/two.esn.median():.2f}×)")
    print(f"  Hybrid/ESN 中位 {(three.hybrid/three.esn).median():.2f}×")
    print(f"  PINN/lam0  中位 {(four.pinn/four.lam0).median():.2f}× (>1→物理损失仿真内有用; 预期≈1 或略<1)")
    print(f"\n写出 four_way_horizons.csv / horizon_compare_50.csv / hybrid_summary_50.csv / model.pt×3")


if __name__ == "__main__":
    main()

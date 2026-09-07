"""
tools/pinn_epoch_transfer.py — PINN 训练时长 vs sim→real 迁移 的权衡曲线

疑点：缓存 data/pinn/model.pt 仿真内只 0.217s 却迁移到真实 0.467s；
我新训 300ep 仿真内 0.738s 却迁移真实仅 0.283s。
假设：训得越久越过拟合干净（无阻尼）仿真 → 迁移到带阻尼真实越差；
存在一个 early-stop 甜点，仿真内偏弱但真实迁移最好（≈复现缓存的 0.467）。

对每个 (epoch, seed) 训 λ=1 PINN，测：
  - 仿真内 chaos 子集（four_way esn<9.9&pinn<9.9, N=29）中位视界
  - 真实 29 片（剔1392）中位视界
画出权衡，定迁移最优 epoch。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "tools"))

import pinn as pinn_mod              # noqa: E402
import real_validation_multi as rv  # noqa: E402

SIM = ROOT / "data" / "sim"
OUT = ROOT / "data" / "epoch_transfer"
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = [30, 60, 100, 150, 200, 300, 500]
SEEDS = [42, 7]
REAL_TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
             '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
             '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
             '1459', '1460']
LEAD_S, PRED_S = 4.0, 4.0


def sim_chaos_median(net):
    fw = pd.read_csv(ROOT / "data/fair_compare/four_way_horizons.csv")
    chaos = sorted(fw.trial[(fw.esn < 9.9) & (fw.pinn < 9.9)].astype(int))
    hs = []
    for i in chaos:
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[2400].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, 1200, dt=1 / 120)
        truth = df[["th1", "th2"]].values[2400:3600]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=120)
        hs.append(h)
    return float(np.median(hs))


def real_median(net, clips):
    hs = []
    for C in clips.values():
        i0, n, fps, dt = C['lead_n'], C['pred_n'], C['fps'], C['dt']
        s0 = C['state'][i0].astype(np.float32)
        p1, p2 = rv.run_torch(net, s0, n, dt)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        h, _ = rv.horizon_s(e, fps)
        hs.append(h)
    return float(np.median(hs))


def main():
    clips = {tg: rv.load_clip(tg, LEAD_S, PRED_S) for tg in REAL_TAGS}
    clips = {k: v for k, v in clips.items() if v['pred_n'] >= 10}
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=20.0)

    rows = []
    print(f"{'epoch':>6} {'seed':>5} | {'sim_chaos':>9} | {'real_med':>8}")
    print("-" * 40)
    for ep in EPOCHS:
        for sd in SEEDS:
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=ep, batch=512, lr=2e-3,
                           lambda_phys=1.0, n_collocation=2048, seed=sd)
            net.eval()
            sm = sim_chaos_median(net)
            rm = real_median(net, clips)
            rows.append({"epoch": ep, "seed": sd, "sim_chaos": sm, "real": rm})
            print(f"{ep:>6} {sd:>5} | {sm:>9.4f} | {rm:>8.4f}")
            pd.DataFrame(rows).to_csv(OUT / "epoch_transfer.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== 按 epoch 聚合（2 seed 均值）===")
    agg = df.groupby("epoch").agg(sim=("sim_chaos", "mean"), real=("real", "mean"))
    print(agg.to_string())
    best = agg.real.idxmax()
    print(f"\n真实迁移最优 epoch = {best}  (real={agg.real.max():.4f})")
    print("缓存模型 data/pinn/model.pt: sim 0.217 / real 0.467 作参照")


if __name__ == "__main__":
    main()

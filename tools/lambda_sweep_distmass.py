"""
tools/lambda_sweep_distmass.py — 分布质量下物理损失权重 λ 的 in-sim vs real 权衡

背景：B(分布质量)下，λ=1 的 PINN 仿真内反而被物理损失拖低(0.66 vs λ0 的 1.58)，
但真实迁移 λ=1(0.40) > λ0(0.375)。需要一个 λ 在"仿真内准"和"迁移好"间取平衡，
既作 canonical PINN 的取值依据，也作论文的物理损失消融。

对每个 (λ, seed)：300ep 训练（data/sim 已是分布质量），测
  - 仿真内 own-chaos(<9.9s) 中位 + all-50 中位
  - 真实 29 片中位
输出 data/lambda_sweep/lambda_sweep.csv
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
OUT = ROOT / "data" / "lambda_sweep"
OUT.mkdir(parents=True, exist_ok=True)

N_TRIALS = 50
FPS = 120
N_TRAIN = 2400
N_PRED = 1200
QUASI = 9.9

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0]
SEEDS = [42, 7]
REAL_TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
             '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
             '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
             '1459', '1460']


def sim_medians(net):
    net.eval()
    hs = []
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
        truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=FPS)
        hs.append(h)
    hs = np.array(hs)
    ch = hs[hs < QUASI]
    return (float(np.median(ch)) if len(ch) else float("nan"), float(np.median(hs)))


def real_median(net, clips):
    hs = []
    for C in clips.values():
        i0, n, dt = C['lead_n'], C['pred_n'], C['dt']
        s0 = C['state'][i0].astype(np.float32)
        p1, p2 = rv.run_torch(net, s0, n, dt)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        h, _ = rv.horizon_s(e, C['fps'])
        hs.append(h)
    return float(np.median(hs))


def main():
    clips = {tg: rv.load_clip(tg, 4.0, 4.0) for tg in REAL_TAGS}
    clips = {k: v for k, v in clips.items() if v['pred_n'] >= 10}
    print(f"真实片 {len(clips)} 条")
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=20.0)

    rows = []
    print(f"{'lam':>5} {'seed':>5} | {'sim_chaos':>9} {'sim_all':>8} | {'real':>6}")
    print("-" * 46)
    for lam in LAMBDAS:
        for sd in SEEDS:
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                           lambda_phys=lam, n_collocation=2048, seed=sd)
            sc, sa = sim_medians(net)
            rm = real_median(net, clips)
            rows.append({"lam": lam, "seed": sd, "sim_chaos": sc, "sim_all": sa, "real": rm})
            print(f"{lam:>5} {sd:>5} | {sc:>9.4f} {sa:>8.4f} | {rm:>6.4f}")
            pd.DataFrame(rows).to_csv(OUT / "lambda_sweep.csv", index=False)

    df = pd.DataFrame(rows)
    agg = df.groupby("lam").agg(sim_chaos=("sim_chaos", "mean"),
                                sim_all=("sim_all", "mean"),
                                real=("real", "mean"))
    print("\n=== 按 λ 聚合（2 seed 均值）===")
    print(agg.to_string())
    print("\nESN 参照: 仿真 own-chaos 0.371 / 真实 0.167")
    print("挑 canonical λ：真实尽量高、仿真内别被拖太低。")


if __name__ == "__main__":
    main()

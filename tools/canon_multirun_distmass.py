"""
tools/canon_multirun_distmass.py — B(分布质量) canonical 诚实多 run（仿真内 + 真实 一体）

PINN 训练非确定性大（同配置单次抽样可在 0.4–1.3 间跳），故所有报告数取多 run 中位 + 跨 run 分布，
而非单次抽样。每个网络只训一次，同时评 仿真内(50 trial 逐条) 与 真实(29 片中位)，口径一致。

模型：
  ESN(seed×10)        ：仿真内 esn_seed_variance.horizon_for；真实 real_validation_multi.run_esn
  PINN λ=0.1 (×8)     ：canonical（λ-sweep 选定）
  PINN λ=0   (×8)     ：消融（纯数据，无物理损失）
  Hybrid λ=0.5(×8)    ：解析先验 + 残差
数据：data/sim（已分布质量）；真实：data/tracking 29 片（剔 1392/1433/1457）。

输出 data/canon_multirun/{insim_runs,real_runs,summary,four_way_perrun_median}.csv
并据「逐 trial 跨 run 中位」重建 data/fair_compare/four_way_horizons.csv（覆盖单抽样版，原版已备份）。
保存最接近中位的 λ0.1 run 权重为 data/pinn/model.pt（供真实验证复跑）。
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
import real_validation_multi as rv  # noqa: E402

SIM = ROOT / "data" / "sim"
OUT = ROOT / "data" / "canon_multirun"; OUT.mkdir(parents=True, exist_ok=True)
FC = ROOT / "data" / "fair_compare"; FC.mkdir(parents=True, exist_ok=True)
PI = ROOT / "data" / "pinn"; PI.mkdir(parents=True, exist_ok=True)
esv.DATA = SIM

N_TRIALS, FPS, QUASI = 50, 120, 9.9
N_TRAIN, N_PRED = 2400, 1200
TAU_L = 0.762

ESN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]
RUN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3]
TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
        '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
        '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
        '1459', '1460']


def insim_pinn(net):
    net.eval(); hs = []
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
        truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=FPS)
        hs.append(h)
    return np.array(hs)


def insim_hybrid(net):
    net.eval()
    return np.array([hybrid_mod.run_one(net, i, 20.0, 10.0, FPS)["horizon"]
                     for i in range(N_TRIALS)])


def real_net(net, clips):
    hs = []
    for C in clips.values():
        i0, n, dt = C['lead_n'], C['pred_n'], C['dt']
        s0 = C['state'][i0].astype(np.float32)
        p1, p2 = rv.run_torch(net, s0, n, dt)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        h, _ = rv.horizon_s(e, C['fps']); hs.append(h)
    return np.array(hs)


def real_esn(clips, seed):
    hs = []
    for C in clips.values():
        i0, n = C['lead_n'], C['pred_n']
        p1, p2 = rv.run_esn(C, seed=seed)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        h, _ = rv.horizon_s(e, C['fps']); hs.append(h)
    return np.array(hs)


def train_pinn(lam, seed):
    states, accs = pinn_mod.load_dataset(range(5), 20.0)
    net = pinn_mod.make_accel_net(seed=seed, hidden=128, n_layers=4)
    pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                   lambda_phys=lam, n_collocation=2048, seed=seed)
    return net


def train_hyb(seed):
    states, accs = pinn_mod.load_dataset(range(5), 20.0)
    net = hybrid_mod.make_hybrid_net(seed=seed, hidden=64, n_layers=3)
    hybrid_mod.train_hybrid(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                            lambda_phys=0.5, n_collocation=1024, seed=seed)
    return net


def own_chaos(h):
    c = h[h < QUASI]
    return float(np.median(c)) if len(c) else float("nan")


def main():
    clips = {tg: rv.load_clip(tg, 4.0, 4.0) for tg in TAGS}
    clips = {k: v for k, v in clips.items() if v['pred_n'] >= 10}
    print(f"真实片 {len(clips)} 条\n")

    insim_rows, real_rows = [], []      # long form
    insim_mat = {}                      # model -> list of per-trial arrays (for per-trial median)
    real_runmed = {}                    # model -> list of per-run real medians
    insim_runmed = {}                   # model -> list of per-run own-chaos medians
    saved = {"best_seed": None, "best_gap": 1e9, "state": None}

    # ---- ESN ----
    print(f"=== ESN {len(ESN_SEEDS)} seeds ===")
    insim_mat["esn"] = []; real_runmed["ESN"] = []; insim_runmed["ESN"] = []
    for s in ESN_SEEDS:
        hi = np.array([esv.horizon_for(i, seed=s) for i in range(N_TRIALS)])
        hr = real_esn(clips, s)
        insim_mat["esn"].append(hi)
        insim_runmed["ESN"].append(own_chaos(hi)); real_runmed["ESN"].append(float(np.median(hr)))
        for i in range(N_TRIALS): insim_rows.append({"model": "ESN", "run": s, "trial": i, "horizon": float(hi[i])})
        for tg, h in zip(clips, hr): real_rows.append({"model": "ESN", "run": s, "tag": tg, "horizon": float(h)})
        print(f"  seed={s:>3}: insim_chaos {insim_runmed['ESN'][-1]:.3f}  real {real_runmed['ESN'][-1]:.3f}")

    # ---- PINN λ0.1 (canonical), PINN λ0, Hybrid ----
    specs = [("pinn", "PINN", 0.1), ("lam0", "PINN0", 0.0), ("hybrid", "Hybrid", None)]
    cross_med_real = {}
    for key, label, lam in specs:
        print(f"\n=== {label} {len(RUN_SEEDS)} runs ===")
        insim_mat[key] = []; real_runmed[label] = []; insim_runmed[label] = []
        for s in RUN_SEEDS:
            if key == "hybrid":
                net = train_hyb(s); hi = insim_hybrid(net)
            else:
                net = train_pinn(lam, s); hi = insim_pinn(net)
            hr = real_net(net, clips)
            insim_mat[key].append(hi)
            insim_runmed[label].append(own_chaos(hi)); real_runmed[label].append(float(np.median(hr)))
            for i in range(N_TRIALS): insim_rows.append({"model": label, "run": s, "trial": i, "horizon": float(hi[i])})
            for tg, h in zip(clips, hr): real_rows.append({"model": label, "run": s, "tag": tg, "horizon": float(h)})
            print(f"  seed={s:>3}: insim_chaos {insim_runmed[label][-1]:.3f}  real {real_runmed[label][-1]:.3f}")
        # 保存最接近 in-sim 中位的 λ0.1 run 作 canonical model
        if key == "pinn":
            cm = float(np.median(insim_runmed["PINN"]))
            for s, m in zip(RUN_SEEDS, insim_runmed["PINN"]):
                if abs(m - cm) < saved["best_gap"]:
                    saved.update(best_gap=abs(m - cm), best_seed=s)
            # 重训该 seed 存权重（确定取到对应模型）
            net = train_pinn(0.1, saved["best_seed"]); torch.save(net.state_dict(), PI / "model.pt")
            print(f"  [canonical model.pt = λ0.1 seed={saved['best_seed']} (closest to median {cm:.3f})]")

    pd.DataFrame(insim_rows).to_csv(OUT / "insim_runs.csv", index=False)
    pd.DataFrame(real_rows).to_csv(OUT / "real_runs.csv", index=False)

    # ---- 逐 trial 跨 run 中位 → 重建 four_way ----
    def pertrial_median(key):
        return np.median(np.vstack(insim_mat[key]), axis=0)
    fw = pd.DataFrame({"trial": range(N_TRIALS),
                       "esn": pertrial_median("esn"), "pinn": pertrial_median("pinn"),
                       "hybrid": pertrial_median("hybrid"), "lam0": pertrial_median("lam0")})
    fw.to_csv(FC / "four_way_horizons.csv", index=False)

    two = fw[(fw.esn < QUASI) & (fw.pinn < QUASI)]
    three = fw[(fw.esn < QUASI) & (fw.pinn < QUASI) & (fw.hybrid < QUASI)]

    def st(v):
        a = np.array(v, float)
        return dict(median=float(np.median(a)), q25=float(np.percentile(a, 25)),
                    q75=float(np.percentile(a, 75)), lo=float(a.min()), hi=float(a.max()), n=len(a))

    print("\n" + "=" * 72)
    print("B canonical（逐 trial 跨 run 中位重建 four_way）")
    print("=" * 72)
    print(f"两方混沌 N={len(two)}  三方 N={len(three)}  (τ_L={TAU_L})")
    for col, sub in [("esn", two), ("pinn", two), ("hybrid", three), ("lam0", two)]:
        h = sub[col]
        print(f"  {col.upper():>7} 中位 {h.median():.3f}s  均值 {h.mean():.3f}s  /τL {h.median()/TAU_L:.2f}")
    print(f"  PINN/ESN 中位 {two.pinn.median()/two.esn.median():.2f}×   Hybrid/ESN {three.hybrid.median()/three.esn.median():.2f}×")

    print("\n--- 跨 run 分布（per-run 中位的 median[IQR]）---")
    summ = []
    for label in ["ESN", "PINN", "PINN0", "Hybrid"]:
        si = st(insim_runmed[label]); sr = st(real_runmed[label])
        print(f"  {label:>7} | insim {si['median']:.3f}[{si['q25']:.3f},{si['q75']:.3f}]"
              f" | real {sr['median']:.3f}[{sr['q25']:.3f},{sr['q75']:.3f}]")
        summ.append({"model": label, "metric": "insim_chaos", **si})
        summ.append({"model": label, "metric": "real", **sr})
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False)

    er = st(real_runmed["ESN"])["median"]; ei = st(insim_runmed["ESN"])["median"]
    print("\n=== 倍数（跨 run 中位 / ESN）===")
    for label in ["PINN", "PINN0", "Hybrid"]:
        print(f"  {label}: insim {st(insim_runmed[label])['median']/ei:.2f}×  real {st(real_runmed[label])['median']/er:.2f}×")
    pp = st(real_runmed["PINN"])["median"]; p0 = st(real_runmed["PINN0"])["median"]
    print(f"  物理损失真实增益 PINN(λ0.1)/PINN0(λ0) = {pp/p0:.2f}×")
    print(f"\n写出 {OUT}/ + 重建 four_way_horizons.csv + data/pinn/model.pt")


if __name__ == "__main__":
    main()

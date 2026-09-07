"""
tools/multirun_compare.py — 多 run 公平对比（诚实重基线）

⚠ 非论文头条来源：本脚本是【仿真内】诚实重基线工具，下文引用的 3.4× / 0.15s / 0.642s
均为被本工具证伪的【旧点质量单次抽样】值（故意留作动机说明，非现行口径）。
现行权威口径：仿真内见 data/canon_multirun/，真实零样本见 data/canonical_results_B.md。

动机（2026-06-19 发现）：
  论文头条 PINN/ESN≈3.4× 用的是 ESN seed=42（16 seed 里最弱的一个，0.15s）
  和 PINN 缓存模型（0.642s，而 PINN 训练非确定性，同 seed 重训只有 ~0.45s）。
  两端都是对"PINN 胜 ESN"叙事最有利的单次抽样 → 头条被高估。

诚实做法：每个模型多次独立训练（ESN 换 reservoir seed；PINN/Hybrid 非确定性，
重复训练），每个 run 报「该 run 自己的混沌 trial（horizon<9.9s）上的中位视界」，
再看这个 per-run 中位在 run 间的分布（中位 + IQR）。
  - 用「每 run 自己的 <9.9s 子集」避免「子集由单一模型/seed 定义」的选择偏差。
  - 另报每 run 的 all-50 中位作为完全客观的交叉验证。

输出 data/multirun/*.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))

import esn as esn_mod          # noqa: E402
import pinn as pinn_mod        # noqa: E402
import hybrid as hybrid_mod    # noqa: E402
from esn_seed_variance import horizon_for as esn_horizon_for  # noqa: E402

SIM = ROOT / "data" / "sim"
OUT = ROOT / "data" / "multirun"
OUT.mkdir(parents=True, exist_ok=True)

N_TRIALS = 50
FPS = 120
TRAIN_SEC = 20.0
PRED_SEC = 10.0
N_TRAIN = int(TRAIN_SEC * FPS)   # 2400
N_PRED = int(PRED_SEC * FPS)     # 1200
THRESH_DEG = 10.0
QUASI = 9.9

ESN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8, 11, 17, 23, 33, 55, 99]
TRAIN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]   # PINN/Hybrid 多 run


# ---------- 单 net 在 50 trial 上的 horizon ----------

def pinn_horizons(net):
    net.eval()
    hs = []
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
        truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=THRESH_DEG, fps=FPS)
        hs.append(h)
    return np.array(hs)


def hybrid_horizons(net):
    net.eval()
    return np.array([hybrid_mod.run_one(net, i, train_sec=TRAIN_SEC,
                                        pred_sec=PRED_SEC, fps=FPS)["horizon"]
                     for i in range(N_TRIALS)])


# ---------- per-run 统计 ----------

def own_chaos_median(h):
    """该 run 自己的混沌 trial（<9.9s）上的中位视界。"""
    ch = h[h < QUASI]
    return float(np.median(ch)) if len(ch) else float("nan")


def all50_median(h):
    return float(np.median(h))


# ---------- 主流程 ----------

def main():
    # ESN：每 seed 一个 horizon 向量（确定性）
    print(f"=== ESN：{len(ESN_SEEDS)} seeds ===")
    esn_rows, esn_own, esn_all = [], [], []
    for s in ESN_SEEDS:
        h = np.array([esn_horizon_for(i, seed=s) for i in range(N_TRIALS)])
        esn_own.append(own_chaos_median(h)); esn_all.append(all50_median(h))
        for i in range(N_TRIALS):
            esn_rows.append({"model": "ESN", "run": s, "trial": i, "horizon": float(h[i])})
        print(f"  seed={s:>3}: own-chaos median {esn_own[-1]:.4f}  all50 {esn_all[-1]:.4f}")
    pd.DataFrame(esn_rows).to_csv(OUT / "esn_runs.csv", index=False)

    # 训练共享数据
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=TRAIN_SEC)

    def train_eval(kind, seed):
        if kind == "PINN1":
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                           lambda_phys=1.0, n_collocation=2048, seed=seed)
            return pinn_horizons(net)
        if kind == "PINN0":
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                           lambda_phys=0.0, n_collocation=2048, seed=seed)
            return pinn_horizons(net)
        if kind == "Hybrid":
            net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                       residual_scale=hybrid_mod.RESIDUAL_SCALE)
            hybrid_mod.train_hybrid(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                                    lambda_phys=0.5, n_collocation=1024, seed=seed)
            return hybrid_horizons(net)

    run_own = {"ESN": esn_own}
    run_all = {"ESN": esn_all}
    for kind in ["PINN1", "PINN0", "Hybrid"]:
        print(f"\n=== {kind}：{len(TRAIN_SEEDS)} runs ===")
        rows, own, allm = [], [], []
        for s in TRAIN_SEEDS:
            h = train_eval(kind, s)
            own.append(own_chaos_median(h)); allm.append(all50_median(h))
            for i in range(N_TRIALS):
                rows.append({"model": kind, "run": s, "trial": i, "horizon": float(h[i])})
            print(f"  run seed={s:>3}: own-chaos median {own[-1]:.4f}  all50 {allm[-1]:.4f}")
        run_own[kind] = own; run_all[kind] = allm
        pd.DataFrame(rows).to_csv(OUT / f"{kind.lower()}_runs.csv", index=False)

    # ---------- 汇总 ----------
    def stats(vals):
        a = np.array(vals, dtype=float)
        return dict(median=float(np.median(a)), mean=float(a.mean()), std=float(a.std()),
                    q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                    lo=float(a.min()), hi=float(a.max()), n=len(a))

    print("\n" + "=" * 70)
    print("诚实多 run 对比 —— 每 run「自己的混沌 trial(<9.9s) 中位视界」的跨 run 分布")
    print("=" * 70)
    print(f"{'model':>8} | {'runs':>4} | {'median':>7} | {'IQR':>15} | {'[min,max]':>15}")
    print("-" * 70)
    rows_out = []
    for k in ["ESN", "PINN1", "PINN0", "Hybrid"]:
        st = stats(run_own[k])
        print(f"{k:>8} | {st['n']:>4} | {st['median']:>7.3f} | "
              f"[{st['q25']:.3f},{st['q75']:.3f}] | [{st['lo']:.3f},{st['hi']:.3f}]")
        rows_out.append({"model": k, "metric": "own_chaos", **st})

    print("\n  (交叉验证) all-50 中位的跨 run 分布：")
    for k in ["ESN", "PINN1", "PINN0", "Hybrid"]:
        st = stats(run_all[k])
        print(f"  {k:>8}: median {st['median']:.3f}  IQR [{st['q25']:.3f},{st['q75']:.3f}]")
        rows_out.append({"model": k, "metric": "all50", **st})

    esn_med = stats(run_own["ESN"])["median"]
    print("\n=== 诚实提升倍数（按 own-chaos run 中位 / ESN run 中位）===")
    for k in ["PINN1", "PINN0", "Hybrid"]:
        km = stats(run_own[k])["median"]
        print(f"  {k}/ESN = {km / esn_med:.2f}×")
    print("  （论文头条用单次最 favorable 抽样 = 3.4×）")

    pd.DataFrame(rows_out).to_csv(OUT / "summary.csv", index=False)
    print(f"\n写出 {OUT}/summary.csv + 各模型 *_runs.csv")


if __name__ == "__main__":
    main()

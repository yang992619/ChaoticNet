"""
tools/multirun_compare_distmass.py — 分布质量(B) 仿真内诚实多 run 对比

镜像 tools/multirun_compare.py 的协议（16 ESN seed / 各 10 run PINN1·PINN0·Hybrid，
per-run own-chaos<9.9 中位的跨 run 分布），唯一区别：
  - 数据用 data/sim_distmass（分布质量仿真）
  - PINN/Hybrid 物理损失参数临时设为分布质量（baseline_distmass）
  - 不动 canonical 点质量 data/sim、不改任何 .py 文件，只在内存里改

目的：补齐决策报告里 B 的「仿真内 PINN/ESN」那一格，使 A vs B 完全可比。
对照（点质量 A，multirun_compare.py 结果）：ESN 0.25 / PINN 0.74 / Hybrid 1.33，
PINN/ESN=2.9×、Hybrid/ESN=5.25×。

输出 data/multirun_distmass/*.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))

import esn as esn_mod              # noqa: E402
import pinn as pinn_mod            # noqa: E402
import hybrid as hybrid_mod        # noqa: E402
import baseline_distmass as bld    # noqa: E402
import esn_seed_variance as esv    # noqa: E402

DIST = ROOT / "data" / "sim_distmass"
OUT = ROOT / "data" / "multirun_distmass"
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
TRAIN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]

# 注：L2 不在列——分布质量 EOM 用 LC2/I2_A（质心距+转动惯量），不用全长 L2。
# pinn/hybrid 里 L2 仅在模块加载时用于定义 LC2/I2_A，而我们直接覆盖 LC2/I2_A，
# 故 L2 残留点质量值无害（lagrange_residual/解析先验都只读 LC2、I2_A）。
PHYS_KEYS = ["G", "L1", "M1", "M2", "LC1", "LC2", "I1_O", "I2_A"]


def set_dist_params(mod):
    """把模块的物理参数（物理损失/解析先验用）设为分布质量。缺键直接报错，不静默。"""
    for k in PHYS_KEYS:
        if not hasattr(bld, k):
            raise AttributeError(f"baseline_distmass 缺少物理参数 {k}")
        setattr(mod, k, getattr(bld, k))


def point_into_distmass():
    """把三模型的数据路径 + 物理参数都指向分布质量（仅内存）。"""
    set_dist_params(pinn_mod)
    set_dist_params(hybrid_mod)
    pinn_mod.DATA = DIST
    hybrid_mod.DATA = DIST
    esv.DATA = DIST            # ESN 复现函数读 esv.DATA（ESN 无物理参数，仅读轨迹）
    print("已把 pinn/hybrid/esn 指向 data/sim_distmass")
    print(f"  pinn  M1={pinn_mod.M1:.4f} M2={pinn_mod.M2:.4f} I1_O={pinn_mod.I1_O:.6f}")
    print(f"  hybrid M1={hybrid_mod.M1:.4f} M2={hybrid_mod.M2:.4f} I1_O={hybrid_mod.I1_O:.6f}")
    print(f"  esn DATA={esv.DATA}\n")


def pinn_horizons(net):
    net.eval()
    hs = []
    for i in range(N_TRIALS):
        df = pd.read_csv(DIST / f"trial_{i:03d}.csv")
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


def own_chaos_median(h):
    ch = h[h < QUASI]
    return float(np.median(ch)) if len(ch) else float("nan")


def all50_median(h):
    return float(np.median(h))


def stats(vals):
    a = np.array(vals, dtype=float)
    return dict(median=float(np.median(a)), mean=float(a.mean()), std=float(a.std()),
                q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                lo=float(a.min()), hi=float(a.max()), n=len(a))


def main():
    point_into_distmass()

    # 健全性：ESN 读分布数据，trial_000 horizon 应与点质量不同
    h0_dist = esv.horizon_for(0, seed=42)
    print(f"[sanity] ESN seed=42 trial_000 (distmass) horizon = {h0_dist:.4f}s "
          f"(点质量基线为 0.1167，应不同)\n")

    # ===== ESN：16 seed（确定性）=====
    print(f"=== ESN：{len(ESN_SEEDS)} seeds（分布质量数据）===")
    esn_rows, esn_own, esn_all = [], [], []
    for s in ESN_SEEDS:
        h = np.array([esv.horizon_for(i, seed=s) for i in range(N_TRIALS)])
        esn_own.append(own_chaos_median(h)); esn_all.append(all50_median(h))
        for i in range(N_TRIALS):
            esn_rows.append({"model": "ESN", "run": s, "trial": i, "horizon": float(h[i])})
        print(f"  seed={s:>3}: own-chaos median {esn_own[-1]:.4f}  all50 {esn_all[-1]:.4f}")
    pd.DataFrame(esn_rows).to_csv(OUT / "esn_runs.csv", index=False)

    # 训练共享数据（分布质量 trial 0-4）
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
        print(f"\n=== {kind}：{len(TRAIN_SEEDS)} runs（分布质量）===")
        rows, own, allm = [], [], []
        for s in TRAIN_SEEDS:
            h = train_eval(kind, s)
            own.append(own_chaos_median(h)); allm.append(all50_median(h))
            for i in range(N_TRIALS):
                rows.append({"model": kind, "run": s, "trial": i, "horizon": float(h[i])})
            print(f"  run seed={s:>3}: own-chaos median {own[-1]:.4f}  all50 {allm[-1]:.4f}")
        run_own[kind] = own; run_all[kind] = allm
        pd.DataFrame(rows).to_csv(OUT / f"{kind.lower()}_runs.csv", index=False)

    print("\n" + "=" * 70)
    print("分布质量(B) 仿真内诚实多 run —— per-run own-chaos(<9.9s) 中位的跨 run 分布")
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
    print("\n=== B 诚实提升倍数（own-chaos run 中位 / ESN run 中位）===")
    for k in ["PINN1", "PINN0", "Hybrid"]:
        km = stats(run_own[k])["median"]
        print(f"  {k}/ESN = {km / esn_med:.2f}×")
    print("  对照 A(点质量): PINN/ESN=2.9× Hybrid/ESN=5.25×")

    pd.DataFrame(rows_out).to_csv(OUT / "summary.csv", index=False)
    print(f"\n写出 {OUT}/summary.csv + 各模型 *_runs.csv")


if __name__ == "__main__":
    main()

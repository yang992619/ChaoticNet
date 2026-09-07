"""
tools/esn_seed_variance.py — ESN reservoir 随机 seed 方差测试

目的：证明 ESN 的"弱视界"（median horizon ~0.15s，点质量期值；分布质量真实零样本
ESN≈0.167s，见 data/canonical_results_B.md）在 reservoir 随机种子间稳健，
不是某个幸运/倒霉的抽样。镜像 tools/multi_seed.py 的汇报风格。

做法：精确复现 rc/esn.py 里 run_one(trial, plot=False) 的全部逻辑
（编码、训练/预测切分、closed-loop 预测、horizon 计算），唯一区别是给
ESN(...) 传入显式 seed。不改 rc/esn.py。

seeds: {42, 7, 13, 21, 100}，每个 seed 跑全部 50 个 trial。

chaos 子集：从 data/fair_compare/four_way_horizons.csv 的 canonical seed=42 基线里
一次性定义为 (esn<9.9 AND pinn<9.9) 的 N=29 两路子集，对所有 seed 固定不变。

输出 data/esn_seed/esn_seed_results.csv（每 seed 每 trial）
    + data/esn_seed/esn_seed_summary.csv（每 seed 中位数）
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import esn as esn_mod  # noqa: E402

DATA = ROOT / "data" / "sim"
OUT = ROOT / "data" / "esn_seed"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_CSV = ROOT / "data" / "fair_compare" / "four_way_horizons.csv"

SEEDS = [42, 7, 13, 21, 100]
N_TRIALS = 50
QUASI = 9.9


def horizon_for(trial_idx, seed, train_sec=20.0, pred_sec=10.0, fps=120):
    """精确复现 esn.run_one(trial_idx, plot=False)['horizon']，但用显式 seed。

    与 rc/esn.py 的 run_one 函数体逐行对应（编码、切分、训练、预测、horizon），
    唯一区别：ESN(...) 多传一个 seed 参数。
    """
    df = pd.read_csv(DATA / f"trial_{trial_idx:03d}.csv")

    # 编码（与 run_one 一致）
    U = esn_mod.encode(df)              # shape (T, 6)
    Y = np.roll(U, -1, axis=0)          # 下一帧目标
    U, Y = U[:-1], Y[:-1]               # 去最后一行

    # 切训练 / 测试（与 run_one 一致）
    n_train = int(train_sec * fps)
    n_pred = int(pred_sec * fps)
    U_tr, Y_tr = U[:n_train], Y[:n_train]
    U_te = U[n_train:n_train + n_pred]  # ground truth

    # 训练 —— 唯一区别：显式 seed
    e = esn_mod.ESN(n_in=6, n_out=6, n_res=800,
                    spectral_radius=0.95, leak=0.25, ridge=1e-5,
                    seed=seed)
    train_mse, x_final = e.train(U_tr, Y_tr, washout=200)

    # 闭环预测（与 run_one 一致）
    pred = e.predict(x_final, U_tr[-1], n_pred)

    # horizon（与 run_one 一致）
    horizon = esn_mod.prediction_horizon(pred, U_te, threshold_deg=10.0, fps=fps)
    return horizon


def sanity_check():
    """seed=42 时复现值必须与 esn.run_one 完全一致（至少 trial 0,1,2，1e-6 内）。"""
    print("=== SANITY CHECK：seed=42 复现 vs esn.run_one ===")
    ok = True
    for trial in (0, 1, 2):
        ref = esn_mod.run_one(trial, plot=False)["horizon"]
        mine = horizon_for(trial, seed=42)
        diff = abs(ref - mine)
        status = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"  trial_{trial:03d}: run_one={ref:.6f}  mine={mine:.6f}  diff={diff:.2e}  [{status}]")
        assert diff < 1e-6, (
            f"复现与 run_one 不一致 trial={trial}: {ref} vs {mine} (diff={diff})"
        )
        ok = ok and (diff < 1e-6)
    print("  → sanity check 通过\n")
    return ok


def load_chaos_subset():
    """从 canonical seed=42 基线一次性定义 chaos 子集：esn<9.9 AND pinn<9.9。"""
    base = pd.read_csv(BASELINE_CSV)
    mask = (base["esn"] < QUASI) & (base["pinn"] < QUASI)
    chaos_trials = sorted(base.loc[mask, "trial"].astype(int).tolist())
    print(f"=== chaos 子集（固定，来自 seed=42 基线 esn<9.9 & pinn<9.9）===")
    print(f"  N_chaos = {len(chaos_trials)}")
    print(f"  trials = {chaos_trials}\n")
    return chaos_trials


def main():
    sanity_ok = sanity_check()
    chaos_trials = load_chaos_subset()
    n_chaos = len(chaos_trials)
    chaos_set = set(chaos_trials)

    rows = []
    per_seed_all50_median = {}
    per_seed_chaos_median = {}

    print(f"=== sweep：seed ∈ {SEEDS} × {N_TRIALS} trials ===")
    for seed in SEEDS:
        horizons = np.array([horizon_for(i, seed=seed) for i in range(N_TRIALS)])
        for i in range(N_TRIALS):
            rows.append({"seed": seed, "trial": i,
                         "horizon": float(horizons[i]),
                         "in_chaos": i in chaos_set})

        all50_med = float(np.median(horizons))
        chaos_horizons = horizons[[i for i in range(N_TRIALS) if i in chaos_set]]
        chaos_med = float(np.median(chaos_horizons))
        per_seed_all50_median[seed] = all50_med
        per_seed_chaos_median[seed] = chaos_med
        print(f"  seed={seed:>4}:  all50 median={all50_med:.4f}s   "
              f"chaos(N={n_chaos}) median={chaos_med:.4f}s")

    # 保存 per trial per seed
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "esn_seed_results.csv", index=False)

    # 保存 per seed 中位数
    summ_df = pd.DataFrame([
        {"seed": s,
         "all50_median": per_seed_all50_median[s],
         "chaos_median": per_seed_chaos_median[s],
         "n_chaos": n_chaos}
        for s in SEEDS
    ])
    summ_df.to_csv(OUT / "esn_seed_summary.csv", index=False)

    all50_vals = np.array([per_seed_all50_median[s] for s in SEEDS])
    chaos_vals = np.array([per_seed_chaos_median[s] for s in SEEDS])

    all50_mean, all50_std = float(all50_vals.mean()), float(all50_vals.std())
    chaos_mean, chaos_std = float(chaos_vals.mean()), float(chaos_vals.std())

    print("\n=== 跨 5 个 seed 聚合 ===")
    print(f"{'seed':>6} | {'all50 median':>14} | {'chaos median':>14}")
    print("-" * 42)
    for s in SEEDS:
        print(f"{s:>6} | {per_seed_all50_median[s]:>14.4f} | "
              f"{per_seed_chaos_median[s]:>14.4f}")
    print("-" * 42)
    print(f"{'mean':>6} | {all50_mean:>14.4f} | {chaos_mean:>14.4f}")
    print(f"{'std':>6} | {all50_std:>14.4f} | {chaos_std:>14.4f}")

    print("\n=== 目标论断核对 ===")
    print(f"  ESN all-50 median horizon = {all50_mean:.4f} ± {all50_std:.4f} s  (跨 {len(SEEDS)} seed)")
    print(f"  ESN chaos(N={n_chaos}) median horizon = {chaos_mean:.4f} ± {chaos_std:.4f} s")
    print(f"  结果文件：{OUT / 'esn_seed_results.csv'}")
    print(f"           {OUT / 'esn_seed_summary.csv'}")

    return {
        "sanity_check_passed": bool(sanity_ok),
        "n_chaos": n_chaos,
        "per_seed_all50_median": per_seed_all50_median,
        "per_seed_chaos_median": per_seed_chaos_median,
        "all50_median_mean": all50_mean,
        "all50_median_std": all50_std,
        "chaos_median_mean": chaos_mean,
        "chaos_median_std": chaos_std,
    }


if __name__ == "__main__":
    main()

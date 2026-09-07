"""
tools/hp_sweep.py — PINN 超参 / 架构自动 sweep loop

三维 search space（默认 6 组，~12 分钟）：
  λ_phys   ∈ {0.5, 1.0, 2.0}
  (hidden, n_layers) ∈ {(64, 3), (128, 4)}

每个 config：训练 200 epoch + 50 组完整评估 + 记录 horizon 分布。
输出 leaderboard 排序，并标注最优 config。

下次想扩 search 改下 LAMBDA_LIST / ARCH_LIST 即可，不动主逻辑。

用法：
    python3 tools/hp_sweep.py                # 默认 6 组
    python3 tools/hp_sweep.py --quick        # 100 epoch、3 组（快速 debug）
    python3 tools/hp_sweep.py --full         # 12 组完整 search（~25 分钟）
"""

import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod        # noqa: E402

DATA = ROOT / "data" / "sim"
OUT = ROOT / "data" / "hp_sweep"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 120
TRAIN_SEC = 20.0
PRED_SEC = 10.0
N_EVAL = 50
QUASI_THRESH = 9.9


def evaluate(net, n_eval=N_EVAL):
    horizons = []
    for i in range(n_eval):
        df = pd.read_csv(DATA / f"trial_{i:03d}.csv")
        n_train = int(TRAIN_SEC * FPS)
        n_pred = int(PRED_SEC * FPS)
        s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / FPS)
        pred_th = pred[:, [0, 2]]
        truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
        h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=FPS)
        horizons.append(h_s)
    return np.array(horizons)


def run_config(states, accs, lambda_phys, hidden, n_layers, n_epochs, seed=42):
    """训练 + 评估单组配置，返回 dict 含统计"""
    t0 = time.time()
    net = pinn_mod.AccelNet(hidden=hidden, n_layers=n_layers)
    pinn_mod.train(net, states, accs, n_epochs=n_epochs, batch=512, lr=2e-3,
                   lambda_phys=lambda_phys, n_collocation=2048, seed=seed)
    train_time = time.time() - t0

    horizons = evaluate(net)
    chaos = horizons[horizons < QUASI_THRESH]
    return {
        "lambda_phys": lambda_phys,
        "hidden": hidden,
        "n_layers": n_layers,
        "n_epochs": n_epochs,
        "n_chaos": int(len(chaos)),
        "horizon_median": float(np.median(chaos)) if len(chaos) else float("nan"),
        "horizon_mean":   float(np.mean(chaos))   if len(chaos) else float("nan"),
        "horizon_std":    float(np.std(chaos))    if len(chaos) else float("nan"),
        "train_time_s": float(train_time),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="100 epoch × 3 组（~3 分钟，debug 用）")
    ap.add_argument("--full", action="store_true",
                    help="12 组完整 search（~25 分钟）")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if args.quick:
        configs = [
            (0.5, 64, 3), (1.0, 128, 4), (2.0, 128, 4),
        ]
        n_epochs = args.epochs or 100
    elif args.full:
        configs = []
        for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for h, nl in [(64, 3), (128, 4)]:
                configs.append((lam, h, nl))
        configs.append((10.0, 128, 4))
        configs.append((0.0, 128, 4))   # 纯数据 PINN baseline 作对照
        n_epochs = args.epochs or 200
    else:
        configs = []
        for lam in [0.5, 1.0, 2.0]:
            for h, nl in [(64, 3), (128, 4)]:
                configs.append((lam, h, nl))
        n_epochs = args.epochs or 200

    print(f"=== HP Sweep ===")
    print(f"  共 {len(configs)} 组 × {n_epochs} epoch")
    print(f"  预计 ~{len(configs) * (n_epochs * 0.6 + 8):.0f} 秒\n")

    # 加载训练数据一次
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=TRAIN_SEC)
    print(f"  训练样本：{len(states)}\n")

    results = []
    for i, (lam, h, nl) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] λ={lam}  hidden={h}  layers={nl}")
        try:
            r = run_config(states, accs, lam, h, nl, n_epochs)
            print(f"  → horizon 中位 {r['horizon_median']:.3f}s  N_chaos={r['n_chaos']}  "
                  f"训练 {r['train_time_s']:.0f}s")
            results.append(r)
            # 增量落盘以防中途崩
            pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
        except Exception as e:
            print(f"  ! 此配置训练失败: {e}")
            results.append({"lambda_phys": lam, "hidden": h, "n_layers": nl,
                            "n_epochs": n_epochs, "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(OUT / "results.csv", index=False)

    # leaderboard
    print("\n=== Leaderboard（按 horizon 中位倒序）===")
    good = df.dropna(subset=["horizon_median"]).sort_values(
        "horizon_median", ascending=False
    ).head(10)
    cols = ["lambda_phys", "hidden", "n_layers",
            "horizon_median", "horizon_mean", "n_chaos", "train_time_s"]
    print(good[cols].to_string(index=False))

    if len(good):
        best = good.iloc[0]
        # 0.642s = 点质量旧 PINN 基线，仅作本 sweep 工具内相对参照，非论文头条口径
        # （现行权威见 data/canonical_results_B.md）
        improvement = (best.horizon_median / 0.642 - 1) * 100
        print(f"\n  最优：λ_phys={best.lambda_phys}, hidden={int(best.hidden)}, "
              f"layers={int(best.n_layers)}")
        print(f"        horizon 中位 {best.horizon_median:.3f}s  "
              f"vs baseline 0.642s ({improvement:+.0f}%)")
        with open(OUT / "best.json", "w") as f:
            json.dump({k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, (np.floating, float, int)) else v)
                       for k, v in best.items()}, f, indent=2, default=str)

    print(f"\n写出 {OUT}/results.csv  +  best.json")


if __name__ == "__main__":
    main()

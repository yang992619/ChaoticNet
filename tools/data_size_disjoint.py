#!/usr/bin/env python3
"""tools/data_size_disjoint.py — 训练数据量消融（训练集/评估集完全不相交）

动机
----
论文 §消融「训练数据量边际效益」已给出 N_train 每翻倍约 2× 视界的趋势，但那份实验
在 N_train=50 时训练集与评估集 100% 重叠（内样本评估），故 canonical 保守地只用
5 个 trial 训练（range(5)），此时仍与 50 组评估集有 10% 重叠。

data/sim200（2026-09-04 生成，trial_000–049 与 data/sim 逐字节一致，050–199 为新增）
让"大训练集 + 零重叠评估"成为可能：
    训练池 = trial 050–199（150 组，全部为评估集未见过的轨迹）
    评估集 = trial 000–049（与论文既有评估集完全相同）

于是可以干净地回答：把训练数据从 5 组加到 150 组，仿真内视界能涨多少？

口径
----
- 评估用 pinn.predict_aligned（2026-09-04 修复的同帧对齐；旧的 predict 超前真值一帧，
  会系统性低估 PINN/Hybrid 视界）。
- 建网用 pinn.make_accel_net（先播种再构造，初始权重可复现；旧写法先建后播种）。
- 其余超参与 canonical 一致：300 epoch、batch 512、lr 2e-3、λ_phys=0.1、collocation 2048、
  train_sec=20、pred_sec=10、fps=120、视界阈值 10°。
- full_median = 50 个 trial 全体中位；chaos_median = 该 run 自身 horizon<9.9s 子集的中位。
  注意 chaos_median 在模型变强时会因子集萎缩而系统性偏低，跨数据量比较请以 full_median 为准。

输出
----
data/data_size_disjoint/results.csv   逐 (N_train, seed) 一行
data/data_size_disjoint/summary.csv   逐 N_train 汇总
data/data_size_disjoint/curve.png     视界 vs 训练数据量
"""

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import pinn as pinn_mod  # noqa: E402

SIM200 = ROOT / "data" / "sim200"
EVAL_DIR = ROOT / "data" / "sim"
OUT = ROOT / "data" / "data_size_disjoint"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_POOL = list(range(50, 200))     # 150 组，与评估集零重叠
EVAL_TRIALS = list(range(50))
N_TRAIN_GRID = [5, 10, 20, 50, 100, 150]
SEEDS = [42, 7, 13]

TRAIN_SEC, PRED_SEC, FPS = 20.0, 10.0, 120
N_TRAIN_FRAMES, N_PRED = int(TRAIN_SEC * FPS), int(PRED_SEC * FPS)
EPOCHS, BATCH, LR, LAM, N_COLL = 300, 512, 2e-3, 0.1, 2048
QUASI = 9.9


def train_one(n_train_trials, seed, epochs=EPOCHS):
    """在训练池前 n_train_trials 个 trial 上训练一个 PINN。"""
    trials = TRAIN_POOL[:n_train_trials]
    states, accs = pinn_mod.load_dataset(trials, TRAIN_SEC, FPS, data_dir=SIM200)
    net = pinn_mod.make_accel_net(seed=seed, hidden=128, n_layers=4)
    pinn_mod.train(net, states, accs, n_epochs=epochs, batch=BATCH, lr=LR,
                   lambda_phys=LAM, n_collocation=N_COLL, seed=seed)
    net.eval()
    return net, len(states)


def evaluate(net):
    """在 trial 000–049 上闭环预测，返回 50 个视界。"""
    hs = []
    for i in EVAL_TRIALS:
        df = pd.read_csv(EVAL_DIR / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN_FRAMES].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
        truth = df[["th1", "th2"]].values[N_TRAIN_FRAMES:N_TRAIN_FRAMES + N_PRED]
        h, _ = pinn_mod.horizon(pred[:, [0, 2]], truth, threshold_deg=10.0, fps=FPS)
        hs.append(h)
    return np.array(hs)


def probe():
    """计时标定：用少量 epoch 估算全网格总耗时。"""
    print("=== 计时标定（每档 3 epoch）===")
    per_epoch = {}
    for n in (5, 50, 150):
        t0 = time.time()
        train_one(n, 42, epochs=3)
        dt = (time.time() - t0) / 3
        per_epoch[n] = dt
        print(f"  N_train={n:>3}: {dt:.2f} s/epoch")
    # 线性外插到网格各档
    k = (per_epoch[150] - per_epoch[5]) / (150 - 5)
    total = 0.0
    for n in N_TRAIN_GRID:
        est = (per_epoch[5] + k * (n - 5)) * EPOCHS
        total += est * len(SEEDS)
    print(f"\n  全网格估算：{total/60:.1f} 分钟"
          f"（{len(N_TRAIN_GRID)} 档 × {len(SEEDS)} seed × {EPOCHS} epoch）")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只做计时标定，不跑全网格")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--grid", type=int, nargs="*", default=None,
                    help="覆盖 N_train 网格，例如 --grid 5 50 150")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    grid = args.grid if args.grid else N_TRAIN_GRID
    print(f"=== 训练集/评估集不相交消融 ===")
    print(f"  训练池 trial {TRAIN_POOL[0]}–{TRAIN_POOL[-1]}（{len(TRAIN_POOL)} 组，来自 data/sim200）")
    print(f"  评估集 trial 0–49（data/sim，与训练池零重叠）")
    print(f"  网格 {grid} × seed {SEEDS} × {args.epochs} epoch\n")

    rows = []
    t_start = time.time()
    for n in grid:
        for seed in SEEDS:
            t0 = time.time()
            net, n_samples = train_one(n, seed, epochs=args.epochs)
            hs = evaluate(net)
            chaos = hs[hs < QUASI]
            row = {
                "n_train_trials": n, "seed": seed, "n_samples": n_samples,
                "full_median": float(np.median(hs)),
                "chaos_median": float(np.median(chaos)) if len(chaos) else float("nan"),
                "n_chaos": int(len(chaos)),
                "full_mean": float(hs.mean()),
                "secs": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  N={n:>3} seed={seed:>3}: full_median={row['full_median']:.3f}s  "
                  f"chaos_median={row['chaos_median']:.3f}s (n_chaos={row['n_chaos']})  "
                  f"[{row['secs']:.0f}s]")
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

    df = pd.DataFrame(rows)
    summ = (df.groupby("n_train_trials")
              .agg(n_samples=("n_samples", "first"),
                   full_median=("full_median", "median"),
                   full_mean=("full_median", "mean"),
                   full_std=("full_median", "std"),
                   chaos_median=("chaos_median", "median"),
                   n_chaos_mean=("n_chaos", "mean"))
              .reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)
    print("\n=== 汇总（full_median 为跨 seed 中位）===")
    print(summ.to_string(index=False))

    base = summ.loc[summ.n_train_trials == grid[0], "full_median"].iloc[0]
    print(f"\n相对 N_train={grid[0]} 的倍数：")
    for _, r in summ.iterrows():
        print(f"  N={int(r.n_train_trials):>3}: {r.full_median:.3f}s  "
              f"{r.full_median/base:.2f}×")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for seed in SEEDS:
            d = df[df.seed == seed]
            ax.plot(d.n_train_trials, d.full_median, "o--", alpha=.45, lw=1,
                    label=f"seed {seed}")
        ax.plot(summ.n_train_trials, summ.full_median, "ko-", lw=2, ms=7,
                label="跨 seed 中位")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("训练 trial 数（训练池与评估集零重叠）")
        ax.set_ylabel("仿真内视界中位 (s)")
        ax.set_title("训练数据量 → PINN 预测视界（不相交划分，同帧对齐口径）")
        ax.grid(alpha=.3, which="both"); ax.legend()
        fig.tight_layout()
        (OUT / "curve.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "curve.png", dpi=140)
        print(f"\n图 → {OUT / 'curve.png'}")
    except Exception as e:  # 画图失败不影响数据
        print(f"[warn] 画图失败：{e}")

    print(f"\n总耗时 {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

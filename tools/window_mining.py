#!/usr/bin/env python3
"""tools/window_mining.py — 从既有实测录像中挖掘多预测窗（U3）

动机
----
29 段实测录像共 412,762 帧已完成追踪（中位每段 12,620 帧 ≈ 210 s @60fps），但论文的
真实零样本评估每段只取了 1 个窗口（引入 3.98 s + 预测 3.98 s ≈ 8 s），即已追踪数据
仅用了约 4%，样本量 N=29。

本工具沿时间轴滑窗，把每段切成多个预测窗，样本量提升一到两个数量级，用于加强统计
功效（置信区间、配对检验、分布分离度），并给出"视界 vs 摆幅"的完整分布而非 29 个散点。

与论文头条口径的关系
--------------------
**本分析不替换头条。** 头条口径是"每段释放后第一个窗口"，对应高摆幅强混沌区段；
滑窗会纳入后段衰减后的低摆幅区段（单摆 τ≈130 s），那里本就更易预测。两者测的是
不同的样本总体，故本工具同时输出：
  (a) 首窗子集 —— 与论文头条同口径，用作回归校验；
  (b) 全窗集合 —— 按摆幅分层报告。

统计独立性
----------
同段内相邻窗共享同一条轨迹，彼此相关，**不可当作独立样本做朴素自助**。汇总时须用
按段聚类（block/clustered bootstrap）重采样，即以"段"为重采样单元。本工具输出保留
clip 列供下游正确处理，并在 summary 中同时给出朴素与聚类两种口径以显示差异。

口径要点
--------
- ESN 引入段改为**固定长度**窗（U[i0-lead_n : i0-1]），而非 real_validation_multi
  原写法的 U[:i0-1]（i0 之前全部历史）。否则越靠后的窗 ESN 训练数据越多，构成系统性
  混淆。注意在首窗（i0 == lead_n）两种写法完全等价，故可用首窗做回归校验。
- PINN/Hybrid 用 pinn.predict_aligned（2026-09-04 修复的同帧对齐）。
- 视界 = 两摆角度欧氏偏差首次超 10°；窗内未超者标右删失，取窗长为下界。
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import esn as esn_mod          # noqa: E402
import pinn as pinn_mod        # noqa: E402
import real_validation_multi as rvm  # noqa: E402

OUT = ROOT / "data" / "window_mining"
OUT.mkdir(parents=True, exist_ok=True)
REF = ROOT / "data" / "real_validation" / "summary_multi.csv"

THRESH_DEG = 10.0
ESN_KW = dict(n_res=300, sr=0.95, leak=0.3, ridge=1e-3, washout=50, seed=42)


def run_esn_fixed_lead(C, i0, lead_n, n_pred, **kw):
    """ESN 自训于 i0 前**固定长度** lead_n 帧，再闭环预测 n_pred 帧。

    i0 == lead_n 时与 rvm.run_esn 完全等价（训练片段同为 U[:i0-1]）。
    """
    kw = {**ESN_KW, **kw}
    df = pd.DataFrame({"th1": C["th1"], "w1": C["w1"],
                       "th2": C["th2"], "w2": C["w2"]})
    U = esn_mod.encode(df)
    Y = np.roll(U, -1, axis=0)
    a, b = i0 - lead_n, i0 - 1
    net = esn_mod.ESN(n_in=6, n_out=6, n_res=kw["n_res"],
                      spectral_radius=kw["sr"], leak=kw["leak"],
                      ridge=kw["ridge"], seed=kw["seed"])
    _, xf = net.train(U[a:b], Y[a:b],
                      washout=min(kw["washout"], max((b - a) // 3, 1)))
    pred = net.predict(xf, U[b], n_pred)
    t1, _, t2, _ = esn_mod.decode(pred)
    return t1, t2


def horizon_and_cens(p1, p2, t1, t2, fps):
    err = rvm.angle_err_deg(p1, p2, t1, t2)
    h, _ = rvm.horizon_s(err, fps)
    cens = bool(err.max() <= THRESH_DEG)   # 整窗未超阈 → 右删失
    return float(h), cens, float(err.max())


def mine_clip(tag, lead_s, pred_s, stride_s, nets, max_windows=None):
    """对一段录像滑窗，返回逐窗结果。"""
    C = rvm.load_clip(str(tag), lead_s, pred_s)
    fps, dt, N = C["fps"], C["dt"], C["N"]
    # 帧数用四舍五入而非截断重算，修掉往返掉帧：
    # summary_multi.csv 存的是 round(lead_n/fps, 2)（real_validation_multi.py:172），
    # 而 load_clip:70 读回时用 int(lead_s*fps) 截断。fps=59.975 而非整 60，
    # int(3.98*59.975)=238 但正确值是 239 —— 29 段里 18 段窗口起点整体前移一帧，
    # 这正是首窗回归校验对不上的真因（此前一度误判为权重漂移）。
    lead_n = min(int(round(lead_s * fps)), N // 2)
    n_pred = min(int(round(pred_s * fps)), N - lead_n - 1)
    stride = max(int(stride_s * fps), 1)

    rows = []
    i0 = lead_n
    while i0 + n_pred < N:
        if max_windows and len(rows) >= max_windows:
            break
        tt1 = C["th1"][i0:i0 + n_pred]
        tt2 = C["th2"][i0:i0 + n_pred]
        s0 = C["state"][i0].astype(np.float32)

        # 摆幅代理：预测窗内上摆角绝对值最大值（度）
        amp = float(np.degrees(np.max(np.abs(tt1))))
        w_rms = float(np.sqrt(np.mean(C["w1"][i0:i0 + n_pred] ** 2)))

        row = dict(clip=int(tag), i0=int(i0), t0=round(float(C["t"][i0]), 3),
                   amp_deg=round(amp, 2), w1_rms=round(w_rms, 3),
                   is_first=(i0 == lead_n))

        e1, e2 = run_esn_fixed_lead(C, i0, lead_n, n_pred)
        h, cens, emax = horizon_and_cens(e1, e2, tt1, tt2, fps)
        row.update(ESN_h=h, ESN_cens=cens, ESN_errmax=round(emax, 2))

        for name, net in nets.items():
            p1, p2 = rvm.run_torch(net, s0, n_pred, dt)
            h, cens, emax = horizon_and_cens(p1, p2, tt1, tt2, fps)
            row.update({f"{name}_h": h, f"{name}_cens": cens,
                        f"{name}_errmax": round(emax, 2)})
        rows.append(row)
        i0 += stride
    return rows


def clustered_bootstrap_median(df, col, n_boot=2000, seed=42):
    """按段聚类的自助中位数 95% CI —— 重采样单元是 clip，不是行。"""
    rng = np.random.default_rng(seed)
    clips = df["clip"].unique()
    groups = {c: df.loc[df["clip"] == c, col].values for c in clips}
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(clips, size=len(clips), replace=True)
        meds[b] = np.median(np.concatenate([groups[c] for c in pick]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def naive_bootstrap_median(vals, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    meds = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                     for _ in range(n_boot)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-s", type=float, default=4.0,
                    help="滑窗步长（秒）。默认 4.0 ≈ 预测窗长，预测段互不重叠")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="每段最多挖多少窗（调试用）")
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    args = ap.parse_args()

    ref = pd.read_csv(REF)
    tags = args.clips if args.clips else [int(t) for t in ref["tag"]]
    lead_by_tag = dict(zip(ref["tag"].astype(int), ref["lead_s"].astype(float)))
    pred_by_tag = dict(zip(ref["tag"].astype(int), ref["pred_s"].astype(float)))

    print("=== 实测录像多窗挖掘 ===")
    print(f"  段数 {len(tags)} | 引入/预测窗长取自既有 summary_multi | 步长 {args.stride_s}s")

    nets = {"PINN": rvm.load_pinn(), "Hybrid": rvm.load_hybrid()}
    lam0 = ROOT / "data" / "fair_compare" / "lam0_model.pt"
    if lam0.exists():
        import torch
        n = pinn_mod.AccelNet(hidden=128, n_layers=4)
        n.load_state_dict(torch.load(lam0, map_location=pinn_mod.DEVICE))
        n.to(pinn_mod.DEVICE).eval()
        nets["MLP"] = n
        print("  已载入 MLP(λ=0) 对照")
    print()

    all_rows, t0 = [], time.time()
    for k, tag in enumerate(tags, 1):
        try:
            rows = mine_clip(tag, lead_by_tag[tag], pred_by_tag[tag],
                             args.stride_s, nets, args.max_windows)
        except Exception as e:
            print(f"  [跳过] clip {tag}: {e}")
            continue
        all_rows += rows
        print(f"  [{k:>2}/{len(tags)}] clip {tag}: {len(rows):>3} 窗  "
              f"(累计 {len(all_rows)}, {time.time()-t0:.0f}s)")
        pd.DataFrame(all_rows).to_csv(OUT / "windows.csv", index=False)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("没有挖到任何窗口"); return
    models = [m for m in ("ESN", "MLP", "PINN", "Hybrid") if f"{m}_h" in df]

    # ---- 回归校验：首窗应与论文既有逐段结果同口径 ----
    first = df[df["is_first"]]
    print(f"\n=== 回归校验：首窗 vs data/real_validation/summary_multi.csv ===")
    m = first.merge(ref, left_on="clip", right_on="tag", suffixes=("", "_ref"))
    for mo in ("ESN", "PINN", "Hybrid"):
        if f"{mo}_h_ref" in m:
            d = (m[f"{mo}_h"] - m[f"{mo}_h_ref"]).abs()
            print(f"  {mo:>6}: 最大偏差 {d.max():.4f}s  中位偏差 {d.median():.4f}s")

    # ---- 汇总 ----
    print(f"\n=== 全窗汇总（N={len(df)} 窗，来自 {df["clip"].nunique()} 段）===")
    print(f"{'模型':>7} {'中位(s)':>9} {'聚类95%CI':>20} {'朴素95%CI':>20} {'删失率':>8}")
    summ = []
    for mo in models:
        v = df[f"{mo}_h"].values
        med = float(np.median(v))
        clo, chi = clustered_bootstrap_median(df, f"{mo}_h")
        nlo, nhi = naive_bootstrap_median(v)
        cens = float(df[f"{mo}_cens"].mean())
        summ.append(dict(model=mo, n=len(v), median=med,
                         ci_clustered_lo=clo, ci_clustered_hi=chi,
                         ci_naive_lo=nlo, ci_naive_hi=nhi, cens_rate=cens))
        print(f"{mo:>7} {med:>9.3f} {f'[{clo:.3f}, {chi:.3f}]':>20} "
              f"{f'[{nlo:.3f}, {nhi:.3f}]':>20} {cens:>7.1%}")
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False)

    # ---- 按摆幅分层 ----
    df["amp_bin"] = pd.cut(df["amp_deg"], [0, 20, 40, 60, 80, 200],
                           labels=["<20°", "20–40°", "40–60°", "60–80°", ">80°"])
    strat = df.groupby("amp_bin", observed=True).agg(
        n=("clip", "size"), **{f"{mo}_med": (f"{mo}_h", "median") for mo in models})
    for mo in models:
        if mo != "ESN":
            strat[f"{mo}/ESN"] = (strat[f"{mo}_med"] / strat["ESN_med"]).round(2)
    strat.to_csv(OUT / "by_amplitude.csv")
    print(f"\n=== 按摆幅分层 ===\n{strat.to_string()}")

    print(f"\n产物 → {OUT}/windows.csv, summary.csv, by_amplitude.csv")
    print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

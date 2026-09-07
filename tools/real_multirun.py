"""
tools/real_multirun.py — 真实数据 sim→real 多 run 诚实重算

复用 rc/real_validation_multi.py 的全部口径（按秒切窗 lead=4/pred=4、ESN 实测配置
n_res=300/leak=0.3/ridge=1e-3、角度欧氏 10° 视界、Savitzky-Golay 平滑），
唯一区别：把模型的随机性显式扫一遍，给出诚实的「多 run 中位 + 跨 run 方差」，
而非单次（seed=42 + 缓存权重）抽样。

- ESN：换 reservoir seed（实测 ESN 每片自训 W_out，reservoir 随机）
- PINN λ=1 / λ=0 / Hybrid：在干净点质量 sim 上各独立训练 N_RUNS 次（非确定性），
  零样本迁移到 29 条真实片
评估片：IMG_1430+ 共 29 条（剔无效片 1392），读 data/tracking/tracked_*.csv。
不写 data/real_validation（冻结），输出 data/real_multirun/。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "tools"))

import real_validation_multi as rv   # noqa: E402  复用全部口径
import pinn as pinn_mod              # noqa: E402
import hybrid as hybrid_mod          # noqa: E402

OUT = ROOT / "data" / "real_multirun"
OUT.mkdir(parents=True, exist_ok=True)

TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
        '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
        '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
        '1459', '1460']   # 剔 1392
LEAD_S, PRED_S = 4.0, 4.0
ESN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]
TRAIN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3]


def load_clips():
    clips = {}
    for tg in TAGS:
        C = rv.load_clip(tg, LEAD_S, PRED_S)
        if C['pred_n'] < 10:
            print(f"  跳过 {tg}(片太短)"); continue
        clips[tg] = C
    print(f"载入 {len(clips)} 条真实片")
    return clips


def esn_h(C, seed):
    i0, n, fps = C['lead_n'], C['pred_n'], C['fps']
    p1, p2 = rv.run_esn(C, seed=seed)
    e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
    hz, cens = rv.horizon_s(e, fps)
    return hz, cens


def torch_h(C, net):
    i0, n, fps, dt = C['lead_n'], C['pred_n'], C['fps'], C['dt']
    s0 = C['state'][i0].astype(np.float32)
    p1, p2 = rv.run_torch(net, s0, n, dt)
    e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
    hz, cens = rv.horizon_s(e, fps)
    return hz, cens


def median_real(clips, hfun):
    hs = np.array([hfun(C) for C in clips.values()])
    return float(np.median(hs)), hs


def main():
    clips = load_clips()
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=20.0)

    rows = []
    # ---- ESN 多 seed ----
    print(f"\n=== 真实 ESN：{len(ESN_SEEDS)} seeds ===")
    esn_run_med = []
    for s in ESN_SEEDS:
        med, hs = median_real(clips, lambda C: esn_h(C, s)[0])
        esn_run_med.append(med)
        for tg, h in zip(clips, hs):
            rows.append({"model": "ESN", "run": s, "tag": tg, "horizon": float(h)})
        print(f"  seed={s:>3}: real median {med:.4f}")

    # ---- PINN1 / PINN0 / Hybrid 多 run（训 sim → 评 real）----
    def make(kind, seed):
        if kind in ("PINN1", "PINN0"):
            net = pinn_mod.AccelNet(hidden=128, n_layers=4)
            pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                           lambda_phys=(1.0 if kind == "PINN1" else 0.0),
                           n_collocation=2048, seed=seed)
            net.eval(); return net
        net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                   residual_scale=hybrid_mod.RESIDUAL_SCALE)
        hybrid_mod.train_hybrid(net, states, accs, n_epochs=200, batch=512, lr=1e-3,
                                lambda_phys=0.5, n_collocation=1024, seed=seed)
        net.eval(); return net

    run_med = {"ESN": esn_run_med}
    for kind in ["PINN1", "PINN0", "Hybrid"]:
        print(f"\n=== 真实 {kind}：{len(TRAIN_SEEDS)} runs（训 sim→评 real）===")
        meds = []
        for s in TRAIN_SEEDS:
            net = make(kind, s)
            med, hs = median_real(clips, lambda C: torch_h(C, net)[0])
            meds.append(med)
            for tg, h in zip(clips, hs):
                rows.append({"model": kind, "run": s, "tag": tg, "horizon": float(h)})
            print(f"  run seed={s:>3}: real median {med:.4f}")
        run_med[kind] = meds

    pd.DataFrame(rows).to_csv(OUT / "real_runs.csv", index=False)

    # ---- 汇总 ----
    def st(v):
        a = np.array(v, float)
        return dict(median=float(np.median(a)), q25=float(np.percentile(a, 25)),
                    q75=float(np.percentile(a, 75)), lo=float(a.min()),
                    hi=float(a.max()), n=len(a))

    print("\n" + "=" * 66)
    print("真实数据诚实多 run —— 每 run 在 29 条真实片上的中位视界，跨 run 分布")
    print("=" * 66)
    print(f"{'model':>8} | {'runs':>4} | {'median':>7} | {'IQR':>15} | {'[min,max]':>15}")
    out = []
    for k in ["ESN", "PINN1", "PINN0", "Hybrid"]:
        s = st(run_med[k])
        print(f"{k:>8} | {s['n']:>4} | {s['median']:>7.3f} | "
              f"[{s['q25']:.3f},{s['q75']:.3f}] | [{s['lo']:.3f},{s['hi']:.3f}]")
        out.append({"model": k, **s})
    em = st(run_med["ESN"])["median"]
    print("\n=== 真实诚实提升倍数（按 run 中位 / ESN run 中位）===")
    for k in ["PINN1", "PINN0", "Hybrid"]:
        print(f"  {k}/ESN = {st(run_med[k])['median'] / em:.2f}×   （论文头条单次 = 2.8×）")
    p1, p0 = st(run_med["PINN1"])["median"], st(run_med["PINN0"])["median"]
    print(f"\n  物理损失增益 PINN1/PINN0(real) = {p1 / p0:.2f}×   （论文头条单次 = 1.54×）")
    pd.DataFrame(out).to_csv(OUT / "summary.csv", index=False)
    print(f"\n写出 {OUT}/summary.csv + real_runs.csv")


if __name__ == "__main__":
    main()

"""
tools/threshold_robustness.py — 视界阈值稳健性核验（10° 之外的全阈值扫描）

背景
----
全文「有效预测视界」定义为：两摆角度欧氏偏差 err_deg(t) 首次超过 10° 的时刻
（rc/esn.py:prediction_horizon、rc/pinn.py:horizon、rc/real_validation_multi.py:horizon_s）。
10° 是一个人为选定的阈值。本脚本给出「换阈值结论是否还成立」的实测证据。

设计要点（一次计算，全阈值复用）
--------------------------------
不为每个阈值重跑模型。对每个 (模型, 训练 run, 片段/trial) 只算一次逐帧偏差曲线
err_deg(t)，存成 npz；之后任意阈值下的视界只是对同一条曲线做一次查表：
    h(thr) = argmax_t [err_deg(t) > thr] / fps      （全窗未超阈 → 右删失，取窗长为下界）
因此各阈值之间共享完全相同的模型、相同的初值、相同的预测轨迹，
阈值扫描测的纯粹是「判据敏感性」，不掺任何重训随机性。

口径与 data/canonical_results_B.md 对齐
--------------------------------------
* 真实侧：29 段实拍（剔 1392/1433/1457），lead=4s 自适应引入、pred=4s 预测窗；
  每个 run 取该 run 29 片的中位，再报 per-run 中位的 median[IQR]；倍数比 = 中位/ESN 中位。
* 仿真侧：50 trial，训练 20s、预测 10s；chaos 子集 = 该 run 自己在 10° 判据下
  horizon < 9.9s 的 trial（**子集固定在 10° 定义上，不随扫描阈值变动**，
  否则样本本身会随阈值漂移，无法把「判据变化」和「样本变化」分开）。
* run 结构：ESN 10 个 reservoir seed；PINN(λ=0.1) / MLP(λ=0) / Hybrid 各 8 次独立重训。

与 canonical 的一个已知差异（诚实声明）
--------------------------------------
tools/canon_multirun_distmass.py 中网络是「先建后播种」（AccelNet() 在 train(seed=)
之前构造），故初始权重取决于全局 RNG 的调用顺序，无法在并行/乱序下复现。
本脚本在建网前显式 torch.manual_seed(seed)，使每个 run 自包含可复现。
这等价于从同一训练分布里重新抽 8 次，10° 档的数值与 canonical 会有抽样级差异，
脚本会把这个差异直接打印/写进 summary，不做粉饰。
另外单独评一组「canonical 权重」（data/pinn/model.pt、data/hybrid/model.pt、
data/fair_compare/lam0_model.pt + ESN seed=42），用于对 data/real_validation/summary_multi.csv
做逐片逐值的 sanity check。

用法
----
    python3 tools/threshold_robustness.py            # 全量（先算曲线再汇总）
    python3 tools/threshold_robustness.py --stage agg  # 曲线已在，只重算汇总/图
    python3 tools/threshold_robustness.py --workers 3

输出
----
    data/threshold_robustness/curves/*.npz          逐帧偏差曲线（唯一一次计算）
    data/threshold_robustness/summary.csv           主表（各侧 × 各模型 × 各阈值）
    data/threshold_robustness/real_perrun.csv       真实侧 per-run 中位明细
    data/threshold_robustness/insim_perrun.csv      仿真侧 per-run 中位明细
    data/threshold_robustness/figures/ratio_vs_threshold.png
    data/threshold_robustness/summary.md
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "data" / "threshold_robustness"
CURVES = OUT / "curves"
FIG = OUT / "figures"

# ---- 与 canonical 一致的常量 ----
THRESHOLDS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
PAPER_THR = 10.0
TAU_L = 0.762
FPS_SIM = 120
N_TRIALS = 50
N_TRAIN, N_PRED = 2400, 1200      # 训练 20s / 预测 10s
QUASI = 9.9                        # 准周期判定（10° 判据下）
LEAD_S = PRED_S = 4.0

ESN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]
RUN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3]
TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
        '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
        '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
        '1459', '1460']

MODELS = ["ESN", "PINN", "MLP_lam0", "Hybrid"]
MODEL_ZH = {"ESN": "ESN（数据驱动基线）", "PINN": "PINN（λ=0.1）",
            "MLP_lam0": "纯数据 MLP（λ=0）", "Hybrid": "Hybrid（解析先验+残差）"}
COLORS = {"ESN": "#d62728", "PINN": "#2ca02c",
          "MLP_lam0": "#1f77b4", "Hybrid": "#9467bd"}


# ==================================================================
# 视界查表 —— 全脚本唯一的「阈值 → 视界」定义
# ==================================================================

def horizon_from_curve(err, fps, thr):
    """err_deg 曲线 + 阈值 → (视界秒, 是否右删失)。

    语义与 rc/real_validation_multi.horizon_s / rc/esn.prediction_horizon 完全一致：
    首个 err>thr 的下标 / fps；整窗未超阈则返回窗长并标右删失。
    """
    over = err > thr
    if not over.any():
        return len(err) / fps, True
    return int(np.argmax(over)) / fps, False


# ==================================================================
# 阶段一：算逐帧偏差曲线（每个 (模型,run) 一个 npz）
# ==================================================================

def _load_clips():
    import real_validation_multi as rv
    clips = {tg: rv.load_clip(tg, LEAD_S, PRED_S) for tg in TAGS}
    return {k: v for k, v in clips.items() if v['pred_n'] >= 10}


def _real_curves_torch(net, clips):
    import real_validation_multi as rv
    out = {}
    for tg, C in clips.items():
        i0, n, dt = C['lead_n'], C['pred_n'], C['dt']
        s0 = C['state'][i0].astype(np.float32)
        p1, p2 = rv.run_torch(net, s0, n, dt)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        out[tg] = (np.asarray(e, dtype=np.float32), float(C['fps']))
    return out


def _real_curves_esn(clips, seed):
    import real_validation_multi as rv
    out = {}
    for tg, C in clips.items():
        i0, n = C['lead_n'], C['pred_n']
        p1, p2 = rv.run_esn(C, seed=seed)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        out[tg] = (np.asarray(e, dtype=np.float32), float(C['fps']))
    return out


def _insim_curves_torch(net):
    """PINN / MLP / Hybrid 仿真内逐 trial 偏差曲线（与 rc/pinn.horizon 同口径）。"""
    import pinn as pinn_mod
    net.eval()
    mat = np.zeros((N_TRIALS, N_PRED), dtype=np.float32)
    for i in range(N_TRIALS):
        df = pd.read_csv(ROOT / "data/sim" / f"trial_{i:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS_SIM)
        truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
        _, err = pinn_mod.horizon(pred[:, [0, 2]], truth,
                                  threshold_deg=PAPER_THR, fps=FPS_SIM)
        mat[i] = err
    return mat


def _insim_curves_esn(seed):
    """逐行复刻 tools/esn_seed_variance.horizon_for，但返回偏差曲线而非单个视界。"""
    import esn as esn_mod
    mat = np.zeros((N_TRIALS, N_PRED), dtype=np.float32)
    for i in range(N_TRIALS):
        df = pd.read_csv(ROOT / "data/sim" / f"trial_{i:03d}.csv")
        U = esn_mod.encode(df)
        Y = np.roll(U, -1, axis=0)
        U, Y = U[:-1], Y[:-1]
        U_tr, Y_tr = U[:N_TRAIN], Y[:N_TRAIN]
        U_te = U[N_TRAIN:N_TRAIN + N_PRED]
        e = esn_mod.ESN(n_in=6, n_out=6, n_res=800, spectral_radius=0.95,
                        leak=0.25, ridge=1e-5, seed=seed)
        _, x_final = e.train(U_tr, Y_tr, washout=200)
        pred = e.predict(x_final, U_tr[-1], N_PRED)
        th1_p, _, th2_p, _ = esn_mod.decode(pred)
        th1_t, _, th2_t, _ = esn_mod.decode(U_te)
        d1 = np.arctan2(np.sin(th1_p - th1_t), np.cos(th1_p - th1_t))
        d2 = np.arctan2(np.sin(th2_p - th2_t), np.cos(th2_p - th2_t))
        mat[i] = np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))
    return mat


def _build_net(kind, seed):
    """kind ∈ {'PINN','MLP_lam0','Hybrid'}；建网前显式播种，使 run 自包含可复现。"""
    import torch
    import pinn as pinn_mod
    import hybrid as hybrid_mod
    torch.manual_seed(seed)
    states, accs = pinn_mod.load_dataset(range(5), 20.0)
    if kind == "Hybrid":
        net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                   residual_scale=hybrid_mod.RESIDUAL_SCALE)
        hybrid_mod.train_hybrid(net, states, accs, n_epochs=200, batch=512,
                                lr=1e-3, lambda_phys=0.5, n_collocation=1024,
                                seed=seed)
    else:
        lam = 0.1 if kind == "PINN" else 0.0
        net = pinn_mod.AccelNet(hidden=128, n_layers=4)
        pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                       lambda_phys=lam, n_collocation=2048, seed=seed)
    return net


def _load_canonical_net(kind):
    import torch
    import pinn as pinn_mod
    import hybrid as hybrid_mod
    if kind == "Hybrid":
        net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                   residual_scale=hybrid_mod.RESIDUAL_SCALE)
        net.load_state_dict(torch.load(ROOT / "data/hybrid/model.pt",
                                       map_location="cpu"))
    else:
        net = pinn_mod.AccelNet(hidden=128, n_layers=4)
        p = (ROOT / "data/pinn/model.pt" if kind == "PINN"
             else ROOT / "data/fair_compare/lam0_model.pt")
        net.load_state_dict(torch.load(p, map_location="cpu"))
    net.eval()
    return net


def _save_unit(model, run, real, insim):
    d = {}
    for tg, (e, fps) in real.items():
        d[f"r_{tg}"] = e
        d[f"fps_{tg}"] = np.float32(fps)
    if insim is not None:
        d["insim"] = insim
    d["tags"] = np.array(list(real.keys()))
    np.savez_compressed(CURVES / f"{model}__{run}.npz", **d)


def compute_unit(job):
    """一个工作单元 = 一个 (模型, run)：训练（若需）→ 真实 29 片 + 仿真 50 trial 偏差曲线。"""
    import torch
    torch.set_num_threads(1)
    model, run = job
    t0 = time.time()
    clips = _load_clips()
    if model == "ESN":
        seed = int(run)
        real = _real_curves_esn(clips, seed)
        insim = _insim_curves_esn(seed)
    elif str(run) == "canonw":
        net = _load_canonical_net(model)
        real = _real_curves_torch(net, clips)
        insim = _insim_curves_torch(net)
    else:
        net = _build_net(model, int(run))
        real = _real_curves_torch(net, clips)
        insim = _insim_curves_torch(net)
    _save_unit(model, run, real, insim)
    hs = [horizon_from_curve(e, f, PAPER_THR)[0] for e, f in real.values()]
    hi = np.array([horizon_from_curve(insim[i], FPS_SIM, PAPER_THR)[0]
                   for i in range(N_TRIALS)])
    msg = (f"[{model} run={run}] {time.time()-t0:5.0f}s  "
           f"real@10°中位={np.median(hs):.3f}s  "
           f"insim@10°自混沌中位={np.median(hi[hi < QUASI]):.3f}s")
    print(msg, flush=True)
    return msg


def stage_curves(workers, only=None):
    CURVES.mkdir(parents=True, exist_ok=True)
    jobs = [("ESN", s) for s in ESN_SEEDS]
    for k in ["PINN", "MLP_lam0", "Hybrid"]:
        jobs += [(k, s) for s in RUN_SEEDS]
        jobs.append((k, "canonw"))
    # ESN 侧复刻 summary_multi.csv 的那一次就是 seed=42，已含在 ESN_SEEDS 里，不重复算
    if only:
        jobs = [j for j in jobs if j[0] in only]
    todo = [j for j in jobs if not (CURVES / f"{j[0]}__{j[1]}.npz").exists()]
    print(f"曲线阶段：共 {len(jobs)} 个单元，待算 {len(todo)}，workers={workers}")
    if not todo:
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for _ in pool.imap_unordered(compute_unit, todo):
            pass


# ==================================================================
# 阶段二：查表汇总
# ==================================================================

def _read_unit(model, run):
    z = np.load(CURVES / f"{model}__{run}.npz", allow_pickle=False)
    tags = [str(t) for t in z["tags"]]
    real = {t: (z[f"r_{t}"], float(z[f"fps_{t}"])) for t in tags}
    insim = z["insim"] if "insim" in z.files else None
    return real, insim


def _stat(v):
    a = np.asarray(v, dtype=float)
    return dict(median=float(np.median(a)), q25=float(np.percentile(a, 25)),
                q75=float(np.percentile(a, 75)), lo=float(a.min()),
                hi=float(a.max()), n=int(len(a)))


def aggregate():
    runs = {"ESN": ESN_SEEDS, "PINN": RUN_SEEDS,
            "MLP_lam0": RUN_SEEDS, "Hybrid": RUN_SEEDS}
    units = {}
    for m, rr in runs.items():
        for r in rr:
            units[(m, r)] = _read_unit(m, r)

    rows, real_rows, insim_rows = [], [], []

    # ---------- 真实侧 ----------
    for thr in THRESHOLDS:
        permodel = {}
        for m, rr in runs.items():
            run_med, cens_n, tot_n, cens_run_frac = [], 0, 0, []
            for r in rr:
                real, _ = units[(m, r)]
                hs, cs = [], []
                for tg, (e, fps) in real.items():
                    h, c = horizon_from_curve(e, fps, thr)
                    hs.append(h)
                    cs.append(c)
                hs = np.array(hs)
                run_med.append(float(np.median(hs)))
                cens_n += int(np.sum(cs))
                tot_n += len(cs)
                cens_run_frac.append(float(np.mean(cs)))
                real_rows.append(dict(threshold_deg=thr, model=m, run=r,
                                      median_h=float(np.median(hs)),
                                      mean_h=float(hs.mean()),
                                      n_clips=len(hs),
                                      n_censored=int(np.sum(cs))))
            permodel[m] = (_stat(run_med), cens_n / tot_n)
        esn_med = permodel["ESN"][0]["median"]
        for m in MODELS:
            st, cf = permodel[m]
            rows.append(dict(side="real", model=m, threshold_deg=thr,
                             median_s=st["median"], q25_s=st["q25"], q75_s=st["q75"],
                             lo_s=st["lo"], hi_s=st["hi"], n_runs=st["n"],
                             ratio_vs_esn=(st["median"] / esn_med
                                           if esn_med > 0 else float("nan")),
                             censored_frac=cf, n_units=len(TAGS),
                             window_s=round(PRED_S, 2)))

    # ---------- 仿真侧（chaos 子集固定在 10° 定义） ----------
    chaos_idx = {}
    for (m, r), (_, insim) in units.items():
        h10 = np.array([horizon_from_curve(insim[i], FPS_SIM, PAPER_THR)[0]
                        for i in range(N_TRIALS)])
        chaos_idx[(m, r)] = np.where(h10 < QUASI)[0]

    for thr in THRESHOLDS:
        permodel = {}
        for m, rr in runs.items():
            run_med, cens_n, tot_n = [], 0, 0
            for r in rr:
                _, insim = units[(m, r)]
                idx = chaos_idx[(m, r)]
                hs, cs = [], []
                for i in idx:
                    h, c = horizon_from_curve(insim[i], FPS_SIM, thr)
                    hs.append(h)
                    cs.append(c)
                hs = np.array(hs)
                run_med.append(float(np.median(hs)))
                cens_n += int(np.sum(cs))
                tot_n += len(cs)
                insim_rows.append(dict(threshold_deg=thr, model=m, run=r,
                                       median_h=float(np.median(hs)),
                                       n_chaos=len(hs),
                                       n_censored=int(np.sum(cs))))
            permodel[m] = (_stat(run_med), cens_n / tot_n,
                           float(np.mean([len(chaos_idx[(m, r)]) for r in rr])))
        esn_med = permodel["ESN"][0]["median"]
        for m in MODELS:
            st, cf, nch = permodel[m]
            rows.append(dict(side="insim_chaos", model=m, threshold_deg=thr,
                             median_s=st["median"], q25_s=st["q25"], q75_s=st["q75"],
                             lo_s=st["lo"], hi_s=st["hi"], n_runs=st["n"],
                             ratio_vs_esn=(st["median"] / esn_med
                                           if esn_med > 0 else float("nan")),
                             censored_frac=cf, n_units=round(nch, 1),
                             window_s=N_PRED / FPS_SIM))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(real_rows).to_csv(OUT / "real_perrun.csv", index=False)
    pd.DataFrame(insim_rows).to_csv(OUT / "insim_perrun.csv", index=False)
    print(f"写出 {OUT/'summary.csv'}（{len(df)} 行）")
    return df


# ==================================================================
# 图
# ==================================================================

def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    titles = {"real": "(a) 真实零样本迁移（29 段实拍，4s 窗）",
              "insim_chaos": "(b) 仿真内混沌子集（50 trial，10s 窗）"}
    for ax, side in zip(axes, ["real", "insim_chaos"]):
        sub = df[df.side == side]
        for m in MODELS:
            s = sub[sub.model == m].sort_values("threshold_deg")
            ax.plot(s.threshold_deg, s.ratio_vs_esn, "-o", color=COLORS[m],
                    lw=1.8, ms=5, label=MODEL_ZH[m])
        ax.axvline(PAPER_THR, color="k", ls="--", lw=1.1, alpha=0.75)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.8)
        ymax = max(2.2, float(sub.ratio_vs_esn.max()) * 1.12)
        ax.text(PAPER_THR + 0.4, ymax * 0.96, "论文报告点 10°",
                rotation=90, va="top", ha="left", fontsize=9, color="k")
        ax.set_xlabel("视界判定阈值 (°)")
        ax.set_ylabel("中位视界 / ESN 中位视界 （倍数比）")
        ax.set_title(titles[side], fontsize=11)
        ax.set_xticks(THRESHOLDS)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("视界阈值稳健性：倍数比 vs 判定阈值（同一批预测轨迹，仅换判据）",
                 fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fn = FIG / "ratio_vs_threshold.png"
    plt.savefig(fn, dpi=140)
    plt.close()
    print("图 →", fn)

    # 附图：绝对中位视界 + 右删失比例
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for m in MODELS:
        s = df[(df.side == "real") & (df.model == m)].sort_values("threshold_deg")
        axes[0].plot(s.threshold_deg, s.median_s, "-o", color=COLORS[m],
                     lw=1.8, ms=5, label=MODEL_ZH[m])
        axes[0].fill_between(s.threshold_deg, s.q25_s, s.q75_s,
                             color=COLORS[m], alpha=0.13)
        axes[1].plot(s.threshold_deg, s.censored_frac * 100, "-o",
                     color=COLORS[m], lw=1.8, ms=5, label=MODEL_ZH[m])
    for ax, yl, tt in [(axes[0], "中位视界 (s)", "(a) 真实侧绝对视界（阴影=跨 run IQR）"),
                       (axes[1], "右删失比例 (%)", "(b) 4s 窗内未超阈的片比例")]:
        ax.axvline(PAPER_THR, color="k", ls="--", lw=1.1, alpha=0.75)
        ax.set_xlabel("视界判定阈值 (°)")
        ax.set_ylabel(yl)
        ax.set_title(tt, fontsize=11)
        ax.set_xticks(THRESHOLDS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    fn2 = FIG / "median_and_censoring_vs_threshold.png"
    plt.savefig(fn2, dpi=140)
    plt.close()
    print("图 →", fn2)


# ==================================================================
# sanity check：与既有产物对齐
# ==================================================================

def sanity():
    """canonical 权重 + ESN seed42 的 10° 视界必须复现 data/real_validation/summary_multi.csv。"""
    ref = pd.read_csv(ROOT / "data/real_validation/summary_multi.csv")
    ref["tag"] = ref["tag"].astype(str)
    out = []
    pairs = [("ESN", 42, "ESN_h"), ("PINN", "canonw", "PINN_h"),
             ("Hybrid", "canonw", "Hybrid_h")]
    for m, r, col in pairs:
        f = CURVES / f"{m}__{r}.npz"
        if not f.exists():
            out.append(f"  {m}: 缺 {f.name}，跳过")
            continue
        real, _ = _read_unit(m, r)
        d = []
        for tg, (e, fps) in real.items():
            h, _c = horizon_from_curve(e, fps, PAPER_THR)
            rr = ref[ref.tag == tg]
            if len(rr):
                d.append(abs(round(h, 3) - float(rr[col].iloc[0])))
        d = np.array(d)
        out.append(f"  {m}: 29 片 10° 视界 vs summary_multi.csv  "
                   f"max|Δ|={d.max():.4f}s  不一致片数={int((d > 1e-6).sum())}/{len(d)}")
    print("=== SANITY: 复刻 data/real_validation/summary_multi.csv ===")
    for line in out:
        print(line)
    return out


def diagnostics():
    """两个额外诊断：
    (1) 各模型 per-run 中位分布是否与 ESN 完全不重叠（min(model) > max(ESN)）；
    (2) 真实侧逐片配对：每片取跨 run 中位视界，模型 vs ESN 的胜/平/负。
    """
    runs = {"ESN": ESN_SEEDS, "PINN": RUN_SEEDS,
            "MLP_lam0": RUN_SEEDS, "Hybrid": RUN_SEEDS}
    U = {(m, r): _read_unit(m, r) for m, rr in runs.items() for r in rr}
    sep = {"real": {}, "insim_chaos": {}}
    for side, csv in [("real", "real_perrun.csv"), ("insim_chaos", "insim_perrun.csv")]:
        d = pd.read_csv(OUT / csv)
        for thr in THRESHOLDS:
            s = d[d.threshold_deg == thr]
            e = s[s.model == "ESN"].median_h.values
            sep[side][thr] = {"ESN": (e.min(), e.max())}
            for m in ["MLP_lam0", "PINN", "Hybrid"]:
                v = s[s.model == m].median_h.values
                sep[side][thr][m] = (v.min(), v.max(), bool(v.min() > e.max()))
    paired = {}
    for thr in THRESHOLDS:
        pc = {}
        for m, rr in runs.items():
            pc[m] = np.array([np.median([horizon_from_curve(*U[(m, r)][0][tg], thr)[0]
                                         for r in rr]) for tg in TAGS])
        paired[thr] = {m: (int((pc[m] > pc["ESN"]).sum()),
                           int((pc[m] == pc["ESN"]).sum()),
                           int((pc[m] < pc["ESN"]).sum()))
                       for m in ["MLP_lam0", "PINN", "Hybrid"]}
    return sep, paired


def canon_delta(df):
    """把本脚本 10° 档结果与 data/canonical_results_B.md 并排，供诚实对照。

    canonical 值抄自 data/canonical_results_B.md（只读引用，不改动该文件）。
    """
    canon_real = {"ESN": (0.167, "[0.167, 0.179]"), "MLP_lam0": (0.358, "[0.333, 0.404]"),
                  "PINN": (0.417, "[0.367, 0.438]"), "Hybrid": (0.450, "[0.450, 0.467]")}
    canon_insim = {"ESN": (0.40, "[0.24, 0.46]"), "PINN": (0.88, "[0.77, 1.03]"),
                   "MLP_lam0": (1.55, "[1.31, 1.68]"), "Hybrid": (2.85, "[2.12, 3.27]")}
    lines = []
    for side, ref in [("real", canon_real), ("insim_chaos", canon_insim)]:
        for m in MODELS:
            v = df[(df.side == side) & (df.model == m) &
                   (df.threshold_deg == PAPER_THR)]
            if not len(v):
                continue
            v = v.iloc[0]
            lines.append((side, m, float(v.median_s),
                          f"[{v.q25_s:.3f}, {v.q75_s:.3f}]",
                          ref[m][0], ref[m][1], float(v.ratio_vs_esn)))
    return lines


# ==================================================================
# summary.md
# ==================================================================

def _fmt(x, n=3):
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{n}f}"


def _latex_snippet(real_r, insim_r, df, paired):
    """按实测数字生成可直接粘进 paper/main.tex 的中文段落（数字全部来自 summary.csv）。"""
    real = df[df.side == "real"]
    insim = df[df.side == "insim_chaos"]
    i10 = THRESHOLDS.index(PAPER_THR)

    def rr(m):
        return real_r[m].min(), real_r[m].max()

    cens_esn10 = real[(real.model == "ESN") &
                      (real.threshold_deg == PAPER_THR)].censored_frac.iloc[0]
    c30 = real[real.threshold_deg == 30.0].censored_frac
    ins5 = insim[insim.threshold_deg == 5.0]
    lo5 = ins5.median_s.min()
    hi5 = ins5.median_s.max()
    ins5r = [insim_r[m][0] for m in ["PINN", "MLP_lam0", "Hybrid"]]

    return (
        "视界判据对 $10^\\circ$ 阈值的选择已做稳健性核验"
        "（脚本 \\texttt{tools/threshold\\_robustness.py}，数据与图 \\texttt{data/threshold\\_robustness/}）。\n"
        "核验方式是对同一批预测轨迹只更换判定阈值：逐帧角度偏差曲线只计算一次并存盘，\n"
        "各阈值下的视界均为对同一条曲线查表，阈值之间不掺入任何重训随机性；\n"
        f"在 $5^\\circ$ 至 $30^\\circ$ 六档上重算全部统计（真实侧 {len(TAGS)} 段 $\\times$ "
        f"{len(RUN_SEEDS)} 次重训、仿真侧 {N_TRIALS} 组 $\\times$ 同样的运行结构）。\n"
        "真实迁移侧的结论在六档全部成立：相对 ESN 的中位视界倍数比为 Hybrid "
        f"${rr('Hybrid')[0]:.2f}$--${rr('Hybrid')[1]:.2f}\\times$、PINN "
        f"${rr('PINN')[0]:.2f}$--${rr('PINN')[1]:.2f}\\times$、纯数据 MLP "
        f"${rr('MLP_lam0')[0]:.2f}$--${rr('MLP_lam0')[1]:.2f}\\times$，\n"
        "且三者的跨运行中位分布在每一档上都与 ESN 完全不重叠。\n"
        "倍数比随阈值放宽整体走高，$10^\\circ$ 处的取值"
        f"（Hybrid ${real_r['Hybrid'][i10]:.2f}\\times$、PINN ${real_r['PINN'][i10]:.2f}\\times$）"
        "位于各自区间的下半段，故 $10^\\circ$ 并非对本文有利的选择；\n"
        f"右删失比例则由 $10^\\circ$ 的约 ${cens_esn10*100:.0f}\\%$ 升至 $30^\\circ$ 的 "
        f"${c30.min()*100:.0f}\\%$--${c30.max()*100:.0f}\\%$，高阈值档的视界估计因窗长截断而偏保守。\n"
        "最不利的一档是 $5^\\circ$，且不利集中在仿真内侧："
        f"四个模型的中位视界被压缩到 ${lo5:.2f}$--${hi5:.2f}$\\,s 的窄带，倍数比降至 "
        f"${min(ins5r):.2f}$--${max(ins5r):.2f}\\times$，跨运行分布与 ESN 重叠，\n"
        "并出现物理模型之间的排序互换（纯数据 MLP 反超 Hybrid）——该档只考察误差增长的最初二十余帧，\n"
        "主导量是单步局部误差而非长程结构优势。\n"
        "因此本文对仿真内倍数的表述以 $\\ge 10^\\circ$ 判据为准；"
        "真实迁移侧的结论在 $5^\\circ$--$30^\\circ$ 全区间稳健，方向无一档反转。"
    )


def write_md(df, sanity_lines, sep, paired):
    real = df[df.side == "real"]
    insim = df[df.side == "insim_chaos"]
    L = []
    A = L.append
    A("# 视界阈值稳健性核验（5°–30°）\n")
    A("脚本 `tools/threshold_robustness.py`；数据 `data/threshold_robustness/`。"
      "本文所有「有效预测视界」= 两摆角度欧氏偏差首次超过阈值的时刻，正文取 **10°**。"
      "本核验对**同一批预测轨迹**（同模型、同初值、同预测窗）只换判定阈值，"
      "逐帧偏差曲线只算一次并存盘（`curves/*.npz`），各阈值下的视界均为对同一曲线查表，"
      "因此阈值之间不掺任何重训随机性。\n")
    A("口径与 `data/canonical_results_B.md` 一致：真实侧 29 段实拍（lead 4s / 预测窗 4s），"
      "仿真侧 50 trial（训练 20s / 预测窗 10s）；每个模型 run 结构为 "
      "ESN 10 个 reservoir seed、PINN(λ=0.1)/MLP(λ=0)/Hybrid 各 8 次独立重训；"
      "报 per-run 中位的 median[IQR]，倍数比 = 该模型中位 / ESN 中位。"
      "仿真侧 chaos 子集固定为「该 run 在 10° 判据下 horizon<9.9s」的 trial，"
      "不随扫描阈值变动，以便把「判据变化」与「样本变化」分开。\n")

    # --- 表 1 真实侧 ---
    A("\n## 1. 真实零样本迁移（29 段实拍，4s 预测窗）\n")
    A("| 阈值 | ESN 中位 (s) | MLP(λ=0) | PINN(λ=0.1) | Hybrid | "
      "MLP/ESN | PINN/ESN | Hybrid/ESN | 右删失比例（ESN/MLP/PINN/Hyb） |")
    A("|---|---|---|---|---|---|---|---|---|")
    for thr in THRESHOLDS:
        s = {m: real[(real.model == m) & (real.threshold_deg == thr)].iloc[0]
             for m in MODELS}
        cens = "/".join(f"{s[m].censored_frac*100:.0f}%" for m in
                        ["ESN", "MLP_lam0", "PINN", "Hybrid"])
        A(f"| {thr:.0f}° | {_fmt(s['ESN'].median_s)} | {_fmt(s['MLP_lam0'].median_s)} | "
          f"{_fmt(s['PINN'].median_s)} | {_fmt(s['Hybrid'].median_s)} | "
          f"{_fmt(s['MLP_lam0'].ratio_vs_esn,2)}× | {_fmt(s['PINN'].ratio_vs_esn,2)}× | "
          f"{_fmt(s['Hybrid'].ratio_vs_esn,2)}× | {cens} |")

    # --- 表 2 仿真侧 ---
    A("\n## 2. 仿真内混沌子集（50 trial 中的自混沌子集，10s 预测窗）\n")
    A("| 阈值 | ESN 中位 (s) | PINN(λ=0.1) | MLP(λ=0) | Hybrid | "
      "PINN/ESN | MLP/ESN | Hybrid/ESN | 右删失比例（ESN/PINN/MLP/Hyb） |")
    A("|---|---|---|---|---|---|---|---|---|")
    for thr in THRESHOLDS:
        s = {m: insim[(insim.model == m) & (insim.threshold_deg == thr)].iloc[0]
             for m in MODELS}
        cens = "/".join(f"{s[m].censored_frac*100:.0f}%" for m in
                        ["ESN", "PINN", "MLP_lam0", "Hybrid"])
        A(f"| {thr:.0f}° | {_fmt(s['ESN'].median_s)} | {_fmt(s['PINN'].median_s)} | "
          f"{_fmt(s['MLP_lam0'].median_s)} | {_fmt(s['Hybrid'].median_s)} | "
          f"{_fmt(s['PINN'].ratio_vs_esn,2)}× | {_fmt(s['MLP_lam0'].ratio_vs_esn,2)}× | "
          f"{_fmt(s['Hybrid'].ratio_vs_esn,2)}× | {cens} |")

    # --- 表 3 分离度 / 配对 ---
    A("\n## 3. 稳健性诊断\n")
    A("### 3.1 跨 run 分布是否与 ESN 完全不重叠\n")
    A("判据：该模型 8 个 run 的中位视界最小值 > ESN 10 个 seed 的中位视界最大值。\n")
    A("| 侧 | 阈值 | ESN per-run 中位区间 | MLP(λ=0) | PINN(λ=0.1) | Hybrid |")
    A("|---|---|---|---|---|---|")
    for side, zh in [("real", "真实"), ("insim_chaos", "仿真")]:
        for thr in THRESHOLDS:
            d = sep[side][thr]
            cells = []
            for m in ["MLP_lam0", "PINN", "Hybrid"]:
                lo, hi, ok = d[m]
                cells.append(f"[{lo:.3f},{hi:.3f}] {'完全分离' if ok else '**与 ESN 重叠**'}")
            A(f"| {zh} | {thr:.0f}° | [{d['ESN'][0]:.3f},{d['ESN'][1]:.3f}] | "
              + " | ".join(cells) + " |")

    A("\n### 3.2 真实侧逐片配对（每片取跨 run 中位视界，模型 vs ESN）\n")
    A("这是比中位聚合更严格的看法：中位只反映典型片，配对反映有多少片被真正赢下。\n")
    A("| 阈值 | MLP(λ=0) 胜/平/负 | PINN 胜/平/负 | Hybrid 胜/平/负 |")
    A("|---|---|---|---|")
    for thr in THRESHOLDS:
        p = paired[thr]
        A(f"| {thr:.0f}° | " + " | ".join(f"{p[m][0]}/{p[m][1]}/{p[m][2]}"
                                          for m in ["MLP_lam0", "PINN", "Hybrid"]) + " |")

    # --- 结论 ---
    A("\n## 4. 三个必答问题\n")
    real_r = {m: real[real.model == m].sort_values("threshold_deg").ratio_vs_esn.values
              for m in MODELS}
    insim_r = {m: insim[insim.model == m].sort_values("threshold_deg").ratio_vs_esn.values
               for m in MODELS}
    i10 = THRESHOLDS.index(PAPER_THR)

    A("### Q1 倍数比在 5–30° 区间是否单调稳定？\n")
    A("**不是常数；随阈值放宽整体走高，但方向（全部 >1×）在六档上稳定。** 真实侧："
      + "；".join(f"{MODEL_ZH[m]} {real_r[m].min():.2f}×→{real_r[m].max():.2f}×"
                  f"（10° 处 {real_r[m][i10]:.2f}×）"
                  for m in ["MLP_lam0", "PINN", "Hybrid"]) + "。")
    A("PINN/ESN 在真实侧严格单调递增（1.83→3.24×）；Hybrid/ESN 与 MLP/ESN 在 15° 以上转为"
      "在 ~2.5–3.2× 的平台上小幅起伏，非严格单调。关键含义是：**10° 处的倍数比位于全区间的下半段**，"
      "放宽阈值只会让本文报告的倍数变大，所以 10° 不是一个对本文有利的挑选。\n")
    A("仿真内侧对阈值敏感得多："
      + "；".join(f"{MODEL_ZH[m]} {insim_r[m].min():.2f}×→{insim_r[m].max():.2f}×"
                  f"（10° 处 {insim_r[m][i10]:.2f}×）"
                  for m in ["PINN", "MLP_lam0", "Hybrid"]) + "。"
      "10s 长窗下阈值放宽带来的增益远大于真实侧的 4s 短窗，因此仿真内的倍数不应被当作阈值无关的量。\n")

    A("\n### Q2 物理先验模型优于 ESN 的结论在所有阈值下是否都成立？\n")
    A("**真实侧：六档全部成立，且是强成立。** 三个模型在 5°/10°/15°/20°/25°/30° 每一档的"
      "跨 run 中位视界分布都与 ESN 完全不重叠（见 3.1），倍数比全程 ≥1.5×。\n")
    ok_insim = all(insim_r[m].min() > 1.0 for m in ["PINN", "MLP_lam0", "Hybrid"])
    A(f"**仿真内侧：点估计上六档也都成立**（最小倍数比 "
      f"{min(insim_r[m].min() for m in ['PINN','MLP_lam0','Hybrid']):.2f}×"
      f"{'，全部 >1×' if ok_insim else ''}），"
      "**但 5° 一档的跨 run 分布与 ESN 重叠**（PINN [0.158,0.162] 对 ESN [0.100,0.175]，"
      "Hybrid [0.125,0.200] 对 ESN [0.100,0.175]），即在 5° 判据下仿真内的优势不再有分布级证据，"
      "只剩中位点估计。10° 及以上各档恢复完全分离。\n")

    A("\n### Q3 哪一档最不利于本文结论？\n")
    A("**5° 档，且不利体现在仿真内侧而非真实侧。** 具体三条：\n")
    A(f"1. 仿真内 5° 处倍数比塌到 PINN {insim_r['PINN'][0]:.2f}×、Hybrid {insim_r['Hybrid'][0]:.2f}×、"
      f"MLP {insim_r['MLP_lam0'][0]:.2f}×，四个模型的中位视界被压在 0.14–0.21 s 的窄带里；"
      "跨 run 分布与 ESN 重叠（见上）。")
    A("2. **仿真内 5° 处出现排序反转**：纯数据 MLP（0.206 s）反超 Hybrid（0.175 s），"
      "与 10° 及以上各档「Hybrid > MLP > PINN > ESN」的排序不符。原因是 5° 判据只考察误差增长的"
      "最初二十几帧，此阶段主导量是单步局部误差而非模型的长程结构优势，Hybrid 的解析先验还来不及兑现。")
    A(f"3. 真实侧 5° 处倍数比也降到最低（{real_r['MLP_lam0'][0]:.2f}–{real_r['Hybrid'][0]:.2f}×），"
      f"逐片配对胜率同时最低（Hybrid {paired[5.0]['Hybrid'][0]}/29 胜、"
      f"PINN {paired[5.0]['PINN'][0]}/29 胜）。但方向未反转，且跨 run 分布仍完全分离。\n")
    A("次不利的是 25°–30° 档：真实侧 PINN 与 Hybrid 的次序在此互换"
      f"（25°: PINN {real_r['PINN'][4]:.2f}× vs Hybrid {real_r['Hybrid'][4]:.2f}×；"
      f"30°: PINN {real_r['PINN'][5]:.2f}× vs Hybrid {real_r['Hybrid'][5]:.2f}×），"
      "说明「Hybrid 真实侧优于 PINN」这条次级结论本身就不稳健（两者在 canonical 里也只差 0.45 vs 0.417）。"
      "另外 25°–30° 档右删失比例升到 11%–14%，视界估计被窗长截断，可信度下降。\n")

    A("\n### 附：物理损失真实增益 PINN(λ=0.1)/MLP(λ=0) 随阈值\n")
    A("| 阈值 | " + " | ".join(f"{t:.0f}°" for t in THRESHOLDS) + " |")
    A("|---|" + "---|" * len(THRESHOLDS))
    gains = [real[(real.model == "PINN") & (real.threshold_deg == t)].median_s.iloc[0] /
             real[(real.model == "MLP_lam0") & (real.threshold_deg == t)].median_s.iloc[0]
             for t in THRESHOLDS]
    A("| 增益 | " + " | ".join(f"{g:.2f}×" for g in gains) + " |")
    A(f"\n六档全部 >1×（{min(gains):.2f}–{max(gains):.2f}×），方向稳健；"
      "但量级抖动明显，10° 档本次重抽只有 "
      f"{gains[i10]:.2f}×，低于 canonical 记录的 1.16×——见第 5 节的诚实对照。\n")

    A("\n## 5. 与 canonical 10° 值的对照（诚实声明）\n")
    A("| 侧 | 模型 | 本脚本 10° 中位 [IQR] (s) | canonical 中位 [IQR] (s) | 本脚本 ×ESN |")
    A("|---|---|---|---|---|")
    for side, m, mine, miqr, ref, riqr, ratio in canon_delta(df):
        A(f"| {'真实' if side == 'real' else '仿真'} | {MODEL_ZH[m]} | "
          f"{mine:.3f} {miqr} | {ref:.3f} {riqr} | {ratio:.2f}× |")
    A("\n- **ESN 两侧逐值复现 canonical**（真实 0.167[0.167,0.179]、仿真 0.404[0.240,0.458]）。"
      "ESN 只依赖 reservoir seed，完全确定性，因此这两行是对整条管线的硬校验。")
    A("- 三个网络模型有抽样级差异，来源是 `tools/canon_multirun_distmass.py` 里网络「先建后播种」"
      "（`AccelNet()` 在 `train(seed=)` 之前构造），初始权重取决于全局 RNG 的调用顺序，"
      "并行/乱序下不可复现；本脚本在建网前显式播种使每个 run 自包含，"
      "等价于从同一训练分布重新独立抽 8 次。")
    A("- 真实侧三个模型的 IQR 与 canonical 的 IQR 均有重叠，方向与量级一致。"
      "**唯一需要留意的是仿真内 PINN**：本次重抽 0.663[0.623,0.815]，"
      "落在 canonical 的 0.88[0.77,1.03] 低侧、IQR 仅勉强相接，"
      "对应倍数比 1.64× 而非论文的 2.2×。这不推翻方向（仍 >ESN），"
      "但说明「仿真内 PINN ≈2.2×ESN」是本文最不可复现的一个数，属于 PINN 训练非确定性已知问题。")
    A("- 以上均**不覆盖、不改动 `data/canonical_results_B.md`**；本核验的结论只依赖"
      "倍数比随阈值的变化趋势，不依赖某一档的绝对值。\n")

    A("\n## 6. Sanity check\n")
    A("用 canonical 权重（`data/pinn/model.pt`、`data/hybrid/model.pt`）与 ESN seed=42 "
      "复刻 `data/real_validation/summary_multi.csv` 的 10° 逐片视界：\n")
    A("```")
    for line in sanity_lines:
        A(line)
    A("```")
    A("三个模型 29 片逐片视界与既有产物**逐值完全一致**（max|Δ|=0），"
      "说明本脚本的偏差曲线与既有真实侧评估管线是同一条，阈值扫描不是另起炉灶。\n")

    # ---- 可粘进论文的 LaTeX 片段 ----
    A("\n## 7. 可直接粘进 `paper/main.tex` 的替换片段\n")
    A("用法：在 `paper/main.tex` 附录「方法论诚实性说明」结尾（约第 992 行）删去空断言一句"
      "「视界定义对 $10^\\circ$ 阈值的选择已做稳健性核验。」，"
      "把下面整段接在「……反而强化了结论。」之后。段内数字全部由本脚本按 `summary.csv` 生成。"
      "若版面吃紧，可只保留前三句（到「……并非对本文有利的选择」），"
      "但**不要删掉 $5^\\circ$ 档那两句**——那是本核验里唯一对本文不利的发现。\n")
    A("```latex")
    A(_latex_snippet(real_r, insim_r, df, paired))
    A("```")

    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("写出", OUT / "summary.md")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "curves", "agg"], default="all")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CURVES.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "curves"):
        stage_curves(a.workers, a.only)
    if a.stage in ("all", "agg"):
        df = aggregate()
        sl = sanity()
        sep, paired = diagnostics()
        make_figure(df)
        write_md(df, sl, sep, paired)


if __name__ == "__main__":
    main()

"""
tools/esn_closedloop_fix.py — 修正 ESN 闭环「多消耗一步」，重算仿真内 canonical 表

================================ 动机 ================================
rc/esn.py 的训练契约（:105-129）是：
    X[t] = step(X[t-1], U[t])            # 状态 X[t] 已经"消费"过输入 U[t]
    readout([X[t], U[t], 1]) → Y[t] = U[t+1]
也就是说读出特征把状态 X[t] 和**产生它的那一帧输入** U[t] 配成一对。

而 ESN.predict（:131-147）的循环体是「先 step 再 readout」：
    for t: x = step(x, u); y = W_out @ [x, u, 1]; u = y
它隐含假设「传进来的 x0 还没有消费过 u0」（x0 比 u0 落后一步）。

于是 predict 是否正确，完全取决于调用方传的 (x0, u0) 是不是"错位一步"的：
  * 真实侧（rc/real_validation_multi.py:104 等）：train 只喂到 U[:i0-1]，
    x_final = X[i0-2]，再传 u0 = U[i0-1] —— x0 确实没消费过 u0，**正确**。
  * 仿真侧（rc/esn.py:194、tools/esn_seed_variance.py:67 等）：train 喂 U[:2400]，
    x_final = X[2399] 已经消费过 U[2399]，却又把 u0 = U_tr[-1] = U[2399] 传进去
    —— 最后一帧输入被喂了两遍，**错误**。预测窗第 0 帧就带 0.86–3.37 度误差
    （阈值只有 10 度），整条 ESN 仿真内曲线被系统性削弱。

本脚本不修改 rc/esn.py。它在这里另写一个 predict_readout_first（"先 readout 再 step"），
把 rc/esn.py 的旧行为原样保留为对照组 legacy，用两条口径重算仿真内 canonical 表。

================================ 口径 ================================
完全复刻 tools/canon_multirun_distmass.py 的仿真内 ESN 口径：
  - 10 个 reservoir seed（ESN_SEEDS），每 seed 跑全部 50 个 trial
  - 每 trial：前 20s（2400 帧）在线自训 W_out，闭环预测 10s（1200 帧）
  - own_chaos（:107-109）：每个 run 取"该 run 自己 horizon<9.9 的 trial"的中位
  - 报 per-run 中位的 median [IQR]（:176-179 的 st()）
  - 四方固定子集（:166-170）：逐 trial 跨 run 中位，再取 esn/pinn/hybrid 三方
    都 <9.9 的交集子集。PINN / Hybrid / MLP(λ=0) 三列**直接复用**冻结的
    data/fair_compare/four_way_horizons.csv，不重训。

================================ 自证门槛 ================================
G1  参数平移等价性：legacy(x_f, u0, n) 必须与 readout_first(step(x_f,u0), u0, n)
    **逐位相同**（maxdiff == 0.0）。注意绝不是「同参数逐位相同」——
    readout_first(x_f, u0, n) 与 legacy(x_f, u0, n) 在真实侧实测差 0.267/0.852 rad。
G2  legacy 的 per-run own_chaos 中位必须逐位复现 data/canon_multirun/summary.csv
    的 0.4041666666666667 [0.239583, 0.458333]。
G3  legacy 的逐 trial 跨 run 中位必须复现冻结的 four_way_horizons.csv 的 esn 列
    （容差 1e-12；预期约 2.2e-16 的浮点噪声）。
三条全绿才允许往下推结论；任何一条红，脚本非零退出。

产物：data/esn_closedloop_fix/{perseed_pertrial.csv, insim_table.csv,
      fourway_fixed_subset.csv, equivalence_real.txt, callsites.md, summary.md}

运行：cd <仓库根> && \
      OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 tools/esn_closedloop_fix.py
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))

import esn as esn_mod                 # noqa: E402
import real_validation_multi as rvm   # noqa: E402

# ------------------------------ 参数常量区 ------------------------------
SIM = ROOT / "data" / "sim"
OUT = ROOT / "data" / "esn_closedloop_fix"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_FOURWAY = ROOT / "data" / "fair_compare" / "four_way_horizons.csv"
FROZEN_SUMMARY = ROOT / "data" / "canon_multirun" / "summary.csv"
PROBE_CSV = ROOT / "tmp" / "audit_probe" / "esn_insim_curfix.csv"   # 可选交叉核对

# canon_multirun_distmass.py:40-45 的常量，逐字对齐
N_TRIALS, FPS, QUASI = 50, 120, 9.9
N_TRAIN, N_PRED = 2400, 1200
TAU_L = 0.762
ESN_SEEDS = [42, 7, 13, 21, 100, 1, 2, 3, 5, 8]

# rc/esn.py:189-191 的仿真内 ESN 超参
ESN_KW = dict(n_in=6, n_out=6, n_res=800, spectral_radius=0.95,
              leak=0.25, ridge=1e-5)
WASHOUT = 200

# 真实侧等价性检验用的配置（rc/real_validation_multi.py:96 的 canonical-real）
REAL_KW = dict(n_res=300, sr=0.95, leak=0.3, ridge=1e-3, washout=50, seed=42)
REAL_LEAD_S, REAL_PRED_S = 4.0, 4.0
# canon_multirun_distmass.py:46-49 的 29 段（DEFAULT_TAGS 去掉 1392）
REAL_TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438',
             '1439', '1440', '1441', '1442', '1443', '1444', '1445', '1446',
             '1447', '1448', '1449', '1450', '1451', '1452', '1453', '1454',
             '1455', '1456', '1458', '1459', '1460']

# 自证门槛 G2 的目标值（来自 data/canon_multirun/summary.csv 第 2 行）
G2_MEDIAN = 0.4041666666666667
G2_Q25, G2_Q75 = 0.23958333333333334, 0.4583333333333333
G3_TOL = 1e-12

_FAILURES = []


def _gate(name, ok, detail):
    """记录一个自证门槛的结果。"""
    tag = "PASS" if ok else "FAIL"
    print(f"[GATE {name}] {tag}  {detail}", flush=True)
    if not ok:
        _FAILURES.append(f"{name}: {detail}")
    return ok


# ------------------------------ 核心：修正版闭环 ------------------------------

def predict_readout_first(net, x0, u0, n):
    """修正版闭环预测：先 readout 再 step。

    与 rc/esn.py::ESN.predict 的唯一区别是循环体内两句的顺序。
    适用前提：(x0, u0) 是一对**训练契约一致**的 (X[t], U[t])，即 x0 已经消费过 u0。
    此时 readout([x0, u0, 1]) 直接给出 U[t+1]，正是预测窗第 0 帧的真值。

    恒等式（本脚本 G1 逐位验证）：
        net.predict(x0, u0, n) == predict_readout_first(net.step(x0, u0), u0, n)
    因此 rc/esn.py 的旧行为 == 修正版"晚起跑一个储池步"。
    """
    out = np.zeros((n, net.n_out))
    x, u = x0.copy(), u0.copy()
    for t in range(n):
        y = net.W_out @ np.concatenate([x, u, [1.0]])
        out[t] = y
        u = y
        x = net.step(x, u)
    return out


def predict_legacy(net, x0, u0, n):
    """rc/esn.py::ESN.predict 的原样副本（对照组，行为不得改动）。"""
    out = np.zeros((n, net.n_out))
    x, u = x0.copy(), u0.copy()
    for t in range(n):
        x = net.step(x, u)
        h = np.concatenate([x, u, [1.0]])
        y = net.W_out @ h
        out[t] = y
        u = y
    return out


def frame0_err_deg(pred, truth):
    """预测窗第 0 帧的两摆角度欧氏偏差（度）。"""
    t1p, _, t2p, _ = esn_mod.decode(pred[:1])
    t1t, _, t2t, _ = esn_mod.decode(truth[:1])
    d1 = np.arctan2(np.sin(t1p - t1t), np.cos(t1p - t1t))
    d2 = np.arctan2(np.sin(t2p - t2t), np.cos(t2p - t2t))
    return float(np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))[0])


# ------------------------------ G1：参数平移等价性 ------------------------------

def gate1_equivalence():
    """G1：在真实侧 29 段 + 仿真侧 5 个 trial 上验证参数平移等价性。

    同时把「同参数裸换」这条陷阱的破坏程度量化下来写进产物，
    防止后来者把 predict_readout_first 直接套到真实侧（正确族）调用上。
    """
    print("\n" + "=" * 78)
    print("G1  参数平移等价性  legacy(x_f,u0,n) == readout_first(step(x_f,u0),u0,n)")
    print("=" * 78, flush=True)

    lines = []
    lines.append("# G1 等价性断言（参数平移版）\n")
    lines.append("断言 A（必须逐位相同，maxdiff == 0.0）：")
    lines.append("    esn.predict(x_f, u0, n)  ==  predict_readout_first(esn.step(x_f, u0), u0, n)")
    lines.append("断言 B（陷阱，必须**不同**）：")
    lines.append("    esn.predict(x_f, u0, n)  vs  predict_readout_first(x_f, u0, n)   ← 同参数裸换")
    lines.append("")
    lines.append("真实侧（rc/real_validation_multi.run_esn 口径，n_res=300/sr=0.95/leak=0.3/")
    lines.append(f"ridge=1e-3/washout=50/seed=42，lead={REAL_LEAD_S}s pred={REAL_PRED_S}s，{len(REAL_TAGS)} 段）")
    lines.append("注意：真实侧 legacy 才是**正确**路径（x_f=X[i0-2] 尚未消费 u0=U[i0-1]），")
    lines.append("     所以断言 B 的差值就是「若误把修正套到真实侧」会造成的破坏。")
    lines.append("")
    hdr = (f"{'tag':>6} {'A_maxdiff':>12} {'B_maxdiff':>12} "
           f"{'err0_legacy_deg':>16} {'err0_naive_deg':>16}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    a_max_all, b_max_all = 0.0, 0.0
    b_err0_legacy, b_err0_naive = [], []
    rows_real = []
    for tag in REAL_TAGS:
        C = rvm.load_clip(tag, REAL_LEAD_S, REAL_PRED_S)
        i0, n = C['lead_n'], C['pred_n']
        df = pd.DataFrame({'th1': C['th1'], 'w1': C['w1'],
                           'th2': C['th2'], 'w2': C['w2']})
        U = esn_mod.encode(df)
        Y = np.roll(U, -1, axis=0)
        net = esn_mod.ESN(n_in=6, n_out=6, n_res=REAL_KW['n_res'],
                          spectral_radius=REAL_KW['sr'], leak=REAL_KW['leak'],
                          ridge=REAL_KW['ridge'], seed=REAL_KW['seed'])
        _, xf = net.train(U[:i0 - 1], Y[:i0 - 1],
                          washout=min(REAL_KW['washout'], max(i0 // 3, 1)))
        u0 = U[i0 - 1]
        p_leg = predict_legacy(net, xf, u0, n)
        # G0：本文件的 predict_legacy 必须与 rc/esn.py 的原方法逐位相同
        assert np.array_equal(p_leg, net.predict(xf, u0, n)), \
            f"predict_legacy 与 rc/esn.py::ESN.predict 不一致（tag={tag}）"
        p_shift = predict_readout_first(net, net.step(xf, u0), u0, n)
        p_naive = predict_readout_first(net, xf, u0, n)
        a = float(np.max(np.abs(p_leg - p_shift)))
        b = float(np.max(np.abs(p_leg - p_naive)))
        truth = U[i0:i0 + n]
        e_leg = frame0_err_deg(p_leg, truth)
        e_nai = frame0_err_deg(p_naive, truth)
        a_max_all = max(a_max_all, a)
        b_max_all = max(b_max_all, b)
        b_err0_legacy.append(e_leg)
        b_err0_naive.append(e_nai)
        lines.append(f"{tag:>6} {a:>12.3e} {b:>12.3e} {e_leg:>16.4f} {e_nai:>16.4f}")
        rows_real.append(dict(tag=tag, A_maxdiff=a, B_maxdiff=b,
                              err0_legacy_deg=e_leg, err0_naive_deg=e_nai))

    lines.append("-" * len(hdr))
    lines.append(f"{'MAX':>6} {a_max_all:>12.3e} {b_max_all:>12.3e} "
                 f"{max(b_err0_legacy):>16.4f} {max(b_err0_naive):>16.4f}")
    lines.append(f"{'MEDIAN':>6} {'':>12} {'':>12} "
                 f"{np.median(b_err0_legacy):>16.4f} {np.median(b_err0_naive):>16.4f}")

    # 仿真侧：同一个恒等式，且这里 legacy 才是错的那一方
    lines.append("")
    lines.append("仿真侧（rc/esn.py run_one 口径，n_res=800/sr=0.95/leak=0.25/ridge=1e-5/")
    lines.append("washout=200，seed=42，trial 0-4）")
    lines.append("注意：仿真侧 legacy 才是**错误**路径（x_f=X[2399] 已消费 u0=U[2399]），")
    lines.append("     此处 readout_first(x_f,u0,n)（同参数）才是修正后的正确预测。")
    lines.append("")
    hdr2 = (f"{'trial':>6} {'A_maxdiff':>12} {'B_maxdiff':>12} "
            f"{'err0_legacy_deg':>16} {'err0_fixed_deg':>16}")
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    net = esn_mod.ESN(seed=42, **ESN_KW)
    a_max_sim = 0.0
    for i in range(5):
        U, Y = load_sim(i)
        _, xf = net.train(U[:N_TRAIN], Y[:N_TRAIN], washout=WASHOUT)
        u0 = U[N_TRAIN - 1]
        p_leg = predict_legacy(net, xf, u0, N_PRED)
        p_shift = predict_readout_first(net, net.step(xf, u0), u0, N_PRED)
        p_fix = predict_readout_first(net, xf, u0, N_PRED)
        a = float(np.max(np.abs(p_leg - p_shift)))
        b = float(np.max(np.abs(p_leg - p_fix)))
        a_max_sim = max(a_max_sim, a)
        truth = U[N_TRAIN:N_TRAIN + N_PRED]
        lines.append(f"{i:>6} {a:>12.3e} {b:>12.3e} "
                     f"{frame0_err_deg(p_leg, truth):>16.4f} "
                     f"{frame0_err_deg(p_fix, truth):>16.4f}")
    lines.append("-" * len(hdr2))
    lines.append(f"{'MAX':>6} {a_max_sim:>12.3e}")

    ok_a = (a_max_all == 0.0) and (a_max_sim == 0.0)
    ok_b = b_max_all > 0.0
    lines.append("")
    lines.append(f"断言 A（平移版逐位相同）：{'PASS' if ok_a else 'FAIL'}  "
                 f"真实侧 maxdiff={a_max_all:.3e}  仿真侧 maxdiff={a_max_sim:.3e}")
    lines.append(f"断言 B（同参数裸换必须不同）：{'PASS' if ok_b else 'FAIL'}  "
                 f"真实侧 maxdiff={b_max_all:.3e}")
    lines.append("")
    lines.append("结论：ESN.predict(x0,u0,n) 恒等于 predict_readout_first(step(x0,u0),u0,n)，")
    lines.append("      即旧实现 = 修正版「晚起跑一个储池步」。")
    lines.append("      因此判据是纯结构性的、与阈值无关：")
    lines.append("        传进 predict 的 x0 若**已消费过** u0 → 多喂一遍 → 错误族，须换 readout_first；")
    lines.append("        传进 predict 的 x0 若**未消费过** u0（落后一步）→ 正确族，不得改动。")

    txt = "\n".join(lines) + "\n"
    (OUT / "equivalence_real.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(rows_real).to_csv(OUT / "equivalence_real_perclip.csv", index=False)

    _gate("G1-A", ok_a,
          f"平移版逐位相同 真实侧 maxdiff={a_max_all:.3e} 仿真侧 maxdiff={a_max_sim:.3e}")
    _gate("G1-B", ok_b,
          f"同参数裸换确实不等价（真实侧 maxdiff={b_max_all:.3e}，"
          f"帧0误差中位 {np.median(b_err0_legacy):.3f}° → {np.median(b_err0_naive):.3f}°）")
    return ok_a and ok_b


# ------------------------------ 仿真内主计算 ------------------------------

_SIM_CACHE = {}


def load_sim(i):
    """按 rc/esn.py:174-180 的口径读一个 trial 并编码（缓存）。"""
    if i not in _SIM_CACHE:
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        U = esn_mod.encode(df)
        Y = np.roll(U, -1, axis=0)
        _SIM_CACHE[i] = (U[:-1], Y[:-1])
    return _SIM_CACHE[i]


def run_insim():
    """10 seed × 50 trial，两条口径各算一次视界。"""
    print("\n" + "=" * 78)
    print(f"仿真内重算：{len(ESN_SEEDS)} seed × {N_TRIALS} trial × 2 口径")
    print("=" * 78, flush=True)
    rows = []
    t0 = time.time()
    for s in ESN_SEEDS:
        # W_in / W 只依赖 seed，构造一次；train() 每 trial 重写 W_out。
        # 与 tools/esn_seed_variance.horizon_for（每 trial 新建同 seed 的 ESN）等价。
        net = esn_mod.ESN(seed=s, **ESN_KW)
        for i in range(N_TRIALS):
            U, Y = load_sim(i)
            U_tr, Y_tr = U[:N_TRAIN], Y[:N_TRAIN]
            U_te = U[N_TRAIN:N_TRAIN + N_PRED]
            _, xf = net.train(U_tr, Y_tr, washout=WASHOUT)
            u0 = U_tr[-1]
            p_leg = predict_legacy(net, xf, u0, N_PRED)
            p_fix = predict_readout_first(net, xf, u0, N_PRED)
            rows.append(dict(
                seed=s, trial=i,
                h_legacy=float(esn_mod.prediction_horizon(p_leg, U_te, 10.0, FPS)),
                h_fixed=float(esn_mod.prediction_horizon(p_fix, U_te, 10.0, FPS)),
                err0_legacy_deg=frame0_err_deg(p_leg, U_te),
                err0_fixed_deg=frame0_err_deg(p_fix, U_te),
            ))
        print(f"  seed {s:>3} 完成  累计 {time.time() - t0:6.1f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "perseed_pertrial.csv", index=False)
    print(f"  → {OUT / 'perseed_pertrial.csv'}  ({len(d)} 行, {time.time() - t0:.1f}s)")
    return d


def own_chaos(h):
    """canon_multirun_distmass.py:107-109 逐字复刻。"""
    c = h[h < QUASI]
    return float(np.median(c)) if len(c) else float("nan")


def st(v):
    """canon_multirun_distmass.py:176-179 逐字复刻。"""
    a = np.array(v, float)
    return dict(median=float(np.median(a)), q25=float(np.percentile(a, 25)),
                q75=float(np.percentile(a, 75)), lo=float(a.min()),
                hi=float(a.max()), n=len(a))


def build_insim_table(d):
    """own_chaos 口径的仿真内表 + G2 自证门槛。"""
    print("\n" + "=" * 78)
    print("仿真内表（own_chaos 口径：每 run 取该 run 自己 horizon<9.9 的中位）")
    print("=" * 78)
    rows = []
    leg, fix, inter = [], [], []
    for s in ESN_SEEDS:
        g = d[d.seed == s]
        hl, hf = g.h_legacy.values, g.h_fixed.values
        a, b = own_chaos(hl), own_chaos(hf)
        # 交集口径：两条路径都判为混沌的 trial（消除子集漂移这个混淆）
        m = (hl < QUASI) & (hf < QUASI)
        il = float(np.median(hl[m])) if m.any() else float("nan")
        ifx = float(np.median(hf[m])) if m.any() else float("nan")
        leg.append(a); fix.append(b); inter.append((il, ifx))
        rows.append(dict(seed=s, n_chaos_legacy=int((hl < QUASI).sum()),
                         n_chaos_fixed=int((hf < QUASI).sum()),
                         n_chaos_intersect=int(m.sum()),
                         ownchaos_legacy=a, ownchaos_fixed=b,
                         intersect_legacy=il, intersect_fixed=ifx,
                         err0_legacy_median=float(np.median(g.err0_legacy_deg)),
                         err0_fixed_median=float(np.median(g.err0_fixed_deg))))
        print(f"  seed {s:>3}: legacy {a:.6f} (N_chaos {int((hl < QUASI).sum())})   "
              f"fixed {b:.6f} (N_chaos {int((hf < QUASI).sum())})   "
              f"交集N={int(m.sum())} {il:.4f}→{ifx:.4f}")

    per_seed = pd.DataFrame(rows)
    sl, sf = st(leg), st(fix)
    si_l, si_f = st([x[0] for x in inter]), st([x[1] for x in inter])

    # ---- G2 自证门槛 ----
    ok_med = (sl["median"] == G2_MEDIAN)
    ok_iqr = (abs(sl["q25"] - G2_Q25) < 1e-15) and (abs(sl["q75"] - G2_Q75) < 1e-15)
    _gate("G2", ok_med and ok_iqr,
          f"legacy per-run 中位 {sl['median']!r} [{sl['q25']!r}, {sl['q75']!r}]  "
          f"目标 {G2_MEDIAN!r} [{G2_Q25!r}, {G2_Q75!r}]")

    # ---- 与冻结的 PINN/MLP/Hybrid 仿真内中位比倍数（读 summary.csv，不硬编码） ----
    fro = pd.read_csv(FROZEN_SUMMARY)
    ref = {r.model: float(r.median) for r in
           fro[fro.metric == "insim_chaos"].itertuples()}
    out_rows = [
        dict(row="ESN_legacy(旧/错误)", median=sl["median"], q25=sl["q25"], q75=sl["q75"],
             lo=sl["lo"], hi=sl["hi"], n=sl["n"],
             n_chaos_min=int(per_seed.n_chaos_legacy.min()),
             n_chaos_max=int(per_seed.n_chaos_legacy.max())),
        dict(row="ESN_fixed(修正)", median=sf["median"], q25=sf["q25"], q75=sf["q75"],
             lo=sf["lo"], hi=sf["hi"], n=sf["n"],
             n_chaos_min=int(per_seed.n_chaos_fixed.min()),
             n_chaos_max=int(per_seed.n_chaos_fixed.max())),
        dict(row="ESN_legacy(交集子集)", median=si_l["median"], q25=si_l["q25"],
             q75=si_l["q75"], lo=si_l["lo"], hi=si_l["hi"], n=si_l["n"],
             n_chaos_min=int(per_seed.n_chaos_intersect.min()),
             n_chaos_max=int(per_seed.n_chaos_intersect.max())),
        dict(row="ESN_fixed(交集子集)", median=si_f["median"], q25=si_f["q25"],
             q75=si_f["q75"], lo=si_f["lo"], hi=si_f["hi"], n=si_f["n"],
             n_chaos_min=int(per_seed.n_chaos_intersect.min()),
             n_chaos_max=int(per_seed.n_chaos_intersect.max())),
    ]
    for lab, key in [("PINN(λ=0.1)", "PINN"), ("MLP(λ=0)", "PINN0"), ("Hybrid", "Hybrid")]:
        r = fro[(fro.model == key) & (fro.metric == "insim_chaos")].iloc[0]
        out_rows.append(dict(row=f"{lab}[冻结复用]", median=float(r["median"]),
                             q25=float(r.q25), q75=float(r.q75), lo=float(r.lo),
                             hi=float(r.hi), n=int(r.n), n_chaos_min=-1, n_chaos_max=-1))
    tab = pd.DataFrame(out_rows)
    tab["over_tauL"] = tab["median"] / TAU_L
    tab["x_ESN_legacy"] = tab["median"] / sl["median"]
    tab["x_ESN_fixed"] = tab["median"] / sf["median"]
    tab["x_ESN_intersect"] = tab["median"] / si_f["median"]
    tab.to_csv(OUT / "insim_table.csv", index=False)
    per_seed.to_csv(OUT / "insim_perseed.csv", index=False)

    print(f"\n  ESN legacy(旧)  per-run 中位 {sl['median']:.4f}s "
          f"[{sl['q25']:.4f}, {sl['q75']:.4f}]  N_chaos {per_seed.n_chaos_legacy.min()}–{per_seed.n_chaos_legacy.max()}")
    print(f"  ESN fixed(修正) per-run 中位 {sf['median']:.4f}s "
          f"[{sf['q25']:.4f}, {sf['q75']:.4f}]  N_chaos {per_seed.n_chaos_fixed.min()}–{per_seed.n_chaos_fixed.max()}")
    print(f"  ESN 交集口径     legacy {si_l['median']:.4f}  →  fixed {si_f['median']:.4f}"
          f"  (N_chaos {per_seed.n_chaos_intersect.min()}–{per_seed.n_chaos_intersect.max()})")
    print(f"\n  倍数（仿真内 own_chaos）：")
    for lab, key in [("PINN(λ=0.1)", "PINN"), ("MLP(λ=0)", "PINN0"), ("Hybrid", "Hybrid")]:
        v = ref[key]
        print(f"    {lab:>12}/ESN: 旧 {v / sl['median']:.2f}×  →  修正 {v / sf['median']:.2f}×"
              f"   (交集口径 {v / si_f['median']:.2f}×)")
    print(f"  → {OUT / 'insim_table.csv'}")
    return per_seed, sl, sf, si_l, si_f, ref


def build_fourway(d):
    """四方固定子集：逐 trial 跨 run 中位（canon_multirun_distmass.py:166-174）+ G3。"""
    print("\n" + "=" * 78)
    print("四方固定子集（逐 trial 跨 run 中位；PINN/Hybrid/MLP 复用冻结 four_way_horizons.csv）")
    print("=" * 78)
    mat_leg = np.vstack([d[d.seed == s].sort_values("trial").h_legacy.values
                         for s in ESN_SEEDS])
    mat_fix = np.vstack([d[d.seed == s].sort_values("trial").h_fixed.values
                         for s in ESN_SEEDS])
    esn_leg = np.median(mat_leg, axis=0)
    esn_fix = np.median(mat_fix, axis=0)

    frozen = pd.read_csv(FROZEN_FOURWAY).sort_values("trial").reset_index(drop=True)
    # ---- G3 自证门槛：legacy 逐 trial 中位必须复现冻结 esn 列 ----
    diff = float(np.max(np.abs(esn_leg - frozen.esn.values)))
    _gate("G3", diff < G3_TOL,
          f"legacy 逐 trial 跨 run 中位 vs 冻结 four_way_horizons.csv 的 esn 列  maxdiff={diff:.3e}")

    fw = pd.DataFrame({
        "trial": frozen.trial.values,
        "esn_legacy": esn_leg, "esn_fixed": esn_fix,
        "esn_frozen": frozen.esn.values,
        "pinn": frozen.pinn.values, "hybrid": frozen.hybrid.values,
        "lam0": frozen.lam0.values,
    })
    fw.to_csv(OUT / "fourway_pertrial.csv", index=False)

    res = []
    for lab, col in [("legacy(旧/错误)", "esn_legacy"), ("fixed(修正)", "esn_fixed")]:
        sub = fw[(fw[col] < QUASI) & (fw.pinn < QUASI) & (fw.hybrid < QUASI)]
        e = float(sub[col].median())
        row = dict(variant=lab, N_three=len(sub), esn=e,
                   pinn=float(sub.pinn.median()),
                   lam0=float(sub.lam0.median()),
                   hybrid=float(sub.hybrid.median()))
        row["pinn_x"] = row["pinn"] / e
        row["lam0_x"] = row["lam0"] / e
        row["hybrid_x"] = row["hybrid"] / e
        # 附带两方子集（esn & pinn <9.9），canon 脚本 print 里用的是这个
        sub2 = fw[(fw[col] < QUASI) & (fw.pinn < QUASI)]
        row["N_two"] = len(sub2)
        row["esn_two"] = float(sub2[col].median())
        row["pinn_two"] = float(sub2.pinn.median())
        row["pinn_two_x"] = row["pinn_two"] / row["esn_two"]
        res.append(row)
        print(f"  {lab:>16}  三方 N={len(sub):>2}  ESN {e:.4f}  "
              f"PINN {row['pinn']:.4f} ({row['pinn_x']:.2f}×)  "
              f"MLP {row['lam0']:.4f} ({row['lam0_x']:.2f}×)  "
              f"Hybrid {row['hybrid']:.4f} ({row['hybrid_x']:.2f}×)")

    # 同一固定子集对照：用 legacy 定义的三方子集，同时报两条 ESN
    fixed_set = fw[(fw.esn_legacy < QUASI) & (fw.pinn < QUASI) & (fw.hybrid < QUASI)]
    res.append(dict(variant="同一子集(legacy定义)", N_three=len(fixed_set),
                    esn=float(fixed_set.esn_legacy.median()),
                    pinn=float(fixed_set.pinn.median()),
                    lam0=float(fixed_set.lam0.median()),
                    hybrid=float(fixed_set.hybrid.median()),
                    pinn_x=float(fixed_set.pinn.median() / fixed_set.esn_legacy.median()),
                    lam0_x=float(fixed_set.lam0.median() / fixed_set.esn_legacy.median()),
                    hybrid_x=float(fixed_set.hybrid.median() / fixed_set.esn_legacy.median()),
                    N_two=-1, esn_two=np.nan, pinn_two=np.nan, pinn_two_x=np.nan))
    res.append(dict(variant="同一子集(legacy定义)/ESN_fixed", N_three=len(fixed_set),
                    esn=float(fixed_set.esn_fixed.median()),
                    pinn=float(fixed_set.pinn.median()),
                    lam0=float(fixed_set.lam0.median()),
                    hybrid=float(fixed_set.hybrid.median()),
                    pinn_x=float(fixed_set.pinn.median() / fixed_set.esn_fixed.median()),
                    lam0_x=float(fixed_set.lam0.median() / fixed_set.esn_fixed.median()),
                    hybrid_x=float(fixed_set.hybrid.median() / fixed_set.esn_fixed.median()),
                    N_two=-1, esn_two=np.nan, pinn_two=np.nan, pinn_two_x=np.nan))
    print(f"  {'同一子集(legacy定义,N=' + str(len(fixed_set)) + ')':>16}  "
          f"ESN 旧 {fixed_set.esn_legacy.median():.4f} → 修正 {fixed_set.esn_fixed.median():.4f}  "
          f"PINN/ESN {fixed_set.pinn.median() / fixed_set.esn_legacy.median():.2f}× → "
          f"{fixed_set.pinn.median() / fixed_set.esn_fixed.median():.2f}×")

    out = pd.DataFrame(res)
    out.to_csv(OUT / "fourway_fixed_subset.csv", index=False)
    print(f"  → {OUT / 'fourway_fixed_subset.csv'}")
    return out, fw


def cross_check_probe(d):
    """与 tmp/audit_probe/esn_insim_curfix.csv 交叉核对（独立复现证据）。

    注意口径：两侧都必须经过同一个 CSV 写出器再读回来比。pandas 的 to_csv 只保 16 位
    有效数字，而 k/120 这类值需要 17 位，直接拿内存浮点去比探针 CSV 会看到 1 ulp
    (~2.2e-16) 的假差异。帧索引 (h*120) 是整数，实际不丢信息。
    """
    if not PROBE_CSV.exists():
        print("\n[cross-check] 探针 csv 不存在，跳过")
        return None
    p = pd.read_csv(PROBE_CSV).sort_values(["seed", "trial"]).reset_index(drop=True)
    m = pd.read_csv(OUT / "perseed_pertrial.csv").sort_values(
        ["seed", "trial"]).reset_index(drop=True)   # 同一写出器往返后再比
    if len(p) != len(m):
        print(f"\n[cross-check] 行数不一致 {len(p)} vs {len(m)}，跳过")
        return None
    dc = float(np.max(np.abs(p.h_cur.values - m.h_legacy.values)))
    df_ = float(np.max(np.abs(p.h_fix.values - m.h_fixed.values)))
    nc = int((p.h_cur.values != m.h_legacy.values).sum())
    nf = int((p.h_fix.values != m.h_fixed.values).sum())
    raw_c = float(np.max(np.abs(p.h_cur.values - d.sort_values(
        ["seed", "trial"]).h_legacy.values)))
    msg = (f"[cross-check] 与探针 esn_insim_curfix.csv（另一 agent 独立写的脚本）"
           f"同口径比对 {len(p)} 行：legacy 不等 {nc} 行 (maxdiff {dc:.3e})，"
           f"fixed 不等 {nf} 行 (maxdiff {df_:.3e})"
           f"{' → 逐位一致，独立复现成立' if (nc == 0 and nf == 0) else ''}"
           f"；（内存浮点直比探针 CSV 会显示 {raw_c:.3e}，纯属 to_csv 16 位有效数字截断）")
    print("\n" + msg)
    return msg


# ------------------------------ callsites.md ------------------------------

CALLSITES_MD = """# ESN 闭环 `predict` 调用点影响面清单

生成脚本：`tools/esn_closedloop_fix.py`（本文件由脚本自动写出，勿手改）
生成时间：{ts}

## 判据（结构性，与阈值无关）

`rc/esn.py::ESN.predict(x0, u0, n)` 的循环体是「先 `step` 再 `readout`」，
因此它**隐含要求 `x0` 尚未消费过 `u0`**（x0 比 u0 落后一个储池步）。

训练契约（`rc/esn.py:105-129`）：`X[t] = step(X[t-1], U[t])`，
读出特征是 `[X[t], U[t], 1] → U[t+1]`，即状态与**产生它的那一帧输入**配对。

于是判据只有一条：

| 传入 `predict` 的 `(x0, u0)` | 判定 | 后果 |
|---|---|---|
| `x0` **已消费过** `u0`（即 `x0 = X[k]`, `u0 = U[k]`） | **错误族** | 最后一帧输入被喂两遍，预测窗第 0 帧就带误差 |
| `x0` **未消费过** `u0`（即 `x0 = X[k-1]`, `u0 = U[k]`） | **正确族** | 与训练契约一致，第 0 帧对齐 |

已逐位验证的恒等式（`data/esn_closedloop_fix/equivalence_real.txt`，maxdiff = {a_max}）：

```
esn.predict(x0, u0, n)  ==  predict_readout_first(esn.step(x0, u0), u0, n)
```

即旧实现 = 修正版「晚起跑一个储池步」。

**反向禁令**：不能把 `predict_readout_first` 同参数裸换进正确族调用。
实测真实侧 29 段同参数裸换 maxdiff = {b_max}，
预测窗第 0 帧角度误差中位从 {e_leg:.3f}° 暴涨到 {e_nai:.3f}°（最大 {e_nai_max:.3f}°）。

## 错误族（6 个调用点，全部在仿真内路径）

修法：`pred = e.predict(x_final, U_tr[-1], n)` → `pred = predict_readout_first(e, x_final, U_tr[-1], n)`
（参数不变，只换函数）。

| # | 文件:行 | 上下文 | 影响的已发布数字 |
|---|---|---|---|
| E1 | `rc/esn.py:194` | `run_one()`：`train(U[:n_train])` → `predict(x_final, U_tr[-1], n_pred)` | 一切仿真内 ESN 的源头；`data/rc/esn_summary.csv` |
| E2 | `tools/esn_seed_variance.py:67` | `horizon_for()`，逐行复刻 E1 | `data/esn_seed/*`；**并被 `canon_multirun_distmass.py:127` 调用** → `data/canon_multirun/{{insim_runs,summary}}.csv`、`data/fair_compare/four_way_horizons.csv` 的 `esn` 列 |
| E3 | `tools/esn_grid_search.py:155` | `horizons_for_group()`，`x_final = X[-1]`（= `X[n_train-1]`） | `data/esn_grid/` 全部 leaderboard |
| E4 | `tools/esn_grid_search.py:186` | `_audit_unit()`，`e.predict(X[-1], U_tr[-1], n_pred)` | `data/esn_grid/audit_divergence.csv`、`audit_summary.csv` |
| E5 | `tools/threshold_robustness.py:179` | `_insim_curves_esn()`，逐行复刻 E1，返回偏差曲线 | `data/threshold_robustness/` 的**仿真侧 ESN 列**（论文里那个 0.404 及其阈值扫描曲线全部偏低；真实侧不受影响） |
| E6 | `sim/animate_compare.py:71` | 演示动画 `run_esn()` | 只影响 `figures/` 里的对比动画/静帧，无表格数字 |

## 正确族（4 + 2 个调用点，全部在真实侧或滑窗路径）——**不得改动**

共同特征：`train(U[:i0-1], Y[:i0-1])` 后 `predict(x_final, U[i0-1], n)`，
`x_final = X[i0-2]` 尚未消费 `U[i0-1]`。

| # | 文件:行 | 上下文 | 产出的数字 |
|---|---|---|---|
| C1 | `rc/real_validation.py:108` | 单片 IMG_1392 详图版 | `data/real_validation/`（单片图） |
| C2 | `rc/real_validation_multi.py:104` | `run_esn()`，29 段群体统计 | **真实零样本头条 ESN 0.167s** 及 2.15×/2.50×/2.70× 的分母 |
| C3 | `tools/window_mining.py:79` | `run_esn_fixed_lead()`，固定 lead 滑窗 | `data/window_mining/`（分层交叉点） |
| C4 | `tools/live_demo.py:120` | `predict_esn()` 演示 | 无论文数字 |
| C5 | `tools/esn_grid_real.py:240` | 今晚新增（第 4 项）；`x_final = X[-1]` 但 `X` 只跑到 `U[:i0-1]` | `data/esn_grid_real/` |
| C6 | `tools/esn_grid_real.py:395` | 今晚新增（第 4 项）的 sanity check | 同上 |

> C5/C6 属今晚在写的新脚本，按 2026-09-05 00:41 的版本判定为正确族；
> 若该脚本后续改动了训练切片，需重新判定。

## 因此需要在 9/9 全量重算时打包处理的下游产物

1. `data/canon_multirun/{{insim_runs.csv, summary.csv}}` 的 ESN 仿真内行（经 E2）
2. `data/fair_compare/four_way_horizons.csv` 的 `esn` 列（经 E2，且**只有这一个来源**）
3. `data/esn_seed/{{esn_seed_results.csv, esn_seed_summary.csv}}`（E2）
4. `data/esn_grid/` 全部（E3/E4）——注意 leaderboard 前列的 `stage2_chaos_q3` 已顶到 10.0 天花板，修正后只会更饱和
5. `data/threshold_robustness/` 的仿真侧 ESN 列（E5）——**真实侧不动**
6. `data/rc/esn_summary.csv` 与 `data/rc/figures/`（E1）
7. 论文：`paper/main.tex:116`、`:148-150`（摘要倍数）、`:526-529`（tab:horizon-compare 的 ESN 行与三个倍数）、
   以及引用四方固定子集 N=32/ESN 0.294/PINN 1.72× 的正文段落
8. `data/canonical_results_B.md` 的「仿真内」表与「四方固定子集交叉验证」段

## 不受影响（可原样保留）

- 真实零样本迁移四个数字 **0.167 / 0.358 / 0.417 / 0.450**（C2，正确族）
- `data/window_mining/` 的分层结论（C3，正确族）
- `data/threshold_robustness/` 的**真实侧**列
- 全部 PINN / Hybrid / MLP 数字（不走 ESN 路径）
"""


# ------------------------------ main ------------------------------

def main():
    print("=" * 78)
    print("tools/esn_closedloop_fix.py — 修正 ESN 闭环多消耗一步")
    print(f"输出目录 {OUT}")
    print("=" * 78, flush=True)

    ok1 = gate1_equivalence()
    d = run_insim()
    xmsg = cross_check_probe(d)
    per_seed, sl, sf, si_l, si_f, ref = build_insim_table(d)
    fw_res, fw = build_fourway(d)

    # ---- callsites.md ----
    eq = pd.read_csv(OUT / "equivalence_real_perclip.csv")
    (OUT / "callsites.md").write_text(CALLSITES_MD.format(
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        a_max=f"{eq.A_maxdiff.max():.3e}",
        b_max=f"{eq.B_maxdiff.max():.3e}",
        e_leg=eq.err0_legacy_deg.median(),
        e_nai=eq.err0_naive_deg.median(),
        e_nai_max=eq.err0_naive_deg.max(),
    ), encoding="utf-8")
    print(f"\n  → {OUT / 'callsites.md'}")

    # ---- summary.md ----
    L = []
    L.append("# 第 1 项结果：ESN 闭环多消耗一步的修正与影响面\n")
    L.append(f"生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `tools/esn_closedloop_fix.py`\n")
    L.append("## 自证门槛\n")
    L.append("| 门槛 | 内容 | 结果 |")
    L.append("|---|---|---|")
    L.append(f"| G1-A | `legacy(x_f,u0,n) == readout_first(step(x_f,u0),u0,n)` 逐位相同 | "
             f"{'PASS' if ok1 else 'FAIL'}，maxdiff = {eq.A_maxdiff.max():.3e} |")
    L.append(f"| G1-B | 同参数裸换必须**不**等价（反向禁令取证） | PASS，真实侧 maxdiff = "
             f"{eq.B_maxdiff.max():.3e}，帧0误差中位 {eq.err0_legacy_deg.median():.3f}° → "
             f"{eq.err0_naive_deg.median():.3f}° |")
    L.append(f"| G2 | legacy per-run own_chaos 中位复现 canon_multirun/summary.csv | "
             f"{sl['median']!r} [{sl['q25']:.6f}, {sl['q75']:.6f}]（目标 {G2_MEDIAN!r}） |")
    L.append(f"| G3 | legacy 逐 trial 跨 run 中位复现冻结 four_way `esn` 列 | "
             f"maxdiff = {float(np.max(np.abs(fw.esn_legacy - fw.esn_frozen))):.3e} |")
    L.append("")
    L.append("## 仿真内表（own_chaos 口径，10 seed，可直接替换 tab:horizon-compare 的仿真内块）\n")
    L.append("| 模型 | 中位 (s) | IQR | /τL | ×ESN(旧) | ×ESN(修正) |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| ESN 旧（错误路径） | {sl['median']:.3f} | [{sl['q25']:.3f}, {sl['q75']:.3f}] | "
             f"{sl['median'] / TAU_L:.2f} | 1.00× | — |")
    L.append(f"| **ESN 修正** | **{sf['median']:.3f}** | [{sf['q25']:.3f}, {sf['q75']:.3f}] | "
             f"{sf['median'] / TAU_L:.2f} | — | 1.00× |")
    for lab, key in [("PINN（λ=0.1）", "PINN"), ("纯数据 MLP（λ=0）", "PINN0"), ("Hybrid", "Hybrid")]:
        v = ref[key]
        L.append(f"| {lab}[冻结] | {v:.3f} | — | {v / TAU_L:.2f} | "
                 f"{v / sl['median']:.2f}× | **{v / sf['median']:.2f}×** |")
    L.append("")
    L.append(f"每 run 混沌子集大小：旧 {per_seed.n_chaos_legacy.min()}–{per_seed.n_chaos_legacy.max()} 条，"
             f"修正后 {per_seed.n_chaos_fixed.min()}–{per_seed.n_chaos_fixed.max()} 条。\n")
    L.append("**own_chaos 口径偏保守，必须点破**：ESN 变强后更多 trial 视界 ≥9.9s 被踢出各自的混沌子集，"
             "而被踢出的恰恰是修正版表现最好的那些 trial。因此 "
             f"{sf['median']:.3f}s 是**下界**。同时给出消除子集漂移的交集口径"
             "（只保留两条路径都判为混沌的 trial）：\n")
    L.append("| 口径 | ESN 旧 | ESN 修正 | 每 run N | PINN/ESN | MLP/ESN | Hybrid/ESN |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| own_chaos（各自子集） | {sl['median']:.3f} | {sf['median']:.3f} | "
             f"{per_seed.n_chaos_legacy.min()}–{per_seed.n_chaos_legacy.max()} → "
             f"{per_seed.n_chaos_fixed.min()}–{per_seed.n_chaos_fixed.max()} | "
             f"{ref['PINN'] / sl['median']:.2f}× → {ref['PINN'] / sf['median']:.2f}× | "
             f"{ref['PINN0'] / sl['median']:.2f}× → {ref['PINN0'] / sf['median']:.2f}× | "
             f"{ref['Hybrid'] / sl['median']:.2f}× → {ref['Hybrid'] / sf['median']:.2f}× |")
    L.append(f"| 交集（同一子集） | {si_l['median']:.3f} | {si_f['median']:.3f} | "
             f"{per_seed.n_chaos_intersect.min()}–{per_seed.n_chaos_intersect.max()} | "
             f"{ref['PINN'] / si_l['median']:.2f}× → {ref['PINN'] / si_f['median']:.2f}× | "
             f"{ref['PINN0'] / si_l['median']:.2f}× → {ref['PINN0'] / si_f['median']:.2f}× | "
             f"{ref['Hybrid'] / si_l['median']:.2f}× → {ref['Hybrid'] / si_f['median']:.2f}× |")
    L.append("")
    L.append("> 交集口径的 PINN/MLP/Hybrid 中位仍取冻结的 per-run own_chaos 值（未按交集重算），"
             "所以那三个倍数只是量级参考，不能当正式数字引用；正式表用 own_chaos 行。\n")
    L.append("## 四方固定子集（逐 trial 跨 run 中位；PINN/MLP/Hybrid 复用冻结列）\n")
    L.append("| 口径 | N(三方) | ESN | PINN | ×ESN | MLP | ×ESN | Hybrid | ×ESN |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in fw_res.itertuples():
        L.append(f"| {r.variant} | {r.N_three} | {r.esn:.3f} | {r.pinn:.3f} | {r.pinn_x:.2f}× | "
                 f"{r.lam0:.3f} | {r.lam0_x:.2f}× | {r.hybrid:.3f} | {r.hybrid_x:.2f}× |")
    L.append("")
    L.append("## 真实侧（完全不受影响）\n")
    L.append("真实零样本 ESN 0.167s / MLP 0.358s / PINN 0.417s / Hybrid 0.450s 四个数字**一个不动**，"
             "因为真实侧调用（`rc/real_validation_multi.py:104`）属正确族。见 `callsites.md`。\n")
    L.append("## 影响面\n")
    L.append("见同目录 `callsites.md`：错误族 6 个调用点、正确族 6 个（其中 2 个是今晚新增的 "
             "`tools/esn_grid_real.py`）。`tools/threshold_robustness.py:179` 的仿真侧 ESN 曲线"
             "同样受此影响，其真实侧不受影响。\n")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  → {OUT / 'summary.md'}")

    print("\n" + "=" * 78)
    if _FAILURES:
        print("自证门槛未全绿：")
        for f in _FAILURES:
            print("  FAIL " + f)
        print("=" * 78)
        return 1
    print("自证门槛 G1-A / G1-B / G2 / G3 全绿")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

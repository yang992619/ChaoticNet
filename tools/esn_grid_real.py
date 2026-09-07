"""
tools/esn_grid_real.py — 真实侧 ESN 的「引入窗长度 × 超参」二维扫描

============================ 动机 ============================
论文真实侧头条「物理先验 ≈ ESN 的 2.5–2.7×」的分母（ESN 0.167s），是在
  (a) 一个从未调过参的配置上，且
  (b) 与论文附录超参表写的配置不一致（附录写 800/0.95/0.25/1e-5，真实侧实跑的是
      rc/real_validation_multi.py:96 的 300/0.95/0.3/1e-3/washout=50），且
  (c) 只给 ESN 4 秒引入窗
算出来的。评委问一句「你给 ESN 只喂 4 秒它当然差」就能击穿。

与其被问，不如自己把交叉曲线测出来：**纯数据 ESN 需要每段多少秒实测数据现场自训，
才能追平一秒真实数据都不用的零样本物理模型？** 这个「跳过物理先验要付多少数据代价」
的定量答案，比一个单点倍数强得多。

============================ 两条硬约束（务必保留） ============================
【硬约束 1】绝不把仿真侧的 predict_readout_first 修正套到真实侧调用上。
  rc/esn.py::predict 的循环体是「先 x=step(x,u) 再 readout」。仿真侧
  (rc/esn.py:194 等 6 个调用点) 传入的 u0 = U_tr[-1] 是**训练段最后一帧**，该帧已被
  train() 的 collect_states 消费过，于是被喂了两遍 → 预测窗第 0 帧就带误差，属"错误族"。
  真实侧 (rc/real_validation_multi.py:104) 传入的 u0 = U[i0-1] 是**训练切片之外的下一帧**
  (训练用 U[:i0-1])，没有被消费过 → 姿势正确，属"正确族"。
  同参数裸换成 readout-first 会把 0.167/0.358/0.417/0.450 改坏
  （已实测：帧 0 误差从 0.25 度暴涨到 9.59 度）。本脚本一律用 rc/esn.py 原生 predict，
  且 u0 严格取 U[i0-1]。等价性已逐位验证：见 equivalence_rvm.txt（maxdiff 0.0）。

【硬约束 2】lead≥16s 之后 ESN 中位会顶到 4s 评估窗上沿。每一格都必须同时落盘
  删失数(censored count) 与删失率，交叉点成立但**交叉后的倍数只能报为下界**。
  没有删失列的表不许出。右删失下：朴素中位是真中位的**下界**；因此
  「X/ESN 倍数」在 ESN 删失时是**上界**，"倍数<1（ESN 反超）"这个方向是安全的。

============================ 设计 ============================
- 片集 = rvm.DEFAULT_TAGS 去掉 1392，共 29 段（与论文真实侧完全同集）。
- pred_s 固定 4.0 秒不动（保证与论文口径可比）；只动 lead_s。
- 轴 1 lead_s ∈ {2,4,8,16,32,64}（已验证 29 片全部可行：最短片 N=9716，
  lead=64 时 lead_n=3838 < N//2=4858）。另有 Stage3 细扫补中间档定位交叉点。
- 轴 2：沿用 tools/esn_grid_search.py 的 144 组网格
  n_res∈{400,800,1600} × ρ∈{0.7,0.9,0.95,1.1} × leak∈{0.1,0.25,0.5,1.0} × ridge∈{1e-8,1e-6,1e-4}
  + canonical-real(300/0.95/0.3/1e-3, washout50) + paper-ESN(800/0.95/0.25/1e-5, washout50)
  + paper-ESN-w200(同上但 washout=200，检验 washout 口径是否影响结论) = 147 组。
- washout 随 lead 自适应：min(washout, max(lead_n//3, 1))，与 rvm.run_esn 逐行一致。
- 两阶段 + 细扫：
  Stage1 单 seed(42) 全网格 × 6 个 lead 档；
  Stage2 每个 lead 档 top-8（按删失感知的严格视界排名）+ 两个参照配置 × seed{42,7,13}；
  Stage3 少数关键配置 × 11 个 lead 细档，用来定位交叉点。
- 每档必带**同窗口**对照 PINN / Hybrid / MLP(λ=0)，排除
  「窗口后移 = 摆幅衰减 = 题目变简单」这个混淆。

============================ 自证门槛（不过就退出） ============================
G1 融合路径等价：fused(canonical-real) 与 rvm.run_esn(C) 逐位相同（实测 0.0）。
G2 ESN 逐片复现：canonical-real + lead=4 + seed=42 逐片对上
   data/real_validation/summary_multi.csv 的 ESN_h，最大差 < 1e-3（实测 4.510e-4，
   纯 csv 三位小数舍入）。
G3 对照组复现：lead=4 → PINN 0.417 / Hybrid 0.484；lead=16 → 0.884 / 1.000；
   lead=32 → 1.284 / 1.267（容差 1e-3）。

============================ 性能 ============================
清单里 4.6h 的估算是"无缓存"口径（每配置每片都重建储池，n_res=1600 时
ESN.__init__ 里的 eigvals 单次就要 0.43s）。本脚本做两处缓存后实测快约 5 倍：
  1) 储池 (W_in, W*mask, eig_max) 只依赖 (n_res, seed)，与 ρ 无关（ρ 只是缩放同一个 W）
     → 每 (n_res, seed) 只建一次。rng 抽取顺序严格保持 W_in→W→mask，否则复现不了。
  2) 固定 (tag, lead, n_res, seed, ρ, leak) 时 collect_states 与 Gram 矩阵只算一次，
     3 个 ridge 共用，只是解不同岭回归。
单进程，OMP/MKL/OPENBLAS/VECLIB/NUMEXPR 线程数全设 1。

输出：data/esn_grid_real/（新目录，不碰任何既有产物目录）
运行：cd <仓库根> && OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 tools/esn_grid_real.py
探针：加 --probe（2 分钟内跑完自证门槛 + 极小子集）
"""

import os

# 必须在 import numpy 之前限制 BLAS 线程（本机 10 核且有其他计算任务在跑，本进程只占 1 核）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                       # noqa: E402
import time                      # noqa: E402
import json                      # noqa: E402
import argparse                  # noqa: E402
import itertools                 # noqa: E402
from pathlib import Path         # noqa: E402

import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import torch                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import esn as esn_mod                    # noqa: E402
import pinn as pinn_mod                  # noqa: E402
import real_validation_multi as rvm      # noqa: E402

torch.set_num_threads(1)

# ======================= 参数常量区 =======================
OUT = ROOT / "data" / "esn_grid_real"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

REF_CSV = ROOT / "data" / "real_validation" / "summary_multi.csv"   # 只读
LAM0_PT = ROOT / "data" / "fair_compare" / "lam0_model.pt"          # 只读

TAGS = [t for t in rvm.DEFAULT_TAGS if t != "1392"]   # 29 段（1392 为无效片，论文已剔）
PRED_S = 4.0                                          # 固定不动，保证与论文口径可比
THRESH_DEG = 10.0                                     # 与 rvm 一致

LEADS_MAIN = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
LEADS_FINE = [2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0, 22.0, 32.0, 45.0, 64.0]

GRID_NRES = [400, 800, 1600]
GRID_SR = [0.7, 0.9, 0.95, 1.1]
GRID_LEAK = [0.1, 0.25, 0.5, 1.0]
GRID_RIDGE = [1e-8, 1e-6, 1e-4]
BASE_WASHOUT = 50                       # rvm.run_esn 真实侧口径

# 参照配置（不属于网格，单独加）
CANON_REAL = dict(n_res=300, sr=0.95, leak=0.3, ridge=1e-3, washout=50,
                  name="canonical-real")     # rc/real_validation_multi.py:95 实跑值
PAPER_ESN = dict(n_res=800, sr=0.95, leak=0.25, ridge=1e-5, washout=50,
                 name="paper-ESN(w50)")      # 论文附录超参表（washout 按真实侧口径取 50）
PAPER_ESN_W200 = dict(n_res=800, sr=0.95, leak=0.25, ridge=1e-5, washout=200,
                      name="paper-ESN(w200)")  # 附录+仿真侧 washout，检验 washout 敏感性
NAMED = [CANON_REAL, PAPER_ESN, PAPER_ESN_W200]

SEED1 = 42
SEEDS2 = [42, 7, 13]
STAGE2_TOPK = 8

MAX_HOURS = 6.0                # 墙钟预算；超了就跳过剩余阶段并把已有结果落盘
CENS_EPS = 1e-9

# ======================= 缓存 =======================
_CLIP = {}    # tag -> 基础量（与 lead 无关）
_RES = {}     # (n_res, seed) -> (W_in, W_masked, eig_max)
_CTRL = {}    # lead_s -> DataFrame(29 行控制组结果)
_NETS = {}


def clip_base(tag):
    """读一次 csv + savgol，得到与 lead 无关的量。逐行复现 rvm.load_clip 的前半段。"""
    if tag in _CLIP:
        return _CLIP[tag]
    df = pd.read_csv(rvm.TRACK / f"tracked_{tag}.csv")
    t = df["t"].values
    t = t - t[0]
    fps = 1.0 / np.median(np.diff(t))
    dt = 1.0 / fps
    N = len(t)
    th1 = np.unwrap(df["th1"].values)
    th2 = np.unwrap(df["th2"].values)
    win = rvm.SAVGOL_WIN if N > rvm.SAVGOL_WIN else (N // 2) * 2 - 1
    from scipy.signal import savgol_filter
    th1s = savgol_filter(th1, win, rvm.SAVGOL_POLY)
    th2s = savgol_filter(th2, win, rvm.SAVGOL_POLY)
    w1 = np.gradient(th1s, dt)
    w2 = np.gradient(th2s, dt)
    state = np.column_stack([th1s, w1, th2s, w2])
    U = esn_mod.encode(pd.DataFrame({"th1": th1s, "w1": w1, "th2": th2s, "w2": w2}))
    Y = np.roll(U, -1, axis=0)
    _CLIP[tag] = dict(tag=tag, t=t, fps=fps, dt=dt, N=N, th1=th1s, th2=th2s,
                      w1=w1, w2=w2, state=state, U=U, Y=Y)
    return _CLIP[tag]


def clip_view(tag, lead_s, pred_s=PRED_S):
    """给定 lead 的切窗量。lead_n / pred_n / rel_deg 的算法与 rvm.load_clip 逐行一致。"""
    B = clip_base(tag)
    N, fps = B["N"], B["fps"]
    lead_n = min(int(lead_s * fps), N // 2)
    pred_n = min(int(pred_s * fps), N - lead_n - 1)
    th1s = B["th1"]
    rel_deg = float(np.degrees(np.max(np.abs(th1s[:max(lead_n + pred_n, 1)]))))
    i0, n = lead_n, pred_n
    amp_pred = float(np.degrees(np.max(np.abs(th1s[i0:i0 + n]))))
    back = min(i0, int(2.0 * fps))
    amp_leadend = float(np.degrees(np.max(np.abs(th1s[max(i0 - back, 0):i0 + 1]))))
    wmax_pred = float(np.max(np.abs(B["w1"][i0:i0 + n])))
    return dict(B, lead_n=lead_n, pred_n=pred_n, rel_deg=rel_deg,
                amp_pred_deg=amp_pred, amp_leadend_deg=amp_leadend,
                wmax_pred=wmax_pred, lead_s_req=lead_s)


def get_res(n_res, seed):
    """建储池。rng 抽取顺序必须是 W_in → W → mask（与 ESN.__init__ 逐行一致），
    否则复现不了 canonical。返回未做谱半径缩放的 W（已乘 mask）与最大特征值模。"""
    key = (n_res, seed)
    if key not in _RES:
        rng = np.random.default_rng(seed)
        W_in = rng.uniform(-1.0, 1.0, (n_res, 6))     # input_scale=1.0（ESN 默认）
        W = rng.uniform(-1, 1, (n_res, n_res))
        mask = rng.random((n_res, n_res)) < 0.1       # sparsity=0.1（ESN 默认）
        W = W * mask
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        _RES[key] = (W_in, W, eig_max)
    return _RES[key]


# ======================= ESN 核心（真实侧口径） =======================

def esn_group(tag, lead_s, n_res, seed, sr, leak, ridges, washout):
    """固定 (tag, lead, n_res, seed, ρ, leak)：一次 collect_states + 一次 Gram，
    多个 ridge 共用 → 返回 {ridge: 结果 dict}。

    与 rc/real_validation_multi.run_esn 逐位等价（已在 G1 门槛验证 maxdiff=0.0）：
      训练切片 U[:i0-1] / Y[:i0-1]，washout=min(w, max(i0//3,1))；
      x0 = X[-1]；**u0 = U[i0-1]**（训练切片之外的下一帧，未被消费过 → 正确族）；
      闭环用 rc/esn.py 原生 predict（先 step 再 readout）。
    这里的 u0 取法是硬约束 1 的落点：绝不改成 U_tr[-1]，也绝不换 readout-first。
    """
    C = clip_view(tag, lead_s)
    i0, n, fps = C["lead_n"], C["pred_n"], C["fps"]
    U, Y = C["U"], C["Y"]
    W_in, W_raw, eig_max = get_res(n_res, seed)

    e = object.__new__(esn_mod.ESN)
    e.n_in, e.n_out, e.n_res = 6, 6, n_res
    e.leak, e.ridge = leak, ridges[0]
    e.W_in = W_in
    e.W = W_raw * (sr / eig_max)          # 与 ESN.__init__ 同样的乘法顺序
    e.W_out = None

    U_tr, Y_tr = U[:i0 - 1], Y[:i0 - 1]
    wo = min(washout, max(i0 // 3, 1))
    with np.errstate(all="ignore"):
        X = e.collect_states(U_tr)
        H = np.column_stack([X, U_tr, np.ones(len(X))])
        Ht, Yt = H[wo:], Y_tr[wo:]
        G = Ht.T @ Ht
        B = Ht.T @ Yt
        I = np.eye(Ht.shape[1])
        x_final, u0 = X[-1], U[i0 - 1]
        tt1, tt2 = C["th1"][i0:i0 + n], C["th2"][i0:i0 + n]
        out = {}
        for rg in ridges:
            e.ridge = rg
            e.W_out = np.linalg.solve(G + rg * I, B).T
            pred = e.predict(x_final, u0, n)
            finite = np.isfinite(pred).all(axis=1)
            n_bad = int((~finite).sum())
            first_bad = int(np.argmax(~finite)) if n_bad else -1
            p1, _, p2, _ = esn_mod.decode(pred)
            err = rvm.angle_err_deg(p1, p2, tt1, tt2)
            h, cens = rvm.horizon_s(err, fps, THRESH_DEG)
            # 保守视界：一旦出现非有限预测就判失败（防 rc/esn.py::horizon_s 把
            # 全 nan 误判成"整窗未超阈"的假满分）
            h_strict = h if n_bad == 0 else min(h, first_bad / fps)
            cens_strict = bool(cens and n_bad == 0)
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.mean(np.nan_to_num(err, nan=1e6,
                                                          posinf=1e6, neginf=1e6) ** 2)))
            out[rg] = dict(horizon=float(h), censored=int(cens),
                           horizon_strict=float(h_strict),
                           censored_strict=int(cens_strict),
                           n_nonfinite=n_bad, rms=min(rms, 1e6))
    return out, C


# ======================= 控制组（同窗口 PINN / Hybrid / MLP） =======================

def get_nets():
    if not _NETS:
        _NETS["PINN"] = rvm.load_pinn()
        _NETS["Hybrid"] = rvm.load_hybrid()
        m = pinn_mod.AccelNet(hidden=128, n_layers=4)
        m.load_state_dict(torch.load(LAM0_PT, map_location=pinn_mod.DEVICE))
        m.to(pinn_mod.DEVICE).eval()
        _NETS["MLP"] = m
    return _NETS


def controls(lead_s):
    """同窗口对照：三个物理/数据 torch 模型都从同一个 i0 起步、对同一段真值评估。
    这是排除「窗口后移 → 摆幅因阻尼衰减 → 题目变简单」这个混淆的关键。
    它们**一秒真实数据都不用**（零样本），窗口只影响初值与被预测的那一段。"""
    if lead_s in _CTRL:
        return _CTRL[lead_s]
    nets = get_nets()
    rows = []
    for tg in TAGS:
        C = clip_view(tg, lead_s)
        i0, n, fps, dt = C["lead_n"], C["pred_n"], C["fps"], C["dt"]
        tt1, tt2 = C["th1"][i0:i0 + n], C["th2"][i0:i0 + n]
        s0 = C["state"][i0].astype(np.float32)
        r = dict(lead_s=lead_s, tag=tg, fps=fps, lead_n=i0, pred_n=n,
                 win_s=n / fps, rel_deg=C["rel_deg"],
                 amp_pred_deg=C["amp_pred_deg"],
                 amp_leadend_deg=C["amp_leadend_deg"], wmax_pred=C["wmax_pred"])
        for nm, net in nets.items():
            p1, p2 = rvm.run_torch(net, s0, n, dt)
            err = rvm.angle_err_deg(p1, p2, tt1, tt2)
            h, cens = rvm.horizon_s(err, fps, THRESH_DEG)
            r[f"{nm}_h"] = float(h)
            r[f"{nm}_cens"] = int(cens)
            r[f"{nm}_rms"] = float(np.sqrt(np.mean(err ** 2)))
        rows.append(r)
    df = pd.DataFrame(rows)
    _CTRL[lead_s] = df
    return df


# ======================= 统计工具 =======================

def agg_cell(hs, cs, hstrict, cstrict, nbad, rms):
    """一格（一个配置 × 一个 lead × 29 片）的汇总。

    右删失说明：删失片的视界值等于评估窗长(≈3.985s)，是真值的**下界**。
    因此朴素中位是真中位的下界；当删失率 ≥50% 时中位不可估计，只能报 '>窗长'。
    """
    hs = np.asarray(hs, float)
    hstrict = np.asarray(hstrict, float)
    n = len(hs)
    cr = float(np.mean(cs))
    return dict(
        n_clips=n,
        median=float(np.median(hs)),
        q1=float(np.percentile(hs, 25)), q3=float(np.percentile(hs, 75)),
        mean=float(np.mean(hs)),
        n_censored=int(np.sum(cs)), cens_rate=cr,
        median_estimable=int(cr < 0.5),          # 删失率≥50% → 中位不可估计，只能报下界
        median_is_lower_bound=int(np.sum(cs) > 0),
        median_strict=float(np.median(hstrict)),
        n_censored_strict=int(np.sum(cstrict)),
        n_clips_nonfinite=int(np.sum(np.asarray(nbad) > 0)),
        rms_median=float(np.median(rms)),
        median_uncens=(float(np.median(hs[~np.asarray(cs, bool)]))
                       if (~np.asarray(cs, bool)).any() else float("nan")),
    )


def cfg_id(c):
    return (f"nres{c['n_res']}_sr{c['sr']}_leak{c['leak']}_"
            f"ridge{c['ridge']:g}_wo{c['washout']}")


def build_configs():
    cfgs = []
    for n, sr, lk, rg in itertools.product(GRID_NRES, GRID_SR, GRID_LEAK, GRID_RIDGE):
        cfgs.append(dict(n_res=n, sr=sr, leak=lk, ridge=rg,
                         washout=BASE_WASHOUT, name=""))
    have = {cfg_id(c) for c in cfgs}
    for c in NAMED:
        cc = dict(c)
        if cfg_id(cc) in have:                # 名字挂到已有网格格子上
            for x in cfgs:
                if cfg_id(x) == cfg_id(cc):
                    x["name"] = cc["name"]
        else:
            cfgs.append(cc)
            have.add(cfg_id(cc))
    return cfgs


def group_configs(cfgs):
    """按 (n_res, sr, leak, washout) 归并，同组共享 collect_states 与 Gram。"""
    g = {}
    for c in cfgs:
        k = (c["n_res"], c["sr"], c["leak"], c["washout"])
        g.setdefault(k, [])
        if c["ridge"] not in g[k]:
            g[k].append(c["ridge"])
    return [(k, sorted(v)) for k, v in sorted(g.items())]


# ======================= 自证门槛 =======================

def gates(strict=True):
    print("=" * 78, flush=True)
    print("自证门槛", flush=True)
    print("=" * 78, flush=True)
    lines = []
    ok = True

    # G1 融合路径 vs rvm.run_esn 逐位等价
    C = rvm.load_clip("1430", 4.0, PRED_S)
    a1, a2 = rvm.run_esn(C)               # 默认即 canonical-real
    res, Cv = esn_group("1430", 4.0, CANON_REAL["n_res"], SEED1, CANON_REAL["sr"],
                        CANON_REAL["leak"], [CANON_REAL["ridge"]], CANON_REAL["washout"])
    # 重算一次角度序列用于逐位比对
    W_in, W_raw, eig = get_res(CANON_REAL["n_res"], SEED1)
    e = object.__new__(esn_mod.ESN)
    e.n_in, e.n_out, e.n_res = 6, 6, CANON_REAL["n_res"]
    e.leak, e.ridge = CANON_REAL["leak"], CANON_REAL["ridge"]
    e.W_in = W_in
    e.W = W_raw * (CANON_REAL["sr"] / eig)
    i0 = Cv["lead_n"]
    Utr, Ytr = Cv["U"][:i0 - 1], Cv["Y"][:i0 - 1]
    wo = min(CANON_REAL["washout"], max(i0 // 3, 1))
    X = e.collect_states(Utr)
    H = np.column_stack([X, Utr, np.ones(len(X))])
    Ht, Yt = H[wo:], Ytr[wo:]
    e.W_out = np.linalg.solve(Ht.T @ Ht + e.ridge * np.eye(Ht.shape[1]), Ht.T @ Yt).T
    pred = e.predict(X[-1], Cv["U"][i0 - 1], Cv["pred_n"])
    b1, _, b2, _ = esn_mod.decode(pred)
    d1 = float(np.max(np.abs(a1 - b1)))
    d2 = float(np.max(np.abs(a2 - b2)))
    g1 = (d1 == 0.0 and d2 == 0.0)
    ok &= g1
    msg = f"G1 融合路径 vs rvm.run_esn 逐位等价：θ1 maxdiff={d1:.3e} θ2 maxdiff={d2:.3e}  [{'PASS' if g1 else 'FAIL'}]"
    print("  " + msg, flush=True)
    lines.append(msg)

    # G2 ESN 逐片复现 summary_multi.csv
    ref = pd.read_csv(REF_CSV)
    ref["tag"] = ref["tag"].astype(str)
    refm = dict(zip(ref["tag"], ref["ESN_h"]))
    dmax, hs = 0.0, []
    for tg in TAGS:
        r, C2 = esn_group(tg, 4.0, CANON_REAL["n_res"], SEED1, CANON_REAL["sr"],
                          CANON_REAL["leak"], [CANON_REAL["ridge"]], CANON_REAL["washout"])
        h = r[CANON_REAL["ridge"]]["horizon"]
        hs.append(h)
        dmax = max(dmax, abs(h - refm[tg]))
    g2 = dmax < 1e-3
    ok &= g2
    msg = (f"G2 canonical-real+lead4+seed42 逐片复现 summary_multi.csv 的 ESN_h："
           f"29 片最大差={dmax:.3e} (<1e-3)，中位={np.median(hs):.4f}s  [{'PASS' if g2 else 'FAIL'}]")
    print("  " + msg, flush=True)
    lines.append(msg)

    # G3 对照组复现
    exp = {4.0: (0.417, 0.484), 16.0: (0.884, 1.000), 32.0: (1.284, 1.267)}
    for ld, (ep, eh) in exp.items():
        d = controls(ld)
        mp, mh = float(np.median(d["PINN_h"])), float(np.median(d["Hybrid_h"]))
        gg = abs(mp - ep) < 1e-3 and abs(mh - eh) < 1e-3
        ok &= gg
        msg = (f"G3 lead={ld:>4.0f}s 对照组：PINN={mp:.4f}(期望{ep}) "
               f"Hybrid={mh:.4f}(期望{eh})  [{'PASS' if gg else 'FAIL'}]")
        print("  " + msg, flush=True)
        lines.append(msg)

    (OUT / "equivalence_rvm.txt").write_text(
        "真实侧 ESN 融合路径等价性 + 自证门槛（tools/esn_grid_real.py）\n"
        "口径：u0 = U[i0-1]（训练切片外的下一帧，未被消费过），闭环用 rc/esn.py 原生 predict。\n"
        "绝不套用仿真侧 predict_readout_first 修正——真实侧本来就是正确族。\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → 写 {OUT/'equivalence_rvm.txt'}", flush=True)
    if strict and not ok:
        print("\n[FATAL] 自证门槛未通过，停止。不要带着未解释的偏差往下推结论。", flush=True)
        sys.exit(2)
    print(flush=True)
    return ok


# ======================= 扫描主体 =======================

_PC_PATH = OUT / "per_clip_raw.csv"
_pc_header_done = False


def flush_perclip(rows):
    """逐组增量落盘，脚本被中断时已有结果不丢。"""
    global _pc_header_done
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(_PC_PATH, mode="a", header=not _pc_header_done, index=False)
    _pc_header_done = True


def scan(stage, leads, cfgs, seeds, t_start, budget_s, tag_subset=None):
    """跑一批 (lead × 配置 × seed × 片)。返回聚合表 DataFrame。"""
    tags = tag_subset or TAGS
    groups = group_configs(cfgs)
    name_by_id = {cfg_id(c): c.get("name", "") for c in cfgs}
    n_units = len(leads) * len(groups) * len(seeds)
    print(f"=== {stage}：{len(cfgs)} 配置 → {len(groups)} 个状态组 × "
          f"{len(leads)} lead 档 × {len(seeds)} seed × {len(tags)} 片 "
          f"= {n_units} 组单元 ===", flush=True)
    agg_rows, done = [], 0
    aborted = False
    for lead in leads:
        controls(lead)      # 保证该 lead 的同窗口对照已算
        for seed in seeds:
            for (n_res, sr, leak, wo), ridges in groups:
                if time.time() - t_start > budget_s:
                    print(f"  [预算耗尽] 已用 {(time.time()-t_start)/3600:.2f}h > "
                          f"{budget_s/3600:.2f}h，中止 {stage} 剩余部分", flush=True)
                    aborted = True
                    break
                t0 = time.time()
                acc = {rg: dict(h=[], c=[], hs=[], cs=[], nb=[], rms=[], tags=[],
                                amp=[], amp_le=[], rel=[]) for rg in ridges}
                pc = []
                for tg in tags:
                    res, C = esn_group(tg, lead, n_res, seed, sr, leak, ridges, wo)
                    for rg in ridges:
                        r = res[rg]
                        a = acc[rg]
                        a["h"].append(r["horizon"]); a["c"].append(r["censored"])
                        a["hs"].append(r["horizon_strict"]); a["cs"].append(r["censored_strict"])
                        a["nb"].append(r["n_nonfinite"]); a["rms"].append(r["rms"])
                        a["tags"].append(tg); a["amp"].append(C["amp_pred_deg"])
                        a["amp_le"].append(C["amp_leadend_deg"]); a["rel"].append(C["rel_deg"])
                        cid = cfg_id(dict(n_res=n_res, sr=sr, leak=leak, ridge=rg, washout=wo))
                        pc.append(dict(stage=stage, lead_s=lead, seed=seed, tag=tg,
                                       config_id=cid, name=name_by_id.get(cid, ""),
                                       n_res=n_res, sr=sr, leak=leak, ridge=rg, washout=wo,
                                       lead_n=C["lead_n"], pred_n=C["pred_n"],
                                       win_s=C["pred_n"] / C["fps"],
                                       rel_deg=C["rel_deg"], amp_pred_deg=C["amp_pred_deg"],
                                       amp_leadend_deg=C["amp_leadend_deg"],
                                       **{k: r[k] for k in ("horizon", "censored",
                                                            "horizon_strict", "censored_strict",
                                                            "n_nonfinite", "rms")}))
                flush_perclip(pc)
                for rg in ridges:
                    a = acc[rg]
                    cid = cfg_id(dict(n_res=n_res, sr=sr, leak=leak, ridge=rg, washout=wo))
                    agg_rows.append(dict(
                        stage=stage, lead_s=lead, seed=seed, config_id=cid,
                        name=name_by_id.get(cid, ""), n_res=n_res, sr=sr, leak=leak,
                        ridge=rg, washout=wo,
                        **agg_cell(a["h"], a["c"], a["hs"], a["cs"], a["nb"], a["rms"])))
                done += 1
                if done % 8 == 0 or done == 1:
                    print(f"  [{stage} {done}/{n_units}] lead={lead:g} seed={seed} "
                          f"n_res={n_res} sr={sr} leak={leak} wo={wo} "
                          f"({time.time()-t0:.2f}s/组, 累计 {(time.time()-t_start)/60:.1f} min)",
                          flush=True)
            if aborted:
                break
        if aborted:
            break
    df = pd.DataFrame(agg_rows)
    print(f"  {stage} 完成 {len(df)} 格，累计 {(time.time()-t_start)/60:.1f} min\n", flush=True)
    return df, aborted


# ======================= 汇总 / 出图 =======================

def ctrl_table(leads):
    rows = []
    for ld in leads:
        d = controls(ld)
        for m in ("PINN", "Hybrid", "MLP"):
            h = d[f"{m}_h"].values
            c = d[f"{m}_cens"].values
            rows.append(dict(lead_s=ld, model=m,
                             **agg_cell(h, c, h, c, np.zeros(len(h)), d[f"{m}_rms"].values)))
    return pd.DataFrame(rows)


def make_summary(s1, s2, s3, ctrl, t_start, notes):
    L = []
    A = L.append

    def cell(df, lead, cid, seed=SEED1):
        q = df[(df.lead_s == lead) & (df.config_id == cid) & (df.seed == seed)]
        return q.iloc[0] if len(q) else None

    can_id = cfg_id(CANON_REAL)
    pap_id = cfg_id(PAPER_ESN)
    all_esn = pd.concat([x for x in (s1, s2, s3) if x is not None and len(x)],
                        ignore_index=True)

    A("# 真实侧 ESN「引入窗长度 × 超参」二维扫描")
    A("")
    A("> 一句话结论候选：**纯数据 ESN 需要每段多少秒实测数据现场自训，才能追平**")
    A("> **一秒真实数据都不用的零样本物理模型？** 下面把这条交叉曲线测出来。")
    A("")
    A("脚本 `tools/esn_grid_real.py`（新建，不改 rc/esn.py 与 rc/real_validation_multi.py）。")
    A(f"产物目录 `data/esn_grid_real/`。单进程，BLAS 线程全设 1。用时 "
      f"{(time.time()-t_start)/60:.1f} min。")
    A("")
    A("## 0. 自证门槛（全部通过才有下文）")
    A("")
    for ln in (OUT / "equivalence_rvm.txt").read_text(encoding="utf-8").splitlines():
        if ln.startswith(("G1", "G2", "G3")):
            A(f"- {ln}")
    A("")
    A("## 1. 论文缺失的真实侧 ESN 配置（必须补进附录）")
    A("")
    A("| 名称 | n_res | ρ | leak | ridge | washout | 出处 |")
    A("|---|---|---|---|---|---|---|")
    A("| canonical-real（**真实侧实跑值**） | 300 | 0.95 | 0.3 | 1e-3 | 50 | `rc/real_validation_multi.py:95` |")
    A("| paper-ESN（**论文附录超参表**） | 800 | 0.95 | 0.25 | 1e-5 | 200(仿真侧) | 论文附录 / `rc/esn.py:189-191` |")
    A("")
    A("两者**不是同一套**。论文真实侧报的 0.167s 来自上面那一行，读者按附录那一行根本复现不出来。")
    A("washout 在真实侧还会被 `min(washout, lead_n//3)` 截断（lead=4 时 lead_n=239 → 79），")
    A("所以 paper-ESN 在 washout=50 与 200 两种写法下也不同，本报告两种都跑（见 leaderboard）。")
    A("")

    A("## 2. 主表：各 lead 档的中位视界（29 片，4s 预测窗，seed=42）")
    A("")
    A("**删失说明（硬性）**：预测窗固定 4.0s（实际 239 帧 ≈ 3.985s）。整窗未超 10° 阈值的片记为")
    A("**右删失**，其视界值等于窗长，是真值的**下界**。因此表中的中位在有删失时是**下界**；")
    A("删失率 ≥50% 时中位不可估计，只能写 `>3.985`。")
    A("")
    A("| lead (s) | ESN canonical-real | 删失 | ESN paper-cfg | 删失 | ESN 网格最优 | 删失 | 最优配置 | PINN | Hybrid | MLP(λ=0) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")

    def fmt(r):
        if r is None:
            return "—", "—"
        m = f"{r['median']:.3f}"
        if r["median_is_lower_bound"]:
            m = "≥" + m
        if not r["median_estimable"]:
            m = f">{r['median']:.3f}"
        return m, f"{int(r['n_censored'])}/{int(r['n_clips'])} ({100*r['cens_rate']:.0f}%)"

    best_by_lead = {}
    for ld in LEADS_MAIN:
        c = cell(s1, ld, can_id)
        p = cell(s1, ld, pap_id)
        g = s1[(s1.lead_s == ld) & (s1.seed == SEED1)]
        g = g.sort_values(["median_strict", "median", "mean"], ascending=False,
                          kind="mergesort")
        b = g.iloc[0] if len(g) else None
        best_by_lead[ld] = b
        cm, cc = fmt(c); pm, pc_ = fmt(p); bm, bc = fmt(b)
        ct = ctrl[ctrl.lead_s == ld].set_index("model")
        pn = f"{ct.loc['PINN','median']:.3f}" if 'PINN' in ct.index else "—"
        hy = f"{ct.loc['Hybrid','median']:.3f}" if 'Hybrid' in ct.index else "—"
        ml = f"{ct.loc['MLP','median']:.3f}" if 'MLP' in ct.index else "—"
        for m_, k_ in (('PINN', 'pn'), ('Hybrid', 'hy'), ('MLP', 'ml')):
            if m_ in ct.index and ct.loc[m_, 'n_censored'] > 0:
                v = {'pn': pn, 'hy': hy, 'ml': ml}[k_]
                s = f"≥{v}(删{int(ct.loc[m_,'n_censored'])})"
                if k_ == 'pn':
                    pn = s
                elif k_ == 'hy':
                    hy = s
                else:
                    ml = s
        bname = (f"{int(b.n_res)}/{b.sr}/{b.leak}/{b.ridge:g}" if b is not None else "—")
        A(f"| {ld:g} | {cm} | {cc} | {pm} | {pc_} | {bm} | {bc} | {bname} | {pn} | {hy} | {ml} |")
    A("")

    A("## 3. 倍数表（分母 = ESN）与交叉点")
    A("")
    A("倍数在 ESN 删失时是**上界**（因为 ESN 中位是下界）。所以「倍数 < 1，即 ESN 反超」")
    A("这个方向的结论是**安全的**；而交叉之后倍数具体是多少，只能报为下界（1/上界）。")
    A("")
    A("| lead (s) | PINN/ESN(canon) | Hybrid/ESN(canon) | MLP/ESN(canon) | PINN/ESN(best) | Hybrid/ESN(best) | ESN 删失(canon/best) |")
    A("|---|---|---|---|---|---|---|")
    cross = {}
    for ld in LEADS_MAIN:
        c = cell(s1, ld, can_id)
        b = best_by_lead[ld]
        ct = ctrl[ctrl.lead_s == ld].set_index("model")

        def rat(mo, e):
            if e is None or e["median"] <= 0 or mo not in ct.index:
                return "—", None
            v = ct.loc[mo, "median"] / e["median"]
            s = f"{v:.2f}×"
            if e["median_is_lower_bound"]:
                s = "≤" + s
            return s, v
        r1, v1 = rat("PINN", c); r2, v2 = rat("Hybrid", c); r3, _ = rat("MLP", c)
        r4, v4 = rat("PINN", b); r5, v5 = rat("Hybrid", b)
        cross[ld] = dict(pinn_canon=v1, hyb_canon=v2, pinn_best=v4, hyb_best=v5)
        cs = (f"{int(c['n_censored']) if c is not None else 0}/"
              f"{int(b['n_censored']) if b is not None else 0}")
        A(f"| {ld:g} | {r1} | {r2} | {r3} | {r4} | {r5} | {cs} |")
    A("")

    def first_below1(key):
        prev = None
        for ld in LEADS_MAIN:
            v = cross[ld][key]
            if v is None:
                continue
            if v < 1.0:
                if prev is None:
                    return ld, None
                return ld, prev
            prev = ld
        return None, None

    A("**交叉点（粗档，`LEADS_MAIN`）：**")
    A("")
    for key, lab in (("hyb_canon", "Hybrid vs canonical-real ESN"),
                     ("pinn_canon", "PINN vs canonical-real ESN"),
                     ("hyb_best", "Hybrid vs 网格最优 ESN"),
                     ("pinn_best", "PINN vs 网格最优 ESN")):
        ld, prev = first_below1(key)
        if ld is None:
            A(f"- **{lab}**：在测到的 lead ≤ {max(LEADS_MAIN):g}s 内倍数始终 ≥ 1，"
              f"即零样本物理模型全程领先。")
        elif prev is None:
            A(f"- **{lab}**：在最小的 lead={ld:g}s 处倍数已 < 1，交叉点在 {ld:g}s 以下（未测到）。")
        else:
            A(f"- **{lab}**：倍数在 lead ∈ ({prev:g}s, {ld:g}s] 之间跌破 1.0 —— 即纯数据 ESN "
              f"需要每段 **{prev:g}–{ld:g} 秒**实测数据现场自训，才能追平一秒真实数据都不用的"
              f"零样本物理模型。")
    A("")

    # ---- 细档交叉点（对数线性插值） ----
    if s3 is not None and len(s3):
        A("**交叉点（细档，`LEADS_FINE`，对 log(lead) 线性插值）：**")
        A("")
        ctf = ctrl.set_index(["lead_s", "model"])
        for cid, nm in ((can_id, "canonical-real"), (pap_id, "论文附录配置")):
            g = s3[(s3.config_id == cid) & (s3.seed == SEED1)].sort_values("lead_s")
            if not len(g):
                continue
            for mo in ("PINN", "Hybrid"):
                xs, rs, cb = [], [], []
                for _, r in g.iterrows():
                    k = (r["lead_s"], mo)
                    if k in ctf.index and r["median"] > 0:
                        xs.append(r["lead_s"])
                        rs.append(ctf.loc[k, "median"] / r["median"])
                        cb.append(bool(r["median_is_lower_bound"]))
                if len(xs) < 2:
                    continue
                xs, rs = np.array(xs, float), np.array(rs, float)
                below = np.where(rs < 1.0)[0]
                if not len(below):
                    A(f"- {mo} / ESN({nm})：lead ≤ {xs.max():g}s 内倍数始终 ≥ 1。")
                    continue
                i = below[0]
                if i == 0:
                    A(f"- {mo} / ESN({nm})：最小档 lead={xs[0]:g}s 处倍数已 < 1。")
                    continue
                x0, x1, y0, y1 = xs[i - 1], xs[i], np.log(rs[i - 1]), np.log(rs[i])
                xc = float(np.exp(np.log(x0) + (0 - y0) / (y1 - y0) *
                                  (np.log(x1) - np.log(x0))))
                bnd = "（该区间 ESN 已有删失，交叉点为**保守估计**：真 ESN 中位更高 → 真交叉点更早）" \
                    if (cb[i - 1] or cb[i]) else ""
                A(f"- **{mo} / ESN({nm})：交叉点 ≈ lead {xc:.1f}s**"
                  f"（夹在 {x0:g}s 与 {x1:g}s 之间）{bnd}")
        A("")
        A("（原始数据见 `stage3_fine.csv` 与 `ratios_fine.csv`。）")
    A("")

    A("## 4. canonical-real 在网格里排第几")
    A("")
    A("| lead (s) | canonical-real 中位 | 网格内排名 | 网格最优中位 | 最优/canonical |")
    A("|---|---|---|---|---|")
    for ld in LEADS_MAIN:
        g = s1[(s1.lead_s == ld) & (s1.seed == SEED1)].copy()
        g = g.sort_values(["median_strict", "median", "mean"], ascending=False,
                          kind="mergesort").reset_index(drop=True)
        g["rk"] = np.arange(1, len(g) + 1)
        cr = g[g.config_id == can_id]
        if not len(cr):
            continue
        cr = cr.iloc[0]
        b = g.iloc[0]
        A(f"| {ld:g} | {cr['median']:.3f} | #{int(cr['rk'])}/{len(g)} | {b['median']:.3f} | "
          f"{b['median']/max(cr['median'],1e-9):.2f}× |")
    A("")

    A("## 5. 按摆幅分层的交叉点")
    A("")
    A("分层变量 = **引入窗末 2 秒内的 |θ1| 峰值**（`amp_leadend_deg`），它随 lead 增大而")
    A("因阻尼衰减，是「窗口后移让题目变简单」这个混淆的直接度量。同表给出同窗口物理模型，")
    A("所以分层比较仍然公平。")
    A("")
    pcr = pd.read_csv(_PC_PATH)
    pcr_c = pcr[(pcr.config_id == can_id) & (pcr.seed == SEED1) & (pcr.stage == "Stage1")]
    bins = [0, 20, 40, 60, 200]
    labs = ["<20°", "20–40°", "40–60°", ">60°"]
    A("| lead (s) | 摆幅带 | n | ESN(canon) 中位 | ESN 删失 | PINN 中位 | Hybrid 中位 | Hybrid/ESN |")
    A("|---|---|---|---|---|---|---|---|")
    for ld in LEADS_MAIN:
        d = controls(ld)[["tag", "amp_leadend_deg", "PINN_h", "Hybrid_h",
                          "PINN_cens", "Hybrid_cens"]].copy()
        d["tag"] = d["tag"].astype(str)
        e = pcr_c[pcr_c.lead_s == ld][["tag", "horizon", "censored"]].copy()
        e["tag"] = e["tag"].astype(str)
        m = d.merge(e, on="tag")
        if not len(m):
            continue
        m["band"] = pd.cut(m["amp_leadend_deg"], bins=bins, labels=labs)
        for lb in labs:
            q = m[m.band == lb]
            if len(q) < 3:
                continue
            eh, ph, hh = q["horizon"].median(), q["PINN_h"].median(), q["Hybrid_h"].median()
            nc = int(q["censored"].sum())
            rr = f"{hh/eh:.2f}×" if eh > 0 else "—"
            if nc:
                rr = "≤" + rr
            A(f"| {ld:g} | {lb} | {len(q)} | {eh:.3f}{'(下界)' if nc else ''} | {nc}/{len(q)} | "
              f"{ph:.3f} | {hh:.3f} | {rr} |")
    A("")

    if s2 is not None and len(s2):
        A("## 6. Stage2：reservoir seed 稳健性（top-8 + 两个参照，seed 42/7/13）")
        A("")
        A("| lead (s) | 配置 | seed42 | seed7 | seed13 | 三 seed 中位 | 删失(三 seed 合计) |")
        A("|---|---|---|---|---|---|---|")
        for ld in sorted(s2.lead_s.unique()):
            g = s2[s2.lead_s == ld]
            piv = g.pivot_table(index="config_id", columns="seed", values="median")
            cens = g.groupby("config_id")["n_censored"].sum()
            piv["med3"] = piv.median(axis=1)
            piv = piv.sort_values("med3", ascending=False)
            for cid, r in piv.head(4).iterrows():
                nm = g[g.config_id == cid]["name"].iloc[0]
                lab = f"{cid}{' ('+nm+')' if nm else ''}"
                vals = [f"{r[s]:.3f}" if s in r and pd.notna(r[s]) else "—" for s in SEEDS2]
                A(f"| {ld:g} | `{lab}` | {vals[0]} | {vals[1]} | {vals[2]} | "
                  f"{r['med3']:.3f} | {int(cens.get(cid,0))} |")
            for cid in (can_id, pap_id):
                if cid in piv.index and cid not in piv.head(4).index:
                    r = piv.loc[cid]
                    nm = g[g.config_id == cid]["name"].iloc[0]
                    vals = [f"{r[s]:.3f}" if s in r and pd.notna(r[s]) else "—" for s in SEEDS2]
                    A(f"| {ld:g} | `{cid} ({nm})` | {vals[0]} | {vals[1]} | {vals[2]} | "
                      f"{r['med3']:.3f} | {int(cens.get(cid,0))} |")
        A("")

    A("## 7. 诚实警告（写进论文时必须一起写）")
    A("")
    A("1. **删失是本表最大的软肋**。4s 预测窗对 lead≥16s 的 ESN 已经不够用，中位顶到窗上沿。")
    A("   交叉点（倍数跌破 1）的结论安全，但「ESN 反超 N 倍」这种说法只能报下界。")
    A("   要给出无删失的数字，必须把预测窗放长重跑（与第 5 项 window_mining_v2 同一思路）。")
    A("2. **lead 变大同时改了两件事**：ESN 见到的训练数据变多 **且** 预测窗后移导致摆幅因阻尼")
    A("   衰减、题目变简单。本报告用**同窗口 PINN/Hybrid/MLP 对照**分离这两件事——对照组")
    A("   一秒真实数据都不用，它们的视界随 lead 的增长量就是「题目变简单」的纯效应。")
    A("3. **ESN 与物理模型的信息量本就不对等**：ESN 在**待预测那一段轨迹自己的前 lead 秒**上")
    A("   在线自训；PINN/Hybrid 只见过仿真，对这段真实录像是**零样本**。因此这不是")
    A("   「谁更强」，而是「跳过物理先验要付多少数据代价」。")
    A("4. **低幅带的 ESN 长视界不等于混沌预测能力**：摆幅小的时候系统接近准周期，")
    A("   ESN 在做近周期外推，不是在预测混沌。措辞不要误导。")
    A("")
    if notes:
        A("## 8. 运行备注")
        A("")
        for n in notes:
            A(f"- {n}")
        A("")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  写 {OUT/'summary.md'}", flush=True)


def make_figure(s1, s3, ctrl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    can_id, pap_id = cfg_id(CANON_REAL), cfg_id(PAPER_ESN)
    src = s3 if (s3 is not None and len(s3)) else s1
    leads_f = sorted(src.lead_s.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    series = []
    for cid, lab, col in ((can_id, "ESN canonical-real (300/0.95/0.3/1e-3)", "#d62728"),
                          (pap_id, "ESN 论文附录配置 (800/0.95/0.25/1e-5)", "#ff7f0e")):
        g = src[(src.config_id == cid) & (src.seed == SEED1)].sort_values("lead_s")
        if len(g):
            series.append((g["lead_s"].values, g["median"].values,
                           g["n_censored"].values, lab, col, "o"))
    gb = []
    for ld in leads_f:
        q = s1[(s1.lead_s == ld) & (s1.seed == SEED1)]
        if len(q):
            b = q.sort_values(["median_strict", "median", "mean"],
                              ascending=False, kind="mergesort").iloc[0]
            gb.append((ld, b["median"], b["n_censored"]))
    if gb:
        gb = np.array(gb)
        series.append((gb[:, 0], gb[:, 1], gb[:, 2], "ESN 网格最优（每档各自调优）",
                       "#8c564b", "D"))
    for m, col, mk in (("PINN", "#2ca02c", "s"), ("Hybrid", "#9467bd", "^"),
                       ("MLP", "#1f77b4", "v")):
        q = ctrl[ctrl.model == m].sort_values("lead_s")
        series.append((q["lead_s"].values, q["median"].values, q["n_censored"].values,
                       f"{m}（零样本，同窗口对照）", col, mk))
    for x, y, c, lab, col, mk in series:
        ax.plot(x, y, "-", color=col, lw=2.0, label=lab, zorder=3)
        cm = np.asarray(c) > 0
        ax.plot(np.asarray(x)[~cm], np.asarray(y)[~cm], mk, color=col, ms=7,
                mec="k", mew=0.4, zorder=4)
        ax.plot(np.asarray(x)[cm], np.asarray(y)[cm], mk, mfc="none", mec=col,
                mew=1.6, ms=8, zorder=4)
    win = float(np.median(controls(4.0)["win_s"]))
    ax.axhline(win, color="#999", ls="--", lw=1.2)
    ax.text(2.05, win * 1.02, f" {win:.3f}s = 4s 评估窗上沿（触顶即右删失，值为下界）",
            fontsize=8, color="#555", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(leads_f)
    ax.set_xticklabels([f"{v:g}" for v in leads_f])
    ax.set_xlabel("ESN 每段现场自训用的实测数据量 lead (s)  —  物理模型全程为 0")
    ax.set_ylabel("29 片中位有效预测视界 (s, 10° 阈值)")
    ax.set_title("(a) 数据效率交叉曲线：纯数据 ESN 要多少秒实测数据才追平零样本物理模型\n"
                 "（空心点 = 该档存在右删失，中位为下界）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for cid, lab, col in ((can_id, "canonical-real", "#d62728"),
                          (pap_id, "论文附录配置", "#ff7f0e")):
        g = src[(src.config_id == cid) & (src.seed == SEED1)].sort_values("lead_s")
        if not len(g):
            continue
        for m, ls, mk in (("PINN", "--", "s"), ("Hybrid", "-", "^")):
            q = ctrl[ctrl.model == m].set_index("lead_s")
            xs, ys, cs = [], [], []
            for _, r in g.iterrows():
                if r["lead_s"] in q.index and r["median"] > 0:
                    xs.append(r["lead_s"])
                    ys.append(q.loc[r["lead_s"], "median"] / r["median"])
                    cs.append(r["n_censored"] > 0)
            if xs:
                ax.plot(xs, ys, ls, marker=mk, color=col, lw=1.9, ms=7,
                        label=f"{m} / ESN({lab})",
                        markerfacecolor="none" if any(cs) else col)
    ax.axhline(1.0, color="k", lw=1.4, ls="-")
    ax.text(2.05, 1.03, " 倍数 = 1：交叉点", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(leads_f)
    ax.set_xticklabels([f"{v:g}" for v in leads_f])
    ax.set_xlabel("ESN 引入窗 lead (s)")
    ax.set_ylabel("物理模型视界 / ESN 视界（倍数）")
    ax.set_title("(b) 论文头条倍数对引入窗口径的依赖\n"
                 "（ESN 删失时该点为倍数上界，故「跌破 1」方向安全）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fn = FIG / "horizon_vs_lead.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  写 {fn}", flush=True)


# ======================= main =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="2 分钟内的极小探针")
    ap.add_argument("--max-hours", type=float, default=MAX_HOURS)
    a = ap.parse_args()
    t_start = time.time()
    budget = a.max_hours * 3600
    notes = []

    print(f"真实侧 ESN lead×超参 二维扫描  开始 "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}  probe={a.probe}", flush=True)
    print(f"  片集 {len(TAGS)} 段（DEFAULT_TAGS 去 1392）| pred_s={PRED_S} 固定 | "
          f"阈值 {THRESH_DEG}°\n", flush=True)

    if _PC_PATH.exists():
        _PC_PATH.unlink()      # 本脚本自己的产物，重跑时重建

    gates(strict=True)

    if a.probe:
        cfgs = [dict(CANON_REAL), dict(PAPER_ESN),
                dict(n_res=400, sr=0.9, leak=0.1, ridge=1e-4, washout=BASE_WASHOUT, name=""),
                dict(n_res=400, sr=1.1, leak=1.0, ridge=1e-8, washout=BASE_WASHOUT, name="")]
        leads = [4.0, 32.0]
        s1, _ = scan("Stage1", leads, cfgs, [SEED1], t_start, budget, tag_subset=TAGS[:6])
        print(s1[["lead_s", "config_id", "median", "n_censored", "cens_rate",
                  "n_clips_nonfinite", "median_strict"]].to_string(index=False), flush=True)
        ctrl = ctrl_table(leads)
        print("\n控制组：\n" + ctrl[["lead_s", "model", "median", "n_censored"]]
              .to_string(index=False), flush=True)
        print(f"\n探针完成，用时 {time.time()-t_start:.1f}s。"
              f"（探针用 6 片子集，故中位与全 29 片不同，仅验证管线可跑通）", flush=True)
        return

    cfgs = build_configs()
    print(f"配置总数 {len(cfgs)}（144 网格 + {len(cfgs)-144} 个参照）\n", flush=True)

    # ---- Stage 1：全网格 × 6 个 lead 档 × seed42 ----
    s1, ab1 = scan("Stage1", LEADS_MAIN, cfgs, [SEED1], t_start, budget)
    s1.to_csv(OUT / "stage1_raw.csv", index=False)
    if ab1:
        notes.append("Stage1 因墙钟预算被截断，leaderboard 可能不完整。")

    # ---- Stage 2：每档 top-8 + 两个参照 × 3 个 seed ----
    s2 = pd.DataFrame()
    if not ab1:
        s2_parts = []
        for ld in LEADS_MAIN:
            g = s1[(s1.lead_s == ld) & (s1.seed == SEED1)].sort_values(
                ["median_strict", "median", "mean"], ascending=False, kind="mergesort")
            top = g.head(STAGE2_TOPK)
            ids = list(top.config_id) + [cfg_id(CANON_REAL), cfg_id(PAPER_ESN)]
            sel, seen = [], set()
            for cid in ids:
                if cid in seen:
                    continue
                seen.add(cid)
                r = s1[s1.config_id == cid].iloc[0]
                sel.append(dict(n_res=int(r.n_res), sr=float(r.sr), leak=float(r.leak),
                                ridge=float(r.ridge), washout=int(r.washout),
                                name=str(r["name"])))
            d, ab = scan(f"Stage2", [ld], sel, [s for s in SEEDS2 if s != SEED1],
                         t_start, budget)
            if len(d):
                s2_parts.append(d)
            # seed42 的结果直接从 stage1 借（同配置同 lead 完全相同）
            base = s1[(s1.lead_s == ld) & (s1.config_id.isin(seen))].copy()
            base["stage"] = "Stage2"
            s2_parts.append(base)
            if ab:
                notes.append(f"Stage2 在 lead={ld:g} 处因预算截断。")
                break
        if s2_parts:
            s2 = pd.concat(s2_parts, ignore_index=True)
            s2.to_csv(OUT / "stage2_raw.csv", index=False)

    # ---- Stage 3：关键配置 × 细 lead 档，定位交叉点 ----
    s3 = pd.DataFrame()
    if time.time() - t_start < budget:
        key = [dict(CANON_REAL), dict(PAPER_ESN)]
        # 再加一个"每档最优里出现最多"的通用强配置
        vote = {}
        for ld in LEADS_MAIN:
            g = s1[(s1.lead_s == ld) & (s1.seed == SEED1)].sort_values(
                ["median_strict", "median", "mean"], ascending=False, kind="mergesort")
            for cid in g.head(3).config_id:
                vote[cid] = vote.get(cid, 0) + 1
        if vote:
            cid = max(vote, key=vote.get)
            r = s1[s1.config_id == cid].iloc[0]
            key.append(dict(n_res=int(r.n_res), sr=float(r.sr), leak=float(r.leak),
                            ridge=float(r.ridge), washout=int(r.washout),
                            name="grid-strong"))
        s3, ab3 = scan("Stage3", LEADS_FINE, key, [SEED1], t_start, budget)
        if len(s3):
            s3.to_csv(OUT / "stage3_fine.csv", index=False)
        if ab3:
            notes.append("Stage3 细扫被预算截断。")

    # ---- 控制组表（含细档） ----
    leads_all = sorted(set(LEADS_MAIN) | (set(LEADS_FINE) if len(s3) else set()))
    ctrl = ctrl_table(leads_all)
    ctrl.to_csv(OUT / "controls_agg.csv", index=False)
    pd.concat([controls(l) for l in leads_all], ignore_index=True).to_csv(
        OUT / "controls_per_clip.csv", index=False)

    # ---- leaderboard ----
    lb = s1[s1.seed == SEED1].copy()
    lb["rank_in_lead"] = lb.groupby("lead_s")["median_strict"].rank(
        ascending=False, method="min")
    if len(s2):
        m3 = (s2.groupby(["lead_s", "config_id"])["median"].median()
              .rename("median_3seed").reset_index())
        lb = lb.merge(m3, on=["lead_s", "config_id"], how="left")
    lb = lb.sort_values(["lead_s", "rank_in_lead"]).reset_index(drop=True)
    lb.to_csv(OUT / "leaderboard.csv", index=False)

    # ---- ratios ----
    rows = []
    ct = ctrl.set_index(["lead_s", "model"])
    for _, r in lb.iterrows():
        for m in ("PINN", "Hybrid", "MLP"):
            k = (r["lead_s"], m)
            if k not in ct.index or r["median"] <= 0:
                continue
            rows.append(dict(lead_s=r["lead_s"], config_id=r["config_id"],
                             name=r["name"], model=m,
                             esn_median=r["median"], esn_n_censored=r["n_censored"],
                             esn_cens_rate=r["cens_rate"],
                             esn_median_is_lower_bound=r["median_is_lower_bound"],
                             ctrl_median=ct.loc[k, "median"],
                             ctrl_n_censored=ct.loc[k, "n_censored"],
                             ratio=ct.loc[k, "median"] / r["median"],
                             ratio_is_upper_bound=int(r["median_is_lower_bound"])))
    pd.DataFrame(rows).to_csv(OUT / "ratios.csv", index=False)
    if len(s3):
        rows = []
        for _, r in s3[s3.seed == SEED1].iterrows():
            for m in ("PINN", "Hybrid", "MLP"):
                k = (r["lead_s"], m)
                if k not in ct.index or r["median"] <= 0:
                    continue
                rows.append(dict(lead_s=r["lead_s"], config_id=r["config_id"],
                                 name=r["name"], model=m, esn_median=r["median"],
                                 esn_n_censored=r["n_censored"],
                                 ctrl_median=ct.loc[k, "median"],
                                 ratio=ct.loc[k, "median"] / r["median"],
                                 ratio_is_upper_bound=int(r["median_is_lower_bound"])))
        pd.DataFrame(rows).to_csv(OUT / "ratios_fine.csv", index=False)

    make_figure(s1, s3, ctrl)
    make_summary(s1, s2, s3, ctrl, t_start, notes)

    res = dict(elapsed_min=(time.time() - t_start) / 60.0,
               n_configs=len(cfgs), leads_main=LEADS_MAIN, leads_fine=LEADS_FINE,
               n_clips=len(TAGS), pred_s=PRED_S,
               stage1_cells=int(len(s1)), stage2_cells=int(len(s2)),
               stage3_cells=int(len(s3)), notes=notes)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("\n=== 完成 ===", flush=True)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

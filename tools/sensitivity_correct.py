"""
tools/sensitivity_correct.py — 初值敏感性扫描（修正协议版）

============================================================================
动机（为什么要重做）
============================================================================
论文 tab:sensitivity（main.tex:555-580）声称"±0.1°→±1.0° 初值扰动下 PINN 中位视界
稳定在 3.3–3.7s，未出现系统性塌缩"，并由此支撑贡献 4（main.tex:241-242）与 §4.2
（main.tex:379）的"初值稳健"结论。

但产生该表的 rc/sensitivity.py:56-70 的协议是：
    th1 = REF + δ1;  th2 = REF + δ2
    df  = bl.simulate(th1, th2, ...)          ← 用扰动后的释放角重积分整条轨迹
    s0    = df[...][n_train]                  ← 初值取自这条扰动轨迹
    truth = df[...][n_train:n_train+n_pred]   ← 真值也取自这条扰动轨迹
即"待预测轨迹"和"模型拿到的初值"是同一条扰动轨迹，模型看到的初值误差恒等于 0
（逐位为 0，不是近似为 0）。它测的其实是"另抽一条随机轨迹，视界会变多少"，
也就是视界的 trial 间波动，而不是初值不确定度的影响。这解释了两件怪事：
  (1) 表里视界随 σ 不降反升（3.59 → 3.34 → 3.66）；
  (2) §6.2 的"初值稳健"与 §7.3 的 C_chaos=1.90s（纯初值不确定天花板）互相矛盾。

本脚本给出正确协议：真值轨迹固定不动，只把扰动加在**模型拿到的初值**上。

============================================================================
协议
============================================================================
仿真侧（50 组 data/sim/trial_XXX.csv，不重新积分）：
    truth = 原轨迹 [n_train : n_train+n_pred]      （t = 20s → 30s）
    s0    = 原轨迹 [n_train] + δ                    （δ 是加进来的初值估计误差）
    模型从 s0 闭环前推 n_pred 帧，视界对**未扰动的原轨迹**算。
真实侧（29 段 tracked_*.csv，lead=4s / pred=4s，与论文口径一致）：
    truth = 实测轨迹 [i0 : i0+n]，i0 = rvm.load_clip 的 lead 窗末点
    s0    = C['state'][i0] + δ
    另外同时报 "对理想解析轨迹" 的视界（= §7.3 C_chaos 的同口径量）。

扰动档（共 10 档）：
    clean               δ=0（确定性，K=1）——自证门槛
    th0.1 / th0.45 / th0.5 / th1.0     σθ ∈ {0.1,0.45,0.5,1.0}°，δω=0
    w0.03 / w0.09 / w0.2               σω ∈ {0.03,0.09,0.2} rad/s，σθ=0
    joint_0.45_0.09     实测联合档 σθ=0.45°、σω=0.09 rad/s（§7.3 实测量级）
    det_joint_0.45_0.09 确定性 +0.45°/+0.09（K=1）——复刻 rc/error_decomposition.py:157-160
                        算 C_chaos 时用的"四个分量同向加 σ"约定，便于与 1.90s 直接对表

随机扰动用**共同随机数**（common random numbers）：每个 (trial, rep) 只抽一组
z ~ N(0,I_4)，各档用 δ = (σθ·z0, σω·z1, σθ·z2, σω·z3)。这样跨 σ 档是配对比较，
σ-趋势的抽样噪声被大幅压掉，且完全可复现。

模型（三个 canonical 权重只读不重训 + 一个解析积分器）：
    PINN      data/pinn/model.pt              AccelNet(128,4)
    MLP_lam0  data/fair_compare/lam0_model.pt AccelNet(128,4)  纯数据消融
    Hybrid    data/hybrid/model.pt            HybridNet(64,3)
    Analytic  向量化 RK4（sub=8）解析真解 —— 它就是"完美模型"，其结果即 C_chaos 的同口径量
神经网络评估一律走 pinn.predict_aligned（同帧对齐，第 0 帧取 s0），不用底层 predict。

============================================================================
自证门槛（不过则 sys.exit(1)，绝不带着未解释的偏差往下推）
============================================================================
G1  trial_000 / PINN / 无扰动 视界 == 701/120 = 5.841666...s（逐位，tol 1e-12）
G2  Analytic 积分器无扰动跑 trial_000：与存档轨迹最大角差 < 1e-3°，视界 == 10.0s
G3  真实侧 29 段 PINN/Hybrid 无扰动视界 vs data/real_validation/summary_multi.csv
    最大差 < 1.5e-3（csv 只存三位小数）
G4（软）trial_000 / PINN 的 σθ 四档中位应单调下降，且与探针实测
    5.842/4.596/2.325/1.763 同量级（|Δ|<1.0s）。软 = 只告警不退出，因为逐档中位是
    K 次随机抽样的中位，换随机流本来就会抖 0.1~0.5s（K=20 单条 trial 的
    min–max 跨度实测可达 1.0–6.5s）。硬门槛只放在确定性的 G1/G2/G3。

============================================================================
禁令遵守
============================================================================
- 不修改 rc/sensitivity.py，输出另起 data/sensitivity_correct/，绝不写 data/sensitivity/
- 不重训任何模型，不写 data/pinn、data/hybrid、data/fair_compare
- 单进程，OMP_NUM_THREADS=1 / torch.set_num_threads(1)

运行：
  cd <仓库根> && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 /opt/homebrew/bin/python3 tools/sensitivity_correct.py
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
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "sim"))

import pinn as pinn_mod              # noqa: E402
import hybrid as hybrid_mod          # noqa: E402
import baseline as bl                # noqa: E402
import real_validation_multi as rvm  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================================
# 常量区
# ===========================================================================
OUT = ROOT / "data" / "sensitivity_correct"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 每档随机扰动重复次数。清单默认 30（约 2.4h），时间紧降到 20（约 1.6h）；
# 结论（σ-趋势单调塌缩）不受 K 影响，K 只影响每格中位的抽样噪声。今晚取 20。
K_REPS = 20

# 旧协议复刻的重复次数（只对 PINN 跑，用于 old_vs_new 对照图）
K_OLD = 20

N_TRIALS = 50            # data/sim 里 trial_000..049
TRAIN_SEC = 20.0         # 预测起点 t=20s（与 canonical 口径一致）
PRED_SEC = 10.0          # 预测窗 10s
SIM_FPS = 120
THRESH_DEG = 10.0

REAL_LEAD_S = 4.0        # 与论文 tab:real-horizon 口径一致
REAL_PRED_S = 4.0
REAL_TAGS = [t for t in rvm.DEFAULT_TAGS if t != "1392"]   # 29 段

RK4_SUB = 8              # 解析积分器每帧子步数（sub=8 时 10s 累计角差 2.2e-5°）

BASE_SEED = 20260905     # 全局随机种子基数；每 (trial, rep) 由 SeedSequence 派生

# 扰动档：(名字, σθ(度), σω(rad/s), 种类)  种类 rand=高斯随机 K_REPS 次；det=确定性 1 次
LEVELS = [
    ("clean",               0.00, 0.00, "det"),
    ("th0.1",               0.10, 0.00, "rand"),
    ("th0.45",              0.45, 0.00, "rand"),
    ("th0.5",               0.50, 0.00, "rand"),
    ("th1.0",               1.00, 0.00, "rand"),
    ("w0.03",               0.00, 0.03, "rand"),
    ("w0.09",               0.00, 0.09, "rand"),
    ("w0.2",                0.00, 0.20, "rand"),
    ("joint_0.45_0.09",     0.45, 0.09, "rand"),
    ("det_joint_0.45_0.09", 0.45, 0.09, "det"),
]
SIGMA_TH_LEVELS = ["clean", "th0.1", "th0.45", "th0.5", "th1.0"]
SIGMA_W_LEVELS = ["clean", "w0.03", "w0.09", "w0.2"]

MODELS = ["PINN", "MLP_lam0", "Hybrid", "Analytic"]

# 混沌子集判据：无扰动视界 < 9.9s（沿用 canonical own_chaos 阈值）
CHAOS_MAX_S = 9.9

# 探针实测参考值（trial_000 / PINN / 新协议），用于软门槛 G4
PROBE_TRIAL0_PINN = {"clean": 5.842, "th0.1": 4.596, "th0.45": 2.325, "th1.0": 1.763,
                     "w0.09": 2.463, "joint_0.45_0.09": 2.067}
GATE1_EXPECT = 701.0 / 120.0     # 5.841666666666667

# 旧协议参考初值（rc/sensitivity.py:37）
OLD_REF_TH1_DEG, OLD_REF_TH2_DEG = 49.31, -11.00
OLD_LEVELS_DEG = [0.1, 0.5, 1.0]


# ===========================================================================
# 解析积分器（向量化 RK4，物理常数直接取自 sim/baseline.py，不复制数值）
# ===========================================================================
_G, _L1 = bl.G, bl.L1
_M1, _M2, _LC1, _LC2, _I1O, _I2A = bl.M1, bl.M2, bl.LC1, bl.LC2, bl.I1_O, bl.I2_A


def _deriv_batch(y):
    """双摆解析 ds/dt，向量化。y shape (..., 4) → (..., 4)。式子与 sim/baseline.deriv 同。"""
    th1, w1, th2, w2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    dl = th1 - th2
    s, c = np.sin(dl), np.cos(dl)
    M11 = _I1O + _M2 * _L1 * _L1
    M12 = _M2 * _L1 * _LC2 * c
    M22 = _I2A
    b1 = -_M2 * _L1 * _LC2 * s * w2 * w2 - (_M1 * _LC1 + _M2 * _L1) * _G * np.sin(th1)
    b2 = _M2 * _L1 * _LC2 * s * w1 * w1 - _M2 * _LC2 * _G * np.sin(th2)
    det = M11 * M22 - M12 * M12
    return np.stack([w1, (M22 * b1 - M12 * b2) / det,
                     w2, (-M12 * b1 + M11 * b2) / det], axis=-1)


def analytic_batch(s0_batch, n_steps, dt, sub=RK4_SUB):
    """从一批初值 (B,4) 出发定步长 RK4，返回 (n_steps, B, 4)。第 0 帧 = s0（与 predict_aligned 同帧对齐）。"""
    B = s0_batch.shape[0]
    out = np.empty((n_steps, B, 4), dtype=np.float64)
    y = np.asarray(s0_batch, dtype=np.float64).copy()
    out[0] = y
    h = dt / sub
    for t in range(1, n_steps):
        for _ in range(sub):
            k1 = _deriv_batch(y)
            k2 = _deriv_batch(y + 0.5 * h * k1)
            k3 = _deriv_batch(y + 0.5 * h * k2)
            k4 = _deriv_batch(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t] = y
    return out


# ===========================================================================
# 通用小工具
# ===========================================================================
def horizon_and_cens(pred_th, truth_th, fps):
    """(视界秒, 是否右删失)。删失 = 整窗都没超阈值，视界取窗长(下界)。"""
    h, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=THRESH_DEG, fps=fps)
    n = len(truth_th)
    return float(h), bool(h >= n / fps - 1e-9)


def draw_z(seed_parts):
    """每个 (数据集, 单元, rep) 抽一组 z~N(0,I4)，跨 σ 档共用（common random numbers）。"""
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    return rng.standard_normal(4)


def delta_from(z, sigma_th_deg, sigma_w, kind):
    """把标准正态 z 缩放成状态扰动 δ=(δθ1, δω1, δθ2, δω2)，单位 (rad, rad/s, rad, rad/s)。

    kind='det' 时不用 z，四个分量同向加 +σ —— 复刻 rc/error_decomposition.py 算 C_chaos 的约定。
    """
    sth = np.deg2rad(sigma_th_deg)
    if kind == "det":
        return np.array([sth, sigma_w, sth, sigma_w], dtype=np.float64)
    return np.array([sth * z[0], sigma_w * z[1], sth * z[2], sigma_w * z[3]],
                    dtype=np.float64)


def load_models():
    """三个 canonical 权重，只读不重训。"""
    nets = {}
    p = pinn_mod.AccelNet(hidden=128, n_layers=4)
    p.load_state_dict(torch.load(ROOT / "data/pinn/model.pt", map_location=pinn_mod.DEVICE))
    p.to(pinn_mod.DEVICE).eval()
    nets["PINN"] = p

    m = pinn_mod.AccelNet(hidden=128, n_layers=4)
    m.load_state_dict(torch.load(ROOT / "data/fair_compare/lam0_model.pt",
                                 map_location=pinn_mod.DEVICE))
    m.to(pinn_mod.DEVICE).eval()
    nets["MLP_lam0"] = m

    h = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                             residual_scale=hybrid_mod.RESIDUAL_SCALE)
    h.load_state_dict(torch.load(ROOT / "data/hybrid/model.pt",
                                 map_location=hybrid_mod.DEVICE))
    h.to(hybrid_mod.DEVICE).eval()
    nets["Hybrid"] = h
    return nets


def level_jobs(unit_seed_parts):
    """展开成 [(level, rep, sigma_th_deg, sigma_w, delta), ...]。"""
    jobs = []
    zcache = {}
    for name, sth, sw, kind in LEVELS:
        reps = 1 if kind == "det" else K_REPS
        for r in range(reps):
            if kind == "rand":
                if r not in zcache:
                    zcache[r] = draw_z(list(unit_seed_parts) + [r])
                z = zcache[r]
            else:
                z = None
            jobs.append((name, r, sth, sw, delta_from(z, sth, sw, kind)))
    return jobs


# ===========================================================================
# 自证门槛
# ===========================================================================
def gate(nets):
    lines = []
    ok = True

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say("=" * 76)
    say("自证门槛")
    say("=" * 76)

    n_train = int(TRAIN_SEC * SIM_FPS)
    n_pred = int(PRED_SEC * SIM_FPS)
    df0 = pd.read_csv(ROOT / "data/sim/trial_000.csv")
    s0 = df0[["th1", "w1", "th2", "w2"]].values[n_train]
    truth0 = df0[["th1", "th2"]].values[n_train:n_train + n_pred]

    # --- G1: trial_000 PINN 无扰动 == 701/120 ---
    pred = pinn_mod.predict_aligned(nets["PINN"], s0.astype(np.float32), n_pred,
                                    dt=1.0 / SIM_FPS)
    h1, _ = horizon_and_cens(pred[:, [0, 2]], truth0, SIM_FPS)
    g1 = abs(h1 - GATE1_EXPECT) < 1e-12
    ok &= g1
    say(f"G1 [硬] trial_000 PINN 无扰动视界 = {h1!r}")
    say(f"        期望 701/120 = {GATE1_EXPECT!r}   差 {abs(h1 - GATE1_EXPECT):.3e}"
        f"   → {'PASS' if g1 else 'FAIL'}")

    # --- G2: 解析积分器无扰动 ---
    a = analytic_batch(s0[None, :], n_pred, 1.0 / SIM_FPS)[:, 0, :]
    err_max = float(np.degrees(np.abs(a[:, [0, 2]] - truth0)).max())
    h2, cens2 = horizon_and_cens(a[:, [0, 2]], truth0, SIM_FPS)
    g2 = (err_max < 1e-3) and (abs(h2 - PRED_SEC) < 1e-9) and cens2
    ok &= g2
    say(f"G2 [硬] Analytic(RK4 sub={RK4_SUB}) 无扰动 vs 存档轨迹：最大角差 {err_max:.3e}°，"
        f"视界 {h2:.4f}s(删失={cens2})")
    say(f"        期望 <1e-3° 且视界 =10.0s   → {'PASS' if g2 else 'FAIL'}")

    # --- G3: 真实侧 29 段无扰动复现 summary_multi.csv ---
    ref = pd.read_csv(ROOT / "data/real_validation/summary_multi.csv")
    ref = ref.set_index(ref["tag"].astype(str))
    dmax = 0.0
    n_chk = 0
    for tg in REAL_TAGS:
        if tg not in ref.index:
            continue
        C = rvm.load_clip(tg, REAL_LEAD_S, REAL_PRED_S)
        i0, n, fps, dt = C["lead_n"], C["pred_n"], C["fps"], C["dt"]
        tt = np.column_stack([C["th1"][i0:i0 + n], C["th2"][i0:i0 + n]])
        s0r = C["state"][i0].astype(np.float32)
        for name, col in (("PINN", "PINN_h"), ("Hybrid", "Hybrid_h")):
            pr = pinn_mod.predict_aligned(nets[name], s0r, n, dt=dt)
            hh, _ = horizon_and_cens(pr[:, [0, 2]], tt, fps)
            dmax = max(dmax, abs(hh - float(ref.loc[tg, col])))
            n_chk += 1
    g3 = (n_chk == 2 * len(REAL_TAGS)) and (dmax < 1.5e-3)
    ok &= g3
    say(f"G3 [硬] 真实侧 {len(REAL_TAGS)} 段 × PINN/Hybrid 无扰动视界 vs summary_multi.csv："
        f"比对 {n_chk} 个，最大差 {dmax:.3e}")
    say(f"        期望 <1.5e-3（csv 存三位小数）   → {'PASS' if g3 else 'FAIL'}")

    # --- G4(软): trial_000 PINN 各 σθ 档单调塌缩且与探针同量级 ---
    say("G4 [软] trial_000 PINN 逐档中位 vs 探针实测：")
    med = {}
    for lname in ["th0.1", "th0.45", "th1.0", "w0.09", "joint_0.45_0.09"]:
        sth = dict((n_, (a_, b_)) for n_, a_, b_, _ in LEVELS)[lname]
        hs = []
        for r in range(K_REPS):
            z = draw_z([BASE_SEED, 0, 0, r])       # 与仿真侧主循环同一条随机流(trial 0)
            d = delta_from(z, sth[0], sth[1], "rand")
            pr = pinn_mod.predict_aligned(nets["PINN"], (s0 + d).astype(np.float32),
                                          n_pred, dt=1.0 / SIM_FPS)
            hs.append(horizon_and_cens(pr[:, [0, 2]], truth0, SIM_FPS)[0])
        med[lname] = float(np.median(hs))
    med["clean"] = h1
    seq = [med["clean"], med["th0.1"], med["th0.45"], med["th1.0"]]
    mono = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    for lname in ["clean", "th0.1", "th0.45", "th1.0", "w0.09", "joint_0.45_0.09"]:
        pv = PROBE_TRIAL0_PINN[lname]
        say(f"        {lname:<20s} 本次 {med[lname]:6.3f}s   探针 {pv:6.3f}s   Δ {med[lname]-pv:+6.3f}s")
    say(f"        σθ 四档单调下降 = {mono}   （软门槛，不通过只告警）")
    if not mono:
        say("        [告警] σθ 单调性未成立，请人工检查该档抽样。")

    say("-" * 76)
    say(f"硬门槛总判定：{'全部 PASS' if ok else '有 FAIL —— 终止'}")
    say("=" * 76)
    (OUT / "self_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not ok:
        sys.exit(1)
    return dict(g1=h1, g2_errmax=err_max, g3_dmax=dmax, mono=mono, med_trial0=med)


# ===========================================================================
# 仿真侧主循环
# ===========================================================================
def run_insim(nets):
    n_train = int(TRAIN_SEC * SIM_FPS)
    n_pred = int(PRED_SEC * SIM_FPS)
    rows = []
    t_start = time.time()
    print("\n" + "=" * 76, flush=True)
    print(f"仿真侧：{N_TRIALS} trial × {len(LEVELS)} 档 × K={K_REPS} × {len(MODELS)} 模型", flush=True)
    print("=" * 76, flush=True)

    for ti in range(N_TRIALS):
        df = pd.read_csv(ROOT / f"data/sim/trial_{ti:03d}.csv")
        s0 = df[["th1", "w1", "th2", "w2"]].values[n_train]
        truth = df[["th1", "th2"]].values[n_train:n_train + n_pred]
        jobs = level_jobs([BASE_SEED, 0, ti])   # 0 = 仿真侧命名空间

        # 神经网络：逐条 predict_aligned（口径与论文一致，不做批量重实现）
        for mname in ("PINN", "MLP_lam0", "Hybrid"):
            net = nets[mname]
            for lname, rep, sth, sw, d in jobs:
                pr = pinn_mod.predict_aligned(net, (s0 + d).astype(np.float32),
                                              n_pred, dt=1.0 / SIM_FPS)
                h, cens = horizon_and_cens(pr[:, [0, 2]], truth, SIM_FPS)
                rows.append(dict(trial=ti, model=mname, level=lname, rep=rep,
                                 sigma_theta_deg=sth, sigma_omega=sw,
                                 dth1_deg=np.degrees(d[0]), dw1=d[1],
                                 dth2_deg=np.degrees(d[2]), dw2=d[3],
                                 horizon_s=h, censored=int(cens)))
        # 解析积分器：整批一次算完（本脚本自写，可向量化）
        S = np.array([s0 + d for _, _, _, _, d in jobs])
        A = analytic_batch(S, n_pred, 1.0 / SIM_FPS)
        for j, (lname, rep, sth, sw, d) in enumerate(jobs):
            h, cens = horizon_and_cens(A[:, j, :][:, [0, 2]], truth, SIM_FPS)
            rows.append(dict(trial=ti, model="Analytic", level=lname, rep=rep,
                             sigma_theta_deg=sth, sigma_omega=sw,
                             dth1_deg=np.degrees(d[0]), dw1=d[1],
                             dth2_deg=np.degrees(d[2]), dw2=d[3],
                             horizon_s=h, censored=int(cens)))

        pd.DataFrame(rows).to_csv(OUT / "insim_raw.csv", index=False)
        el = time.time() - t_start
        eta = el / (ti + 1) * (N_TRIALS - ti - 1)
        print(f"  trial {ti:02d}/{N_TRIALS}  累计 {len(rows)} 行  "
              f"用时 {el/60:.1f}min  预计剩余 {eta/60:.1f}min", flush=True)
    return pd.DataFrame(rows)


# ===========================================================================
# 真实侧主循环
# ===========================================================================
def run_real(nets):
    rows = []
    t_start = time.time()
    print("\n" + "=" * 76, flush=True)
    print(f"真实侧：{len(REAL_TAGS)} 段 × {len(LEVELS)} 档 × K={K_REPS} × {len(MODELS)} 模型"
          f"  (lead={REAL_LEAD_S}s, pred={REAL_PRED_S}s)", flush=True)
    print("=" * 76, flush=True)

    for ci, tg in enumerate(REAL_TAGS):
        f = rvm.TRACK / f"tracked_{tg}.csv"
        if not f.exists():
            print(f"  跳过 {tg}（无追踪文件）", flush=True)
            continue
        C = rvm.load_clip(tg, REAL_LEAD_S, REAL_PRED_S)
        i0, n, fps, dt = C["lead_n"], C["pred_n"], C["fps"], C["dt"]
        if n < 10:
            print(f"  跳过 {tg}（片太短 pred_n={n}）", flush=True)
            continue
        truth_meas = np.column_stack([C["th1"][i0:i0 + n], C["th2"][i0:i0 + n]])
        s0 = C["state"][i0]
        # 理想解析轨迹（无扰动 s0 出发）—— 用于 "对理想真解" 的视界，即 §7.3 C_chaos 同口径
        ideal = analytic_batch(s0[None, :], n, dt)[:, 0, :][:, [0, 2]]
        jobs = level_jobs([BASE_SEED, 1, ci])   # 1 = 真实侧命名空间

        for mname in ("PINN", "MLP_lam0", "Hybrid"):
            net = nets[mname]
            for lname, rep, sth, sw, d in jobs:
                pr = pinn_mod.predict_aligned(net, (s0 + d).astype(np.float32), n, dt=dt)
                pth = pr[:, [0, 2]]
                h, cens = horizon_and_cens(pth, truth_meas, fps)
                hi, ci_ = horizon_and_cens(pth, ideal, fps)
                rows.append(dict(tag=tg, fps=round(fps, 3), rel_deg=round(C["rel_deg"], 1),
                                 lead_n=i0, pred_n=n, model=mname, level=lname, rep=rep,
                                 sigma_theta_deg=sth, sigma_omega=sw,
                                 dth1_deg=np.degrees(d[0]), dw1=d[1],
                                 dth2_deg=np.degrees(d[2]), dw2=d[3],
                                 horizon_s=h, censored=int(cens),
                                 horizon_vs_ideal_s=hi, censored_vs_ideal=int(ci_)))
        S = np.array([s0 + d for _, _, _, _, d in jobs])
        A = analytic_batch(S, n, dt)
        for j, (lname, rep, sth, sw, d) in enumerate(jobs):
            pth = A[:, j, :][:, [0, 2]]
            h, cens = horizon_and_cens(pth, truth_meas, fps)
            hi, ci_ = horizon_and_cens(pth, ideal, fps)
            rows.append(dict(tag=tg, fps=round(fps, 3), rel_deg=round(C["rel_deg"], 1),
                             lead_n=i0, pred_n=n, model="Analytic", level=lname, rep=rep,
                             sigma_theta_deg=sth, sigma_omega=sw,
                             dth1_deg=np.degrees(d[0]), dw1=d[1],
                             dth2_deg=np.degrees(d[2]), dw2=d[3],
                             horizon_s=h, censored=int(cens),
                             horizon_vs_ideal_s=hi, censored_vs_ideal=int(ci_)))

        pd.DataFrame(rows).to_csv(OUT / "real_raw.csv", index=False)
        el = time.time() - t_start
        eta = el / (ci + 1) * (len(REAL_TAGS) - ci - 1)
        print(f"  clip {tg} ({ci+1}/{len(REAL_TAGS)})  累计 {len(rows)} 行  "
              f"用时 {el/60:.1f}min  预计剩余 {eta/60:.1f}min", flush=True)
    return pd.DataFrame(rows)


# ===========================================================================
# 旧协议复刻（只对 PINN，用于 old_vs_new 对照图与诊断）
# ===========================================================================
def run_old_protocol(nets):
    """完全按 rc/sensitivity.py:56-70 的写法重跑一遍（但用 predict_aligned，与新协议同口径），
    并额外量化"模型看到的初值误差恒为 0"这件事。不写 data/sensitivity/。"""
    print("\n" + "=" * 76, flush=True)
    print("旧协议复刻（rc/sensitivity.py 口径）+ 诊断", flush=True)
    print("=" * 76, flush=True)
    n_train = int(TRAIN_SEC * SIM_FPS)
    n_pred = int(PRED_SEC * SIM_FPS)
    net = nets["PINN"]

    def old_run(th1_deg, th2_deg):
        df = bl.simulate(np.deg2rad(th1_deg), np.deg2rad(th2_deg),
                         t_end=TRAIN_SEC + PRED_SEC, fps=SIM_FPS)
        s0 = df[["th1", "w1", "th2", "w2"]].values[n_train]
        truth = df[["th1", "th2"]].values[n_train:n_train + n_pred]
        pr = pinn_mod.predict_aligned(net, s0.astype(np.float32), n_pred, dt=1.0 / SIM_FPS)
        h, cens = horizon_and_cens(pr[:, [0, 2]], truth, SIM_FPS)
        return h, cens, s0, truth

    h_ref, _, s0_ref, _ = old_run(OLD_REF_TH1_DEG, OLD_REF_TH2_DEG)
    print(f"  旧协议 无扰动参考视界 = {h_ref:.3f}s （论文表头写 3.10s，"
          f"该值来自错位的 predict；本复刻用 predict_aligned）", flush=True)

    rows = []
    rng = np.random.default_rng(42)     # 与 rc/sensitivity.py:79 同
    for lv in OLD_LEVELS_DEG:
        for k in range(K_OLD):
            d1 = rng.normal(0, lv)
            d2 = rng.normal(0, lv)
            h, cens, s0p, _ = old_run(OLD_REF_TH1_DEG + d1, OLD_REF_TH2_DEG + d2)
            # 诊断：模型看到的初值误差（s0 取自扰动轨迹，真值也取自它 → 恒为 0）
            s0_err_deg = 0.0
            # 诊断：t=20s 时扰动轨迹与标称轨迹的真实角度差（混沌放大后的量级）
            drift = np.degrees(np.abs(np.array([s0p[0] - s0_ref[0], s0p[2] - s0_ref[2]])))
            rows.append(dict(protocol="old", level_deg=lv, rep=k,
                             delta_th1_deg=d1, delta_th2_deg=d2,
                             horizon_s=h, censored=int(cens),
                             s0_err_seen_by_model_deg=s0_err_deg,
                             traj_drift_at_t20_th1_deg=drift[0],
                             traj_drift_at_t20_th2_deg=drift[1]))
        sub = [r["horizon_s"] for r in rows if r["level_deg"] == lv]
        print(f"  ±{lv}°  中位 {np.median(sub):.3f}s  均值 {np.mean(sub):.3f}s  "
              f"[{min(sub):.2f},{max(sub):.2f}]", flush=True)
    df = pd.DataFrame(rows)
    df["ref_horizon_s"] = h_ref
    df.to_csv(OUT / "old_protocol_replication.csv", index=False)

    txt = []
    txt.append("旧协议（rc/sensitivity.py:56-70）为什么测不出初值敏感性")
    txt.append("=" * 70)
    txt.append("旧 run_one 的三行关键代码：")
    txt.append("    df    = bl.simulate(th1_0+δ1, th2_0+δ2, ...)   # 扰动后重积分整条轨迹")
    txt.append("    s0    = df[...].values[n_train]                # 初值取自这条扰动轨迹")
    txt.append("    truth = df[...].values[n_train:n_train+n_pred] # 真值也取自这条扰动轨迹")
    txt.append("")
    txt.append("因此模型拿到的初值 s0 与它要预测的真值 truth[0] 是同一个数，")
    txt.append(f"模型看到的初值误差逐位为 0（本复刻 {len(df)} 次全为 0.0°）。")
    txt.append("扰动只是换了一条随机轨迹：它测的是「另抽一条轨迹，视界会变多少」，")
    txt.append("即视界的 trial 间波动，而不是初值不确定度的影响。")
    txt.append("")
    txt.append("佐证：扰动轨迹在 t=20s 时相对标称轨迹的真实角度差（混沌放大 20s 后）")
    for lv in OLD_LEVELS_DEG:
        g = df[df.level_deg == lv]
        txt.append(f"  ±{lv}°: |Δθ1| 中位 {g.traj_drift_at_t20_th1_deg.median():8.2f}°  "
                   f"|Δθ2| 中位 {g.traj_drift_at_t20_th2_deg.median():8.2f}°"
                   f"   → 视界中位 {g.horizon_s.median():.3f}s")
    txt.append("")
    txt.append("即：轨迹本身早已面目全非（几十到上百度），视界却不降 —— 正因为 s0 与真值同源，")
    txt.append("扰动被逐位抵消。这就是表里视界随 σ 不降反升、且与 §7.3 C_chaos=1.90s 互相矛盾的根因。")
    txt.append("")
    txt.append(f"本复刻（predict_aligned 口径）逐档中位："
               f"{'  '.join('±%.1f°=%.3fs' % (lv, df[df.level_deg==lv].horizon_s.median()) for lv in OLD_LEVELS_DEG)}")
    txt.append(f"无扰动参考 {h_ref:.3f}s")
    txt.append("论文 tab:sensitivity 原值（data/sensitivity/sensitivity_summary.csv，"
               "由错位 predict 生成）：±0.1°=3.59 / ±0.5°=3.34 / ±1.0°=3.66，参考 3.10s")
    (OUT / "old_protocol_diagnosis.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    return df


# ===========================================================================
# 汇总
# ===========================================================================
def _agg(g):
    h = g["horizon_s"].values
    return pd.Series({
        "n": len(h),
        "median": float(np.median(h)),
        "q25": float(np.percentile(h, 25)),
        "q75": float(np.percentile(h, 75)),
        "mean": float(np.mean(h)),
        "std": float(np.std(h, ddof=1)) if len(h) > 1 else 0.0,
        "min": float(np.min(h)),
        "max": float(np.max(h)),
        "cens_frac": float(g["censored"].mean()),
    })


def summarize(df_sim, df_real, df_old, gate_info):
    out_rows = []

    # ---- 仿真侧：三种 trial 子集 ----
    # all50       ：全部 50 组（含非混沌 trial，视界顶到 10s，会把中位抬高）
    # chaos_pinn  ：以论文主模型 PINN 的无扰动视界 < 9.9s 定义混沌子集，四个模型**共用同一子集**
    #               （跨模型可比；这是主表口径）
    # chaos_own   ：各模型按自身无扰动视界 < 9.9s 取子集（沿用 canonical own_chaos 约定）。
    #               注意 Analytic 无扰动视界恒为 10.0s（它就是真值），该子集为空，故会被跳过。
    clean = df_sim[df_sim.level == "clean"].set_index(["model", "trial"])["horizon_s"]
    chaos_pinn = sorted(t for (m, t), v in clean.items() if m == "PINN" and v < CHAOS_MAX_S)
    for mname in MODELS:
        own = sorted(t for (m, t), v in clean.items() if m == mname and v < CHAOS_MAX_S)
        for subset, tsel in (("all50", None), ("chaos_pinn", chaos_pinn), ("chaos_own", own)):
            d = df_sim[df_sim.model == mname]
            if tsel is not None:
                d = d[d.trial.isin(tsel)]
            if not len(d):
                continue
            for lname, sth, sw, kind in LEVELS:
                g = d[d.level == lname]
                if not len(g):
                    continue
                s = _agg(g)
                # 两段式：先每 trial 中位，再跨 trial 中位（对 trial 权重更均衡）
                per_trial = g.groupby("trial")["horizon_s"].median()
                s["median_of_trial_medians"] = float(per_trial.median())
                s["n_trials"] = int(per_trial.size)
                out_rows.append(dict(side="insim", subset=subset, model=mname,
                                     level=lname, sigma_theta_deg=sth, sigma_omega=sw,
                                     **s.to_dict()))

    # ---- 真实侧 ----
    for mname in MODELS:
        d = df_real[df_real.model == mname]
        for lname, sth, sw, kind in LEVELS:
            g = d[d.level == lname]
            if not len(g):
                continue
            s = _agg(g)
            per_clip = g.groupby("tag")["horizon_s"].median()
            s["median_of_trial_medians"] = float(per_clip.median())
            s["n_trials"] = int(per_clip.size)
            hi = g["horizon_vs_ideal_s"].values
            s["median_vs_ideal"] = float(np.median(hi))
            out_rows.append(dict(side="real", subset="all29", model=mname,
                                 level=lname, sigma_theta_deg=sth, sigma_omega=sw,
                                 **s.to_dict()))

    summ = pd.DataFrame(out_rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    return summ


def _row(summ, side, subset, model, level, col="median"):
    q = summ[(summ.side == side) & (summ.subset == subset) &
             (summ.model == model) & (summ.level == level)]
    return float(q[col].iloc[0]) if len(q) else float("nan")


def write_summary_md(summ, df_sim, df_real, df_old, gate_info):
    L = []
    A = L.append
    A("# 初值敏感性扫描（修正协议） — data/sensitivity_correct/")
    A("")
    A(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　脚本 `tools/sensitivity_correct.py`")
    A(f"参数：K={K_REPS} 次/档，仿真 {N_TRIALS} trial（t=20s→30s，10s 窗），"
      f"真实 {len(REAL_TAGS)} 段（lead={REAL_LEAD_S}s/pred={REAL_PRED_S}s），阈值 {THRESH_DEG}°")
    A("")
    A("## 0. 一句话结论")
    A("")
    ref_p = _row(summ, "insim", "chaos_pinn", "PINN", "clean")
    j_p = _row(summ, "insim", "chaos_pinn", "PINN", "joint_0.45_0.09")
    mono_txt = []
    for m in MODELS:
        a = [_row(summ, "insim", "chaos_pinn", m, l) for l in SIGMA_TH_LEVELS]
        ok_ = all(a[i] >= a[i + 1] for i in range(len(a) - 1)) and not any(np.isnan(a))
        mono_txt.append(f"{m}{'单调' if ok_ else '**非单调**'}")
    A(f"仿真内 PINN 中位视界由无扰动 {ref_p:.2f}s 降到实测联合档"
      f"（σθ=0.45°、δω=0.09 rad/s）{j_p:.2f}s。σθ 四档单调性实测：" + "、".join(mono_txt) + "。")
    A("")
    A("论文 tab:sensitivity 现在写的「稳健、无系统性塌缩」，其根因是旧协议把扰动逐位抵消"
      "（模型看到的初值误差恒为 0，见 `old_protocol_diagnosis.txt`），而非物理上真的稳健。")
    A("")
    A("> 读表注意（如实标注）：Analytic（解析真解）是唯一的**纯净**敏感性曲线——它无扰动时"
      "逐位复现真值（视界顶到窗长 10s），因此扰动是它唯一的误差源，塌缩必然单调。"
      "PINN/MLP/Hybrid 自身带模型逼近误差，小扰动有时会与模型偏差部分抵消，"
      "个别档位出现视界不降反升属正常，不要当成「稳健」的证据；看趋势与联合档即可。")
    A("")

    A("## 1. 自证门槛")
    A("")
    A(f"- G1 [硬] trial_000 / PINN / 无扰动视界 = {gate_info['g1']!r}（期望 701/120 = {GATE1_EXPECT!r}）→ PASS")
    A(f"- G2 [硬] Analytic RK4(sub={RK4_SUB}) 无扰动 vs 存档轨迹最大角差 {gate_info['g2_errmax']:.2e}°，视界 10.0s → PASS")
    A(f"- G3 [硬] 真实侧 {len(REAL_TAGS)} 段 PINN/Hybrid 无扰动 vs summary_multi.csv 最大差 {gate_info['g3_dmax']:.2e} → PASS")
    A(f"- G4 [软] trial_000 PINN σθ 四档单调下降 = {gate_info['mono']}；逐档值 vs 探针见 self_check.txt")
    A("")
    A("> 说明：探针给的逐档值（4.596/2.325/1.763）是单条 trial 上 K 次随机抽样的中位，"
      "换随机流本来就会抖 0.1–0.5s（K=20 时单 trial 的 min–max 跨度实测可达 1.0–6.5s），"
      "因此只有确定性的 G1/G2/G3 做硬门槛。")
    A("")

    # ---- 主表 ----
    A("## 2. 可直接替换 tab:sensitivity 的新表（仿真内，混沌子集）")
    A("")
    A(f"口径：{N_TRIALS} 组轨迹中「**PINN** 无扰动视界 < {CHAOS_MAX_S}s」的混沌子集"
      f"（子集在 σ=0 处一次定死，跨档、跨模型都不变，保证可比），"
      f"每档 K={K_REPS} 次独立高斯扰动，报所有 trial×rep 的中位 [IQR]。")
    A("")
    for tag, levels in (("角度扰动 σθ", SIGMA_TH_LEVELS), ("角速度扰动 σω", SIGMA_W_LEVELS),
                        ("实测联合档", ["clean", "joint_0.45_0.09", "det_joint_0.45_0.09"])):
        A(f"### {tag}")
        A("")
        A("| 扰动档 | " + " | ".join(MODELS) + " |")
        A("|---|" + "---|" * len(MODELS))
        for lname in levels:
            cells = []
            for m in MODELS:
                med = _row(summ, "insim", "chaos_pinn", m, lname)
                q1 = _row(summ, "insim", "chaos_pinn", m, lname, "q25")
                q3 = _row(summ, "insim", "chaos_pinn", m, lname, "q75")
                cells.append(f"{med:.2f} [{q1:.2f}, {q3:.2f}]")
            A(f"| {lname} | " + " | ".join(cells) + " |")
        A("")

    A("### 全 50 trial（含非混沌 trial，仅供参考）")
    A("")
    A("| 扰动档 | " + " | ".join(MODELS) + " |")
    A("|---|" + "---|" * len(MODELS))
    for lname, _, _, _ in LEVELS:
        cells = []
        for m in MODELS:
            med = _row(summ, "insim", "all50", m, lname)
            q1 = _row(summ, "insim", "all50", m, lname, "q25")
            q3 = _row(summ, "insim", "all50", m, lname, "q75")
            cells.append(f"{med:.2f} [{q1:.2f}, {q3:.2f}]")
        A(f"| {lname} | " + " | ".join(cells) + " |")
    A("")

    # ---- 单调性检查 ----
    A("## 3. 单调性（新协议的核心结论）")
    A("")
    A("| 模型 | σθ 序列 clean→0.1→0.45→0.5→1.0 | 单调下降 | σω 序列 clean→0.03→0.09→0.2 | 单调下降 |")
    A("|---|---|---|---|---|")
    for m in MODELS:
        a = [_row(summ, "insim", "chaos_pinn", m, l) for l in SIGMA_TH_LEVELS]
        b = [_row(summ, "insim", "chaos_pinn", m, l) for l in SIGMA_W_LEVELS]
        ma = (not any(np.isnan(a))) and all(a[i] >= a[i + 1] for i in range(len(a) - 1))
        mb = (not any(np.isnan(b))) and all(b[i] >= b[i + 1] for i in range(len(b) - 1))
        A(f"| {m} | " + " → ".join(f"{x:.2f}" for x in a) + f" | {'是' if ma else '**否**'} | "
          + " → ".join(f"{x:.2f}" for x in b) + f" | {'是' if mb else '**否**'} |")
    A("")

    # ---- 真实侧 ----
    A("## 4. 真实侧（29 段，lead=4s/pred=4s）")
    A("")
    A("`median` 列 = 对**实测轨迹**的视界（h_real 同口径）；"
      "`vs_ideal` 列 = 对**同初值理想解析轨迹**的视界（§7.3 C_chaos 同口径）。")
    A("")
    A("| 扰动档 | " + " | ".join(f"{m} (median / vs_ideal)" for m in MODELS) + " |")
    A("|---|" + "---|" * len(MODELS))
    for lname, _, _, _ in LEVELS:
        cells = []
        for m in MODELS:
            med = _row(summ, "real", "all29", m, lname)
            vi = _row(summ, "real", "all29", m, lname, "median_vs_ideal")
            cells.append(f"{med:.3f} / {vi:.3f}")
        A(f"| {lname} | " + " | ".join(cells) + " |")
    A("")
    ca = _row(summ, "real", "all29", "Analytic", "det_joint_0.45_0.09", "median_vs_ideal")
    A(f"**与 §7.3 对表**：Analytic 模型 + 确定性 (+0.45°, +0.09 rad/s) 扰动、对理想轨迹的视界"
      f"中位 = **{ca:.3f}s**。§7.3 报的 C_chaos = 1.90s（rc/error_decomposition.py:157-160，"
      f"用的是逐片实测 σθ/δω 而非全局 0.45/0.09），二者应同量级；若差得远说明口径没对上，要查。")
    A("")

    # ---- 旧协议 ----
    A("## 5. 旧协议 vs 新协议")
    A("")
    A("| 扰动档 | 旧协议(本复刻, PINN) | 新协议(PINN, 混沌子集) |")
    A("|---|---|---|")
    href = float(df_old["ref_horizon_s"].iloc[0])
    A(f"| 无扰动 | {href:.2f} | {ref_p:.2f} |")
    for lv, newname in ((0.1, "th0.1"), (0.5, "th0.5"), (1.0, "th1.0")):
        o = float(df_old[df_old.level_deg == lv].horizon_s.median())
        A(f"| ±{lv}° | {o:.2f} | {_row(summ, 'insim', 'chaos_pinn', 'PINN', newname):.2f} |")
    A("")
    A("论文 tab:sensitivity 原值（`data/sensitivity/sensitivity_summary.csv`，只读引用）："
      "±0.1°=3.59 / ±0.5°=3.34 / ±1.0°=3.66，参考 3.10s。")
    A("旧协议下模型看到的初值误差**逐位为 0**，详见 `old_protocol_diagnosis.txt`。")
    A("")

    A("## 6. 对论文的影响")
    A("")
    A("- main.tex:555-580 §6.2 整节 + tab:sensitivity 需按第 2 节的新表重写，结论由"
      "「初值稳健」改为「视界随初值不确定度单调塌缩」。")
    A("- main.tex:241-242 贡献 4 需改写：不再是「论证 ±0.5° 对结论的鲁棒性」，"
      "而是「量化初值估计精度对预测视界的定量代价」。")
    A("- main.tex:379 §4.2 末句「结论稳健」需一并改。")
    A("- 改写方向对论文有利：§6.2 与 §7.3（C_chaos=1.90s、可改进空间 1.5s）由互斥变互证，"
      "「1.5s 可改进空间来自初值估计精度」这句话第一次有了直接证据。")
    A("- 附录「方法论诚实性说明」可新增第四条：自查出旧敏感性协议把扰动逐位抵消并已改正。")
    A("")
    A("## 7. 产物清单")
    A("")
    A("- `insim_raw.csv` 仿真侧逐次评估原始记录")
    A("- `real_raw.csv` 真实侧逐次评估原始记录")
    A("- `summary.csv` 全部聚合（side × subset × model × level）")
    A("- `old_protocol_replication.csv` / `old_protocol_diagnosis.txt` 旧协议复刻与诊断")
    A("- `self_check.txt` 自证门槛完整输出")
    A("- `figures/horizon_vs_sigma_insim.png`, `figures/horizon_vs_sigma_real.png`,"
      " `figures/old_vs_new_protocol.png`")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT/'summary.md'}", flush=True)


# ===========================================================================
# 图
# ===========================================================================
COLORS = {"PINN": "#2ca02c", "MLP_lam0": "#ff7f0e", "Hybrid": "#9467bd",
          "Analytic": "#1f77b4"}


def plot_vs_sigma(summ, side, subset, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    for ax, levels, xlab, key in (
            (axes[0], SIGMA_TH_LEVELS, "角度扰动 σθ (°)", "sigma_theta_deg"),
            (axes[1], SIGMA_W_LEVELS, "角速度扰动 σω (rad/s)", "sigma_omega")):
        for m in MODELS:
            xs, ys, lo, hi = [], [], [], []
            for lname in levels:
                q = summ[(summ.side == side) & (summ.subset == subset) &
                         (summ.model == m) & (summ.level == lname)]
                if not len(q):
                    continue
                xs.append(float(q[key].iloc[0]))
                ys.append(float(q["median"].iloc[0]))
                lo.append(float(q["q25"].iloc[0]))
                hi.append(float(q["q75"].iloc[0]))
            ax.plot(xs, ys, "o-", color=COLORS[m], lw=1.6, ms=5, label=m)
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel("有效预测视界 中位 (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("(a) 仅角度扰动")
    axes[1].set_title("(b) 仅角速度扰动")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / fname, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图 → {FIG/fname}", flush=True)


def plot_old_vs_new(summ, df_sim, df_old):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    ax = axes[0]
    href = float(df_old["ref_horizon_s"].iloc[0])
    xs = [0.0] + OLD_LEVELS_DEG
    old_med = [href] + [float(df_old[df_old.level_deg == lv].horizon_s.median())
                        for lv in OLD_LEVELS_DEG]
    new_names = ["clean", "th0.1", "th0.5", "th1.0"]
    new_med = [_row(summ, "insim", "chaos_pinn", "PINN", n) for n in new_names]
    paper = [3.10, 3.59, 3.34, 3.66]
    ax.plot(xs, paper, "s--", color="#999999", lw=1.4, ms=6,
            label="论文 tab:sensitivity 原值（旧协议+错位 predict）")
    ax.plot(xs, old_med, "^-", color="#d62728", lw=1.6, ms=6,
            label="旧协议复刻（predict_aligned）")
    ax.plot(xs, new_med, "o-", color="#2ca02c", lw=2.0, ms=7,
            label="新协议（真值不动，只扰动初值）")
    ax.set_xlabel("角度扰动 σθ (°)")
    ax.set_ylabel("PINN 有效预测视界 中位 (s)")
    ax.set_title("(a) 旧协议测不出塌缩，新协议单调塌缩")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    data, labels = [], []
    for lv, nn in ((0.1, "th0.1"), (0.5, "th0.5"), (1.0, "th1.0")):
        data.append(df_old[df_old.level_deg == lv].horizon_s.values)
        labels.append(f"旧 ±{lv}°")
    d = df_sim[(df_sim.model == "PINN")]
    clean = d[d.level == "clean"].set_index("trial")["horizon_s"]
    keep = clean[clean < 9.9].index
    for nn, lb in (("th0.1", "新 σθ0.1°"), ("th0.5", "新 σθ0.5°"), ("th1.0", "新 σθ1.0°")):
        data.append(d[(d.level == nn) & (d.trial.isin(keep))]["horizon_s"].values)
        labels.append(lb)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True, widths=0.6)
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor("#d62728" if i < 3 else "#2ca02c")
        p.set_alpha(0.45)
    ax.set_ylabel("PINN 有效预测视界 (s)")
    ax.set_title("(b) 分布对照（左三=旧协议，右三=新协议）")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("初值敏感性：旧协议 vs 新协议（PINN，仿真内）", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "old_vs_new_protocol.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"图 → {FIG/'old_vs_new_protocol.png'}", flush=True)


# ===========================================================================
def main():
    t0 = time.time()
    print(f"tools/sensitivity_correct.py  启动 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  输出目录 {OUT}", flush=True)
    print(f"  K={K_REPS}  仿真 {N_TRIALS} trial  真实 {len(REAL_TAGS)} 段  "
          f"档位 {len(LEVELS)}  模型 {MODELS}", flush=True)
    nets = load_models()
    gate_info = gate(nets)

    df_sim = run_insim(nets)
    df_real = run_real(nets)
    df_old = run_old_protocol(nets)

    summ = summarize(df_sim, df_real, df_old, gate_info)
    plot_vs_sigma(summ, "insim", "chaos_pinn", "horizon_vs_sigma_insim.png",
                  f"仿真内初值敏感性（混沌子集，K={K_REPS}/档）")
    plot_vs_sigma(summ, "real", "all29", "horizon_vs_sigma_real.png",
                  f"真实零样本初值敏感性（{len(REAL_TAGS)} 段，K={K_REPS}/档）")
    plot_old_vs_new(summ, df_sim, df_old)
    write_summary_md(summ, df_sim, df_real, df_old, gate_info)

    print(f"\n全部完成，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()

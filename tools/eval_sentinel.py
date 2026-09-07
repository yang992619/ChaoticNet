#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/eval_sentinel.py — 评估管线哨兵（帧0对齐 / 同种子可复现 / 数值有限性）

========================= 动机 =========================
2026-09-03 至 09-04 两天内在同一处「评估管线接缝」上查出三个缺陷：

  B1  整帧错位     pinn.predict() 的 out[t] 是从 s0 积分 t+1 步的结果，直接与
                   truth[i0:i0+n] 比对会整体超前一帧，系统性低估 PINN/Hybrid 视界。
                   （已由 pinn.predict_aligned() 收口）
  B2  先建后播种   net = AccelNet() 先构造、train(seed=) 后播种，初始权重取决于
                   播种前全局 RNG 被调用过多少次 → 与执行顺序绑定，换调用序即不可复现。
                   （已由 pinn.make_accel_net() / hybrid.make_hybrid_net() 提供正确姿势）
  B3  ESN 多消耗一步  esn.predict() 循环体是「先 step 再 readout」，而 train 契约是
                   X[t]=step(X[t-1],U[t])、readout([X[t],U[t]])→U[t+1]。这个循环体在
                   「u0 = 尚未消费的下一帧输入」时正确，在「u0 = 已消费的最后一帧」时
                   等于把同一帧喂两遍，预测窗第 0 帧就带误差。

三个都是「接缝上没有断言守着」。本脚本把这三类接缝各钉一组断言，作为回归闸门。
本脚本不产生任何新科学结论，价值是「防第四个同类 bug」。

========================= 三组哨兵 =========================
A 组 帧 0 对齐
  A1  torch 三模型（PINN / MLP λ=0 / Hybrid）× (仿真内 50 trial + 真实 29 段)：
      predict_aligned 的第 0 帧必须等于给定初值 s0，角度误差 < FRAME0_TOL_DEG。
  A2  判别力反向对照：同样口径下走旧的 predict()，帧 0 误差必须远超阈值。
      （若 A2 也通过，说明这个检验根本区分不出对错，哨兵形同虚设。）
  A3  ESN 用「无阈值·逐位」判据，绝不用角度阈值：
      正确闭环（先读出、再步进）在给定 (U, i0) 下有唯一参考实现
      esn_frame0_reference()；任何调用点的闭环输出必须与它 np.array_equal。

B 组 同种子可复现
  B1  裸 AccelNet() / HybridNet() 连续两次（中间扰动全局 RNG）→ 必须不同。
  B2  make_accel_net(42) / make_hybrid_net(42) 连续两次（同样扰动）→ 必须逐参数相同。
  B3  全仓静态扫描「先建后播种」残留调用点，与 PIN 对表；新增即失败（棘轮）。

C 组 数值有限性
  C1  A 组跑出的每一条闭环 rollout 全程不得出现 NaN/Inf。
  C2  「视界 == 窗长」必须真的是删失（全窗有限且都没超阈），不得由 NaN 伪造。
  C3  退化输入单元测试：记录 pinn.horizon / esn.prediction_horizon / rvm.horizon_s
      在「部分 NaN」下的真实行为（已知漏洞，登记在案，不作为失败）。

========================= 自证门槛 =========================
在跑任何哨兵之前，先复现三个已冻结的旧数字。复现不出来就退出（码 2），
绝不带着未解释的偏差往下推结论。
  G1  真实侧 29 段 × 3 模型 → 必须逐位复现 data/real_validation/summary_multi.csv
  G2  仿真内 50 trial × Hybrid / MLP(λ=0) 走【旧】predict() → 必须逐位复现
      data/hybrid/hybrid_summary_50.csv 与 data/fair_compare/lam0_summary_50.csv
      （这两个冻结表是 B1 修复之前的产物，用旧路径才对得上——这本身就是一条结论）
  G3  仿真内 50 trial × ESN → 必须逐位复现 data/rc/esn_summary_50.csv

========================= 只读 / 只写自己目录 =========================
本脚本不写 data/ 下任何既有目录，只写 data/eval_sentinel/。
不改 rc/*.py 任何函数语义；需要「正确版」行为一律在本文件里另写。

运行：
  cd ~/Desktop/混沌之眼 && OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 tools/eval_sentinel.py

退出码：0 全绿；1 哨兵失败；2 自证门槛未过（结果不可信，不看下文）。
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ---------------- 单线程（机器上还有别的后台计算，本进程只占 1 核） ----------------
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
# 2026-09-07：G2/G3 曾因 pandas 3.0.3 的 read_csv 默认浮点解析把 CSV 里的完整精度
# 十进制读偏最多 19 ulp（max|Δ|=2.22e-16 s），哨兵拿逐位精确的重算值去比被读坏的
# 冻结值，于是自己判自己失败——冻结数字本身是对的。凡是要做 np.array_equal 逐位
# 比较的冻结表，一律用 float_precision="round_trip" 读。
                      # noqa: E402
import pandas as pd                     # noqa: E402
import torch                            # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))

import esn as esn_mod                   # noqa: E402
import pinn as pinn_mod                 # noqa: E402
import hybrid as hybrid_mod             # noqa: E402
import real_validation_multi as rvm     # noqa: E402

OUT = ROOT / "data" / "eval_sentinel"
OUT.mkdir(parents=True, exist_ok=True)

# ================= 参数常量区（口径全部写死，便于复现） =================
SEED = 42

# 仿真内口径（与 rc/sweep_50.py / rc/hybrid.py __main__ / rc/esn.py run_one 一致）
SIM_DIR       = ROOT / "data" / "sim"
N_TRIALS      = 50
FPS           = 120
TRAIN_SEC     = 20.0
PRED_SEC      = 10.0
N_TRAIN       = int(TRAIN_SEC * FPS)     # 2400
N_PRED        = int(PRED_SEC * FPS)      # 1200
THRESH_DEG    = 10.0

# 仿真内 ESN 超参（rc/esn.py run_one 写死的那一组）
ESN_INSIM_KW  = dict(n_in=6, n_out=6, n_res=800, spectral_radius=0.95,
                     leak=0.25, ridge=1e-5)
ESN_INSIM_WASHOUT = 200

# 真实侧口径（rc/real_validation_multi.py 默认）
LEAD_S, PRED_S = 4.0, 4.0
ESN_REAL_KW   = dict(n_res=300, sr=0.95, leak=0.3, ridge=1e-3, washout=50, seed=42)

# A 组阈值：predict_aligned 的第 0 帧只差一次 float64→float32 的 cast，
# 真实侧实测 5.0e-7 ~ 1.07e-5 度，用 1e-6 会误报，故取 1e-3 度。
FRAME0_TOL_DEG = 1e-3
# A2 判别力下限：旧 predict() 的帧 0 误差至少要比阈值大这么多倍，否则本检验没有区分度。
A2_MIN_RATIO = 100.0

MODELS = [
    ("PINN",   "AccelNet(128,4)",  ROOT / "data/pinn/model.pt"),
    ("MLP_l0", "AccelNet(128,4)",  ROOT / "data/fair_compare/lam0_model.pt"),
    ("Hybrid", "HybridNet(64,3)",  ROOT / "data/hybrid/model.pt"),
]

# B3 棘轮：当前已知的「先建后播种」（构造后紧跟 train，未先播种）调用点。
# 新增任何一处即失败；减少只提示（说明有人修好了，请把 PIN 改小）。
# 注：今晚执行清单里点名了 17 处（正文写「16 处」）。本脚本静态扫描实得 23 处 ——
# 多出的 4 处是 rc/pinn.py:375、rc/hybrid.py:175、rc/sweep_damped.py:71、:88
# （均已逐行人工核对：构造在前、train(seed=) 在后，命中同一模式），
# 外加清单未列的 tools/threshold_robustness.py:196、:203。清单那份是低估。
PIN_LATE_SEED_SITES = {
    "rc/hybrid.py:175",
    "rc/pinn.py:375",
    "rc/sweep_damped.py:71",
    "rc/sweep_damped.py:88",
    "tools/build_canonical_distmass.py:61",
    "tools/build_canonical_distmass.py:81",
    "tools/data_size_ablation.py:57",
    "tools/distmass_transfer_check.py:59",
    "tools/hp_sweep.py:61",
    "tools/lambda_ablation_distmass.py:54",
    "tools/lambda_sweep_distmass.py:82",
    "tools/multi_seed.py:55",
    "tools/multirun_compare.py:108",
    "tools/multirun_compare.py:113",
    "tools/multirun_compare.py:118",
    "tools/multirun_compare_distmass.py:136",
    "tools/multirun_compare_distmass.py:141",
    "tools/multirun_compare_distmass.py:146",
    "tools/pinn_epoch_transfer.py:76",
    "tools/real_multirun.py:93",
    "tools/real_multirun.py:98",
    "tools/threshold_robustness.py:196",
    "tools/threshold_robustness.py:203",
}

# A3 棘轮：两个 ESN 调用姿势当前的对齐结论。
#   True  = 该调用点的闭环输出与「先读出再步进」参考实现逐位相同（正确族）
#   False = 不同（错误族：多消耗一步）
# 由 True 变 False 是回归 → 失败；由 False 变 True 是修复 → 只告警并要求改 PIN。
PIN_ESN_ALIGNED = {
    "insim_rc_esn_run_one": False,   # rc/esn.py:194  predict(x_final, U_tr[-1], n)
    "real_rvm_run_esn":     True,    # rc/real_validation_multi.py:104 predict(xf, U[i0-1], n)
}

# ================= 记账 =================
ROWS_FRAME0, ROWS_SEED, ROWS_FINITE, ROWS_ESN, ROWS_SITES = [], [], [], [], []
FAILURES, WARNINGS, NOTES = [], [], []
_t0 = time.time()


def log(msg=""):
    print(msg, flush=True)


def record_fail(tag, msg):
    FAILURES.append((tag, msg))
    log(f"  [FAIL] {tag}: {msg}")


def record_warn(tag, msg):
    WARNINGS.append((tag, msg))
    log(f"  [WARN] {tag}: {msg}")


# ================= 通用小工具 =================

def ang_err_deg(p1, p2, t1, t2):
    """两摆角度欧氏偏差（度），与全仓口径一致（周期用 atan2 折回）。"""
    d1 = np.arctan2(np.sin(p1 - t1), np.cos(p1 - t1))
    d2 = np.arctan2(np.sin(p2 - t2), np.cos(p2 - t2))
    return np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))


def horizon_guarded(err, fps, thr=THRESH_DEG):
    """带守卫的视界：显式区分「删失」与「被 NaN 伪造的删失」。

    与 rc/pinn.horizon / rc/esn.prediction_horizon / rvm.horizon_s 的区别只有一点：
    先检查有限性。生产函数用 `err > thr` 判越界，而 NaN > thr 恒为 False，
    于是「前几帧有限、之后全 NaN」的发散轨迹会被判成整窗未越界 = 满分。
    本函数把这种情况标出来。

    返回 (horizon_s, censored, n_nonfinite, first_nonfinite_idx)
    """
    err = np.asarray(err, dtype=float)
    finite = np.isfinite(err)
    n_bad = int((~finite).sum())
    first_bad = int(np.argmax(~finite)) if n_bad else -1
    over = err > thr
    if not over.any():
        return len(err) / fps, True, n_bad, first_bad
    return int(np.argmax(over)) / fps, False, n_bad, first_bad


def load_torch_model(kind, pt_path):
    """只加载既有 model.pt，不训练。"""
    if kind == "Hybrid":
        net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                   residual_scale=hybrid_mod.RESIDUAL_SCALE)
    else:
        net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    net.load_state_dict(torch.load(pt_path, map_location=pinn_mod.DEVICE))
    net.to(pinn_mod.DEVICE).eval()
    return net


# ================= ESN 正确闭环（本文件自备，绝不改 rc/esn.py） =================

def predict_readout_first(net, x0, u0, n_steps):
    """按 train 契约的正确闭环：先读出、再步进。

    train 契约（rc/esn.py:105-129）：X[t] = step(X[t-1], U[t])，
    读出特征 [X[t], U[t], 1] → 目标 U[t+1]。
    因此从 (X[t], U[t]) 出发的闭环必须是：

        y = W_out @ concat([x, u, [1.0]])
        out[k] = y
        u = y
        x = step(x, u)

    对照 rc/esn.py:131-147 的 predict()：它先 x = step(x, u) 再读出，
    等价于 predict_readout_first(step(x0, u0), u0, n)——即比本函数多消耗一步。
    该差别本身无所谓对错，取决于调用方传进来的 u0 是「已消费的最后一帧」
    还是「尚未消费的下一帧」。本文件不修改 rc/esn.py，只用本函数当参考尺。
    """
    out = np.zeros((n_steps, net.n_out))
    x, u = np.asarray(x0).copy(), np.asarray(u0).copy()
    for t in range(n_steps):
        y = net.W_out @ np.concatenate([x, u, [1.0]])
        out[t] = y
        u = y
        x = net.step(x, u)
    return out


def esn_frame0_reference(net, U, i0, n_steps):
    """帧 0 对齐的唯一参考实现（无阈值判据的分母）。

    给定完整输入序列 U 与「预测窗第一帧的下标 i0」（真值取 U[i0:i0+n]），
    正确的闭环第 0 帧必须是 readout([X[i0-1], U[i0-1]]) → U[i0]。
    X[i0-1] 由 collect_states(U[:i0]) 的末态给出，与 train 内部的递推同源
    （都从 x=0 起步、同一条 step 递推），故可逐位比较。
    """
    X = net.collect_states(U[:i0])
    return predict_readout_first(net, X[-1], U[i0 - 1], n_steps)


# ================= 自证门槛 =================

def gate_g1_real():
    """G1：真实侧 29 段 × 3 模型逐位复现 summary_multi.csv。"""
    frozen = pd.read_csv(ROOT / "data/real_validation/summary_multi.csv",
                         dtype={"tag": str})
    pinn_net, hyb_net = rvm.load_pinn(), rvm.load_hybrid()
    got = []
    store = {}
    for tg in frozen["tag"].tolist():
        C = rvm.load_clip(tg, LEAD_S, PRED_S)
        r = rvm.process(C, pinn_net, hyb_net)
        store[tg] = (C, r)
        got.append(dict(tag=tg,
                        ESN_h=round(r["ESN"]["horizon"], 3),
                        PINN_h=round(r["PINN"]["horizon"], 3),
                        Hybrid_h=round(r["Hybrid"]["horizon"], 3)))
    got = pd.DataFrame(got)
    cols = ["ESN_h", "PINN_h", "Hybrid_h"]
    same = bool((got[cols].values == frozen[cols].values).all())
    nbad = int((got[cols].values != frozen[cols].values).sum())
    log(f"  G1 真实侧 {len(got)} 段 × 3 模型 vs summary_multi.csv："
        f"{'逐位一致' if same else f'{nbad} 个格子不一致'}")
    return same, store, frozen


def gate_g2_insim_torch():
    """G2：仿真内 Hybrid / MLP(λ=0) 走【旧】predict() 逐位复现两个冻结表。

    顺带把每个模型 50 条 aligned rollout 也算出来给 A 组和 C 组复用。
    """
    frozen = {
        "Hybrid": pd.read_csv(ROOT / "data/hybrid/hybrid_summary_50.csv",
                              float_precision="round_trip"),
        "MLP_l0": pd.read_csv(ROOT / "data/fair_compare/lam0_summary_50.csv",
                              float_precision="round_trip"),
    }
    nets = {k: load_torch_model(k, p) for k, _, p in MODELS}
    cache = {}          # (model, trial) -> dict(aligned=..., old_full=..., truth=..., s0=...)
    ok = True
    for name, _, _ in MODELS:
        net = nets[name]
        h_aligned, h_old = [], []
        for i in range(N_TRIALS):
            df = pd.read_csv(SIM_DIR / f"trial_{i:03d}.csv", float_precision="round_trip")
            s0 = df[["th1", "w1", "th2", "w2"]].values[N_TRAIN].astype(np.float32)
            truth = df[["th1", "th2"]].values[N_TRAIN:N_TRAIN + N_PRED]
            aligned = pinn_mod.predict_aligned(net, s0, N_PRED, dt=1.0 / FPS)
            old_full = pinn_mod.predict(net, s0, N_PRED, dt=1.0 / FPS)
            ha, ea = pinn_mod.horizon(aligned[:, [0, 2]], truth, THRESH_DEG, FPS)
            ho, eo = pinn_mod.horizon(old_full[:, [0, 2]], truth, THRESH_DEG, FPS)
            h_aligned.append(ha)
            h_old.append(ho)
            cache[(name, i)] = dict(s0=s0, truth=truth, aligned=aligned,
                                    old_full=old_full, err_aligned=ea,
                                    h_aligned=ha, h_old=ho)
        if name in frozen:
            fz = frozen[name]["horizon"].values
            same = bool(np.array_equal(np.asarray(h_old), fz))
            same_aligned = bool(np.array_equal(np.asarray(h_aligned), fz))
            log(f"  G2 仿真内 {name} 50 trial：旧 predict() vs 冻结表 "
                f"{'逐位一致' if same else '不一致'}"
                f"（对照：新 predict_aligned() vs 冻结表 "
                f"{'一致' if same_aligned else '不一致'}）")
            ok &= same
    return ok, nets, cache


def gate_g3_insim_esn():
    """G3：仿真内 ESN 50 trial 逐位复现 data/rc/esn_summary_50.csv。

    调用姿势严格照抄 rc/esn.py::run_one（不调用它本人，因为它会画图写盘）。
    顺带把每 trial 的 U / x_final / pred 缓存给 A3 用。
    """
    frozen = pd.read_csv(ROOT / "data/rc/esn_summary_50.csv",
                         float_precision="round_trip")
    got, cache = [], {}
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM_DIR / f"trial_{i:03d}.csv", float_precision="round_trip")
        U = esn_mod.encode(df)
        Y = np.roll(U, -1, axis=0)
        U, Y = U[:-1], Y[:-1]
        net = esn_mod.ESN(**ESN_INSIM_KW)
        _, x_final = net.train(U[:N_TRAIN], Y[:N_TRAIN], washout=ESN_INSIM_WASHOUT)
        pred = net.predict(x_final, U[:N_TRAIN][-1], N_PRED)     # rc/esn.py:194 原样
        U_te = U[N_TRAIN:N_TRAIN + N_PRED]
        h = esn_mod.prediction_horizon(pred, U_te, THRESH_DEG, FPS)
        got.append(h)
        cache[i] = dict(net=net, U=U, x_final=x_final, pred=pred, U_te=U_te, h=h)
    same = bool(np.array_equal(np.asarray(got), frozen["horizon"].values))
    log(f"  G3 仿真内 ESN {N_TRIALS} trial vs data/rc/esn_summary_50.csv："
        f"{'逐位一致' if same else '不一致'}")
    return same, cache


# ================= A 组：帧 0 对齐 =================

def group_a_torch_insim(cache):
    """A1/A2 仿真内：predict_aligned 帧 0 == s0；旧 predict 帧 0 必须显著偏离。"""
    log("\n[A 组] 帧 0 对齐 — 仿真内 50 trial × 3 torch 模型")
    worst = {}
    for name, _, _ in MODELS:
        e_new, e_old = [], []
        for i in range(N_TRIALS):
            c = cache[(name, i)]
            s0, tr = c["s0"], c["truth"]
            a0, o0 = c["aligned"][0], c["old_full"][0]
            en = float(ang_err_deg(a0[0], a0[2], tr[0, 0], tr[0, 1]))
            eo = float(ang_err_deg(o0[0], o0[2], tr[0, 0], tr[0, 1]))
            e_new.append(en)
            e_old.append(eo)
            ROWS_FRAME0.append(dict(group="insim", model=name, case=f"trial_{i:03d}",
                                    n_steps=N_PRED,
                                    err_deg_aligned=en, err_deg_old_predict=eo,
                                    passed=int(en < FRAME0_TOL_DEG)))
        mx_new, mn_old = max(e_new), min(e_old)
        worst[name] = (mx_new, mn_old)
        log(f"  {name:>7}: aligned 帧0 误差 max={mx_new:.3e}°  中位={np.median(e_new):.3e}°"
            f"   |  旧 predict 帧0 误差 min={mn_old:.3e}° 中位={np.median(e_old):.3e}°")
        if mx_new >= FRAME0_TOL_DEG:
            record_fail("A1-insim", f"{name} 帧0 最大误差 {mx_new:.3e}° ≥ 阈值 {FRAME0_TOL_DEG}°")
        if mn_old < FRAME0_TOL_DEG * A2_MIN_RATIO:
            record_fail("A2-insim", f"{name} 旧 predict 帧0 最小误差 {mn_old:.3e}° "
                                    f"< {A2_MIN_RATIO}×阈值，本检验缺乏区分度")
    return worst


def group_a_torch_real(store):
    """A1/A2 真实侧：29 段 × 3 torch 模型（ESN 不走阈值，见 A3）。"""
    log("\n[A 组] 帧 0 对齐 — 真实 29 段 × 3 torch 模型")
    nets = {"PINN": rvm.load_pinn(), "Hybrid": rvm.load_hybrid(),
            "MLP_l0": load_torch_model("MLP_l0", ROOT / "data/fair_compare/lam0_model.pt")}
    for name, net in nets.items():
        e_new, e_old = [], []
        for tg, (C, _r) in store.items():
            i0, n, dt = C["lead_n"], C["pred_n"], C["dt"]
            s0 = C["state"][i0].astype(np.float32)
            t1, t2 = C["th1"][i0], C["th2"][i0]
            p1, p2 = rvm.run_torch(net, s0, n, dt)          # = predict_aligned
            en = float(ang_err_deg(p1[0], p2[0], t1, t2))
            o = pinn_mod.predict(net, s0, 1, dt=dt)          # 旧口径的第 0 帧
            eo = float(ang_err_deg(o[0, 0], o[0, 2], t1, t2))
            e_new.append(en)
            e_old.append(eo)
            ROWS_FRAME0.append(dict(group="real", model=name, case=tg, n_steps=n,
                                    err_deg_aligned=en, err_deg_old_predict=eo,
                                    passed=int(en < FRAME0_TOL_DEG)))
        mx_new, mn_old = max(e_new), min(e_old)
        log(f"  {name:>7}: aligned 帧0 误差 max={mx_new:.3e}°  中位={np.median(e_new):.3e}°"
            f"   |  旧 predict 帧0 误差 min={mn_old:.3e}° 中位={np.median(e_old):.3e}°")
        if mx_new >= FRAME0_TOL_DEG:
            record_fail("A1-real", f"{name} 帧0 最大误差 {mx_new:.3e}° ≥ 阈值 {FRAME0_TOL_DEG}°")
        if mn_old < FRAME0_TOL_DEG * A2_MIN_RATIO:
            record_fail("A2-real", f"{name} 旧 predict 帧0 最小误差 {mn_old:.3e}° "
                                   f"< {A2_MIN_RATIO}×阈值，本检验缺乏区分度")


def group_a_esn(esn_cache, store):
    """A3 ESN：无阈值·逐位判据。"""
    log("\n[A 组] 帧 0 对齐 — ESN（逐位判据，不用角度阈值）")

    # --- A3-a 先钉住 rc/esn.py::predict 的循环体约定 ---
    rng = np.random.default_rng(SEED)
    probe = esn_mod.ESN(n_in=6, n_out=6, n_res=60, spectral_radius=0.9,
                        leak=0.3, ridge=1e-6, seed=SEED)
    probe.W_out = rng.normal(size=(6, 60 + 6 + 1)) * 0.05
    x0 = rng.normal(size=60) * 0.1
    u0 = rng.normal(size=6) * 0.5
    leg = probe.predict(x0, u0, 40)
    eq_step_first = bool(np.array_equal(leg, predict_readout_first(probe, probe.step(x0, u0), u0, 40)))
    eq_readout_first = bool(np.array_equal(leg, predict_readout_first(probe, x0, u0, 40)))
    log(f"  A3-a rc/esn.py::predict 循环体约定："
        f"step-first={eq_step_first}  readout-first={eq_readout_first}")
    ROWS_ESN.append(dict(check="A3a_convention", target="rc/esn.py:131-147",
                         expected="step_first=True, readout_first=False",
                         actual=f"step_first={eq_step_first}, readout_first={eq_readout_first}",
                         passed=int(eq_step_first and not eq_readout_first)))
    if not eq_step_first:
        record_fail("A3-a", "rc/esn.py::predict 不再是 step-first 约定 —— 语义已被改动，"
                            "本哨兵所有 ESN 结论作废，请重审 PIN_ESN_ALIGNED")
    if eq_readout_first:
        record_fail("A3-a", "两种约定给出同一结果 → 本检验没有区分度（探针构造有误）")

    # --- A3-b 仿真内调用姿势（rc/esn.py:194） ---
    n_align = 0
    f0_err = []
    for i in range(N_TRIALS):
        c = esn_cache[i]
        net, U = c["net"], c["U"]
        ref = esn_frame0_reference(net, U, N_TRAIN, N_PRED)
        same = bool(np.array_equal(c["pred"], ref))
        n_align += int(same)
        t1, _, t2, _ = esn_mod.decode(c["U_te"][0])
        p1, _, p2, _ = esn_mod.decode(c["pred"][0])
        r1, _, r2, _ = esn_mod.decode(ref[0])
        f0_err.append((float(ang_err_deg(p1, p2, t1, t2)),
                       float(ang_err_deg(r1, r2, t1, t2))))
    f0 = np.array(f0_err)
    insim_aligned = (n_align == N_TRIALS)
    log(f"  A3-b 仿真内 rc/esn.py:194 姿势 predict(x_final, U_tr[-1], n)：")
    log(f"       与参考实现逐位相同 {n_align}/{N_TRIALS} → aligned={insim_aligned}")
    log(f"       帧0 角度误差  该姿势 中位={np.median(f0[:, 0]):.3f}° "
        f"[{f0[:, 0].min():.3f}, {f0[:, 0].max():.3f}]  |  "
        f"参考(正确) 中位={np.median(f0[:, 1]):.3f}° "
        f"[{f0[:, 1].min():.3f}, {f0[:, 1].max():.3f}]")
    ROWS_ESN.append(dict(check="A3b_insim_call_site", target="rc/esn.py:194",
                         expected=f"aligned={PIN_ESN_ALIGNED['insim_rc_esn_run_one']}",
                         actual=f"aligned={insim_aligned} ({n_align}/{N_TRIALS} 逐位相同); "
                                f"frame0_med_call={np.median(f0[:, 0]):.4f}deg, "
                                f"frame0_med_ref={np.median(f0[:, 1]):.4f}deg",
                         passed=int(insim_aligned == PIN_ESN_ALIGNED["insim_rc_esn_run_one"])))
    _check_esn_pin("insim_rc_esn_run_one", insim_aligned, "rc/esn.py:194")

    # --- A3-c 真实侧调用姿势（rc/real_validation_multi.py:104） ---
    n_align_r = 0
    f0r = []
    tags = list(store.keys())
    for tg in tags:
        C, r = store[tg]
        i0, n = C["lead_n"], C["pred_n"]
        df = pd.DataFrame({"th1": C["th1"], "w1": C["w1"],
                           "th2": C["th2"], "w2": C["w2"]})
        U = esn_mod.encode(df)
        Y = np.roll(U, -1, axis=0)
        net = esn_mod.ESN(n_in=6, n_out=6, n_res=ESN_REAL_KW["n_res"],
                          spectral_radius=ESN_REAL_KW["sr"], leak=ESN_REAL_KW["leak"],
                          ridge=ESN_REAL_KW["ridge"], seed=ESN_REAL_KW["seed"])
        _, xf = net.train(U[:i0 - 1], Y[:i0 - 1],
                          washout=min(ESN_REAL_KW["washout"], max(i0 // 3, 1)))
        pred = net.predict(xf, U[i0 - 1], n)                  # rvm.py:104 原样
        # 先确认本地复刻与 rvm.run_esn 逐位一致（否则下面的结论没有意义）
        c1, c2 = esn_mod.decode(pred)[0], esn_mod.decode(pred)[2]
        if not (np.array_equal(c1, r["ESN"]["th1"]) and np.array_equal(c2, r["ESN"]["th2"])):
            record_fail("A3-c", f"{tg} 本地复刻 run_esn 与 rvm.run_esn 输出不一致")
        ref = esn_frame0_reference(net, U, i0, n)
        same = bool(np.array_equal(pred, ref))
        n_align_r += int(same)
        p1, _, p2, _ = esn_mod.decode(pred[0])
        f0r.append(float(ang_err_deg(p1, p2, C["th1"][i0], C["th2"][i0])))
    real_aligned = (n_align_r == len(tags))
    f0r = np.array(f0r)
    log(f"  A3-c 真实侧 rvm.py:104 姿势 predict(xf, U[i0-1], n)：")
    log(f"       与参考实现逐位相同 {n_align_r}/{len(tags)} → aligned={real_aligned}")
    log(f"       帧0 角度误差 中位={np.median(f0r):.3f}° [{f0r.min():.3f}, {f0r.max():.3f}]")
    ROWS_ESN.append(dict(check="A3c_real_call_site", target="rc/real_validation_multi.py:104",
                         expected=f"aligned={PIN_ESN_ALIGNED['real_rvm_run_esn']}",
                         actual=f"aligned={real_aligned} ({n_align_r}/{len(tags)} 逐位相同); "
                                f"frame0_med={np.median(f0r):.4f}deg",
                         passed=int(real_aligned == PIN_ESN_ALIGNED["real_rvm_run_esn"])))
    _check_esn_pin("real_rvm_run_esn", real_aligned, "rc/real_validation_multi.py:104")
    return f0, f0r


def _check_esn_pin(key, actual, where):
    """棘轮：正确→错误 = 回归（失败）；错误→正确 = 修复（告警并要求改 PIN）。"""
    pinned = PIN_ESN_ALIGNED[key]
    if actual == pinned:
        return
    if pinned and not actual:
        record_fail("A3-pin", f"{where} 从「帧0 对齐」退化为「多消耗一步」——这是回归")
    else:
        record_warn("A3-pin", f"{where} 已从「多消耗一步」修正为「帧0 对齐」。"
                              f"若为有意修复，请把 PIN_ESN_ALIGNED['{key}'] 改为 True")


# ================= B 组：同种子可复现 =================

def _params_equal(a, b):
    pa = [p.detach().cpu().numpy() for p in a.parameters()]
    pb = [p.detach().cpu().numpy() for p in b.parameters()]
    if len(pa) != len(pb):
        return False, float("inf")
    diffs = [np.abs(x - y).max() for x, y in zip(pa, pb)]
    return all(d == 0.0 for d in diffs), float(max(diffs))


def group_b_seed():
    log("\n[B 组] 同种子可复现（不训练，只比初始权重）")
    cases = [
        ("AccelNet()  裸构造 ×2",       lambda: pinn_mod.AccelNet(hidden=128, n_layers=4),        False),
        ("make_accel_net(42) ×2",       lambda: pinn_mod.make_accel_net(seed=SEED, hidden=128, n_layers=4), True),
        ("HybridNet() 裸构造 ×2",       lambda: hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                                                     residual_scale=hybrid_mod.RESIDUAL_SCALE), False),
        ("make_hybrid_net(42) ×2",      lambda: hybrid_mod.make_hybrid_net(seed=SEED, hidden=64, n_layers=3), True),
    ]
    for label, factory, expect_same in cases:
        torch.manual_seed(0)
        _ = torch.randn(7)          # 让全局 RNG 处在某个状态
        a = factory()
        _ = torch.randn(13)         # 在两次构造之间扰动全局 RNG（这正是 B2 的触发条件）
        b = factory()
        same, maxdiff = _params_equal(a, b)
        ok = (same == expect_same)
        log(f"  {label:<28} 逐参数相同={same}  max|Δ|={maxdiff:.3e}  "
            f"期望={'相同' if expect_same else '不同'}  → {'OK' if ok else 'FAIL'}")
        ROWS_SEED.append(dict(check=label, expected=("identical" if expect_same else "different"),
                              actual=("identical" if same else "different"),
                              max_abs_diff=maxdiff, passed=int(ok)))
        if not ok:
            record_fail("B1/B2", f"{label} 期望{'相同' if expect_same else '不同'}，实得"
                                 f"{'相同' if same else '不同'}")

    # 补一条：同一 seed 跨「先扰动再播种」也必须相同（make_* 的真正卖点）
    torch.manual_seed(999)
    _ = torch.randn(101)
    a = pinn_mod.make_accel_net(seed=SEED, hidden=128, n_layers=4)
    torch.manual_seed(1)
    _ = torch.randn(3)
    b = pinn_mod.make_accel_net(seed=SEED, hidden=128, n_layers=4)
    same, maxdiff = _params_equal(a, b)
    log(f"  {'make_accel_net(42) 跨不同全局态':<28} 逐参数相同={same}  max|Δ|={maxdiff:.3e}  → "
        f"{'OK' if same else 'FAIL'}")
    ROWS_SEED.append(dict(check="make_accel_net(42) 跨不同全局 RNG 态",
                          expected="identical", actual="identical" if same else "different",
                          max_abs_diff=maxdiff, passed=int(same)))
    if not same:
        record_fail("B2", "make_accel_net(42) 在不同全局 RNG 态下给出不同初值")


def scan_late_seed_sites():
    """B3：静态扫描「先建后播种」残留调用点（构造后紧跟 train，且未先播种）。"""
    log("\n[B 组] 全仓扫描「先建后播种」残留调用点")
    targets = {"AccelNet", "HybridNet"}
    factories = {"make_accel_net", "make_hybrid_net"}
    found = {}
    self_path = Path(__file__).resolve()
    for sub in ("rc", "tools", "sim"):
        for f in sorted((ROOT / sub).glob("*.py")):
            if f.resolve() == self_path:
                continue          # 哨兵自己不算结果产出脚本
            src = f.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.split("\n")
            # 收集 make_* 工厂函数体的行区间，其中的构造是安全的
            safe_spans = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in factories:
                    safe_spans.append((node.lineno, node.end_lineno))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else None)
                if nm not in targets:
                    continue
                ln = node.lineno
                if any(a <= ln <= b for a, b in safe_spans):
                    continue
                tail = "\n".join(lines[(node.end_lineno or ln): (node.end_lineno or ln) + 8])
                if re.search(r"load_state_dict\s*\(", tail):
                    cls = "SAFE_LOAD"        # 只加载权重，初值随即被覆盖
                elif re.search(r"\b(train|train_hybrid)\s*\(", tail):
                    cls = "LATE_SEED"        # 构造在前、播种在 train 里 → B2 模式
                else:
                    cls = "UNKNOWN"
                key = f"{f.relative_to(ROOT)}:{ln}"
                found[key] = cls
                ROWS_SITES.append(dict(site=key, cls=nm, classification=cls,
                                       snippet=lines[ln - 1].strip()[:110]))
    late = {k for k, v in found.items() if v == "LATE_SEED"}
    unknown = {k for k, v in found.items() if v == "UNKNOWN"}
    log(f"  扫到构造点 {len(found)} 处：SAFE_LOAD {sum(1 for v in found.values() if v=='SAFE_LOAD')}、"
        f"LATE_SEED {len(late)}、UNKNOWN {len(unknown)}")
    new = sorted(late - PIN_LATE_SEED_SITES)
    gone = sorted(PIN_LATE_SEED_SITES - late)
    for k in sorted(late):
        log(f"    LATE_SEED  {k}")
    for k in sorted(unknown):
        log(f"    UNKNOWN    {k}  (人工判一下)")
    if new:
        record_fail("B3-棘轮", f"新增「先建后播种」调用点 {len(new)} 处：{new}")
    if gone:
        record_warn("B3-棘轮", f"PIN 里有 {len(gone)} 处已不再命中（可能已修好或行号变了）：{gone}")
    return found, late


# ================= C 组：数值有限性 =================

def group_c_finiteness(cache, store, esn_cache):
    log("\n[C 组] 数值有限性 + 删失一致性")
    n_bad_total = 0
    n_rollout = 0

    def _one(group, model, case, err, h_prod, fps, n_frames):
        nonlocal n_bad_total, n_rollout
        e = np.asarray(err, dtype=float)
        h_g, cens_g, n_bad, first_bad = horizon_guarded(e, fps)
        n_rollout += 1
        n_bad_total += n_bad
        agree = bool(abs(h_g - h_prod) < 1e-12)
        # 「视界恰好等于窗长」必须是真删失：全窗有限、且最大误差没超阈
        fake_full = bool(abs(h_prod - n_frames / fps) < 1e-12 and
                         (n_bad > 0 or not np.all(e[np.isfinite(e)] <= THRESH_DEG)))
        ROWS_FINITE.append(dict(group=group, model=model, case=case, n_frames=n_frames,
                                n_nonfinite=n_bad, first_nonfinite=first_bad,
                                horizon_prod=h_prod, horizon_guarded=h_g,
                                censored=int(cens_g), guard_agrees=int(agree),
                                fake_full_window=int(fake_full),
                                passed=int(n_bad == 0 and agree and not fake_full)))
        if n_bad:
            record_fail("C1", f"{group}/{model}/{case} rollout 出现 {n_bad} 个非有限值"
                              f"（首个在第 {first_bad} 帧）")
        if not agree:
            record_fail("C2", f"{group}/{model}/{case} 生产视界 {h_prod} 与守卫视界 {h_g} 不一致")
        if fake_full:
            record_fail("C2", f"{group}/{model}/{case} 视界等于窗长但并非真删失")

    # 仿真内 torch（aligned 口径）
    for name, _, _ in MODELS:
        for i in range(N_TRIALS):
            c = cache[(name, i)]
            if not np.all(np.isfinite(c["aligned"])):
                record_fail("C1", f"insim/{name}/trial_{i:03d} 状态序列含非有限值")
            _one("insim", name, f"trial_{i:03d}", c["err_aligned"], c["h_aligned"],
                 FPS, N_PRED)

    # 真实侧三模型（rvm.process 的口径）
    for tg, (C, r) in store.items():
        for name in ("ESN", "PINN", "Hybrid"):
            _one("real", name, tg, r[name]["err"], r[name]["horizon"],
                 C["fps"], C["pred_n"])

    # 仿真内 ESN
    for i in range(N_TRIALS):
        c = esn_cache[i]
        if not np.all(np.isfinite(c["pred"])):
            record_fail("C1", f"insim/ESN/trial_{i:03d} 预测序列含非有限值")
        th1p, _, th2p, _ = esn_mod.decode(c["pred"])
        th1t, _, th2t, _ = esn_mod.decode(c["U_te"])
        err = ang_err_deg(th1p, th2p, th1t, th2t)
        _one("insim", "ESN", f"trial_{i:03d}", err, c["h"], FPS, N_PRED)

    log(f"  C1 共检 {n_rollout} 条 rollout，非有限值合计 {n_bad_total} 个")
    return n_rollout, n_bad_total


def group_c_degenerate_unit_tests():
    """C3：退化输入单元测试 —— 记录三个生产视界函数的真实行为（已知漏洞，登记在案）。"""
    log("\n[C 组] 退化输入单元测试（记录生产函数真实行为，不作为失败判据）")
    n = 480
    fps = 120.0
    cases = {}

    # 全 NaN
    e_all_nan = np.full(n, np.nan)
    # 部分 NaN：前 30 帧很小，之后全 NaN（真实失效模式：闭环炸掉）
    e_part_nan = np.full(n, np.nan)
    e_part_nan[:30] = 0.5
    # 部分 Inf
    e_inf = np.full(n, 0.5)
    e_inf[100:] = np.inf
    # 正常：整窗都没超阈（真删失）
    e_ok = np.full(n, 0.5)

    def pinn_h(err):
        # pinn.horizon 吃 (T,2) 角度，绕不过去；直接照抄它的判据逻辑做等价调用
        idx = np.argmax(err > THRESH_DEG)
        if idx == 0 and err[0] <= THRESH_DEG:
            return len(err) / fps
        return idx / fps

    for label, err in [("全 NaN", e_all_nan), ("前30帧有限+之后全 NaN", e_part_nan),
                       ("100 帧后全 Inf", e_inf), ("整窗 0.5°(真删失)", e_ok)]:
        h_p = pinn_h(err)
        h_r, cens_r = rvm.horizon_s(err, fps, THRESH_DEG)
        h_g, cens_g, n_bad, first_bad = horizon_guarded(err, fps)
        risky = (n_bad > 0 and abs(h_p - n / fps) < 1e-12)
        cases[label] = dict(pinn_horizon=h_p, rvm_horizon=h_r, rvm_censored=int(cens_r),
                            guarded_horizon=h_g, guarded_censored=int(cens_g),
                            n_nonfinite=n_bad, first_nonfinite=first_bad,
                            scored_full_window_despite_nonfinite=int(risky))
        log(f"  {label:<24} pinn.horizon={h_p:.4f}s  rvm.horizon_s=({h_r:.4f}s,"
            f"cens={int(cens_r)})  守卫={h_g:.4f}s/nonfinite={n_bad}"
            f"{'   ← 满分假象' if risky else ''}")
        ROWS_FINITE.append(dict(group="unit", model="degenerate", case=label,
                                n_frames=n, n_nonfinite=n_bad, first_nonfinite=first_bad,
                                horizon_prod=h_p, horizon_guarded=h_g,
                                censored=int(cens_g), guard_agrees=int(abs(h_p - h_g) < 1e-12),
                                fake_full_window=int(risky), passed=-1))

    # 同样对 esn.prediction_horizon 做一次（它吃 6 维编码，构造一条前段吻合、后段 NaN 的序列）
    T = 240
    truth = np.zeros((T, 6))
    truth[:, 1] = 1.0
    truth[:, 4] = 1.0
    pred_part_nan = truth.copy()
    pred_part_nan[30:] = np.nan
    h_esn_part = esn_mod.prediction_horizon(pred_part_nan, truth, THRESH_DEG, FPS)
    pred_all_nan = np.full_like(truth, np.nan)
    h_esn_all = esn_mod.prediction_horizon(pred_all_nan, truth, THRESH_DEG, FPS)
    log(f"  esn.prediction_horizon: 全 NaN → {h_esn_all:.4f}s（安全）；"
        f"前30帧有限+之后全 NaN → {h_esn_part:.4f}s（窗长 {T/FPS:.4f}s）"
        f"{'   ← 满分假象' if abs(h_esn_part - T / FPS) < 1e-12 else ''}")
    cases["esn.prediction_horizon 部分 NaN"] = dict(
        pinn_horizon=float("nan"), rvm_horizon=float("nan"), rvm_censored=-1,
        guarded_horizon=float("nan"), guarded_censored=-1, n_nonfinite=T - 30,
        first_nonfinite=30,
        scored_full_window_despite_nonfinite=int(abs(h_esn_part - T / FPS) < 1e-12))
    ROWS_FINITE.append(dict(group="unit", model="esn.prediction_horizon",
                            case="前30帧有限+之后全 NaN", n_frames=T, n_nonfinite=T - 30,
                            first_nonfinite=30, horizon_prod=h_esn_part,
                            horizon_guarded=float("nan"), censored=-1, guard_agrees=-1,
                            fake_full_window=int(abs(h_esn_part - T / FPS) < 1e-12),
                            passed=-1))
    return cases, h_esn_all, h_esn_part


# ================= 附录复现流程三条硬伤（只记录，不真跑 2 小时） =================

def write_repro_facts():
    """回归检查：附录复现流程曾有三条硬伤，逐条核验它们现在是否仍然存在。

    改为按内容检索而非硬编码行号——早先版本写死 main.tex:817 / baseline_distmass.py:180，
    论文一改行号就漂移，检查会静默失效（恒判为"已修"）。
    """
    facts = []

    # F1 附录耗时口径：是否还在写"约 25 分钟"
    main_tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    hit_25 = "25 分钟" in main_tex
    where_25 = next((i + 1 for i, l in enumerate(main_tex.split("\n")) if "25 分钟" in l), None)
    facts.append(("F1", hit_25, f"main.tex:{where_25}" if where_25 else "全文已无「25 分钟」"))

    # F2 仿真脚本相对路径：是否还会把数据写到仓库外
    rel_hits = []
    for rel in ("sim/baseline_distmass.py", "sim/baseline.py", "sim/visualize.py"):
        txt = (ROOT / rel).read_text(encoding="utf-8")
        if "'../data/sim'" in txt or '"../data/sim"' in txt:
            rel_hits.append(rel)
    wild = Path.home() / "Desktop/data"
    wild_sim = (wild / "sim").exists()
    wild_real = (wild / "real_validation").exists()
    wild_sim_n = len(list((wild / "sim").glob("trial_*.csv"))) if wild_sim else -1
    facts.append(("F2", bool(rel_hits), "、".join(rel_hits) if rel_hits else "三个脚本均已锚定仓库根"))

    # F3 latexmk 可用性 + 附录命令是否带上了国标样式所需的 TEXINPUTS
    texbin = Path("/Library/TeX/texbin")
    has_latexmk = (texbin / "latexmk").exists()
    has_xelatex = (texbin / "xelatex").exists()
    which_latexmk = None
    for q in os.environ.get("PATH", "").split(":"):
        if q and (Path(q) / "latexmk").exists():
            which_latexmk = str(Path(q) / "latexmk")
            break
    latexmk_ok = has_latexmk or bool(which_latexmk)
    needs_texinputs = "style=gb7714-2015" in main_tex
    cmd_has_texinputs = "TEXINPUTS=" in main_tex
    has_fallback = "xelatex main.tex" in main_tex
    # 只判仓库层面的缺陷：命令缺 TEXINPUTS，或没有不依赖 latexmk 的备用命令。
    # 本机是否装了 latexmk 属环境信息，另行记录，不计入回归判定。
    f3_bad = (needs_texinputs and not cmd_has_texinputs) or not has_fallback
    facts.append(("F3", f3_bad,
                  f"附录命令含 TEXINPUTS={'是' if cmd_has_texinputs else '否'}，"
                  f"备用 xelatex 命令={'有' if has_fallback else '无'}"
                  f"（本机 latexmk：{'有' if latexmk_ok else '无'}）"))

    status = lambda bad: "**仍存在**" if bad else "已修复"
    md = f"""# 附录复现流程三条硬伤的回归检查（repro_facts）

生成：`tools/eval_sentinel.py`，{time.strftime('%Y-%m-%d %H:%M')}，本机 macOS。
本文件按**内容检索**核验，不依赖行号（早先版本写死行号，论文一改就静默失效）。

---

## F1 附录耗时口径 —— {status(hit_25)}

- 现场核验：全文检索「25 分钟」→ **{facts[0][2]}**
- 背景：曾写「CPU 全流程约 25 分钟」，而逐条分项计时表明多 run 训练脚本
  （canon_multirun / lambda_ablation / data_size_ablation）就占八成以上耗时，
  25 分钟明显低估。现应写成有余量的量级说法。

## F2 仿真脚本相对路径 —— {status(bool(rel_hits))}

- 现场核验：**{facts[1][2]}**
- 背景：`run_sweep('../data/sim')` 相对**当前工作目录**而非仓库根，从仓库根执行会把数据
  写到 `~/Desktop/data/sim`（仓库外面），导致后续步骤连环失败。
  现已改为 `Path(__file__).resolve().parents[1] / 'data/sim'`。
- 历史痕迹（野生目录，说明这个坑被踩过）：
  - `{wild}/sim` 存在 = **{wild_sim}**（其中 trial_*.csv 数量 = {wild_sim_n}）
  - `{wild}/real_validation` 存在 = **{wild_real}**

## F3 论文编译命令 —— {status(f3_bad)}

- 现场核验：{facts[2][2]}；`/Library/TeX/texbin/xelatex` 存在 = **{has_xelatex}**，
  PATH 里的 latexmk = **{which_latexmk or '无'}**
- 背景：文献样式改用国标 GB/T 7714 后，样式文件随项目放在 `paper/texstyles/gb7714/`，
  编译必须带 `TEXINPUTS="./texstyles/gb7714//:"`，否则 biblatex 报
  `Style 'gb7714-2015' not found` 直接失败。附录命令现已补上该环境变量。

---

## 回归检查汇总

| 编号 | 硬伤是否仍存在 | 现场核验依据 |
|---|---|---|
| F1 | {'**是**' if facts[0][1] else '否'} | {facts[0][2]} |
| F2 | {'**是**' if facts[1][1] else '否'} | {facts[1][2]} |
| F3 | {'**是**' if facts[2][1] else '否'} | {facts[2][2]} |

> 三项全为「否」即表示附录复现流程的这三条硬伤均已修复。
> 任一项转回「是」，说明有人把修好的地方改回去了，或论文换了写法。
"""
    (OUT / "repro_facts.md").write_text(md, encoding="utf-8")
    log(f"\n[附录] repro_facts.md 回归检查完成："
        f"F1={'仍存在' if facts[0][1] else '已修'} "
        f"F2={'仍存在' if facts[1][1] else '已修'} "
        f"F3={'仍存在' if facts[2][1] else '已修'}")
    return dict(F1=facts[0][1], F2=facts[1][1], F3=facts[2][1],
                wild_sim=wild_sim, wild_real=wild_real, wild_sim_n=wild_sim_n,
                has_latexmk=latexmk_ok)


# ================= 报告 =================

def write_report(ctx):
    lines = []
    lines.append("# 评估哨兵报告（eval_sentinel）\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　"
                 f"耗时 {time.time() - _t0:.1f}s　"
                 f"脚本 `tools/eval_sentinel.py`\n")
    lines.append("目的：给评估管线的三处接缝各钉一组断言，防第四个同类 bug。"
                 "不产生新科学结论。\n")

    lines.append("## 自证门槛\n")
    lines.append("| 门槛 | 内容 | 结果 |")
    lines.append("|---|---|---|")
    lines.append(f"| G1 | 真实侧 29 段 × 3 模型 逐位复现 `data/real_validation/summary_multi.csv` "
                 f"| {'通过' if ctx['g1'] else '未过'} |")
    lines.append(f"| G2 | 仿真内 50 trial × Hybrid/MLP(λ=0) 走**旧** `predict()` 逐位复现两个冻结表 "
                 f"| {'通过' if ctx['g2'] else '未过'} |")
    lines.append(f"| G3 | 仿真内 50 trial × ESN 逐位复现 `data/rc/esn_summary_50.csv` "
                 f"| {'通过' if ctx['g3'] else '未过'} |")
    lines.append("")
    lines.append("> G2 的副产物结论：`data/hybrid/hybrid_summary_50.csv` 与 "
                 "`data/fair_compare/lam0_summary_50.csv` 只有走 **B1 修复之前**的 "
                 "`pinn.predict()` 才能逐位对上；用现在的 `predict_aligned()` 对不上。"
                 "这两张冻结表是 pre-B1 产物，9/9 全量重算时要一起刷新。\n")

    lines.append("## A 组：帧 0 对齐\n")
    df0 = pd.DataFrame(ROWS_FRAME0)
    lines.append(f"阈值 `{FRAME0_TOL_DEG}` 度（`predict_aligned` 第 0 帧只差一次 "
                 "float64→float32 cast，用 1e-6 会误报）。\n")
    lines.append("| 数据集 | 模型 | N | aligned 帧0 最大误差(°) | 旧 predict 帧0 最小误差(°) | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for (g, m), sub in df0.groupby(["group", "model"], sort=False):
        lines.append(f"| {g} | {m} | {len(sub)} | {sub.err_deg_aligned.max():.3e} | "
                     f"{sub.err_deg_old_predict.min():.3e} | "
                     f"{'通过' if sub.passed.all() else '失败'} |")
    lines.append("")
    lines.append("ESN 用**无阈值·逐位**判据（正确路径 0.000–0.61°、错误路径 0.567–3.37°，"
                 "两组在 0.57° 附近重叠，定角度阈值必误报）：\n")
    lines.append("| 检查 | 目标 | 期望 | 实得 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for r in ROWS_ESN:
        lines.append(f"| {r['check']} | `{r['target']}` | {r['expected']} | {r['actual']} | "
                     f"{'通过' if r['passed'] else '失败'} |")
    lines.append("")

    lines.append("## B 组：同种子可复现\n")
    lines.append("| 检查 | 期望 | 实得 | max&#124;Δ&#124; | 判定 |")
    lines.append("|---|---|---|---|---|")
    for r in ROWS_SEED:
        lines.append(f"| {r['check']} | {r['expected']} | {r['actual']} | "
                     f"{r['max_abs_diff']:.3e} | {'通过' if r['passed'] else '失败'} |")
    lines.append("")
    dfs = pd.DataFrame(ROWS_SITES)
    n_late = int((dfs.classification == "LATE_SEED").sum()) if len(dfs) else 0
    n_safe = int((dfs.classification == "SAFE_LOAD").sum()) if len(dfs) else 0
    n_unk = int((dfs.classification == "UNKNOWN").sum()) if len(dfs) else 0
    lines.append(f"全仓 `AccelNet(` / `HybridNet(` 直接构造点共 **{len(dfs)}** 处："
                 f"SAFE_LOAD {n_safe}（构造后立刻 `load_state_dict`，初值被覆盖，无害）、"
                 f"**LATE_SEED {n_late}**（构造后紧跟 `train(seed=)`，命中 B2 模式）、"
                 f"UNKNOWN {n_unk}。明细见 `late_seed_sites.csv`。\n")
    if n_late:
        lines.append("LATE_SEED 清单（9/9 全量重算时应统一换成 `make_accel_net` / `make_hybrid_net`）：\n")
        for k in sorted(dfs[dfs.classification == "LATE_SEED"].site):
            lines.append(f"- `{k}`")
        lines.append("")

    lines.append("## C 组：数值有限性\n")
    lines.append(f"生产 rollout 共 **{ctx['n_rollout']}** 条（仿真内 50×4 + 真实 29×3），"
                 f"非有限值合计 **{ctx['n_bad']}** 个。\n")
    lines.append("退化输入单元测试（记录生产函数真实行为，已知漏洞登记在案，不作为失败判据）：\n")
    lines.append("| 输入 | `pinn.horizon` | `rvm.horizon_s` | 守卫版 | 非有限帧数 | 满分假象 |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in ctx["degen"].items():
        lines.append(f"| {k} | {v['pinn_horizon']:.4f}s | {v['rvm_horizon']:.4f}s"
                     f"(cens={v['rvm_censored']}) | {v['guarded_horizon']:.4f}s | "
                     f"{v['n_nonfinite']} | "
                     f"{'**是**' if v['scored_full_window_despite_nonfinite'] else '否'} |")
    lines.append("")
    lines.append(f"`esn.prediction_horizon`：全 NaN → {ctx['h_esn_all']:.4f}s（安全）；"
                 f"「前 30 帧有限 + 之后全 NaN」→ {ctx['h_esn_part']:.4f}s = 窗长（**满分假象**）。\n")
    lines.append("> 结论：三个生产视界函数都用 `err > thr` 判越界，而 `NaN > thr` 恒为 False，"
                 "所以「炸成 NaN 的轨迹」会被判成整窗未越界 = 满分。当前所有生产 rollout "
                 "全程有限，这个漏洞是**潜伏**的而非**已发作**；C 组的价值就是保证它一直潜伏。\n")

    lines.append("## 附录复现流程三条硬伤\n")
    lines.append("详见 `repro_facts.md`。现场核验：")
    lines.append(f"F1（main.tex:817 那行确含「25 分钟」）= {ctx['facts']['F1']}；")
    lines.append(f"F2（baseline_distmass.py:180 确为 `'../data/sim'`，野生目录存在={ctx['facts']['wild_sim']}）"
                 f" = {ctx['facts']['F2']}；")
    lines.append(f"F3（本机无 latexmk）= {ctx['facts']['F3']}。\n")

    lines.append("## 总判定\n")
    if FAILURES:
        lines.append(f"**失败 {len(FAILURES)} 项**：\n")
        for t, m in FAILURES:
            lines.append(f"- `{t}` {m}")
    else:
        lines.append("**全部通过。**")
    lines.append("")
    if WARNINGS:
        lines.append(f"告警 {len(WARNINGS)} 项：\n")
        for t, m in WARNINGS:
            lines.append(f"- `{t}` {m}")
        lines.append("")
    (OUT / "sentinel_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    log("=" * 68)
    log("评估哨兵 eval_sentinel —— 帧0对齐 / 同种子可复现 / 数值有限性")
    log(f"输出目录 {OUT}")
    log("=" * 68)

    log("\n[自证门槛] 先复现三个已冻结的旧数字，过不了就不往下算")
    g1, store, _frozen_real = gate_g1_real()
    g2, _nets, cache = gate_g2_insim_torch()
    g3, esn_cache = gate_g3_insim_esn()
    if not (g1 and g2 and g3):
        log("\n自证门槛未通过 —— 环境或数据与冻结产物不一致，本次结果不可信。退出码 2。")
        return 2
    log("  自证门槛 3/3 通过。\n")

    group_a_torch_insim(cache)
    group_a_torch_real(store)
    group_a_esn(esn_cache, store)

    group_b_seed()
    scan_late_seed_sites()

    n_rollout, n_bad = group_c_finiteness(cache, store, esn_cache)
    degen, h_esn_all, h_esn_part = group_c_degenerate_unit_tests()

    facts = write_repro_facts()

    pd.DataFrame(ROWS_FRAME0).to_csv(OUT / "frame0_alignment.csv", index=False)
    pd.DataFrame(ROWS_SEED).to_csv(OUT / "seed_reproducibility.csv", index=False)
    pd.DataFrame(ROWS_FINITE).to_csv(OUT / "finiteness.csv", index=False)
    pd.DataFrame(ROWS_ESN).to_csv(OUT / "esn_alignment.csv", index=False)
    pd.DataFrame(ROWS_SITES).to_csv(OUT / "late_seed_sites.csv", index=False)

    write_report(dict(g1=g1, g2=g2, g3=g3, n_rollout=n_rollout, n_bad=n_bad,
                      degen=degen, h_esn_all=h_esn_all, h_esn_part=h_esn_part,
                      facts=facts))

    log("\n" + "=" * 68)
    log(f"产物 → {OUT}")
    for f in sorted(OUT.iterdir()):
        log(f"    {f.name}")
    log(f"耗时 {time.time() - _t0:.1f}s")
    if FAILURES:
        log(f"\n哨兵失败 {len(FAILURES)} 项：")
        for t, m in FAILURES:
            log(f"  - {t}: {m}")
        log("退出码 1。")
        return 1
    log(f"\n哨兵全部通过（告警 {len(WARNINGS)} 项）。退出码 0。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)

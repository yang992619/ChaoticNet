"""
sim/baseline_damped.py — 带阻尼的双摆仿真（v3 ground truth）

在 baseline.py 的无摩擦模型基础上加粘性阻尼项 −b·ω，让 ground truth 贴近实物。
阻尼系数 b 由单摆衰减时间常数 τ 反推：
  小振幅极限下 amplitude A(t) = A₀ exp(−b·t / 2I)
  → b ≈ 2I / τ
对 P01 摆臂等效 I = m₁·L₁² ≈ 1.56e-3 kg·m²
目标 τ ≈ 30s（BOM 验收阈值）→ b ≈ 1e-4 N·m·s/rad

输出：
  data/sim_damped/trial_000.csv ~ trial_049.csv
  data/sim_damped/summary.csv（含能量衰减率 + 半寿命）
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent

# ---- 真实复摆参数（分布质量，与 sim/baseline.py 一致）----
G = 9.81
L1 = 0.250                         # 关节间距 m
_ARM1, _ARM2 = 0.09156, 0.07324    # 上/下铝臂(均匀杆) kg
_CW = 0.0639                       # 配重 kg，在下摆末端 0.20m
M1 = _ARM1 + 0.0013 + 0.0023 + 0.0013
M2 = _ARM2 + 0.0019 + _CW
LC1 = (_ARM1 * 0.125 + 0.0013 * L1) / M1
LC2 = (_ARM2 * 0.10 + _CW * 0.20) / M2
I1_O = _ARM1 * L1**2 / 3 + 0.0013 * L1**2
I2_A = _ARM2 * 0.20**2 / 3 + _CW * 0.20**2 + 0.5 * _CW * 0.015**2

# 阻尼系数（由 fit_damping 全局池化拟合反推，2026-06-18）
#   粘性(轴承)  −b·ω        b = 1.66e-5 N·m·s/rad
#   二次空气阻力 −c·ω·|ω|    c = 8.72e-6 N·m·s²/rad²
# 单一粘性项无法同时拟合大角度(阻力主导,占总耗散~78%)与小幅尾(粘性主导)，
# 故升级为 粘性+二次阻力 两项模型；详见 fit_damping.py。
B_DEFAULT = 1.66e-5
C_DEFAULT = 8.72e-6


def deriv_damped(t, y, b=B_DEFAULT, c=C_DEFAULT):
    th1, w1, th2, w2 = y
    delta = th1 - th2
    s, cc = np.sin(delta), np.cos(delta)

    M11 = I1_O + M2 * L1 * L1
    M12 = M2 * L1 * LC2 * cc
    M22 = I2_A

    b1 = -M2 * L1 * LC2 * s * w2 * w2 - (M1 * LC1 + M2 * L1) * G * np.sin(th1)
    b2 =  M2 * L1 * LC2 * s * w1 * w1 - M2 * LC2 * G * np.sin(th2)

    # 阻尼力矩（两个关节各一份）：粘性 −b·ω + 二次空气阻力 −c·ω·|ω|
    b1 -= b * w1 + c * w1 * abs(w1)
    b2 -= b * w2 + c * w2 * abs(w2)

    det = M11 * M22 - M12 * M12
    wd1 = ( M22 * b1 - M12 * b2) / det
    wd2 = (-M12 * b1 + M11 * b2) / det
    return [w1, wd1, w2, wd2]


def energy(y):
    th1, w1, th2, w2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    M11 = I1_O + M2 * L1 * L1
    T = (0.5 * M11 * w1**2
         + M2 * L1 * LC2 * np.cos(th1 - th2) * w1 * w2
         + 0.5 * I2_A * w2**2)
    V = -(M1 * LC1 + M2 * L1) * G * np.cos(th1) - M2 * LC2 * G * np.cos(th2)
    return T + V


def simulate(th1_0, th2_0, w1_0=0.0, w2_0=0.0,
             t_end=30.0, fps=120, b=B_DEFAULT, c=C_DEFAULT):
    y0 = [th1_0, w1_0, th2_0, w2_0]
    t_eval = np.linspace(0, t_end, int(t_end * fps) + 1)
    sol = solve_ivp(
        deriv_damped, [0, t_end], y0,
        t_eval=t_eval, args=(b, c),
        method="RK45", rtol=1e-10, atol=1e-12, max_step=0.001,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    y_arr = sol.y.T
    return pd.DataFrame({
        "t":      sol.t,
        "th1":    sol.y[0],
        "w1":     sol.y[1],
        "th2":    sol.y[2],
        "w2":     sol.y[3],
        "energy": energy(y_arr),
    })


def run_sweep(out_dir, n_trials=50, seed=42, t_end=30.0, fps=120, b=B_DEFAULT):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary = []
    for i in range(n_trials):
        th1_0 = rng.uniform(-np.pi / 2, np.pi / 2)
        th2_0 = rng.uniform(-np.pi / 2, np.pi / 2)
        df = simulate(th1_0, th2_0, t_end=t_end, fps=fps, b=b)
        E0 = df["energy"].iloc[0]
        E_end = df["energy"].iloc[-1]
        # 计算半寿命（能量衰减到 E0 / 2 的时间）
        thresh = E0 / 2 if E0 > 0 else 2 * E0
        below = (df["energy"] < thresh).idxmax() if (df["energy"] < thresh).any() else len(df) - 1
        t_half = df["t"].iloc[below] if below else None

        fn = out / f"trial_{i:03d}.csv"
        df.to_csv(fn, index=False)
        summary.append({
            "trial": i,
            "th1_0_deg": np.degrees(th1_0),
            "th2_0_deg": np.degrees(th2_0),
            "E0": E0,
            "E_end": E_end,
            "decay_ratio": E_end / E0 if E0 != 0 else None,
            "file": fn.name,
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>3}/{n_trials}]  "
                  f"θ₁={np.degrees(th1_0):+6.1f}° θ₂={np.degrees(th2_0):+6.1f}°  "
                  f"E end / E0 = {E_end/E0:+.3f}")

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"\n完成 {n_trials} 组 → {out}/")
    print(f"  阻尼系数 b = {b:.2e} N·m·s/rad")


def estimate_tau_single_pendulum(b=B_DEFAULT, theta0_deg=30, t_end=60.0):
    """单摆衰减实验：把 θ₂ 锁定为 θ₁（视作刚体单摆），
    从 θ₀ 静止释放，拟合振幅指数衰减得到 τ。"""
    # 整摆对顶轴的总转动惯量（下摆用平行轴从关节移到顶轴）
    # 实物会用单摆 30° 释放测；这函数仅给数量级 sanity check
    I_eff = I1_O + I2_A + M2 * L1 ** 2
    tau_est = 2 * I_eff / b
    print(f"  小振幅理论：b={b:.2e}, I_eff≈{I_eff:.2e} → τ ≈ {tau_est:.1f}s")
    return tau_est


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="带阻尼双摆仿真")
    ap.add_argument("--b", type=float, default=B_DEFAULT,
                    help=f"阻尼系数 N·m·s/rad（默认 {B_DEFAULT:.0e}，实测后替换）")
    ap.add_argument("--n", type=int, default=50, help="trial 数")
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "data" / "sim_damped"),
                    help="输出目录")
    args = ap.parse_args()

    print("ChaoticNet 阻尼仿真 v1.1")
    print(f"  阻尼 b = {args.b:.2e} N·m·s/rad")
    estimate_tau_single_pendulum(b=args.b)

    print("\n【demo】θ₁ = 60°, θ₂ = 0°, 30s")
    df = simulate(np.deg2rad(60), 0.0, t_end=30.0, b=args.b)
    E0, Ee = df["energy"].iloc[0], df["energy"].iloc[-1]
    print(f"  E 起 = {E0:+.4e}, E 末 = {Ee:+.4e}, 比 = {Ee/E0:.3f}")

    print(f"\n【sweep】{args.n} 组随机初值")
    run_sweep(args.out, n_trials=args.n, b=args.b)

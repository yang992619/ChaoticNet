"""
sim/baseline.py — 双摆数值仿真基线 (ChaoticNet v1.0)

求解双摆拉格朗日方程，输出 θ₁(t), θ₂(t) 时间序列。
做能量守恒检查（无摩擦理想情形，总能量应当守恒到机器精度级别）。

用作 ChaoticNet 项目所有 AI 模型的 ground truth 对照。
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path

# ---- 真实复摆参数（分布质量模型，零件实测反推；见 diagrams/parts/BOM.md）----
# 实物是分布质量物理摆：铝臂质量沿杆分布 + 下摆末端配重，故用质心距 LC 与转动惯量 I 建模，
# 而非把质量当成杆末端一个质点。点质量是本式的特例（LC1=L1, LC2=L2, I1_O=M1·L1², I2_A=M2·L2²），
# 点质量 canonical 快照见 data/_pointmass_canonical_backup_20260619/。deriv/energy 通用式不变，仅参数变。
G  = 9.81                          # 重力加速度 m/s²
L1 = 0.250                         # 上摆长（顶轴→关节）m，几何长度
L2 = 0.200                         # 下摆长（关节→末端）m，几何长度（末端坐标用）
_ARM1, _ARM2 = 0.09156, 0.07324    # 上/下铝臂(均匀杆) kg：总臂重 164.8g 按长度 250:200 分
_CW = 0.0639                       # 下摆末端配重 kg，位于 0.20 m
M1 = _ARM1 + 0.0013 + 0.0023 + 0.0013    # 上摆总质量 ≈ 96.5g (臂+顶轴承+长销+关节轴承)
M2 = _ARM2 + 0.0019 + _CW                # 下摆总质量 ≈ 139g (臂+短销+配重)
LC1 = (_ARM1 * 0.125 + 0.0013 * L1) / M1          # 上摆质心距顶轴 ≈ 122mm
LC2 = (_ARM2 * 0.10 + _CW * 0.20) / M2            # 下摆质心距关节 ≈ 145mm
I1_O = _ARM1 * L1**2 / 3 + 0.0013 * L1**2         # 上摆对顶轴转动惯量 kg·m²
I2_A = _ARM2 * 0.20**2 / 3 + _CW * 0.20**2 + 0.5 * _CW * 0.015**2   # 下摆对关节 kg·m²


def deriv(t, y):
    """双摆复摆(分布质量)状态方程（无摩擦）。

    状态 y = [θ₁, ω₁, θ₂, ω₂]，返回 dy/dt = [ω₁, ω̇₁, ω₂, ω̇₂]。

    Lagrangian L = T − V，Euler-Lagrange 得 M(θ)·[ω̇₁, ω̇₂]ᵀ = b(θ, ω)。
    分布质量下质量矩阵用对顶轴/对关节的转动惯量，重力项用质心距。
    """
    th1, w1, th2, w2 = y
    delta = th1 - th2
    s, c = np.sin(delta), np.cos(delta)

    # 质量矩阵 M (2x2 对称)
    M11 = I1_O + M2 * L1 * L1
    M12 = M2 * L1 * LC2 * c
    M22 = I2_A

    # 力项 b（含科氏 + 重力，重力用质心距 LC）
    b1 = -M2 * L1 * LC2 * s * w2 * w2 - (M1 * LC1 + M2 * L1) * G * np.sin(th1)
    b2 =  M2 * L1 * LC2 * s * w1 * w1 - M2 * LC2 * G * np.sin(th2)

    # 解 [ω̇₁, ω̇₂] = M⁻¹·b
    det = M11 * M22 - M12 * M12
    wd1 = ( M22 * b1 - M12 * b2) / det
    wd2 = (-M12 * b1 + M11 * b2) / det

    return [w1, wd1, w2, wd2]


def energy(y):
    """总能量 E = T + V（分布质量），用于守恒检查。

    可接受单帧 y=(4,) 或时间序列 y=(T,4)。
    """
    th1, w1, th2, w2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    M11 = I1_O + M2 * L1 * L1
    T = (0.5 * M11 * w1**2
         + M2 * L1 * LC2 * np.cos(th1 - th2) * w1 * w2
         + 0.5 * I2_A * w2**2)
    V = -(M1 * LC1 + M2 * L1) * G * np.cos(th1) - M2 * LC2 * G * np.cos(th2)
    return T + V


def simulate(th1_0, th2_0, w1_0=0.0, w2_0=0.0, t_end=30.0, fps=120,
             rtol=1e-10, atol=1e-12):
    """跑一次仿真。

    Args:
        th1_0, th2_0: 初值角度 (rad)。0 = 竖直向下。
        w1_0, w2_0:   初值角速度 (rad/s)。
        t_end:        仿真时长 (s)。
        fps:          输出帧率（与相机一致，便于和实测对比）。
        rtol, atol:   求解器容差。混沌系统对精度敏感，默认很严。

    Returns:
        DataFrame，列 = [t, th1, w1, th2, w2, energy]
    """
    y0 = [th1_0, w1_0, th2_0, w2_0]
    t_eval = np.linspace(0, t_end, int(t_end * fps) + 1)

    sol = solve_ivp(
        deriv, [0, t_end], y0,
        t_eval=t_eval,
        method='RK45',
        rtol=rtol, atol=atol,
        max_step=0.001,
    )
    if not sol.success:
        raise RuntimeError(f"仿真失败：{sol.message}")

    y_arr = sol.y.T
    return pd.DataFrame({
        't':      sol.t,
        'th1':    sol.y[0],
        'w1':     sol.y[1],
        'th2':    sol.y[2],
        'w2':     sol.y[3],
        'energy': energy(y_arr),
    })


def run_sweep(out_dir='data/sim', n_trials=50, seed=42, t_end=30.0, fps=120):
    """扫描 n 组随机初值，存 CSV。

    初值采样：θ₁ θ₂ ∈ [-90°, 90°] 均匀分布（覆盖混沌窗口）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    summary = []
    for i in range(n_trials):
        th1_0 = rng.uniform(-np.pi/2, np.pi/2)
        th2_0 = rng.uniform(-np.pi/2, np.pi/2)

        df = simulate(th1_0, th2_0, t_end=t_end, fps=fps)

        E0 = df['energy'].iloc[0]
        drift = (df['energy'] - E0).abs().max() / abs(E0)

        fn = out / f'trial_{i:03d}.csv'
        df.to_csv(fn, index=False)

        summary.append({
            'trial':         i,
            'th1_0_deg':     np.degrees(th1_0),
            'th2_0_deg':     np.degrees(th2_0),
            'energy_drift':  drift,
            'file':          fn.name,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>3}/{n_trials}] "
                  f"θ₁={np.degrees(th1_0):+6.1f}° θ₂={np.degrees(th2_0):+6.1f}° "
                  f"漂移 {drift:.2e}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / 'summary.csv', index=False)

    max_drift = summary_df['energy_drift'].max()
    print(f"\n完成 {n_trials} 组 → {out}/")
    print(f"  最大能量漂移：{max_drift:.2e}")
    print(f"  平均能量漂移：{summary_df['energy_drift'].mean():.2e}")
    if max_drift > 1e-6:
        print("  ⚠️ 能量漂移过大，建议收紧 rtol/atol 或缩短 max_step")
    else:
        print("  ✅ 能量守恒检查通过（漂移 < 1e-6，求解精度足够）")


if __name__ == '__main__':
    print("ChaoticNet 双摆仿真基线 v1.0")
    print(f"  L₁(关节间距) = {L1*1000:.0f} mm")
    print(f"  质心 LC₁ = {LC1*1000:.0f} mm,  LC₂ = {LC2*1000:.0f} mm")
    print(f"  m₁ = {M1*1000:.0f} g,   m₂ = {M2*1000:.0f} g")
    print(f"  转动惯量 I₁(顶轴) = {I1_O:.3e},  I₂(关节) = {I2_A:.3e}")
    print(f"  g  = {G} m/s²\n")

    # demo: 单次仿真 + 能量守恒打印
    print("【demo】θ₁ = 90°, θ₂ = 0°, 跑 10s @ 120fps")
    df = simulate(np.pi/2, 0.0, t_end=10.0, fps=120)
    drift = (df['energy'] - df['energy'].iloc[0]).abs().max() / abs(df['energy'].iloc[0])
    print(f"  起态 E = {df['energy'].iloc[0]:.6e} J")
    print(f"  末态 E = {df['energy'].iloc[-1]:.6e} J")
    print(f"  能量相对漂移 = {drift:.2e}")
    print(f"  末态 θ₁ = {np.degrees(df['th1'].iloc[-1]):+7.2f}°, "
          f"θ₂ = {np.degrees(df['th2'].iloc[-1]):+7.2f}°\n")

    # sweep: 50 组随机初值
    print("【sweep】50 组随机初值，每组 30s @ 120fps")
    run_sweep(Path(__file__).resolve().parents[1] / 'data/sim', n_trials=50)

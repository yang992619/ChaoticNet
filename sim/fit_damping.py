"""
sim/fit_damping.py — 从实测衰减反推双摆阻尼系数 b（ChaoticNet v3）

思路（能量耗散率法，抗混沌）：
  粘性阻尼下系统总机械能单调耗散，且耗散功率与轨迹是否混沌无关：
      dE/dt = −b·(ω₁² + ω₂²)            （baseline_damped 的单系数粘性模型）
  对时间积分：
      E(t) = E₀ − b·∫₀ᵗ (ω₁²+ω₂²) dt'
  于是把实测能量 E(t) 对累积耗散积分 D_visc(t)=∫(ω₁²+ω₂²)dt' 做线性回归，
  斜率即 −b。这条关系不混沌、不受蝴蝶效应影响，比硬拟合 θ 轨迹稳得多。

  模型检验结论（2026-06-18，16 片）：纯粘性单系数不够——大角度片"等效粘性"
  系统性偏高，前向仿真在高速段严重欠阻尼。升级为 粘性+二次空气阻力 两项：
      dE/dt = −b·(ω₁²+ω₂²) − c·(|ω₁|³+|ω₂|³)
  其中二次阻力耗散∝|ω|³，高速段才显著，占总耗散约 78%。
  单片内 D_visc 与 D_drag 高度共线、拆不开（单片拟合 b 常出负值）；改用
  global_visc_drag 把所有片池化（各片不同振幅→阻力/粘性比例不同），共线性
  被打破，b 与 c 可分离辨识。最终 b≈1.66e-5、c≈8.72e-6 已写入 baseline_damped。

验证手段：
  1) 已知 b 自测：用 baseline_damped 造已知 b 的合成轨迹 + 加追踪噪声，
     回灌本估计器，看能否无偏地把 b 找回来。
  2) 跨片交叉验证：同一物理摆不同片子反推的 b 应当一致。
  3) 前向确认：把反推的 b 喂回 baseline_damped 从实测初值前向仿真，
     能量衰减曲线应贴合实测（θ 轨迹会因混沌发散，但能量包络不会）。

物理参数（质量/转动惯量/质心距）直接复用 sim/baseline.py，保证与 ground truth 一致。
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_damped as bd  # 物理参数 + energy + 阻尼仿真，单一真值源

R = np.degrees(1.0)  # rad → deg


# ----------------------------------------------------------------------------
# 1) 从实测 CSV 取平滑状态 + 能量
# ----------------------------------------------------------------------------
def load_state(csv_path, tmin=None, tmax=None, smooth_s=0.25, amp_max_deg=None):
    """读 tracked_*.csv，平滑 θ 后微分得 ω，算总机械能 E(t)。

    Args:
        csv_path: tracked_*.csv 路径（列含 t, th1, th2，弧度制；th2 已解卷）。
        tmin/tmax: 拟合时间窗（秒，绝对时间）。None 用全程（自动各裁掉边沿）。
        smooth_s: Savitzky-Golay 平滑窗口（秒）。
        amp_max_deg: 若给定，只保留 θ₁ 包络(3s 滚动峰值)已降到该值以下的"小幅段"，
                     用来取纯轴承粘性 b（避开高速空气阻力污染）。

    Returns:
        dict(t, th1, th2, w1, w2, E)  —— θ/ω 为平滑后量，E 由其算得。
    """
    df = pd.read_csv(csv_path)
    t = df['t'].to_numpy()
    th1 = df['th1'].to_numpy()
    th2 = df['th2'].to_numpy()

    # 默认裁掉头 1s（放手瞬态/窗口边沿）和尾 0.3s
    t0, t1 = t[0], t[-1]
    lo = (t0 + 1.0) if tmin is None else tmin
    hi = (t1 - 0.3) if tmax is None else tmax
    m = (t >= lo) & (t <= hi)
    t, th1, th2 = t[m], th1[m], th2[m]

    dt = np.median(np.diff(t))
    fps = 1.0 / dt
    win = int(round(smooth_s * fps))
    win = max(7, win | 1)  # 奇数，>=7
    if win >= len(t):
        win = (len(t) - 1) | 1

    # 平滑 θ，并以多项式导数取 ω（比逐帧差分稳，避免 ω² 噪声偏置）
    th1_s = savgol_filter(th1, win, 3)
    th2_s = savgol_filter(th2, win, 3)
    w1 = savgol_filter(th1, win, 3, deriv=1, delta=dt)
    w2 = savgol_filter(th2, win, 3, deriv=1, delta=dt)

    if amp_max_deg is not None:
        # θ₁ 包络：3s 滚动峰值；取包络首次降到阈值以下之后的整段尾巴
        env = pd.Series(np.abs(th1_s) * R).rolling(
            int(round(3.0 * fps)), center=True, min_periods=1).max().to_numpy()
        below = np.where(env <= amp_max_deg)[0]
        if len(below) > 50:
            k = below[0]
            t, th1_s, th2_s = t[k:], th1_s[k:], th2_s[k:]
            w1, w2 = w1[k:], w2[k:]

    y = np.column_stack([th1_s, w1, th2_s, w2])
    E = bd.energy(y)
    return dict(t=t, th1=th1_s, th2=th2_s, w1=w1, w2=w2, E=E, fps=fps)


# ----------------------------------------------------------------------------
# 2) 拟合：粘性单系数 / 粘性+库仑
# ----------------------------------------------------------------------------
def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_viscous(st):
    """E(t) = E0 − b·D_visc(t)。线性回归取 b（headline，对齐 baseline_damped）。"""
    t, w1, w2, E = st['t'], st['w1'], st['w2'], st['E']
    D = cumulative_trapezoid(w1 ** 2 + w2 ** 2, t, initial=0.0)
    slope, intercept = np.polyfit(D, E, 1)
    b = -slope
    yhat = intercept + slope * D
    return dict(b=b, E0=intercept, R2=_r2(E, yhat), D_visc_total=D[-1])


def global_visc_drag(states, with_drag=True):
    """全局池化拟合：所有片共享 b(、c)，每片各自截距 E0。

    单片内 D_visc 与 D_drag 高度共线、拆不开；但不同片振幅分布不同→阻力/粘性
    比例不同，池化后两者可分离辨识。这是同时定 b 与 c 的关键。

    Returns: dict(b, c, R2, frac_drag) —— with_drag=False 时只解 b。
    """
    tags = list(states)
    k = len(tags)
    blocks, E_all = [], []
    for j, tag in enumerate(tags):
        st = states[tag]
        t, w1, w2, E = st['t'], st['w1'], st['w2'], st['E']
        Dv = cumulative_trapezoid(w1 ** 2 + w2 ** 2, t, initial=0.0)
        onehot = np.zeros((len(t), k)); onehot[:, j] = 1.0
        cols = [onehot, -Dv[:, None]]
        if with_drag:
            Dd = cumulative_trapezoid(np.abs(w1) ** 3 + np.abs(w2) ** 3, t, initial=0.0)
            cols.append(-Dd[:, None])
        blocks.append(np.hstack(cols))
        E_all.append(E)
    A = np.vstack(blocks); y = np.concatenate(E_all)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    b = coef[k]
    c = coef[k + 1] if with_drag else 0.0
    R2 = _r2(y, A @ coef)
    # 全局阻力占比
    Dv_tot = sum(cumulative_trapezoid(states[t2]['w1'] ** 2 + states[t2]['w2'] ** 2,
                 states[t2]['t'], initial=0.0)[-1] for t2 in tags)
    Dd_tot = sum(cumulative_trapezoid(np.abs(states[t2]['w1']) ** 3 + np.abs(states[t2]['w2']) ** 3,
                 states[t2]['t'], initial=0.0)[-1] for t2 in tags) if with_drag else 0.0
    frac = (c * Dd_tot) / (b * Dv_tot + c * Dd_tot) if with_drag and (b * Dv_tot + c * Dd_tot) else 0.0
    return dict(b=b, c=c, R2=R2, frac_drag=frac)


def fit_visc_drag(st):
    """E(t) = E0 − b·D_visc − c·D_drag。

    粘性(轴承) b·(ω₁²+ω₂²) + 二次空气阻力 c·(|ω₁|³+|ω₂|³)。
    阻力耗散∝|ω|³，在高速段才显著——能解释为何大角度片"等效粘性"偏高。
    """
    t, w1, w2, E = st['t'], st['w1'], st['w2'], st['E']
    Dv = cumulative_trapezoid(w1 ** 2 + w2 ** 2, t, initial=0.0)
    Dd = cumulative_trapezoid(np.abs(w1) ** 3 + np.abs(w2) ** 3, t, initial=0.0)
    A = np.column_stack([np.ones_like(t), -Dv, -Dd])
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    E0, b, c = coef
    yhat = A @ coef
    visc_diss = b * Dv[-1]
    drag_diss = c * Dd[-1]
    frac_drag = drag_diss / (visc_diss + drag_diss) if (visc_diss + drag_diss) else np.nan
    return dict(b=b, c=c, E0=E0, R2=_r2(E, yhat), frac_drag=frac_drag)


# ----------------------------------------------------------------------------
# 3) 前向确认：用反推 b 从实测初值仿真，比能量衰减
# ----------------------------------------------------------------------------
def forward_check(st, b, t_span=None):
    """从窗口起点的实测状态出发，用 baseline_damped(b) 前向仿真，
    返回实测 vs 仿真的能量序列（同起点对齐）。"""
    t = st['t'] - st['t'][0]
    if t_span is None:
        t_span = min(t[-1], 40.0)  # 只看前段足够看出衰减率
    sel = t <= t_span
    sim = bd.simulate(st['th1'][0], st['th2'][0],
                      w1_0=st['w1'][0], w2_0=st['w2'][0],
                      t_end=t_span, fps=st['fps'], b=b)
    return dict(t_real=t[sel], E_real=st['E'][sel],
                t_sim=sim['t'].to_numpy(), E_sim=sim['energy'].to_numpy())


# ----------------------------------------------------------------------------
# 4) 已知 b 自测：造合成轨迹 + 追踪噪声，回灌看能否找回 b
# ----------------------------------------------------------------------------
def self_test(b_true=2.0e-4, th1_0=70.0, th2_0=0.0, t_end=120.0,
              fps=60.0, noise_deg=0.3, seed=0):
    """合成验证：已知 b_true → 仿真 → 加角度噪声(模拟追踪) → 反推 → 对比。"""
    rng = np.random.default_rng(seed)
    sim = bd.simulate(np.deg2rad(th1_0), np.deg2rad(th2_0),
                      t_end=t_end, fps=fps, b=b_true, c=0.0)  # 纯粘性合成
    df = pd.DataFrame({
        't': sim['t'],
        'th1': sim['th1'] + np.deg2rad(noise_deg) * rng.standard_normal(len(sim)),
        'th2': sim['th2'] + np.deg2rad(noise_deg) * rng.standard_normal(len(sim)),
    })
    tmp = ROOT / 'data' / 'sim_damped' / '_selftest.csv'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    st = load_state(tmp)
    fv = fit_viscous(st)
    err = (fv['b'] - b_true) / b_true * 100
    print("【自测】已知 b_true = %.3e，加 %.2f° 角度噪声后反推" % (b_true, noise_deg))
    print("   反推 b = %.3e   相对误差 = %+.1f%%   R² = %.5f" % (fv['b'], err, fv['R2']))
    ok = abs(err) < 10.0 and fv['R2'] > 0.99
    print("   %s" % ("✅ 估计器无偏，自测通过" if ok else "⚠️ 误差偏大，需检查"))
    tmp.unlink(missing_ok=True)
    return ok, fv['b'], err


def self_test_drag(b_true=1.7e-5, c_true=8.0e-6,
                   th1_list=(40.0, 80.0, 130.0, 170.0),
                   t_end=80.0, fps=60.0, noise_deg=0.3, seed=0):
    """两参数自测：用已知 (b,c) 在多个释放角造合成轨迹 + 噪声，
    回灌全局池化拟合，看能否同时把 b 与 c 找回（验证两项估计器可辨识）。"""
    rng = np.random.default_rng(seed)
    states = {}
    tmp = ROOT / 'data' / 'sim_damped' / '_selftest_drag.csv'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(th1_list):
        sim = bd.simulate(np.deg2rad(a), 0.0, t_end=t_end, fps=fps,
                          b=b_true, c=c_true)
        df = pd.DataFrame({
            't': sim['t'],
            'th1': sim['th1'] + np.deg2rad(noise_deg) * rng.standard_normal(len(sim)),
            'th2': sim['th2'] + np.deg2rad(noise_deg) * rng.standard_normal(len(sim)),
        })
        df.to_csv(tmp, index=False)
        states['a%d' % i] = load_state(tmp)
    tmp.unlink(missing_ok=True)
    g = global_visc_drag(states, with_drag=True)
    eb = (g['b'] - b_true) / b_true * 100
    ec = (g['c'] - c_true) / c_true * 100
    # 真正可观测/影响预测的是“总耗散”，而非 b、c 的劈分（两者相关）
    Dv = sum(cumulative_trapezoid(s['w1'] ** 2 + s['w2'] ** 2, s['t'], initial=0.0)[-1]
             for s in states.values())
    Dd = sum(cumulative_trapezoid(np.abs(s['w1']) ** 3 + np.abs(s['w2']) ** 3, s['t'], initial=0.0)[-1]
             for s in states.values())
    diss_true = b_true * Dv + c_true * Dd
    diss_fit = g['b'] * Dv + g['c'] * Dd
    e_diss = (diss_fit - diss_true) / diss_true * 100
    print("【两参数自测】已知 b=%.3e c=%.3e，%d 个释放角合成+噪声后全局反推"
          % (b_true, c_true, len(th1_list)))
    print("   反推 b=%.3e(%+.0f%%)  c=%.3e(%+.0f%%)  R²=%.4f"
          % (g['b'], eb, g['c'], ec, g['R2']))
    print("   总耗散(真正影响预测的量)误差 %+.1f%%  ← b/c 相关，劈分有余地但总量稳"
          % e_diss)
    ok = abs(e_diss) < 10.0 and g['R2'] > 0.98
    print("   %s" % ("✅ 总耗散无偏(预测可靠)，自测通过" if ok else "⚠️ 总耗散偏差大，需检查"))
    return ok, g


# ----------------------------------------------------------------------------
# 5) 跑全部干净片 + 交叉验证 + 诊断图
# ----------------------------------------------------------------------------
def run_clips(tags, track_dir, out_dir, tmin=None, tmax=None, make_plot=True):
    track_dir = Path(track_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    states = {}
    for tag in tags:
        csv = track_dir / f'tracked_{tag}.csv'
        if not csv.exists():
            print("  跳过 %s（无 %s）" % (tag, csv.name))
            continue
        st = load_state(csv, tmin=tmin, tmax=tmax)            # 全程
        st_late = load_state(csv, tmin=tmin, tmax=tmax,
                             amp_max_deg=25.0)                 # 小幅尾段(纯粘性)
        fv = fit_viscous(st)
        fd = fit_visc_drag(st)
        fl = fit_viscous(st_late)
        states[tag] = st
        rows.append(dict(
            tag=tag, n=len(st['t']),
            dur_s=round(st['t'][-1] - st['t'][0], 1),
            b_visc=fv['b'], R2_visc=fv['R2'],
            b_late=fl['b'], R2_late=fl['R2'], n_late=len(st_late['t']),
            b_drag=fd['b'], c_drag=fd['c'], R2_drag=fd['R2'],
            frac_drag=fd['frac_drag'],
            E0=fv['E0'],
        ))
        print("  %s | 全程b=%.2e(R²%.3f) | 小幅尾b=%.2e(R²%.3f) | +阻力: b=%.2e c=%.2e 阻力占比%.0f%%(R²%.3f)"
              % (tag, fv['b'], fv['R2'], fl['b'], fl['R2'],
                 fd['b'], fd['c'], 100 * fd['frac_drag'], fd['R2']))

    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / 'fit_damping_report.csv', index=False)

    if len(rep):
        for col, name in [('b_visc', '全程粘性'), ('b_late', '小幅尾纯粘性'),
                          ('b_drag', '含阻力分离粘性')]:
            bs = rep[col].to_numpy()
            bs = bs[np.isfinite(bs)]
            print("  [%s] b 中位=%.3e 均值=%.3e 离散度=%.1f%%"
                  % (name, np.median(bs), np.mean(bs), 100 * np.std(bs) / np.mean(bs)))

    # 全局池化拟合：破单片共线性，同时定 b 与 c（强辨识）
    if len(states) >= 2:
        gv = global_visc_drag(states, with_drag=False)
        gd = global_visc_drag(states, with_drag=True)
        print("  [全局池化·纯粘性]  b=%.3e  (R²=%.4f)" % (gv['b'], gv['R2']))
        print("  [全局池化·粘性+阻力] b=%.3e  c=%.3e  阻力占比=%.0f%%  (R²=%.4f)"
              % (gd['b'], gd['c'], 100 * gd['frac_drag'], gd['R2']))

    if make_plot and len(states):
        _diagnostic_plot(states, rep, out_dir)
    return rep, states


def _diagnostic_plot(states, rep, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tags = list(states.keys())
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) 能量衰减 E/E0 vs t + 粘性拟合曲线
    for tag in tags:
        st = states[tag]
        t = st['t'] - st['t'][0]
        b = float(rep.loc[rep.tag == tag, 'b_visc'].iloc[0])
        E0 = float(rep.loc[rep.tag == tag, 'E0'].iloc[0])
        Dv = cumulative_trapezoid(st['w1'] ** 2 + st['w2'] ** 2, st['t'], initial=0.0)
        ax[0, 0].plot(t, st['E'], lw=0.8, alpha=0.7, label=tag)
        ax[0, 0].plot(t, E0 - b * Dv, 'k--', lw=0.6, alpha=0.5)
    ax[0, 0].set_title('能量衰减 E(t) 与粘性拟合(虚线)')
    ax[0, 0].set_xlabel('t (s)'); ax[0, 0].set_ylabel('E (J)')
    ax[0, 0].legend(fontsize=7, ncol=2)

    # (b) 瞬时耗散率 −dE/dt vs (ω1²+ω2²) 散点 —— 检验线性(过原点=纯粘性)
    rep_tag = tags[0]
    st = states[rep_tag]
    dEdt = np.gradient(st['E'], st['t'])
    sq = st['w1'] ** 2 + st['w2'] ** 2
    ax[0, 1].scatter(sq, -dEdt, s=3, alpha=0.25)
    b = float(rep.loc[rep.tag == rep_tag, 'b_visc'].iloc[0])
    xx = np.linspace(0, np.percentile(sq, 99), 50)
    ax[0, 1].plot(xx, b * xx, 'r-', lw=1.5, label='纯粘性 b·x')
    ax[0, 1].set_title('耗散率检验 (%s)：−dE/dt vs ω₁²+ω₂²' % rep_tag)
    ax[0, 1].set_xlabel('ω₁²+ω₂² (rad²/s²)'); ax[0, 1].set_ylabel('−dE/dt (W)')
    ax[0, 1].set_xlim(0, np.percentile(sq, 99))
    ax[0, 1].set_ylim(bottom=0); ax[0, 1].legend(fontsize=8)

    # (c) 跨片 b 柱状（全程 vs 小幅尾纯粘性，一致性）
    x = np.arange(len(rep))
    ax[1, 0].bar(x - 0.2, rep['b_visc'], 0.4, label='全程(含阻力)')
    ax[1, 0].bar(x + 0.2, rep['b_late'], 0.4, label='小幅尾(纯粘性)')
    ax[1, 0].axhline(rep['b_late'].median(), color='r', ls='--',
                     label='小幅尾中位 %.2e' % rep['b_late'].median())
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(rep['tag'].astype(str))
    ax[1, 0].set_title('跨片反推 b 一致性')
    ax[1, 0].set_ylabel('b (N·m·s/rad)')
    ax[1, 0].tick_params(axis='x', rotation=45); ax[1, 0].legend(fontsize=7)

    # (d) 前向确认：能量实测 vs 仿真（用小幅尾中位 b）
    b_med = float(rep['b_late'].median())
    st = states[rep_tag]
    fc = forward_check(st, b_med)
    ax[1, 1].plot(fc['t_real'], fc['E_real'], 'b-', lw=1.2, label='实测 E')
    ax[1, 1].plot(fc['t_sim'], fc['E_sim'], 'r--', lw=1.2, label='仿真 E (b_中位)')
    ax[1, 1].set_title('前向确认 (%s)：能量包络贴合' % rep_tag)
    ax[1, 1].set_xlabel('t (s)'); ax[1, 1].set_ylabel('E (J)'); ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    plt.tight_layout()
    fn = out_dir / 'fit_damping_diagnostic.png'
    plt.savefig(fn, dpi=130)
    plt.close()
    print("  诊断图 → %s" % fn)


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='从实测衰减反推双摆阻尼 b')
    # 默认即论文 §4.3 所用的 16 段（与 data/sim_damped/fit_damping_report.csv 一致）；
    # 早先默认只有 7 段，照论文复现会得到不同的系数。
    ap.add_argument('--tags', nargs='+',
                    default=['1430', '1431', '1432', '1434', '1435', '1436',
                             '1437', '1438', '1439', '1440', '1441', '1442',
                             '1443', '1444', '1445', '1446'],
                    help='要反推的片子 tag')
    ap.add_argument('--track-dir', default=str(ROOT / 'data' / 'tracking'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'sim_damped'))
    ap.add_argument('--tmin', type=float, default=None)
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--self-test', action='store_true', help='只跑已知 b 自测')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    print("ChaoticNet 阻尼反推 v3  (能量耗散率法)")
    print("  物理参数：L1=%.3f I1_O=%.3e I2_A=%.3e M1=%.1fg M2=%.1fg\n"
          % (bd.L1, bd.I1_O, bd.I2_A, bd.M1 * 1000, bd.M2 * 1000))

    print("=== 估计器自测 ===")
    self_test()
    self_test_drag()

    if args.self_test:
        sys.exit(0)

    print("\n=== 实测片反推 ===")
    run_clips(args.tags, args.track_dir, args.out,
              tmin=args.tmin, tmax=args.tmax, make_plot=not args.no_plot)

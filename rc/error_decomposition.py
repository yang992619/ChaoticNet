"""
rc/error_decomposition.py — 真实短预测视界的成因分解（三天花板框架，经对抗审查修正）

回答评委必问：sim→real 预测视界短，是混沌本身（谁都救不了），
还是无阻尼模型对真实(有阻尼)系统的失配（我们能改）？

同口径视界（复用 real_validation_multi 的窗口/对齐/视界定义；阈值 10°）：
  ── 模型实测 ──
  h_real    : 无阻尼 PINN/Hybrid 从实测初值 s0 前推 vs 真实测量轨迹      = 实际迁移
  h_model   : 同模型 vs 无阻尼 baseline 从 s0 积分的理想 GT             = 模型分布内可预测性（逼近误差+分布内混沌）
  h_damp    : 带阻尼 PINN/Hybrid（训于 sim_damped）从 s0 vs 真实测量    = 换带阻尼模型的真实增益
  ── 模型无关理论天花板（解析积分器，隔离单一因素，经混沌放大）──
  C_dampmis : 无阻尼 baseline 从 s0 vs 带阻尼 baseline 从同一 s0        = 纯阻尼失配天花板
  C_chaos   : 无阻尼 baseline 从 s0 vs 从 s0+δ（δ=每片实测初值不确定度）= 纯初值不确定天花板（Lyapunov）

解读（修正后，对抗审查结论）：
  · 能量耗散慢(~百秒)≠窗内不发散：微小阻尼失配(相对扰动~0.4%)经混沌指数放大，C_dampmis 仍可在 ~1.5s 触及 10°。
    （旧论证"4s 能量损失几% → 不可能发散"是逻辑跳跃，已删除。）
  · 初值不确定度须用对趋势不敏感的噪声估计器（三阶差分）：savgol 残差含被平滑掉的真实曲率(lag-1 自相关~0.5)，
    会系统性高估 σθ→δω、人为压低混沌天花板。诚实估计 σθ≈0.4°，δω≈0.09 rad/s。
  · h_real(~0.45s) 远短于两条理论天花板(C_dampmis~1.5s、C_chaos~2s) → 真实视界受 模型逼近误差 + 初值估计误差
    + 测量噪声 的"组合"压制，早在阻尼失配/混沌天花板生效之前就发散。故 4s 短窗内"修阻尼"无收益
    （带阻尼模型实测不救、Hybrid 几乎不变）；优先级是初值状态估计精度与模型逼近，
    阻尼建模要等 h_real 逼近 ~1.5s 量级后才成为约束。

baseline.deriv == baseline_damped.deriv_damped(b=0,c=0)；PINN/Hybrid 训练于 data/sim(baseline.py)，
带阻尼版训练于 data/sim_damped(baseline_damped.py 默认 b/c)，故 GT 与各自训练分布同源。
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'rc'))
sys.path.insert(0, str(_ROOT / 'sim'))
import real_validation_multi as rvm   # noqa: E402  复用 load_clip/run_torch/angle_err_deg/horizon_s/load_pinn/load_hybrid
import baseline as bl                 # noqa: E402  无阻尼 GT（PINN/Hybrid 训练分布源）
import baseline_damped as bld         # noqa: E402  带阻尼 GT（拟合 b/c）

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
# 字号与论文正文/其余插图协调（原图 14in 无裁边、硬编码 7–8pt，放进正文显著偏小）
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
})

OUT = _ROOT / 'data/real_validation'
FIG = OUT / 'figures'

DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]   # 全局 δω 扫描 (rad/s)，覆盖诚实估计区间
DTH_FALLBACK = 0.007                       # 缺省 θ 扰动 (rad≈0.4°)，正常用每片实测 σθ


# ---------- 解析积分器天花板 ----------
def baseline_gt(s0, n, fps):
    """从 s0 用无阻尼 baseline 积分，返回前 n 帧 (th1, th2)。第 0 帧 = s0。"""
    df = bl.simulate(float(s0[0]), float(s0[2]), w1_0=float(s0[1]), w2_0=float(s0[3]),
                     t_end=(n + 1) / fps, fps=fps)
    return df['th1'].values[:n], df['th2'].values[:n]


def baseline_damped_gt(s0, n, fps):
    """从 s0 用带阻尼 baseline(拟合 b/c) 积分，返回前 n 帧 (th1, th2)。"""
    df = bld.simulate(float(s0[0]), float(s0[2]), w1_0=float(s0[1]), w2_0=float(s0[3]),
                      t_end=(n + 1) / fps, fps=fps)
    return df['th1'].values[:n], df['th2'].values[:n]


def load_damped():
    """加载在 sim_damped(粘性+二次阻力) 训练的 PINN/Hybrid，结构同无阻尼版。"""
    P = rvm.pinn_mod.AccelNet(hidden=128, n_layers=4)
    P.load_state_dict(torch.load(_ROOT / 'data/hybrid_damped/pinn_damped_model.pt',
                                 map_location=rvm.pinn_mod.DEVICE))
    P.to(rvm.pinn_mod.DEVICE).eval()
    H = rvm.hybrid_mod.HybridNet(hidden=64, n_layers=3,
                                 residual_scale=rvm.hybrid_mod.RESIDUAL_SCALE)
    H.load_state_dict(torch.load(_ROOT / 'data/hybrid_damped/hybrid_damped_model.pt',
                                 map_location=rvm.hybrid_mod.DEVICE))
    H.to(rvm.hybrid_mod.DEVICE).eval()
    return P, H


# ---------- 初值不确定度估计（对趋势不敏感） ----------
def sigma_theta_thirddiff(x):
    """三阶差分噪声估计：系数 [1,-3,3,-1]，湮灭≤二次局部趋势，残留主要为测量噪声。
    白噪声下 var(d3)=20σ²，故 σ=std(d3)/√20。x 为弧度角序列。返回 σθ (rad)。"""
    d3 = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return float(np.std(d3) / np.sqrt(20.0))


def estimate_delta(tags, lead_s, n_mc=150):
    """逐片估初值 (σθ, δω)：三阶差分定噪声底 σθ，再蒙特卡洛走 savgol+gradient 管线得 ω[i0] 抖动 δω。"""
    rng = np.random.default_rng(0)
    per_clip, sth_deg, sw = {}, [], []
    for tg in tags:
        f = rvm.TRACK / f'tracked_{tg}.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        t = df['t'].values - df['t'].values[0]
        fps = 1.0 / np.median(np.diff(t))
        dt = 1.0 / fps
        N = len(t)
        win = rvm.SAVGOL_WIN if N > rvm.SAVGOL_WIN else (N // 2) * 2 - 1
        lead_n = min(int(lead_s * fps), N // 2)
        if lead_n < win + 4:
            continue
        sths, sws = [], []
        for col in ('th1', 'th2'):
            raw = np.unwrap(df[col].values)
            sm = rvm.savgol_filter(raw, win, rvm.SAVGOL_POLY)
            sth = sigma_theta_thirddiff(raw[:lead_n])          # 诚实噪声底
            w0 = [np.gradient(rvm.savgol_filter(sm + rng.normal(0, sth, N),
                                                win, rvm.SAVGOL_POLY), dt)[lead_n]
                  for _ in range(n_mc)]
            sths.append(sth)
            sws.append(float(np.std(w0)))
        sth_c = float(np.mean(sths))
        dw_c = float(np.sqrt(np.mean(np.square(sws))))         # 两摆 RMS
        per_clip[tg] = (sth_c, dw_c)
        sth_deg.append(np.degrees(sth_c))
        sw.append(dw_c)
    return per_clip, float(np.median(sth_deg)), float(np.median(sw))


# ---------- 主流程 ----------
def process(C, pinn, hyb, pinn_d, hyb_d, sth_self, dw_self):
    i0, n, fps, dt = C['lead_n'], C['pred_n'], C['fps'], C['dt']
    tt1, tt2 = C['th1'][i0:i0 + n], C['th2'][i0:i0 + n]      # 真实测量轨迹
    s0 = C['state'][i0].astype(np.float32)
    gt1, gt2 = baseline_gt(s0, n, fps)                       # 无阻尼理想 GT（同初值）
    dd1, dd2 = baseline_damped_gt(s0, n, fps)                # 带阻尼理想 GT（同初值）

    row = {}
    for name, net in [('PINN', pinn), ('Hybrid', hyb)]:
        p1, p2 = rvm.run_torch(net, s0, n, dt)
        row[f'h_real_{name}'] = rvm.horizon_s(rvm.angle_err_deg(p1, p2, tt1, tt2), fps)[0]
        row[f'h_model_{name}'] = rvm.horizon_s(rvm.angle_err_deg(p1, p2, gt1, gt2), fps)[0]
    for name, net in [('PINN', pinn_d), ('Hybrid', hyb_d)]:
        p1, p2 = rvm.run_torch(net, s0, n, dt)
        row[f'h_damp_{name}'] = rvm.horizon_s(rvm.angle_err_deg(p1, p2, tt1, tt2), fps)[0]

    # 纯阻尼失配天花板：无阻尼 vs 带阻尼解析积分（完美初值、唯一差别=阻尼项）
    row['C_dampmis'] = rvm.horizon_s(rvm.angle_err_deg(gt1, gt2, dd1, dd2), fps)[0]

    # 纯混沌天花板：用本片实测初值不确定度 (σθ, δω)
    s0s = s0.copy()
    s0s[0] += sth_self; s0s[2] += sth_self
    s0s[1] += dw_self;  s0s[3] += dw_self
    b1, b2 = baseline_gt(s0s, n, fps)
    row['C_chaos_self'] = rvm.horizon_s(rvm.angle_err_deg(gt1, gt2, b1, b2), fps)[0]

    # 全局 δω 扫描（敏感性曲线）
    for dw in DELTAS:
        s0p = s0.copy()
        s0p[0] += DTH_FALLBACK; s0p[2] += DTH_FALLBACK
        s0p[1] += dw;  s0p[3] += dw
        c1, c2 = baseline_gt(s0p, n, fps)
        row[f'C_chaos_{dw}'] = rvm.horizon_s(rvm.angle_err_deg(gt1, gt2, c1, c2), fps)[0]
    return row


def batch(tags, lead_s, pred_s):
    pinn, hyb = rvm.load_pinn(), rvm.load_hybrid()
    pinn_d, hyb_d = load_damped()
    print('  估计逐片初值不确定度 (三阶差分噪声底 + 蒙特卡洛传播)...')
    deltas, sth_med, dw_med = estimate_delta(tags, lead_s)
    print(f'  → σθ 中位 {sth_med:.3f}°,  δω 中位 {dw_med:.3f} rad/s')

    rows = []
    for tg in tags:
        f = rvm.TRACK / f'tracked_{tg}.csv'
        if not f.exists():
            continue
        C = rvm.load_clip(tg, lead_s, pred_s)
        if C['pred_n'] < 10:
            continue
        sth_self, dw_self = deltas.get(tg, (np.radians(sth_med), dw_med))
        r = process(C, pinn, hyb, pinn_d, hyb_d, sth_self, dw_self)
        r.update(tag=tg, rel_deg=round(C['rel_deg'], 1), fps=round(C['fps'], 1),
                 sigth_deg=round(np.degrees(sth_self), 3), dw_self=round(dw_self, 3))
        rows.append(r)
        print(f"  {tg} rel={C['rel_deg']:6.1f}°  "
              f"real(P/H)={r['h_real_PINN']:.2f}/{r['h_real_Hybrid']:.2f}  "
              f"damp(P/H)={r['h_damp_PINN']:.2f}/{r['h_damp_Hybrid']:.2f}  "
              f"C_dampmis={r['C_dampmis']:.2f}  C_chaos(自{dw_self:.2f})={r['C_chaos_self']:.2f}")

    df = pd.DataFrame(rows)
    cols = (['tag', 'rel_deg', 'fps', 'sigth_deg', 'dw_self'] +
            [f'h_{k}_{m}' for k in ('real', 'damp', 'model') for m in ('PINN', 'Hybrid')] +
            ['C_dampmis', 'C_chaos_self'] + [f'C_chaos_{dw}' for dw in DELTAS])
    df = df[cols]
    df.to_csv(OUT / 'error_decomposition.csv', index=False)

    def med(c):
        return df[c].median()

    print('\n=== 三天花板汇总（%d 条，视界中位 s）===' % len(df))
    print(f"  实测   h_real  : PINN {med('h_real_PINN'):.2f}   Hybrid {med('h_real_Hybrid'):.2f}")
    print(f"  实测   h_damp  : PINN {med('h_damp_PINN'):.2f}   Hybrid {med('h_damp_Hybrid'):.2f}   "
          f"(带阻尼模型真实增益：{med('h_damp_PINN')/med('h_real_PINN'):.2f}× / "
          f"{med('h_damp_Hybrid')/med('h_real_Hybrid'):.2f}×)")
    print(f"  实测   h_model : PINN {med('h_model_PINN'):.2f}   Hybrid {med('h_model_Hybrid'):.2f}   (模型分布内上限)")
    print(f"  天花板 C_dampmis = {med('C_dampmis'):.2f}   (纯阻尼失配，完美初值)")
    print(f"  天花板 C_chaos(每片实测δω) = {med('C_chaos_self'):.2f}   (纯初值不确定，完美模型)")
    print('  全局 δω 敏感性 C_chaos 中位:  ' +
          '  '.join(f"δ={dw}:{med(f'C_chaos_{dw}'):.2f}s" for dw in DELTAS))

    hr = med('h_real_PINN')
    print(f"\n  ★ 关键比值（PINN h_real={hr:.2f}s）：")
    print(f"     C_dampmis / h_real  = {med('C_dampmis')/hr:.1f}×   （阻尼失配天花板远高于实测 → 阻尼非 4s 窗主瓶颈）")
    print(f"     C_chaos  / h_real   = {med('C_chaos_self')/hr:.1f}×   （距混沌天花板尚远 → 初值精度+模型逼近仍有改进空间）")

    print('\n  按释放角分组（PINN, 中位）:')
    for lab, lo, hi in [('温和 <60°', 0, 60), ('中等 60-90°', 60, 90), ('大角 >90°', 90, 999)]:
        g = df[(df.rel_deg >= lo) & (df.rel_deg < hi)]
        if len(g):
            print(f"    {lab:>12} (n={len(g):2d}):  h_real={g['h_real_PINN'].median():.2f}  "
                  f"C_dampmis={g['C_dampmis'].median():.2f}  C_chaos={g['C_chaos_self'].median():.2f}  "
                  f"(δω 中位 {g['dw_self'].median():.3f})")
    plot(df, dw_med, sth_med)
    return df


def plot(df, dw_med, sth_med_deg):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) 四层视界 vs 释放角（PINN）
    ax = axes[0, 0]
    ax.scatter(df.rel_deg, df.C_chaos_self, c='#1f77b4', marker='*', s=90, label='C_chaos 混沌天花板(实测δω)', zorder=3)
    ax.scatter(df.rel_deg, df.C_dampmis, c='#ff7f0e', marker='D', s=40, alpha=0.8, label='C_dampmis 阻尼失配天花板')
    ax.scatter(df.rel_deg, df.h_real_PINN, c='#d62728', marker='o', s=45, alpha=0.85, label='h_real 实际迁移')
    ax.set_xlabel('释放角 |θ1| 峰值 (°)'); ax.set_ylabel('预测视界 (s)')
    ax.set_title('(a) PINN 实测视界 vs 两条理论天花板')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (b) δω 敏感性 + 实测 δω 带 + h_real 线
    ax = axes[1, 0]
    medc = [df[f'C_chaos_{dw}'].median() for dw in DELTAS]
    ax.plot(DELTAS, medc, 'o-', color='#1f77b4', lw=1.5, label='C_chaos 中位 vs δω')
    for x, y in zip(DELTAS, medc):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 8), fontsize=10)
    ax.axvline(dw_med, color='#2ca02c', ls='-', lw=1.4, label=f'实测 δω 中位={dw_med:.3f}')
    ax.axhline(df.h_real_PINN.median(), color='#d62728', ls=':', lw=1.4,
               label=f'h_real 中位={df.h_real_PINN.median():.2f}s')
    ax.set_xlabel('初值不确定度 δω (rad/s)'); ax.set_ylabel('混沌天花板中位 (s)')
    ax.set_title('(b) 混沌天花板对 δω 敏感性（诚实 δω→天花板≈实测的数倍）')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (c) 中位条形：四层 + 两天花板
    ax = axes[0, 1]
    labels = ['h_real', 'h_damp', 'h_model', 'C_dampmis', 'C_chaos']
    pv = [df.h_real_PINN.median(), df.h_damp_PINN.median(), df.h_model_PINN.median(),
          df.C_dampmis.median(), df.C_chaos_self.median()]
    hv = [df.h_real_Hybrid.median(), df.h_damp_Hybrid.median(), df.h_model_Hybrid.median(),
          df.C_dampmis.median(), df.C_chaos_self.median()]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, pv, w, label='PINN', color='#2ca02c', alpha=0.85)
    ax.bar(x + w / 2, hv, w, label='Hybrid', color='#9467bd', alpha=0.85)
    for i, (a, b) in enumerate(zip(pv, hv)):
        ax.annotate(f'{a:.2f}', (i - w / 2, a), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=9)
        ax.annotate(f'{b:.2f}', (i + w / 2, b), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.axvspan(2.5, 4.5, color='gray', alpha=0.08)
    ax.set_ylabel('视界中位 (s)'); ax.set_title('(c) 实测视界（左3）vs 理论天花板（右2，灰区）')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')

    # (d) h_real vs C_chaos_self 配对：贴对角线=已触混沌天花板；远在下方=仍有空间
    ax = axes[1, 1]
    mx = max(df.C_chaos_self.max(), df.h_real_PINN.max(), 0.1) * 1.05
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.6, label='h_real=C_chaos（已触天花板）')
    ax.scatter(df.C_chaos_self, df.h_real_PINN, c='#2ca02c', marker='s', s=45, alpha=0.8, label='PINN')
    ax.scatter(df.C_chaos_self, df.h_real_Hybrid, c='#9467bd', marker='^', s=45, alpha=0.8, label='Hybrid')
    ax.set_xlabel('C_chaos 混沌天花板（实测δω, s）'); ax.set_ylabel('h_real 实际迁移 (s)')
    ax.set_title('(d) 实测 vs 自身混沌天花板：多数点在对角线下方=仍有改进空间')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    fn = FIG / 'error_decomposition.png'
    plt.savefig(fn, dpi=130, bbox_inches="tight")
    plt.close()
    print('\n图 →', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=rvm.DEFAULT_TAGS)
    ap.add_argument('--lead', type=float, default=4.0)
    ap.add_argument('--pred', type=float, default=4.0)
    a = ap.parse_args()
    print(f'误差分解（三天花板）：lead={a.lead}s pred={a.pred}s  {len(a.tags)} 条')
    batch(a.tags, a.lead, a.pred)

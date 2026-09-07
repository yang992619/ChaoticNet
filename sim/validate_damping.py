"""
sim/validate_damping.py — 阻尼模型前向验证（决定性裁决：纯粘性 vs 粘性+阻力）

从实测片窗口起点的状态出发，用三套阻尼模型前向仿真，叠在实测能量包络上对比：
  (A) 纯粘性 b=5.4e-5   —— 小幅尾反推值
  (B) 纯粘性 b=6.6e-5   —— 全局池化纯粘性值
  (C) 粘性+二次阻力 b=1.66e-5 c=8.72e-6 —— 全局池化两项值（baseline_damped 新默认）

能量衰减不受混沌影响（θ 轨迹会发散，能量包络不会），故可直接比 E(t)。
判据：能在大角度片(快衰减)与温和片(慢衰减)同时贴合实测的模型胜出。
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_damped as bd
import fit_damping as fd

MODELS = [
    ('纯粘性 b=5.4e-5',          dict(b=5.4e-5, c=0.0),     'tab:green'),
    ('纯粘性 b=6.6e-5(池化)',    dict(b=6.6e-5, c=0.0),     'tab:orange'),
    ('粘性+阻力 b=1.66e-5,c=8.72e-6', dict(b=1.66e-5, c=8.72e-6), 'tab:red'),
]


def rmse(t_real, E_real, t_sim, E_sim):
    Ei = np.interp(t_real, t_sim, E_sim)
    return float(np.sqrt(np.mean((E_real - Ei) ** 2)))


def run(tags, track_dir, out_dir, t_span=120.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    track_dir, out_dir = Path(track_dir), Path(out_dir)
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8), squeeze=False)

    summary = []
    for j, tag in enumerate(tags):
        st = fd.load_state(track_dir / f'tracked_{tag}.csv')
        t = st['t'] - st['t'][0]
        span = min(t[-1], t_span)
        sel = t <= span
        ax = axes[0, j]
        ax.plot(t[sel], st['E'][sel], 'b-', lw=1.0, alpha=0.55, label='实测 E', zorder=1)
        row = dict(tag=tag)
        for name, kw, col in MODELS:
            sim = bd.simulate(st['th1'][0], st['th2'][0],
                              w1_0=st['w1'][0], w2_0=st['w2'][0],
                              t_end=span, fps=st['fps'], **kw)
            ts, Es = sim['t'].to_numpy(), sim['energy'].to_numpy()
            r = rmse(t[sel], st['E'][sel], ts, Es)
            row[name] = r
            ax.plot(ts, Es, '--', color=col, lw=1.4, label='%s (RMSE %.3f)' % (name, r))
        ax.set_title('片 %s 前向能量包络对比' % tag)
        ax.set_xlabel('t (s)'); ax.set_ylabel('E (J)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        summary.append(row)
        print('  片 %s  RMSE(J): %s' % (tag, '  '.join(
            '%s=%.4f' % (k, v) for k, v in row.items() if k != 'tag')))

    plt.tight_layout()
    fn = out_dir / 'validate_damping.png'
    plt.savefig(fn, dpi=130)
    plt.close()
    print('  对比图 → %s' % fn)
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['1430', '1444'])
    ap.add_argument('--track-dir', default=str(ROOT / 'data' / 'tracking'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'sim_damped'))
    ap.add_argument('--t-span', type=float, default=120.0)
    args = ap.parse_args()
    print('阻尼模型前向验证（纯粘性 vs 粘性+阻力）')
    run(args.tags, args.track_dir, args.out, t_span=args.t_span)

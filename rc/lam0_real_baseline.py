"""
rc/lam0_real_baseline.py — 真实数据上 λ=0(无物理损失) vs λ=1 PINN 同口径对照

回答评委必问：物理损失在 sim→real 是否真扩大可信预测视界？
（注意：sim→sim 上 tools/fair_compare.py 显示物理损失冗余甚至略伤，但不能搬到 sim→real，需真实数据自己测。）

两模型除 λ 外训练设置完全一致（均 AccelNet 128×4, load_dataset(range(5)), 300 epoch,
batch=512, lr=2e-3, seed=42）：
  λ=1 : data/pinn/model.pt           （论文 PINN，summary_multi 中位 h_real≈0.45s）
  λ=0 : data/fair_compare/lam0_model.pt（纯数据 MLP，fair_compare 训练）
复用 real_validation_multi 的窗口/对齐/视界口径（lead/pred=4s, 阈值 10°）。
配对 Wilcoxon 符号秩检验判 λ=1 vs λ=0 在真实数据上的差异是否显著。
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'rc'))
import real_validation_multi as rvm   # noqa: E402

OUT = ROOT / 'data/real_validation'


def load_acc(path):
    net = rvm.pinn_mod.AccelNet(hidden=128, n_layers=4)
    net.load_state_dict(torch.load(path, map_location=rvm.pinn_mod.DEVICE))
    net.to(rvm.pinn_mod.DEVICE).eval()
    return net


def run(tags, lead_s, pred_s):
    lam1 = load_acc(ROOT / 'data/pinn/model.pt')
    lam0 = load_acc(ROOT / 'data/fair_compare/lam0_model.pt')
    rows = []
    for tg in tags:
        if not (rvm.TRACK / f'tracked_{tg}.csv').exists():
            continue
        C = rvm.load_clip(tg, lead_s, pred_s)
        if C['pred_n'] < 10:
            continue
        i0, n, fps, dt = C['lead_n'], C['pred_n'], C['fps'], C['dt']
        tt1, tt2 = C['th1'][i0:i0 + n], C['th2'][i0:i0 + n]
        s0 = C['state'][i0].astype(np.float32)
        r = {'tag': tg, 'rel_deg': round(C['rel_deg'], 1), 'fps': round(fps, 1)}
        for nm, net in [('lam1', lam1), ('lam0', lam0)]:
            p1, p2 = rvm.run_torch(net, s0, n, dt)
            h, cens = rvm.horizon_s(rvm.angle_err_deg(p1, p2, tt1, tt2), fps)
            r[f'h_{nm}'] = h
            r[f'cens_{nm}'] = int(cens)
        rows.append(r)
        print(f"  {tg} rel={C['rel_deg']:6.1f}°  λ=1:{r['h_lam1']:.2f}  λ=0:{r['h_lam0']:.2f}  "
              f"Δ={r['h_lam1'] - r['h_lam0']:+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lam0_vs_lam1_real.csv', index=False)

    m1, m0 = df.h_lam1.median(), df.h_lam0.median()
    print('\n=== 真实数据 λ=1 vs λ=0 同口径对照（%d 片，视界 s）===' % len(df))
    print(f"  λ=1(物理损失) h_real 中位 = {m1:.3f}   均值 {df.h_lam1.mean():.3f}")
    print(f"  λ=0(纯数据)   h_real 中位 = {m0:.3f}   均值 {df.h_lam0.mean():.3f}")
    print(f"  中位差 λ=1−λ=0 = {m1 - m0:+.3f}s   ({m1/m0 if m0>0 else float('nan'):.2f}×)")
    wins = int((df.h_lam1 > df.h_lam0).sum())
    ties = int((df.h_lam1 == df.h_lam0).sum())
    print(f"  逐片：λ=1 更久 {wins} 片 / λ=0 更久 {len(df)-wins-ties} 片 / 平 {ties} 片")
    try:
        from scipy.stats import wilcoxon
        diff = df.h_lam1 - df.h_lam0
        nz = diff[diff != 0]
        if len(nz) >= 5:
            stat, p = wilcoxon(df.h_lam1, df.h_lam0)
            print(f"  配对 Wilcoxon 符号秩检验 p = {p:.3f}  "
                  f"({'显著' if p < 0.05 else '不显著'}，物理损失在真实数据上"
                  f"{'确有' if (p<0.05 and m1>m0) else '未见显著'}增益)")
    except Exception as e:
        print('  (Wilcoxon 跳过:', e, ')')
    print('\n  csv →', OUT / 'lam0_vs_lam1_real.csv')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=rvm.DEFAULT_TAGS)
    ap.add_argument('--lead', type=float, default=4.0)
    ap.add_argument('--pred', type=float, default=4.0)
    a = ap.parse_args()
    print(f'真实数据 λ=0 vs λ=1 PINN 对照  lead={a.lead}s pred={a.pred}s  {len(a.tags)} 片')
    run(a.tags, a.lead, a.pred)

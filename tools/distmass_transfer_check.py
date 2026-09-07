"""
tools/distmass_transfer_check.py — 验证「分布质量 sim 迁移到真实更好」

发现：缓存 data/pinn/model.pt（6/17 分布质量期）真实迁移 0.467，
点质量新训只 0.28。假设：真实摆是分布质量，故分布质量 sim 训出的模型迁移更好。

干净实验（不动 canonical 点质量 data/sim，不改 pinn.py 文件，只在内存里改）：
  1. 用 baseline_distmass 生成分布质量 sim → data/sim_distmass
  2. λ=0（纯数据 MLP，无物理损失→无物理参数依赖）：分布 vs 点质量 训，各评真实
  3. λ=1：临时把 pinn 模块物理参数设成分布质量，训分布数据，评真实（期望≈0.467）
对照：点质量 λ0 real≈0.25 / λ1 real≈0.28 / 缓存分布模型 real 0.467
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))

import baseline_distmass as bld     # noqa: E402  分布质量参数+仿真
import pinn as pinn_mod             # noqa: E402
import real_validation_multi as rv  # noqa: E402

DIST = ROOT / "data" / "sim_distmass"
PM = ROOT / "data" / "sim"
REAL_TAGS = ['1430', '1431', '1432', '1434', '1435', '1436', '1437', '1438', '1439',
             '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448',
             '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1458',
             '1459', '1460']
SEEDS = [42, 7]


def gen_distmass():
    if (DIST / "trial_049.csv").exists():
        print("分布质量数据已存在，跳过生成"); return
    print("生成分布质量 sim → data/sim_distmass ...")
    bld.run_sweep(str(DIST), n_trials=50)


def real_median(net, clips):
    hs = []
    for C in clips.values():
        i0, n, dt, fps = C['lead_n'], C['pred_n'], C['dt'], C['fps']
        s0 = C['state'][i0].astype(np.float32)
        p1, p2 = rv.run_torch(net, s0, n, dt)
        e = rv.angle_err_deg(p1, p2, C['th1'][i0:i0 + n], C['th2'][i0:i0 + n])
        h, _ = rv.horizon_s(e, fps)
        hs.append(h)
    return float(np.median(hs))


def train_on(data_dir, lam, seed):
    pinn_mod.DATA = data_dir
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=20.0)
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                   lambda_phys=lam, n_collocation=2048, seed=seed)
    net.eval()
    return net


def set_pinn_params(dist):
    """把 pinn 模块物理参数（物理损失用）设为分布质量或点质量。"""
    if dist:
        for k in ["G", "L1", "L2", "M1", "M2", "LC1", "LC2", "I1_O", "I2_A"]:
            if hasattr(bld, k):
                setattr(pinn_mod, k, getattr(bld, k))
    # 点质量就用 pinn.py 现有的（已是点质量），不动


def main():
    gen_distmass()
    clips = {tg: rv.load_clip(tg, 4.0, 4.0) for tg in REAL_TAGS}
    clips = {k: v for k, v in clips.items() if v['pred_n'] >= 10}
    print(f"真实片 {len(clips)} 条\n")

    res = []
    # λ=0：无物理损失，物理参数无关 —— 干净对比 数据保真度
    for data_dir, name in [(PM, "点质量"), (DIST, "分布质量")]:
        for sd in SEEDS:
            net = train_on(data_dir, lam=0.0, seed=sd)
            rm = real_median(net, clips)
            res.append({"lam": 0, "data": name, "seed": sd, "real": rm})
            print(f"λ=0 {name} seed={sd}: real median {rm:.4f}")

    # λ=1：物理损失，参数须与数据一致
    for data_dir, name, dist in [(PM, "点质量", False), (DIST, "分布质量", True)]:
        set_pinn_params(dist)
        for sd in SEEDS:
            net = train_on(data_dir, lam=1.0, seed=sd)
            rm = real_median(net, clips)
            res.append({"lam": 1, "data": name, "seed": sd, "real": rm})
            print(f"λ=1 {name}({'分布参数' if dist else '点参数'}) seed={sd}: real median {rm:.4f}")

    df = pd.DataFrame(res)
    OUT = ROOT / "data" / "epoch_transfer"; OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "distmass_transfer.csv", index=False)
    print("\n=== 真实迁移中位（2 seed 均值）===")
    print(df.groupby(["lam", "data"]).real.mean().to_string())
    print("\n参照：缓存分布模型 real 0.467；点质量诚实多 run real PINN 0.28")


if __name__ == "__main__":
    main()

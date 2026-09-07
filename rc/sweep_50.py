"""
50 组完整 sweep：ESN + PINN 在 trial 0-49 全部跑一遍，得到统计意义上的视界分布。

⚠️ 旧基线脚本（λ_phys=1.0、FPS=120），**非论文头条数字来源**。
   论文 B(分布质量) canonical 视界由 tools/canon_multirun_distmass.py 产出，请勿用本脚本复现头条表。
   本脚本权重独立缓存在 data/pinn/sweep50_model.pt，**不会覆盖** canonical 的 data/pinn/model.pt
   （后者由 build_canonical_distmass.py 写出、供 rc/real_validation_multi.py 真实验证复用）。

复用：
  - PINN 权重缓存在 data/pinn/sweep50_model.pt（与 canonical 权重隔离）
  - 不重画每个 trial 的 png（plot=False / skip）

输出：
  - data/rc/esn_summary_50.csv
  - data/pinn/pinn_summary_50.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rc"))
import esn as esn_mod        # noqa: E402
import pinn as pinn_mod      # noqa: E402

SIM = ROOT / "data" / "sim"
RC_OUT = ROOT / "data" / "rc"
PINN_OUT = ROOT / "data" / "pinn"

N_TRIALS = 50
FPS = 120
TRAIN_SEC = 20.0
PRED_SEC = 10.0


def esn_sweep():
    print(f"\n=== ESN sweep 0-{N_TRIALS - 1} ===")
    results = []
    for i in range(N_TRIALS):
        r = esn_mod.run_one(i, train_sec=TRAIN_SEC, pred_sec=PRED_SEC,
                            fps=FPS, plot=False)
        results.append(r)
    df = pd.DataFrame(results)
    RC_OUT.mkdir(parents=True, exist_ok=True)
    out = RC_OUT / "esn_summary_50.csv"
    df.to_csv(out, index=False)
    h = df["horizon"]
    print(f"[ESN] 写出 {out}")
    print(f"  视界 均值 {h.mean():.2f}s ± {h.std():.2f}s  范围 {h.min():.2f}–{h.max():.2f}s")
    return df


def pinn_get_net():
    pt = PINN_OUT / "sweep50_model.pt"   # 独立缓存，绝不覆盖 canonical model.pt
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    if pt.exists():
        print(f"[PINN] 加载缓存权重 {pt}")
        net.load_state_dict(torch.load(pt))
    else:
        print("[PINN] 缓存不存在，训练 300 epoch")
        states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=TRAIN_SEC)
        pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                       lambda_phys=1.0, n_collocation=2048)
        PINN_OUT.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), pt)
    net.eval()
    return net


def pinn_sweep(net):
    print(f"\n=== PINN sweep 0-{N_TRIALS - 1} ===")
    results = []
    for i in range(N_TRIALS):
        df = pd.read_csv(SIM / f"trial_{i:03d}.csv")
        n_train = int(TRAIN_SEC * FPS)
        n_pred = int(PRED_SEC * FPS)
        s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
        pred = pinn_mod.predict_aligned(net, s0, n_pred, dt=1.0 / FPS)
        pred_th = pred[:, [0, 2]]
        truth_th = df[["th1", "th2"]].values[n_train:n_train + n_pred]
        h_s, _ = pinn_mod.horizon(pred_th, truth_th, threshold_deg=10.0, fps=FPS)
        results.append({
            "trial": i, "horizon": h_s,
            "mse_th1": float(np.mean((pred[:, 0] - truth_th[:, 0]) ** 2)),
            "mse_th2": float(np.mean((pred[:, 2] - truth_th[:, 1]) ** 2)),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TRIALS}  最近 horizon={h_s:.2f}s")
    df_res = pd.DataFrame(results)
    out = PINN_OUT / "pinn_summary_50.csv"
    df_res.to_csv(out, index=False)
    h = df_res["horizon"]
    print(f"[PINN] 写出 {out}")
    print(f"  视界 均值 {h.mean():.2f}s ± {h.std():.2f}s  范围 {h.min():.2f}–{h.max():.2f}s")
    return df_res


def joint_summary(esn_df, pinn_df):
    """合并比较表 + 排除准周期（视界 >= 9.9s）后的提升倍数"""
    merged = esn_df[["trial", "horizon"]].rename(columns={"horizon": "esn"}).merge(
        pinn_df[["trial", "horizon"]].rename(columns={"horizon": "pinn"}),
        on="trial",
    )
    merged["ratio"] = merged["pinn"] / merged["esn"].replace(0, np.nan)
    out = ROOT / "data" / "horizon_compare_50.csv"
    merged.to_csv(out, index=False)
    chaotic = merged[(merged.esn < 9.9) & (merged.pinn < 9.9)]
    print(f"\n=== 联合 ===")
    print(f"  写出 {out}")
    print(f"  剔除准周期后 N = {len(chaotic)}/{N_TRIALS}")
    print(f"  ESN  平均视界 {chaotic.esn.mean():.3f}s")
    print(f"  PINN 平均视界 {chaotic.pinn.mean():.3f}s")
    print(f"  提升倍数 中位 {chaotic.ratio.median():.1f}x  均值 {chaotic.ratio.mean():.1f}x")
    return merged


if __name__ == "__main__":
    net = pinn_get_net()                  # 先把权重 cache 好（animate 也会用）
    esn_df = esn_sweep()
    pinn_df = pinn_sweep(net)
    joint_summary(esn_df, pinn_df)

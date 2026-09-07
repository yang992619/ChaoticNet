"""
ChaoticNet 三色对比动画：Ground Truth vs ESN vs PINN

视频结构（10 s 视频）：
  前 20 s 训练段不展示；只播预测段 10 s（trial 选 0）
  白色：GT（baseline.py 仿真）
  蓝色：ESN 闭环预测
  红色：PINN 闭环预测
  时间戳 + 实时偏差角度 + 视界标注

权重缓存：data/pinn/model.pt（若不存在则即时训练 ~3-5 分钟）
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "sim"
PINN_DIR = ROOT / "data" / "pinn"
OUT_DIR = SIM / "figures"

sys.path.insert(0, str(ROOT / "rc"))
import esn as esn_mod        # noqa: E402
import pinn as pinn_mod      # noqa: E402

# 物理参数
L1, L2 = 0.250, 0.200
FPS_DATA = 120
TRAIN_SEC = 20.0
PRED_SEC = 10.0
TRIAL = 0


def get_pinn_net():
    """加载或训练 PINN 权重，缓存在 data/pinn/model.pt"""
    model_pt = PINN_DIR / "model.pt"
    net = pinn_mod.AccelNet(hidden=128, n_layers=4)
    if model_pt.exists():
        print(f"[PINN] 加载缓存权重 {model_pt}")
        net.load_state_dict(torch.load(model_pt))
        net.eval()
        return net
    print("[PINN] 缓存不存在，训练 300 epoch（~3-5 分钟）")
    states, accs = pinn_mod.load_dataset(trial_indices=range(5), train_sec=TRAIN_SEC)
    pinn_mod.train(net, states, accs, n_epochs=300, batch=512, lr=2e-3,
                   lambda_phys=1.0, n_collocation=2048)
    PINN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_pt)
    net.eval()
    return net


def run_esn(df):
    """训练 ESN 并做 10s 闭环预测，返回 (th1_seq, th2_seq)"""
    print("[ESN] 训练 + 预测")
    U = esn_mod.encode(df)
    Y = np.roll(U, -1, axis=0)
    U, Y = U[:-1], Y[:-1]
    n_train = int(TRAIN_SEC * FPS_DATA)
    n_pred = int(PRED_SEC * FPS_DATA)
    esn = esn_mod.ESN(n_in=6, n_out=6, n_res=800,
                      spectral_radius=0.95, leak=0.25, ridge=1e-5)
    _, x_final = esn.train(U[:n_train], Y[:n_train], washout=200)
    pred = esn.predict(x_final, U[n_train - 1], n_pred)
    th1, _, th2, _ = esn_mod.decode(pred)
    return th1, th2


def run_pinn(net, df):
    print("[PINN] 闭环预测")
    n_train = int(TRAIN_SEC * FPS_DATA)
    n_pred = int(PRED_SEC * FPS_DATA)
    s0 = df[["th1", "w1", "th2", "w2"]].values[n_train].astype(np.float32)
    pred = pinn_mod.predict(net, s0, n_pred, dt=1.0 / FPS_DATA)
    return pred[:, 0], pred[:, 2]


def horizon_idx(pred_th1, pred_th2, gt_th1, gt_th2, thresh_deg=10.0):
    err = np.sqrt((pred_th1 - gt_th1) ** 2 + (pred_th2 - gt_th2) ** 2) * 180 / np.pi
    above = np.where(err > thresh_deg)[0]
    return int(above[0]) if len(above) else len(err)


def to_xy(th1, th2):
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    return x1, y1, x2, y2


def make_video(trial=TRIAL, fps_out=30, slow=2.0):
    """slow=2.0 表示视频以 1/slow 倍速回放，便于观察分叉。"""
    df = pd.read_csv(SIM / f"trial_{trial:03d}.csv")

    net = get_pinn_net()
    esn_th1, esn_th2 = run_esn(df)
    pinn_th1, pinn_th2 = run_pinn(net, df)

    n_train = int(TRAIN_SEC * FPS_DATA)
    n_pred = int(PRED_SEC * FPS_DATA)
    gt_th1 = df["th1"].values[n_train:n_train + n_pred]
    gt_th2 = df["th2"].values[n_train:n_train + n_pred]
    t = df["t"].values[n_train:n_train + n_pred] - df["t"].iloc[n_train]

    h_esn = horizon_idx(esn_th1, esn_th2, gt_th1, gt_th2) / FPS_DATA
    h_pinn = horizon_idx(pinn_th1, pinn_th2, gt_th1, gt_th2) / FPS_DATA
    print(f"[视界] ESN={h_esn:.2f}s, PINN={h_pinn:.2f}s, 提升 {h_pinn/h_esn:.1f}x")

    # 下采样到 fps_out
    step = max(1, int(FPS_DATA / fps_out / slow))
    idx = np.arange(0, n_pred, step)
    fps_video = FPS_DATA / step / slow

    xy_gt = to_xy(gt_th1[idx], gt_th2[idx])
    xy_esn = to_xy(esn_th1[idx], esn_th2[idx])
    xy_pinn = to_xy(pinn_th1[idx], pinn_th2[idx])
    t_idx = t[idx]

    # 画布
    extent = (L1 + L2) * 1.15
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 6.5), dpi=120,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.patch.set_facecolor("#0c0c10")

    for a in (ax, ax2):
        a.set_facecolor("#0c0c10")
        a.tick_params(colors="#888")
        for s in a.spines.values():
            s.set_color("#444")

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="#bbb")
    ax.set_ylabel("y (m)", color="#bbb")
    ax.set_title(
        f"trial {trial:03d} · 预测段 0–{PRED_SEC:.0f}s · 1/{slow:.0f} 速回放",
        color="#eee", fontsize=11,
    )

    # 三组摆
    rod1_gt, = ax.plot([], [], "-", color="#ffffff", lw=2.3, alpha=0.95)
    rod2_gt, = ax.plot([], [], "-", color="#ffffff", lw=2.3, alpha=0.95)
    bob_gt, = ax.plot([], [], "o", color="#ffffff", ms=14, mec="#222", mew=0.5, label="GT")

    rod1_e, = ax.plot([], [], "-", color="#5fb0ff", lw=1.8, alpha=0.85)
    rod2_e, = ax.plot([], [], "-", color="#5fb0ff", lw=1.8, alpha=0.85)
    bob_e, = ax.plot([], [], "o", color="#5fb0ff", ms=11, mec="#222", mew=0.4, label="ESN")

    rod1_p, = ax.plot([], [], "-", color="#ff5f7a", lw=1.8, alpha=0.85)
    rod2_p, = ax.plot([], [], "-", color="#ff5f7a", lw=1.8, alpha=0.85)
    bob_p, = ax.plot([], [], "o", color="#ff5f7a", ms=11, mec="#222", mew=0.4, label="PINN")

    ax.plot(0, 0, "o", color="#888", ms=5)
    leg = ax.legend(loc="lower right", facecolor="#1a1a22", edgecolor="#444",
                    labelcolor="#eee", fontsize=10)
    leg.get_frame().set_alpha(0.85)

    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        color="#fffacc", fontsize=12, family="monospace",
                        verticalalignment="top")

    # 误差曲线
    err_esn = np.sqrt((esn_th1 - gt_th1) ** 2 + (esn_th2 - gt_th2) ** 2) * 180 / np.pi
    err_pinn = np.sqrt((pinn_th1 - gt_th1) ** 2 + (pinn_th2 - gt_th2) ** 2) * 180 / np.pi
    ax2.plot(t, err_esn, color="#5fb0ff", lw=1.4, alpha=0.6, label=f"ESN  视界 {h_esn:.2f}s")
    ax2.plot(t, err_pinn, color="#ff5f7a", lw=1.4, alpha=0.6, label=f"PINN 视界 {h_pinn:.2f}s")
    ax2.axhline(10.0, color="#ffeb6b", ls="--", lw=0.9, alpha=0.7)
    ax2.text(t[-1] * 0.98, 10.5, "阈值 10°", ha="right", color="#ffeb6b", fontsize=9)
    ax2.set_xlabel("预测时间 t (s)", color="#bbb")
    ax2.set_ylabel("角度欧氏偏差 (°)", color="#bbb")
    ax2.set_xlim(0, PRED_SEC)
    ax2.set_ylim(0, max(err_esn.max(), err_pinn.max()) * 1.05)
    ax2.set_title("预测偏差随时间", color="#eee", fontsize=11)
    leg2 = ax2.legend(loc="upper left", facecolor="#1a1a22", edgecolor="#444",
                      labelcolor="#eee", fontsize=10)
    leg2.get_frame().set_alpha(0.85)

    cur_line = ax2.axvline(0, color="#fffacc", lw=1.2, alpha=0.8)

    def init():
        for line in (rod1_gt, rod2_gt, bob_gt,
                     rod1_e, rod2_e, bob_e,
                     rod1_p, rod2_p, bob_p):
            line.set_data([], [])
        time_text.set_text("")
        return (rod1_gt, rod2_gt, bob_gt,
                rod1_e, rod2_e, bob_e,
                rod1_p, rod2_p, bob_p,
                time_text, cur_line)

    def update(i):
        x1g, y1g, x2g, y2g = xy_gt[0][i], xy_gt[1][i], xy_gt[2][i], xy_gt[3][i]
        x1e, y1e, x2e, y2e = xy_esn[0][i], xy_esn[1][i], xy_esn[2][i], xy_esn[3][i]
        x1p, y1p, x2p, y2p = xy_pinn[0][i], xy_pinn[1][i], xy_pinn[2][i], xy_pinn[3][i]
        rod1_gt.set_data([0, x1g], [0, y1g])
        rod2_gt.set_data([x1g, x2g], [y1g, y2g])
        bob_gt.set_data([x2g], [y2g])
        rod1_e.set_data([0, x1e], [0, y1e])
        rod2_e.set_data([x1e, x2e], [y1e, y2e])
        bob_e.set_data([x2e], [y2e])
        rod1_p.set_data([0, x1p], [0, y1p])
        rod2_p.set_data([x1p, x2p], [y1p, y2p])
        bob_p.set_data([x2p], [y2p])
        time_text.set_text(
            f"t = {t_idx[i]:5.2f}s\n"
            f"ESN  err = {err_esn[int(idx[i])]:5.1f}°\n"
            f"PINN err = {err_pinn[int(idx[i])]:5.1f}°"
        )
        cur_line.set_xdata([t_idx[i], t_idx[i]])
        return (rod1_gt, rod2_gt, bob_gt,
                rod1_e, rod2_e, bob_e,
                rod1_p, rod2_p, bob_p,
                time_text, cur_line)

    anim = FuncAnimation(fig, update, frames=len(idx), init_func=init,
                         interval=1000 / fps_video, blit=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"anim_compare_trial_{trial:03d}.mp4"
    print(f"渲染 {len(idx)} 帧 @ {fps_video:.0f}fps")
    writer = FFMpegWriter(fps=fps_video, codec="libx264", bitrate=3200,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    anim.save(out, writer=writer, savefig_kwargs={"facecolor": "#0c0c10"})
    plt.close(fig)
    print(f"已生成：{out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GT/ESN/PINN 三色对比动画")
    ap.add_argument("--trial", type=int, default=TRIAL,
                    help=f"trial 编号（默认 {TRIAL}）")
    ap.add_argument("--fps", type=int, default=30, help="视频 fps")
    ap.add_argument("--slow", type=float, default=2.0,
                    help="慢放倍率 (2.0 = 半速回放)")
    args = ap.parse_args()
    make_video(trial=args.trial, fps_out=args.fps, slow=args.slow)

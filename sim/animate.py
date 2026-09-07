"""
ChaoticNet 双摆仿真动画
用 baseline.py 已经跑好的数据生成 mp4。

用法：
    python3 animate.py                   # 默认 trial_000，30fps，~30s 视频
    python3 animate.py --trial 3         # 换一组
    python3 animate.py --trial 0 --fps 60 --tail 1.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 物理参数（与 baseline.py 一致）
L1 = 0.250
L2 = 0.200

# 路径
ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "data" / "sim"
OUT_DIR = SIM_DIR / "figures"


def make_animation(trial: int, fps: int, tail_sec: float) -> Path:
    csv = SIM_DIR / f"trial_{trial:03d}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"找不到 {csv}，先跑 baseline.py")

    df = pd.read_csv(csv)
    dt_raw = float(df.t.iloc[1] - df.t.iloc[0])
    fps_raw = round(1 / dt_raw)
    step = max(1, fps_raw // fps)
    df = df.iloc[::step].reset_index(drop=True)
    fps_actual = round(1 / (df.t.iloc[1] - df.t.iloc[0]))

    th1, th2 = df.th1.values, df.th2.values
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    t = df.t.values

    tail_n = int(tail_sec * fps_actual)

    extent = (L1 + L2) * 1.1
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_facecolor("#0c0c10")
    fig.patch.set_facecolor("#0c0c10")
    ax.tick_params(colors="#888")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_xlabel("x (m)", color="#bbb")
    ax.set_ylabel("y (m)", color="#bbb")
    ax.set_title(
        f"Double Pendulum  ·  trial {trial:03d}  ·  "
        f"L1={L1*1000:.0f}mm  L2={L2*1000:.0f}mm",
        color="#eee", fontsize=11,
    )

    pivot = ax.plot(0, 0, "o", color="#888", ms=6)[0]
    rod1, = ax.plot([], [], "-", color="#e0e0e0", lw=2)
    rod2, = ax.plot([], [], "-", color="#e0e0e0", lw=2)
    bob1, = ax.plot([], [], "o", color="#5fb0ff", ms=10, mec="white", mew=0.5)
    bob2, = ax.plot([], [], "o", color="#ff5f7a", ms=14, mec="white", mew=0.5)
    trail, = ax.plot([], [], "-", color="#ff5f7a", lw=1.0, alpha=0.5)
    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color="#fffacc", fontsize=12, family="monospace",
        verticalalignment="top",
    )

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return rod1, rod2, bob1, bob2, trail, time_text

    def update(i):
        rod1.set_data([0, x1[i]], [0, y1[i]])
        rod2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        bob1.set_data([x1[i]], [y1[i]])
        bob2.set_data([x2[i]], [y2[i]])
        s = max(0, i - tail_n)
        trail.set_data(x2[s:i+1], y2[s:i+1])
        time_text.set_text(f"t = {t[i]:5.2f} s")
        return rod1, rod2, bob1, bob2, trail, time_text

    print(f"渲染 {len(df)} 帧（{fps_actual} fps，预计视频 ~{len(df)/fps_actual:.1f}s）…")
    anim = FuncAnimation(
        fig, update, frames=len(df), init_func=init,
        interval=1000/fps_actual, blit=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"anim_trial_{trial:03d}.mp4"
    writer = FFMpegWriter(
        fps=fps_actual, codec="libx264", bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"],
    )
    anim.save(out, writer=writer, savefig_kwargs={"facecolor": "#0c0c10"})
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tail", type=float, default=0.6, help="末端球轨迹尾迹时长（秒）")
    args = p.parse_args()
    out = make_animation(args.trial, args.fps, args.tail)
    print(f"已生成：{out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""
重绘视觉追踪 QC 图（答辩 PPT 第9页 / idx8, ppt media image8.png）。
只读 data/tracking/tracked_1448.csv 重画，不重跑视频追踪。
CSV 已存 th1/th2；pivot(cx,cy) 与连杆长用几何重算（数据已插值无缺帧，代数圆拟合足够）。
相比原图：字号放大、面板标题改中文短标 + θ 角标用 mathtext，投影更清晰。
输出：data/tracking/qc_1448.png（同名同比例 1.625，直接替换 PPT 内嵌图）。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "tracking" / "tracked_1448.csv"
OUT = ROOT / "data" / "tracking" / "qc_1448.png"

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 20, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
})

df = pd.read_csv(CSV)
t = df.t.values

# pivot：对 m1 点做代数最小二乘圆拟合（数据已插值，无缺帧）
x = df.x_m1_px.values.astype(float)
y = df.y_m1_px.values.astype(float)
A = np.c_[2 * x, 2 * y, np.ones_like(x)]
b = x**2 + y**2
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
cx, cy = sol[0], sol[1]

# 连杆长（m1-m2 间距）与自标定基准（中位）
link = np.hypot(df.x_m2_px - df.x_m1_px, df.y_m2_px - df.y_m1_px).values
link_px = float(np.median(link))

fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.46), dpi=150)

ax[0, 0].plot(t, np.rad2deg(df.th1.values), lw=1.2)
ax[0, 0].set_title(r"(a) 上摆角 $\theta_1$")
ax[0, 0].set_xlabel("t (s)"); ax[0, 0].set_ylabel("角度 (°)")
ax[0, 0].grid(alpha=.3)

ax[0, 1].plot(t, np.rad2deg(df.th2.values), lw=1.2, color="firebrick")
ax[0, 1].set_title(r"(b) 下摆角 $\theta_2$")
ax[0, 1].set_xlabel("t (s)"); ax[0, 1].set_ylabel("角度 (°)")
ax[0, 1].grid(alpha=.3)

ax[1, 0].plot(df.x_m2_px, df.y_m2_px, ".", ms=2, label="m2")
ax[1, 0].plot(df.x_m1_px, df.y_m1_px, ".", ms=2, color="tab:blue", alpha=.5, label="m1")
ax[1, 0].plot([cx], [cy], "k+", ms=16, mew=2.5, label="转轴")
ax[1, 0].set_title("(c) m1 / m2 轨迹 + 转轴")
ax[1, 0].set_xlabel("x (px)"); ax[1, 0].set_ylabel("y (px)")
ax[1, 0].invert_yaxis(); ax[1, 0].set_aspect("equal")
ax[1, 0].legend(fontsize=13, loc="best"); ax[1, 0].grid(alpha=.3)

ax[1, 1].plot(t, link, lw=1.2)
ax[1, 1].axhline(link_px, color="g", ls="--", lw=1.5,
                 label=f"基准 {link_px:.0f}px")
ax[1, 1].set_title("(d) 连杆长度 · 刚性核验")
ax[1, 1].set_xlabel("t (s)"); ax[1, 1].set_ylabel("m1-m2 间距 (px)")
ax[1, 1].legend(fontsize=13, loc="best"); ax[1, 1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
plt.close(fig)
print(f"写出 {OUT}  pivot=({cx:.0f},{cy:.0f})  link_px={link_px:.0f}")

from PIL import Image
im = Image.open(OUT)
print(f"尺寸 {im.size}  比例 {im.size[0]/im.size[1]:.3f}")

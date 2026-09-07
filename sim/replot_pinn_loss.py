"""
重绘 PINN 训练损失曲线（答辩 PPT 第7页 / idx6, ppt media image4.png）。
只读 data/pinn/train_history.csv 重画，不重训模型。
相比原图：字号整体放大（投影清晰），y 轴用十进制标签避开缺字体的上标负号字形。
输出：data/pinn/figures/pinn_train_loss.png（与 PPT 内嵌图同名同比例 1.80，直接替换）。
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "pinn" / "train_history.csv"
OUT = ROOT / "data" / "pinn" / "figures" / "pinn_train_loss.png"

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 19, "axes.titlesize": 22, "axes.labelsize": 20,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 18,
})

hist = pd.read_csv(CSV)

# 帧比 1.801（对齐 PPT 内嵌位）：8.0 / 4.44
fig, ax = plt.subplots(figsize=(8.0, 4.44), dpi=170)
ax.semilogy(hist["epoch"], hist["data_loss"], label="data loss", lw=2.0)
ax.semilogy(hist["epoch"], hist["phys_loss"], label="physics loss", lw=2.0)
ax.set_xlabel("epoch")
ax.set_ylabel("loss (对数)")
ax.set_title("PINN 训练曲线")

# y 轴十进制标签，避开字体缺失的上标负号 tofu
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: ("%g" % v)))

ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"写出 {OUT}")

from PIL import Image
im = Image.open(OUT)
print(f"尺寸 {im.size}  比例 {im.size[0]/im.size[1]:.3f}")

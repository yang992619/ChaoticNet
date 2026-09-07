"""
tracking/track.py — 双摆视觉追踪管线（OpenCV，HSV 阈值 + 质心）

实物到货前用合成动画 sim/figures/anim_trial_000.mp4 验证：
  - 上端球（m1）：蓝色 ≈ #5fb0ff
  - 下端球（m2）：白色反光片（低饱和+高亮度，黑背景上靠亮度区分）

管线：
  1. BGR → HSV
  2. 各色掩膜（红/蓝独立阈值）
  3. 形态学 open + close 去噪
  4. 找最大连通区域 → 质心
  5. 输出 csv: t, x_m1_px, y_m1_px, x_m2_px, y_m2_px, conf_m1, conf_m2
  6. 跟仿真 GT 比对：把像素轨迹转回物理坐标 (m)，计算误差

用法：
    python3 tracking/track.py                                  # 默认合成视频
    python3 tracking/track.py --video path/to/实测.mp4          # 实物视频
    python3 tracking/track.py --debug                           # 写出每帧可视化
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SIM_FIG = ROOT / "data" / "sim" / "figures"
OUT = ROOT / "data" / "tracking"
OUT.mkdir(parents=True, exist_ok=True)
CAMERA_PARAMS = ROOT / "camera_params.npz"   # 相机标定参数（去畸变用）

# ---- HSV 颜色阈值 ----
# 蓝色：#5fb0ff → HSV(207°, 63%, 100%)；OpenCV H 范围 [0,180]
BLUE_LOW  = np.array([100, 90, 120], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# 白色反光片（下摆 m2）：无红色片时改用白色。白=低饱和+高亮度，
# 在哑光黑背景上靠"亮且无色"区分；蓝片饱和度高，不会被白掩膜误抓。
WHITE_LOW  = np.array([0,   0, 180], dtype=np.uint8)
WHITE_HIGH = np.array([180, 70, 255], dtype=np.uint8)

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def detect_centroid(hsv, low, high, low2=None, high2=None):
    """HSV 阈值 → 形态学 → 最大连通区域质心。返回 (x, y, area) 或 None。"""
    mask = cv2.inRange(hsv, low, high)
    if low2 is not None:
        mask |= cv2.inRange(hsv, low2, high2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    # 跳过 label 0（背景），找面积最大
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < 20:        # 太小判为噪声
        return None
    k = 1 + int(np.argmax(areas))
    cx, cy = cents[k]
    return float(cx), float(cy), int(stats[k, cv2.CC_STAT_AREA])


def build_undistort_maps(frame_w, frame_h):
    """读 camera_params.npz，返回逐帧去畸变 remap 表 (map1, map2)；无标定文件返回 None。

    视频分辨率与标定照不同则按比例缩放内参；长宽比不一致会告警（标定失效）。
    """
    if not CAMERA_PARAMS.exists():
        print("  (无 camera_params.npz，跳过去畸变；先跑 tools/calibrate_camera.py)")
        return None
    d = np.load(CAMERA_PARAMS)
    mtx = d["mtx"].astype(np.float64).copy()
    dist = d["dist"]
    cw, ch = (int(d["image_size"][0]), int(d["image_size"][1])) \
        if "image_size" in d else (frame_w, frame_h)
    if (frame_w, frame_h) != (cw, ch):
        sx, sy = frame_w / cw, frame_h / ch
        if abs(sx - sy) / max(sx, sy) > 0.02:
            print(f"  ⚠ 视频 {frame_w}x{frame_h} 与标定 {cw}x{ch} 长宽比不符，"
                  f"去畸变可能不准；建议拍摄分辨率/比例与标定照一致")
        mtx[0, 0] *= sx; mtx[0, 2] *= sx
        mtx[1, 1] *= sy; mtx[1, 2] *= sy
        print(f"  内参已按分辨率缩放 ({cw}x{ch} → {frame_w}x{frame_h})")
    newcam, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_w, frame_h), 0)
    map1, map2 = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcam, (frame_w, frame_h), cv2.CV_16SC2)
    print("  ✓ 已加载标定参数，逐帧去畸变开启")
    return map1, map2


def track(video_path: Path, fps_hint=30.0, debug_dir=None, undistort=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"加载 {video_path.name}: {n} 帧 @ {fps:.1f}fps, {fw}x{fh}")
    umaps = build_undistort_maps(fw, fh) if undistort else None

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if umaps is not None:
            frame = cv2.remap(frame, umaps[0], umaps[1], cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b = detect_centroid(hsv, BLUE_LOW, BLUE_HIGH)
        r = detect_centroid(hsv, WHITE_LOW, WHITE_HIGH)
        rows.append({
            "t": frame_idx / fps,
            "x_m1_px": b[0] if b else np.nan, "y_m1_px": b[1] if b else np.nan,
            "x_m2_px": r[0] if r else np.nan, "y_m2_px": r[1] if r else np.nan,
            "area_m1": b[2] if b else 0, "area_m2": r[2] if r else 0,
        })

        if debug_dir and frame_idx % 30 == 0:
            vis = frame.copy()
            if b: cv2.circle(vis, (int(b[0]), int(b[1])), 12, (255, 200, 0), 2)
            if r: cv2.circle(vis, (int(r[0]), int(r[1])), 12, (0, 200, 255), 2)
            cv2.imwrite(str(Path(debug_dir) / f"frame_{frame_idx:05d}.png"), vis)
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    detect_rate_m1 = df.x_m1_px.notna().mean()
    detect_rate_m2 = df.x_m2_px.notna().mean()
    print(f"  检测率：m1 蓝球 {detect_rate_m1*100:.1f}%  m2 白点 {detect_rate_m2*100:.1f}%")
    return df, fps


def compare_to_gt(track_df, fps_video, sim_csv, L1=0.250, L2=0.200,
                  px_per_m=None, pivot_px=None):
    """像素轨迹 → 物理坐标 → 跟仿真 GT 比较，画对比图。

    自动从轨迹估算 pivot（顶点固定，约等于 m1 轨迹圆心）和 m/px 标度（半径 L1 → 像素半径）。
    """
    sim = pd.read_csv(sim_csv)
    # 自动标定：m1 围绕 pivot 旋转半径 = L1（最小二乘拟合圆）
    m1 = track_df[["x_m1_px", "y_m1_px"]].dropna().values
    if len(m1) < 10:
        raise RuntimeError(f"m1 检测点太少 ({len(m1)})，无法拟合 pivot 圆；检查 HSV 阈值")
    x_, y_ = m1[:, 0], m1[:, 1]
    A = np.column_stack([2 * x_, 2 * y_, np.ones_like(x_)])
    b_lsq = x_ ** 2 + y_ ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b_lsq, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r_sq = sol[2] + cx ** 2 + cy ** 2
        if r_sq <= 0:
            raise np.linalg.LinAlgError("拟合半径非正，轨迹可能近似线段")
        r_px = float(np.sqrt(r_sq))
    except np.linalg.LinAlgError as e:
        # 回退：取轨迹中心 + 中位数距离作为粗略 pivot
        print(f"  ! 拟合圆失败 ({e})，回退到轨迹均值 + 中位距离")
        cx, cy = float(m1[:, 0].mean()), float(m1[:, 1].mean())
        r_px = float(np.median(np.linalg.norm(m1 - [cx, cy], axis=1)))
    pivot_px = np.array([cx, cy])
    px_per_m = r_px / L1
    print(f"  最小二乘拟合圆：pivot=({cx:.0f},{cy:.0f}), r={r_px:.0f}px → 标度 {px_per_m:.1f} px/m")

    # 像素 → 物理（y 翻转，OpenCV y 向下）
    def to_phys(xy_px):
        x = (xy_px[:, 0] - pivot_px[0]) / px_per_m
        y = -(xy_px[:, 1] - pivot_px[1]) / px_per_m
        return np.column_stack([x, y])

    m1_xy = to_phys(track_df[["x_m1_px", "y_m1_px"]].values)
    m2_xy = to_phys(track_df[["x_m2_px", "y_m2_px"]].values)

    # 从位置反解 (θ₁, θ₂)
    # m1: x = L1 sin(θ₁), y = -L1 cos(θ₁) → θ₁ = atan2(x, -y)
    th1_track = np.arctan2(m1_xy[:, 0], -m1_xy[:, 1])
    dx, dy = m2_xy[:, 0] - m1_xy[:, 0], m2_xy[:, 1] - m1_xy[:, 1]
    th2_track = np.arctan2(dx, -dy)

    # 跟仿真同步：仿真 fps=120，视频 fps_video=30 → 抽 sim 每 4 帧
    step = int(round(120 / fps_video))
    sim_idx = np.arange(0, len(sim), step)[:len(track_df)]
    th1_gt = sim.th1.values[sim_idx]
    th2_gt = sim.th2.values[sim_idx]

    n = min(len(th1_track), len(th1_gt))
    th1_track, th2_track = th1_track[:n], th2_track[:n]
    th1_gt, th2_gt = th1_gt[:n], th2_gt[:n]
    t = track_df.t.values[:n]

    # 误差
    err1 = np.rad2deg(np.abs(th1_track - th1_gt))
    err2 = np.rad2deg(np.abs(th2_track - th2_gt))
    print(f"  θ₁ 跟踪误差 中位 {np.nanmedian(err1):.2f}°  90% 分位 {np.nanpercentile(err1, 90):.2f}°")
    print(f"  θ₂ 跟踪误差 中位 {np.nanmedian(err2):.2f}°  90% 分位 {np.nanpercentile(err2, 90):.2f}°")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    plt.rcParams["axes.unicode_minus"] = False
    axes[0, 0].plot(t, np.rad2deg(th1_gt), "k-", lw=1.5, label="GT")
    axes[0, 0].plot(t, np.rad2deg(th1_track), "r--", lw=1.0, label="tracked")
    axes[0, 0].set_ylabel("θ₁ (°)"); axes[0, 0].set_title("θ₁ 时间序列")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, np.rad2deg(th2_gt), "k-", lw=1.5, label="GT")
    axes[0, 1].plot(t, np.rad2deg(th2_track), "r--", lw=1.0, label="tracked")
    axes[0, 1].set_ylabel("θ₂ (°)"); axes[0, 1].set_title("θ₂ 时间序列")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, err1, "b-", lw=1, label="|θ₁ err|")
    axes[1, 0].plot(t, err2, "r-", lw=1, label="|θ₂ err|")
    axes[1, 0].set_xlabel("t (s)"); axes[1, 0].set_ylabel("角度误差 (°)")
    axes[1, 0].set_title("逐帧跟踪误差")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale("symlog")

    axes[1, 1].scatter(m2_xy[:, 0], m2_xy[:, 1], s=3, c=t, cmap="viridis", alpha=0.8)
    axes[1, 1].set_aspect("equal"); axes[1, 1].set_title("m2 末端轨迹（像素→物理坐标）")
    axes[1, 1].set_xlabel("x (m)"); axes[1, 1].set_ylabel("y (m)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT / "tracking_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  写出 {out}")
    return float(np.nanmedian(err1)), float(np.nanmedian(err2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str,
                   default=str(SIM_FIG / "anim_trial_000.mp4"))
    p.add_argument("--sim", type=str,
                   default=str(ROOT / "data" / "sim" / "trial_000.csv"))
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-undistort", action="store_true", help="关闭去畸变（合成测试视频用）")
    args = p.parse_args()

    debug_dir = OUT / "debug_frames" if args.debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    df, fps_video = track(Path(args.video), debug_dir=debug_dir,
                          undistort=not args.no_undistort)
    out_csv = OUT / "tracked.csv"
    df.to_csv(out_csv, index=False)
    print(f"  写出 {out_csv}")

    if Path(args.sim).exists():
        compare_to_gt(df, fps_video, Path(args.sim))
    else:
        print(f"[skip GT 比较] {args.sim} 不存在")


if __name__ == "__main__":
    main()

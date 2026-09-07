"""
tools/acceptance.py — 装置到货验收脚本（W3 实测用）

三项强制验收（任意一项失败装置都不允许进入 W4 数据采集阶段）：

  A. τ 衰减测试       单摆 30° 手动放手，目标衰减时间常数 ≥ 30 秒
  B. 初值反演离散度   同一目标手动放手 N 次，视频反演初值 θ 的标准差 ≤ 0.5°
                      （衡量视频反演初值的估计不确定度，非释放机构的机械重复性）
  C. 视觉系统标定     棋盘格 + OpenCV calibrateCamera，反投影误差 ≤ 0.5 px

用法：
  python3 tools/acceptance.py tau --video data/raw/tau_test.mp4 --theta0 30
  python3 tools/acceptance.py repeat --videos data/raw/release_*.mp4
  python3 tools/acceptance.py calib --videos data/raw/checker_*.mp4 --rows 9 --cols 7 --size 0.025
  python3 tools/acceptance.py report   # 汇总 + 输出 acceptance_report.md

每个子命令独立可跑，最后用 report 生成验收报告。
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "acceptance"
OUT.mkdir(parents=True, exist_ok=True)

# 验收阈值（与 BOM.md / 装置设计目标一致）
TAU_THRESHOLD = 30.0       # 单摆衰减时间常数（秒）≥
INIT_STD_THRESH = 0.5      # 视频反演初值状态估计离散度 σ ≤ 度（非释放机构重复性）
CALIB_REPROJ_THRESH = 0.5  # 棋盘格反投影误差 ≤ 像素


# ---------- A. τ 衰减测试 ----------
def cmd_tau(args):
    """从单摆 30° 释放的视频里拟合振幅指数衰减得 τ。"""
    print(f"[τ 衰减测试] video={args.video}, θ₀={args.theta0}°")
    # 复用 tracking 模块追踪 m1 球（单摆只有 m1）
    import sys
    sys.path.insert(0, str(ROOT / "tracking"))
    from track import track as track_video
    df, fps = track_video(Path(args.video))
    if df.x_m1_px.notna().mean() < 0.95:
        raise RuntimeError(f"m1 检测率 {df.x_m1_px.notna().mean()*100:.1f}% < 95%，检查 HSV 阈值或光照")

    # m1 围绕 pivot 振荡，半径 L1。求每帧 θ_t
    m1 = df[["x_m1_px", "y_m1_px"]].dropna().values
    cx, cy = m1.mean(axis=0)                   # pivot ≈ 轨迹中心
    th = np.arctan2(m1[:, 0] - cx, -(m1[:, 1] - cy))
    t = df.t.values[: len(th)]

    # 提取局部极值作为振幅样本
    from scipy.signal import find_peaks
    pk_idx, _ = find_peaks(np.abs(th), height=np.deg2rad(2), distance=int(fps * 0.5))
    t_peaks = t[pk_idx]
    amp_peaks = np.abs(th[pk_idx])
    print(f"  找到 {len(pk_idx)} 个极值")

    # 指数拟合 A(t) = A₀ exp(-t/τ)
    if len(amp_peaks) < 5:
        raise RuntimeError("极值太少，拟合不靠谱。视频可能太短。")
    log_a = np.log(amp_peaks)
    slope, intercept = np.polyfit(t_peaks, log_a, 1)
    tau = -1.0 / slope
    print(f"  τ = {tau:.2f}s  (阈值 ≥ {TAU_THRESHOLD}s)  {'✅' if tau >= TAU_THRESHOLD else '❌'}")

    out = {"tau_s": float(tau), "n_peaks": int(len(pk_idx)),
           "threshold": TAU_THRESHOLD, "pass": bool(tau >= TAU_THRESHOLD)}
    (OUT / "tau.json").write_text(json.dumps(out, indent=2))
    return out


# ---------- B. 初值反演离散度（视频反演初值的估计不确定度，非释放机构重复性）----------
def cmd_repeat(args):
    """对一组同目标手动放手的视频，用视觉反演每段初始 θ，看其离散度（标准差）。
    衡量的是"视频反演初值的估计不确定度"，不是释放机构的机械重复性——
    本研究每段真实轨迹的初值都从视频反演，故关心的是这个量。"""
    print(f"[初值反演离散度] {len(args.videos)} 个视频")
    import sys
    sys.path.insert(0, str(ROOT / "tracking"))
    from track import track as track_video
    initials = []
    for v in args.videos:
        df, fps = track_video(Path(v))
        # 取前 5 帧的平均，作为"释放瞬间"角度
        m1 = df[["x_m1_px", "y_m1_px"]].dropna().head(5).values
        m2 = df[["x_m2_px", "y_m2_px"]].dropna().head(5).values
        if len(m1) < 3 or len(m2) < 3:
            print(f"  ! 跳过 {v}：前段检测不全")
            continue
        # 自动 pivot
        # 简化：用 m1 全程平均；实物上 pivot 应预校准（固定坐标）
        cx, cy = df[["x_m1_px", "y_m1_px"]].dropna().mean().values
        th1 = np.rad2deg(np.arctan2(m1[:, 0].mean() - cx, -(m1[:, 1].mean() - cy)))
        dx = m2[:, 0].mean() - m1[:, 0].mean()
        dy = m2[:, 1].mean() - m1[:, 1].mean()
        th2 = np.rad2deg(np.arctan2(dx, -dy))
        initials.append((th1, th2))
        print(f"  {Path(v).name}: θ₁={th1:+6.2f}° θ₂={th2:+6.2f}°")

    arr = np.array(initials)
    std1, std2 = arr.std(axis=0)
    print(f"  σ(θ₁) = {std1:.3f}°,  σ(θ₂) = {std2:.3f}°  (阈值 ≤ {INIT_STD_THRESH}°)")
    pass_ = bool(max(std1, std2) <= INIT_STD_THRESH)
    print(f"  {'✅' if pass_ else '❌'}")

    out = {"n_repeats": int(len(arr)), "std_th1_deg": float(std1),
           "std_th2_deg": float(std2), "threshold": INIT_STD_THRESH,
           "pass": pass_}
    (OUT / "repeat.json").write_text(json.dumps(out, indent=2))
    return out


# ---------- C. 棋盘格相机标定 ----------
def cmd_calib(args):
    print(f"[相机标定] 棋盘格 {args.rows}×{args.cols}, 方格 {args.size*1000:.0f}mm, 视频 {len(args.videos)} 个")
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.size

    obj_pts, img_pts = [], []
    img_size = None
    for v in args.videos:
        cap = cv2.VideoCapture(v)
        # 每 10 帧取一张
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if img_size is None:
                    img_size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
                if ret:
                    obj_pts.append(objp)
                    img_pts.append(corners)
            i += 1
        cap.release()
    print(f"  收集 {len(obj_pts)} 个有效棋盘格视图")
    if len(obj_pts) < 10:
        raise RuntimeError(f"标定视图太少 ({len(obj_pts)} < 10)，重拍多角度棋盘格")

    rms, mtx, dist, *_ = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    print(f"  反投影 RMS 误差 = {rms:.3f} px  (阈值 ≤ {CALIB_REPROJ_THRESH} px)")
    pass_ = bool(rms <= CALIB_REPROJ_THRESH)
    print(f"  {'✅' if pass_ else '❌'}")

    np.savez(OUT / "camera_calib.npz", mtx=mtx, dist=dist, rms=rms, img_size=img_size)
    out = {"rms_px": float(rms), "n_views": int(len(obj_pts)),
           "threshold": CALIB_REPROJ_THRESH, "pass": pass_,
           "mtx_npz": str(OUT / "camera_calib.npz")}
    (OUT / "calib.json").write_text(json.dumps(out, indent=2))
    return out


# ---------- 汇总 report ----------
def cmd_report(args):
    print("[汇总验收报告]")
    rows = []
    for name in ["tau", "repeat", "calib"]:
        j = OUT / f"{name}.json"
        if not j.exists():
            rows.append((name, "未跑", False))
            continue
        d = json.loads(j.read_text())
        rows.append((name, json.dumps(d, ensure_ascii=False), d.get("pass", False)))

    md = ["# ChaoticNet 装置验收报告", ""]
    md.append(f"日期：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    md.append("")
    md.append("| 项 | 结论 | 详情 |")
    md.append("|---|---|---|")
    for n, det, ok in rows:
        sym = "✅" if ok else "❌"
        md.append(f"| {n} | {sym} | `{det[:80]}{'...' if len(det)>80 else ''}` |")

    all_pass = all(r[2] for r in rows)
    md.append("")
    md.append(f"**总判定：{'✅ GO（可进入 W4 数据采集）' if all_pass else '❌ NO-GO，先解决失败项'}**")

    rpt = OUT / "acceptance_report.md"
    rpt.write_text("\n".join(md))
    print(f"  写出 {rpt}")
    for line in md: print(" ", line)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("tau", help="τ 衰减测试")
    sp.add_argument("--video", required=True)
    sp.add_argument("--theta0", type=float, default=30.0)

    sp = sub.add_parser("repeat", help="初值反演离散度（视频反演初值估计不确定度）")
    sp.add_argument("--videos", nargs="+", required=True)

    sp = sub.add_parser("calib", help="相机棋盘格标定")
    sp.add_argument("--videos", nargs="+", required=True)
    sp.add_argument("--rows", type=int, default=9)
    sp.add_argument("--cols", type=int, default=7)
    sp.add_argument("--size", type=float, default=0.025, help="方格边长 m")

    sub.add_parser("report", help="汇总报告")

    args = p.parse_args()
    {"tau": cmd_tau, "repeat": cmd_repeat,
     "calib": cmd_calib, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

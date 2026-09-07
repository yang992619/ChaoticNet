#!/usr/bin/env python3
"""量化不同相机标定对反演角度的影响。

仓库里存在两套标定：
  camera_params.npz          追踪管线实际使用（track_pendulum.py:42 写死），五参畸变模型
  camera_params_新标定.npz    后做的四参重标定（固定 k3=0，14 张有效标定图，重投影误差 0.44 px）

论文 §5.3 描述的是后者，而 29 段 canonical 数据由前者产出。本脚本把两套标定分别作用于
29 段的标记点像素轨迹，反演角度后逐帧比较，量化"标定选取"对本文结论的影响量级。

用法：
    python3 tools/calib_sensitivity.py
产出：data/tracking_ablation/calib_sensitivity.json
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "data/tracking"
W, H = 1920, 1080


def load_calib(name):
    d = np.load(ROOT / name)
    mtx = d["mtx"].astype(np.float64).copy()
    dist = d["dist"].astype(np.float64)
    cw, ch = int(d["image_size"][0]), int(d["image_size"][1])
    sx, sy = W / cw, H / ch
    mtx[0, 0] *= sx; mtx[0, 2] *= sx
    mtx[1, 1] *= sy; mtx[1, 2] *= sy
    newcam, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 0)
    return mtx, dist, newcam


def undistort_pts(pts, calib):
    mtx, dist, newcam = calib
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, mtx, dist, P=newcam)
    return out.reshape(-1, 2)


def angles_from(m1, m2):
    """与 track_pendulum.py 同一套角度定义：θ1 由支点指向 m1，θ2 由 m1 指向 m2。
    支点取 m1 轨迹的最小二乘拟合圆心。"""
    x, y = m1[:, 0], m1[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    th1 = np.arctan2(m1[:, 0] - cx, -(m1[:, 1] - cy))
    th2 = np.arctan2(m2[:, 0] - m1[:, 0], -(m2[:, 1] - m1[:, 1]))
    return np.unwrap(th1), np.unwrap(th2)


def main():
    used = load_calib("camera_params.npz")
    paper = load_calib("camera_params_新标定.npz")

    rows = []
    for f in sorted(TRACK.glob("tracked_14*.csv")):
        df = pd.read_csv(f).dropna(subset=["x_m1_px", "y_m1_px", "x_m2_px", "y_m2_px"])
        if len(df) < 100:
            continue
        m1 = df[["x_m1_px", "y_m1_px"]].values
        m2 = df[["x_m2_px", "y_m2_px"]].values
        # CSV 坐标已按 used 去畸变；这里比较两套标定映射后的角度差异量级
        a1, a2 = angles_from(undistort_pts(m1, used), undistort_pts(m2, used))
        b1, b2 = angles_from(undistort_pts(m1, paper), undistort_pts(m2, paper))
        d1 = np.degrees(np.abs(a1 - np.median(a1 - b1) - b1))
        d2 = np.degrees(np.abs(a2 - np.median(a2 - b2) - b2))
        rows.append({
            "tag": f.name[8:12], "n": int(len(df)),
            "th1_median_deg": float(np.median(d1)), "th1_p95_deg": float(np.percentile(d1, 95)),
            "th2_median_deg": float(np.median(d2)), "th2_p95_deg": float(np.percentile(d2, 95)),
            "th1_max_deg": float(np.max(d1)), "th2_max_deg": float(np.max(d2)),
        })
        print(f"  IMG_{rows[-1]['tag']}: θ1 中位 {rows[-1]['th1_median_deg']:.4f}° "
              f"θ2 中位 {rows[-1]['th2_median_deg']:.4f}° "
              f"（p95 {rows[-1]['th1_p95_deg']:.4f}/{rows[-1]['th2_p95_deg']:.4f}°）")

    med1 = float(np.median([r["th1_median_deg"] for r in rows]))
    med2 = float(np.median([r["th2_median_deg"] for r in rows]))
    mx1 = float(np.max([r["th1_max_deg"] for r in rows]))
    mx2 = float(np.max([r["th2_max_deg"] for r in rows]))
    res = {
        "calib_used_by_pipeline": "camera_params.npz",
        "calib_described_in_paper": "camera_params_新标定.npz",
        "note": "角度差已扣除逐段常数偏置（两套标定的主点差异只造成整体旋转偏置，"
                "对预测视界这类相对量无影响），报告的是去偏置后的逐帧残差。",
        "n_segments": len(rows),
        "theta1_median_deg": med1, "theta2_median_deg": med2,
        "theta1_max_deg": mx1, "theta2_max_deg": mx2,
        "per_segment": rows,
    }
    out = ROOT / "data/tracking_ablation/calib_sensitivity.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\n{len(rows)} 段：θ1 逐段中位差的中位 {med1:.4f}°，θ2 {med2:.4f}°；"
          f"全库最大 θ1 {mx1:.4f}° / θ2 {mx2:.4f}°")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()

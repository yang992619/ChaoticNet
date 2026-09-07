#!/usr/bin/env python3
"""消融实验：Viterbi 转移代价（时序连续性）对同色假阳性的抑制作用。

论文 §5.2 声称「刚性约束与时序连续性联合判别，比其中任一单独准则都更能压住同色假阳性」。
本脚本把 track_pendulum.py 的选轨策略退化为「只用刚性约束」（转移代价系数 lam=0），
与管线实际使用的「刚性 + 时序」（lam=1）在同一段实拍、同一套检测候选上对比，量化二者
在 m1 选点差异帧数与圆拟合 RMS 残差上的差别。检测阶段只跑一遍、两种策略共用同一批候选，
因此差别完全来自选轨策略本身。

用法（默认参数与 tracking/reproduce_29.sh 的 COMMON 一致）：
    python3 tools/ablation_transition_cost.py --video 实测视频/源视频归档/IMG_1448.MOV

产出：data/tracking_ablation/ —— 论文 §5.2 引用其中的数字。
严阈值（正式参数）与放宽阈值两种场景分别为 strict_threshold.json 与 loose_threshold.json。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracking"))

from track_pendulum import (  # noqa: E402
    build_undistort_maps,
    detect_all,
    fit_circle_robust,
    track_joint,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    # 与 reproduce_29.sh 的 COMMON 保持一致
    ap.add_argument("--blue-hmin", type=int, default=90)
    ap.add_argument("--blue-hmax", type=int, default=140)
    ap.add_argument("--blue-smin", type=int, default=55)
    ap.add_argument("--blue-vmin", type=int, default=160)
    ap.add_argument("--pink-smin", type=int, default=40)
    ap.add_argument("--pink-vmin", type=int, default=120)
    ap.add_argument("--gate-xmax", type=float, default=1780)
    ap.add_argument("--tol-px", type=float, default=5.0,
                    help="两种策略选到的 m1 相距超过该值即计为一帧选点差异")
    ap.add_argument("--out", default=str(ROOT / "data/tracking_ablation/strict_threshold.json"))
    args = ap.parse_args()

    vp = Path(args.video)
    if not vp.is_absolute():
        vp = ROOT / vp
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfull = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = fh / 2160.0
    min_area = max(int(round(120 * scale ** 2)), 15)
    f0 = max(int(round(args.t0 * fps)), 0)
    f1 = min(int(round(args.t1 * fps)), nfull) if args.t1 < 1e8 else nfull
    print(f"加载 {vp.name}: {nfull} 帧 @ {fps:.2f}fps {fw}x{fh}  窗口[{f0},{f1})  MIN_AREA={min_area}")

    P = dict(
        blue_low=(args.blue_hmin, args.blue_smin, args.blue_vmin),
        blue_high=(args.blue_hmax, 255, 255),
        pink1_low=(0, args.pink_smin, args.pink_vmin), pink1_high=(15, 255, 255),
        pink2_low=(160, args.pink_smin, args.pink_vmin), pink2_high=(180, 255, 255),
        xmin=0, xmax=args.gate_xmax, ymin=0, ymax=1e9, y_gate=1e9,
    )
    umaps = build_undistort_maps(fw, fh)

    pink_lists, blue_lists = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if umaps is not None:
            frame = cv2.remap(frame, umaps[0], umaps[1], cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pg, bg = detect_all(hsv, P, min_area)
        pink_lists.append(pg)
        blue_lists.append(bg)
        fi += 1
    cap.release()
    N = len(blue_lists)
    print(f"检测完成：{N} 帧")

    # 与 track_pendulum.main() 同款：临时 m2 = 每帧最大粉斑，用作 m1 的 anchor 与 link 自标定
    m2_prov = [(max(p, key=lambda t: t[0])[1], max(p, key=lambda t: t[0])[2]) if p else None
               for p in pink_lists]

    ds = []
    for i in range(N):
        if m2_prov[i] is None:
            continue
        mx, my = m2_prov[i]
        for a, x, y in blue_lists[i]:
            ds.append(((x - mx) ** 2 + (y - my) ** 2) ** 0.5)
    if len(ds) < 10:
        raise RuntimeError(f"自标定 LINK_PX 失败：候选-m2 间距仅 {len(ds)} 个")
    ds = np.array(sorted(ds))
    win = max(30.0 * scale, 12.0)
    best_c, link_px = -1, 0.0
    for d in ds:
        sel = ds[(ds >= d - win) & (ds <= d + win)]
        if len(sel) > best_c:
            best_c, link_px = len(sel), float(np.median(sel))
    print(f"LINK_PX = {link_px:.1f}px（自标定众数，簇内 {best_c}/{len(ds)}）")

    n_cand = [len(c) for c in blue_lists]
    res = {
        "video": vp.name,
        "frames": N,
        "fps": round(fps, 3),
        "resolution": f"{fw}x{fh}",
        "window": [f0, f1],
        "hsv_params": {"blue_vmin": args.blue_vmin, "blue_smin": args.blue_smin,
                       "gate_xmax": args.gate_xmax},
        "link_px": round(link_px, 1),
        "blue_candidates_per_frame": {
            "median": float(np.median(n_cand)),
            "max": int(np.max(n_cand)),
            "frames_with_multiple": int(np.sum(np.array(n_cand) > 1)),
        },
        "variants": {},
    }

    paths = {}
    for name, lam in (("rigid_plus_temporal", 1.0), ("rigid_only", 0.0)):
        p = track_joint(blue_lists, m2_prov, link_px, lam=lam)
        paths[name] = p
        pts = np.array([(q[0], q[1]) for q in p if q], dtype=float)
        cx, cy, r_px, rms, keep = fit_circle_robust(pts)
        res["variants"][name] = {
            "lam": lam,
            "pivot_px": [round(float(cx), 1), round(float(cy), 1)],
            "L1_px": round(float(r_px), 1),
            "circle_fit_rms_px": round(float(rms), 2),
            "inliers": f"{int(keep.sum())}/{len(keep)}",
            "resolved_frames": int(len(pts)),
        }
        print(f"[{name}] lam={lam}  pivot=({cx:.0f},{cy:.0f})  L1={r_px:.0f}px  "
              f"RMS={rms:.2f}px  内点 {keep.sum()}/{len(keep)}")

    a, b = paths["rigid_plus_temporal"], paths["rigid_only"]
    diff = 0
    for pa, pb in zip(a, b):
        if pa is None or pb is None:
            if pa is not pb:
                diff += 1
            continue
        if ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 > args.tol_px:
            diff += 1
    res["m1_selection_diff_frames"] = diff
    res["m1_selection_diff_ratio"] = round(diff / N, 5) if N else None
    res["tol_px"] = args.tol_px

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\n两种策略 m1 选点相差 > {args.tol_px}px 的帧数：{diff} / {N}")
    print(f"RMS 残差：刚性+时序 {res['variants']['rigid_plus_temporal']['circle_fit_rms_px']}px"
          f"  vs  只用刚性 {res['variants']['rigid_only']['circle_fit_rms_px']}px")
    print(f"已写入 {outp}")


if __name__ == "__main__":
    main()

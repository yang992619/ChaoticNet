#!/usr/bin/env python3
"""统计 29 段实拍在给定 HSV 门限下每帧的蓝色候选连通域个数。

论文 §5.2 用这个数说明「阈值取错时背景会大量涌入候选集」。本脚本复用 track_pendulum.py
的检测器与去畸变，按 reproduce_29.sh 的逐段时间窗抽帧统计，把逐段中位落盘，
使论文引用的区间有可复核的出处。

默认统计「仅把明度门限退回脚本默认（--blue-vmin 40），饱和度与空间门限仍取正式参数」
这一档，即论文所声明的条件。

用法：
    python3 tools/count_blue_candidates.py                    # 默认档，抽 120 帧/段
    python3 tools/count_blue_candidates.py --blue-vmin 160    # 正式参数档做对照
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracking"))
from track_pendulum import build_undistort_maps, detect_all  # noqa: E402

SRC = ROOT / "实测视频/源视频归档"


def load_windows():
    """从 reproduce_29.sh 解析逐段时间窗，保证与 canonical 追踪口径一致。"""
    sh = (ROOT / "tracking/reproduce_29.sh").read_text(encoding="utf8")
    out = {}
    for m in re.finditer(r'"(\d{4})\s+(\S+)\s+(\S+)"', sh):
        tag, t0, t1 = m.group(1), m.group(2), m.group(3)
        out[tag] = (None if t0 == "-" else float(t0),
                    None if t1 == "-" else float(t1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue-hmin", type=int, default=90)
    ap.add_argument("--blue-hmax", type=int, default=140)
    ap.add_argument("--blue-smin", type=int, default=55, help="正式参数值；脚本默认为 70")
    ap.add_argument("--blue-vmin", type=int, default=40, help="脚本默认值；正式参数为 160")
    ap.add_argument("--gate-xmax", type=float, default=1780)
    ap.add_argument("--samples", type=int, default=120, help="每段等距抽样帧数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    windows = load_windows()
    tags = sorted(p.name[8:12] for p in (ROOT / "data/tracking").glob("tracked_14*.csv"))
    P = dict(
        blue_low=(args.blue_hmin, args.blue_smin, args.blue_vmin),
        blue_high=(args.blue_hmax, 255, 255),
        pink1_low=(0, 40, 120), pink1_high=(15, 255, 255),
        pink2_low=(160, 40, 120), pink2_high=(180, 255, 255),
        xmin=0, xmax=args.gate_xmax, ymin=0, ymax=1e9, y_gate=1e9,
    )

    per_seg = {}
    for tg in tags:
        vp = SRC / f"IMG_{tg}.MOV"
        if not vp.exists():
            print(f"  跳过 {tg}（源视频不在）")
            continue
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nfull = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t0, t1 = windows.get(tg, (None, None))
        f0 = int(round(t0 * fps)) if t0 else 0
        f1 = min(int(round(t1 * fps)), nfull) if t1 else nfull
        min_area = max(int(round(120 * (fh / 2160.0) ** 2)), 15)
        umaps = build_undistort_maps(fw, fh)
        counts = []
        for fi in np.linspace(f0, f1 - 1, args.samples, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            if umaps is not None:
                frame = cv2.remap(frame, umaps[0], umaps[1], cv2.INTER_LINEAR)
            _, blue = detect_all(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), P, min_area)
            counts.append(len(blue))
        cap.release()
        if counts:
            per_seg[tg] = {"median": float(np.median(counts)),
                           "max": int(np.max(counts)),
                           "window_frames": f1 - f0,
                           "sampled": len(counts)}
            print(f"  IMG_{tg}: 中位 {per_seg[tg]['median']:.0f}  最大 {per_seg[tg]['max']}")

    meds = [v["median"] for v in per_seg.values()]
    res = {
        "hsv_params": {"blue_hmin": args.blue_hmin, "blue_hmax": args.blue_hmax,
                       "blue_smin": args.blue_smin, "blue_vmin": args.blue_vmin,
                       "gate_xmax": args.gate_xmax},
        "samples_per_segment": args.samples,
        "n_segments": len(per_seg),
        "per_segment": per_seg,
        "median_of_segment_medians": float(np.median(meds)),
        "min_segment_median": float(np.min(meds)),
        "max_segment_median": float(np.max(meds)),
    }
    out = Path(args.out) if args.out else (
        ROOT / f"data/tracking_ablation/blue_candidates_vmin{args.blue_vmin}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\n{len(per_seg)} 段：逐段中位 {res['min_segment_median']:.0f}--"
          f"{res['max_segment_median']:.0f}，全库中位 {res['median_of_segment_medians']:.0f}")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()

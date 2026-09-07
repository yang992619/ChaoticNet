#!/usr/bin/env python3
"""生成随仓库发布的「分析窗数据集」data/tracking_windows/。

data/tracking/ 下的 29 段逐帧追踪结果合计约 440 MB，不适合随仓库分发；而复现本文的
真实零样本结果只需要每段起始的分析窗（引入窗 + 预测窗）。本脚本从每段 tracked_<tag>.csv
截取起始 N 帧（默认 600 帧 = 10 s @ 60fps），写入 data/tracking_windows/，
并生成一份 README 与 MD5 清单，供读者核对。

用法：
    python3 tools/make_tracking_windows.py            # 默认 600 帧
    python3 tools/make_tracking_windows.py --frames 600 --check   # 只校验不覆盖
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/tracking"
DST = ROOT / "data/tracking_windows"


def md5(p: Path) -> str:
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600, help="每段截取的起始帧数")
    ap.add_argument("--check", action="store_true", help="只校验已有产物，不写盘")
    args = ap.parse_args()

    srcs = sorted(SRC.glob("tracked_14*.csv"))
    if not srcs:
        raise SystemExit(f"没找到源数据：{SRC}/tracked_14*.csv")

    if not args.check:
        DST.mkdir(parents=True, exist_ok=True)

    rows, total = [], 0
    short = []
    for s in srcs:
        df = pd.read_csv(s)
        win = df.head(args.frames)
        if len(win) < args.frames:
            short.append((s.name, len(win)))
        out = DST / s.name
        if args.check:
            if not out.exists():
                raise SystemExit(f"缺失：{out}")
        else:
            win.to_csv(out, index=False)
        total += out.stat().st_size
        rows.append((s.name, len(win), float(win["t"].iloc[0]), float(win["t"].iloc[-1]),
                     md5(out)))

    if short:
        print(f"⚠ 有 {len(short)} 段不足 {args.frames} 帧：{short}")

    print(f"{'段':28s} {'帧数':>6s} {'t起':>8s} {'t止':>8s}  md5")
    for name, n, t0, t1, m in rows:
        print(f"{name:28s} {n:6d} {t0:8.3f} {t1:8.3f}  {m[:12]}")
    print(f"\n共 {len(rows)} 段，合计 {total/1024/1024:.2f} MB")

    if not args.check:
        manifest = "\n".join(f"{m}  {name}" for name, _, _, _, m in rows)
        (DST / "MD5SUMS").write_text(manifest + "\n", encoding="utf8")
        (DST / "README.md").write_text(
            "# 分析窗数据集（tracking_windows）\n\n"
            f"29 段真实双摆录像的追踪结果，每段截取起始 {args.frames} 帧"
            f"（{args.frames/60:.0f} s @ 60 fps），合计约 {total/1024/1024:.1f} MB。\n\n"
            "本目录由 `tools/make_tracking_windows.py` 从 `data/tracking/` 下的逐帧全量结果\n"
            "（约 440 MB，不随仓库分发）截取生成，列定义与全量结果完全一致：\n\n"
            "`frame, t, x_m1_px, y_m1_px, x_m2_px, y_m2_px, area_m1, area_m2, th1, th2, w1, w2`\n\n"
            "其中 `th1/th2` 为反演角度（rad），`w1/w2` 为角速度（rad/s），像素坐标为 1920×1080 口径。\n\n"
            "分析窗完整覆盖论文所用的 4 s 引入窗与 4 s 预测窗，因此仅凭本目录即可复现\n"
            "论文的真实零样本结果。逐帧全量结果与源视频（约 13 GB）不随仓库分发，\n"
            "但可由 `tracking/reproduce_29.sh` 从源视频完整重建。\n\n"
            "`MD5SUMS` 为各段校验和，可用 `md5sum -c MD5SUMS`（macOS：`md5 -r`）核对。\n",
            encoding="utf8")
        print(f"已写入 {DST}（含 README.md 与 MD5SUMS）")


if __name__ == "__main__":
    main()

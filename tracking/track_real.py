"""
tracking/track_real.py — 实拍双摆视觉追踪（蓝 m1 中间关节 + 红/粉 m2 末端）

针对 IMG_1392 这类真实贴标记视频。关键与合成版 track.py 的差异：
  - m1 = 蓝色（中间关节），m2 = 红色（末端，实拍发白发粉、低饱和，须放宽阈值）。
  - 蓝色空间门限：丢弃 y>=Y_GATE 的底部假阳性（袋子缝线等）。
  - 刚性下臂先验 + 时间连续性（Viterbi）：画面里常有第二个蓝假斑，且其到红尖
    的距离会偶然也≈LINK_PX，光靠"离 m2 最近 570"会被骗（约 30 帧选错、左右乱跳）。
    故对所有门限蓝斑做最短路径：发射代价=|到 m2 距离-LINK_PX|，转移代价=帧间位移；
    真 m1 路径既贴刚性距离又平滑，假斑要"跳过去再跳回"位移代价爆炸→自动排除。
  - 角度约定对齐 sim：竖直向下为 0，θ1=atan2(m1x-pivx, m1y-pivy)，θ2 从 m1 起算。

用法：
    python3 tracking/track_real.py --video ~/Pictures/IMG_1392.MOV --tag 1392
输出：
    data/tracking/tracked_<tag>.csv         逐帧 像素 + θ1/θ2/w1/w2
    data/tracking/qc_<tag>.png              θ 时序 + m2 轨迹 + 命中/残差
    data/tracking/overlay_<tag>_*.png       叠加抽帧目视
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
OUT = ROOT / "data" / "tracking"
OUT.mkdir(parents=True, exist_ok=True)
CAMERA_PARAMS = ROOT / "camera_params.npz"

# ---- HSV 阈值（在 IMG_1392 上验证过）----
BLUE_LOW, BLUE_HIGH = (90, 70, 40), (140, 255, 255)
# 红/粉末端：低饱和高明度，跨 0/180 两端
PINK1_LOW, PINK1_HIGH = (0, 40, 120), (15, 255, 255)
PINK2_LOW, PINK2_HIGH = (160, 40, 120), (180, 255, 255)

KER = np.ones((5, 5), np.uint8)
Y_GATE = 1800          # 蓝色空间门限：丢弃 y>=此值的底部假阳性（按 4K 全分辨率）
MIN_AREA = 120         # 连通域最小面积
LINK_PX = 570          # m1->m2 刚性下臂先验距离（实测 561~581）
LINK_TOL = 130         # 容差（仅用于发射代价软约束 / 质检参考）
JUMP_PX = 260          # m1 帧间最大合理位移（120fps 下真摆 <~150px/帧），超此判离群


def blobs(mask, min_area):
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out.append((a, float(ce[i][0]), float(ce[i][1])))
    return out


def detect_all(hsv):
    """返回 (m2=(x,y) 或 None, m2_area, 门限内蓝斑候选 [(area,x,y), ...])。"""
    blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    pink = cv2.inRange(hsv, PINK1_LOW, PINK1_HIGH) | cv2.inRange(hsv, PINK2_LOW, PINK2_HIGH)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, KER)
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, KER)
    bb = blobs(blue, MIN_AREA)
    pp = blobs(pink, MIN_AREA)

    m2 = None; m2a = 0
    if pp:
        a, x, y = max(pp, key=lambda t: t[0])
        m2 = (x, y); m2a = a

    gated = [(a, x, y) for (a, x, y) in bb if y < Y_GATE]
    return m2, m2a, gated


def track_m1(cand_lists, m2_xy, lam=1.0):
    """Viterbi 在每帧蓝斑候选里选一条 m1 轨迹：
       发射代价=|到 m2 距离-LINK_PX|（无 m2 则 0），转移代价=lam*帧间位移。
       返回每帧 (x,y,area) 或 None（该帧无候选）。"""
    N = len(cand_lists)
    INF = 1e18
    # 每帧的状态（候选位置）与发射代价
    states, emis = [], []
    for i in range(N):
        cs = cand_lists[i]
        if cs:
            st, em = [], []
            for a, x, y in cs:
                if m2_xy[i] is not None:
                    d = ((x - m2_xy[i][0])**2 + (y - m2_xy[i][1])**2) ** 0.5
                    e = abs(d - LINK_PX)
                else:
                    e = 0.0
                st.append((x, y, a)); em.append(e)
            states.append(st); emis.append(em)
        else:
            states.append([None]); emis.append([0.0])   # 空帧=虚拟节点（转移免费，事后插值）

    cost = [None] * N
    back = [None] * N
    cost[0] = list(emis[0]); back[0] = [-1] * len(emis[0])
    for i in range(1, N):
        cur, prev = states[i], states[i - 1]
        cost[i] = [INF] * len(cur); back[i] = [-1] * len(cur)
        for j, sj in enumerate(cur):
            best, bp = INF, -1
            for k, sk in enumerate(prev):
                tr = 0.0 if (sj is None or sk is None) else \
                    lam * ((sj[0] - sk[0])**2 + (sj[1] - sk[1])**2) ** 0.5
                c = cost[i - 1][k] + tr
                if c < best:
                    best, bp = c, k
            cost[i][j] = best + emis[i][j]; back[i][j] = bp

    j = int(np.argmin(cost[N - 1]))
    out = [None] * N
    for i in range(N - 1, -1, -1):
        out[i] = states[i][j]
        j = back[i][j]
    return out


def build_undistort_maps(fw, fh):
    if not CAMERA_PARAMS.exists():
        print("  (无 camera_params.npz，跳过去畸变)")
        return None
    d = np.load(CAMERA_PARAMS)
    mtx = d["mtx"].astype(np.float64).copy()
    dist = d["dist"]
    cw, ch = (int(d["image_size"][0]), int(d["image_size"][1])) if "image_size" in d else (fw, fh)
    if (fw, fh) != (cw, ch):
        sx, sy = fw / cw, fh / ch
        if abs(sx - sy) / max(sx, sy) > 0.02:
            print(f"  ⚠ 视频 {fw}x{fh} 与标定 {cw}x{ch} 比例不符")
        mtx[0, 0] *= sx; mtx[0, 2] *= sx
        mtx[1, 1] *= sy; mtx[1, 2] *= sy
        print(f"  内参按分辨率缩放 ({cw}x{ch}->{fw}x{fh})")
    newcam, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (fw, fh), 0)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newcam, (fw, fh), cv2.CV_16SC2)
    print("  ✓ 去畸变开启")
    return map1, map2


def fit_circle(pts):
    """最小二乘拟合圆 → (cx, cy, r, rms残差px)。"""
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0)))
    res = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    return cx, cy, r, float(np.sqrt(np.mean(res**2)))


def fit_circle_robust(pts, iters=3, k=2.5):
    """带残差剔除的圆拟合：反复丢弃 >k*RMS 的离群点再重拟合。"""
    keep = np.ones(len(pts), bool)
    cx = cy = r = rms = 0.0
    for _ in range(iters):
        cx, cy, r, rms = fit_circle(pts[keep])
        res = np.abs(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2) - r)
        new = res <= max(k * rms, 5.0)
        if new.sum() == keep.sum() or new.sum() < 10:
            keep = new; break
        keep = new
    return cx, cy, r, rms, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", default="real")
    ap.add_argument("--no-undistort", action="store_true")
    args = ap.parse_args()

    vp = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"加载 {vp.name}: {n} 帧 @ {fps:.2f}fps {fw}x{fh}")
    umaps = None if args.no_undistort else build_undistort_maps(fw, fh)

    m2_xy, m2_area, cand_lists = [], [], []
    raw_frames = {}
    save_idx = set(int(round(k)) for k in np.linspace(0, n - 1, 6))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if umaps is not None:
            frame = cv2.remap(frame, umaps[0], umaps[1], cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m2, m2a, gated = detect_all(hsv)
        m2_xy.append(m2); m2_area.append(m2a); cand_lists.append(gated)
        if fi in save_idx:
            raw_frames[fi] = frame.copy()
        fi += 1
    cap.release()
    N = len(m2_xy)

    # 命中率（原始检测层：m2 有粉斑、m1 至少有一个门限蓝斑）
    hit2_raw = np.mean([m is not None for m in m2_xy])
    hit1_raw = np.mean([len(c) > 0 for c in cand_lists])

    # m1 轨迹：Viterbi 选一条贴刚性距离且平滑的路径
    m1_path = track_m1(cand_lists, m2_xy, lam=1.0)

    df = pd.DataFrame({
        "frame": np.arange(N), "t": np.arange(N) / fps,
        "x_m1_px": [p[0] if p else np.nan for p in m1_path],
        "y_m1_px": [p[1] if p else np.nan for p in m1_path],
        "x_m2_px": [m[0] if m else np.nan for m in m2_xy],
        "y_m2_px": [m[1] if m else np.nan for m in m2_xy],
        "area_m1": [p[2] if p else 0 for p in m1_path],
        "area_m2": m2_area,
    })

    # 限速清离群：m1 若与前后邻都跳 >JUMP_PX，判为坏点置 NaN（事后插值）
    x = df.x_m1_px.values.copy(); y = df.y_m1_px.values.copy()
    bad = np.zeros(N, bool)
    for i in range(N):
        pj = i - 1
        nj = i + 1
        dp = np.hypot(x[i] - x[pj], y[i] - y[pj]) if pj >= 0 and not np.isnan(x[pj]) else 0
        dn = np.hypot(x[i] - x[nj], y[i] - y[nj]) if nj < N and not np.isnan(x[nj]) else 0
        if dp > JUMP_PX and dn > JUMP_PX:
            bad[i] = True
    df.loc[bad, ["x_m1_px", "y_m1_px"]] = np.nan
    n_bad = int(bad.sum())

    print(f"  命中率：m2 红/粉 {hit2_raw*100:.1f}%   m1 蓝(门限内有候选) {hit1_raw*100:.1f}%")
    print(f"  Viterbi 选轨 + 限速清离群：剔除 m1 离群 {n_bad} 帧（插值补回）")

    # 刚性下臂质检（清离群前的链长——用当前 df 未插值值）
    both = df.dropna(subset=["x_m1_px", "x_m2_px"])
    link = np.sqrt((both.x_m2_px - both.x_m1_px)**2 + (both.y_m2_px - both.y_m1_px)**2)
    print(f"  m1-m2 间距：中位 {link.median():.1f}px  std {link.std():.1f}  "
          f"范围 [{link.min():.0f}, {link.max():.0f}]")

    # 丢帧线性插值
    for c in ["x_m1_px", "y_m1_px", "x_m2_px", "y_m2_px"]:
        df[c] = df[c].interpolate(limit_direction="both")

    # 圆拟合 pivot + L1（鲁棒：剔残差离群再拟合）
    m1pts = df[["x_m1_px", "y_m1_px"]].dropna().values
    cx, cy, r_px, rms, keep = fit_circle_robust(m1pts)
    print(f"  圆拟合 pivot=({cx:.0f},{cy:.0f})  L1={r_px:.0f}px  "
          f"RMS残差={rms:.1f}px  内点 {keep.sum()}/{len(keep)}")
    print(f"  L2/L1 像素比 ≈ {link.median()/r_px:.2f}（物理设计 L2≈0.20 / L1≈0.25 = 0.80 参照）")

    # 角度（竖直向下=0，对齐 sim track.py 约定）
    th1 = np.arctan2(df.x_m1_px - cx, df.y_m1_px - cy)
    th2 = np.arctan2(df.x_m2_px - df.x_m1_px, df.y_m2_px - df.y_m1_px)
    th1 = np.unwrap(th1.values); th2 = np.unwrap(th2.values)
    dt = 1.0 / fps
    w1 = np.gradient(th1, dt); w2 = np.gradient(th2, dt)
    df["th1"], df["th2"], df["w1"], df["w2"] = th1, th2, w1, w2

    # θ 连续性自检
    d1 = np.abs(np.diff(np.degrees(th1))); d2 = np.abs(np.diff(np.degrees(th2)))
    print(f"  Δθ1/帧 中位{np.median(d1):.2f}° 95%{np.percentile(d1,95):.1f}° 最大{d1.max():.1f}°  "
          f">20°帧={int((d1>20).sum())}")
    print(f"  Δθ2/帧 中位{np.median(d2):.2f}° 95%{np.percentile(d2,95):.1f}° 最大{d2.max():.1f}°  "
          f">20°帧={int((d2>20).sum())}")
    print(f"  |w1|max {np.abs(w1).max():.1f}  |w2|max {np.abs(w2).max():.1f} rad/s")

    csv = OUT / f"tracked_{args.tag}.csv"
    df.to_csv(csv, index=False)
    print(f"  写出 {csv}")

    # ---- QC 图 ----
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), dpi=120)
    t = df.t.values
    ax[0, 0].plot(t, np.rad2deg(th1), lw=1)
    ax[0, 0].set_title("theta1 (deg) vs t"); ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(t, np.rad2deg(th2), lw=1, color="firebrick")
    ax[0, 1].set_title("theta2 (deg) vs t"); ax[0, 1].grid(alpha=.3)
    ax[1, 0].plot(df.x_m2_px, df.y_m2_px, ".", ms=2)
    ax[1, 0].plot(df.x_m1_px, df.y_m1_px, ".", ms=2, color="tab:blue", alpha=.5)
    ax[1, 0].plot([cx], [cy], "k+", ms=14)
    ax[1, 0].set_title("m1(蓝)/m2 traj (px) + pivot"); ax[1, 0].invert_yaxis()
    ax[1, 0].set_aspect("equal"); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(both.t.values, link.values, lw=1)
    ax[1, 1].axhline(LINK_PX, color="g", ls="--", lw=1)
    ax[1, 1].set_title("m1-m2 link length (px) — rigidity check"); ax[1, 1].grid(alpha=.3)
    fig.tight_layout()
    qc = OUT / f"qc_{args.tag}.png"
    fig.savefig(qc, dpi=130, facecolor="white"); plt.close(fig)
    print(f"  写出 {qc}")

    # ---- 叠加抽帧（用最终 df 的清洗后坐标重画）----
    for k, frame in sorted(raw_frames.items()):
        vis = frame.copy()
        row = df.iloc[k]
        cv2.circle(vis, (int(cx), int(cy)), 18, (0, 255, 0), 3)   # pivot 绿
        if not np.isnan(row.x_m1_px):
            p1 = (int(row.x_m1_px), int(row.y_m1_px))
            cv2.circle(vis, p1, 26, (255, 0, 0), 4)
            cv2.line(vis, (int(cx), int(cy)), p1, (0, 255, 255), 2)
            if not np.isnan(row.x_m2_px):
                p2 = (int(row.x_m2_px), int(row.y_m2_px))
                cv2.circle(vis, p2, 26, (0, 0, 255), 4)
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
        small = cv2.resize(vis, (fw // 3, fh // 3))
        cv2.imwrite(str(OUT / f"overlay_{args.tag}_{k:04d}.png"), small)
    print(f"  写出 {len(raw_frames)} 张叠加抽帧 overlay_{args.tag}_*.png")


if __name__ == "__main__":
    main()

"""
tracking/track_pendulum.py — 通用实拍双摆视觉追踪（蓝 m1 中间关节 + 红/粉 m2 末端）

是 track_real.py（专为早期试拍片 IMG_1392 调好，那条是 4K/120fps，已被剔除、未纳入
正式数据集；正式 29 段全为 1080p/60fps）的分辨率/机位无关泛化版，
为补拍批量（不同分辨率、帧率、机位、打光）准备。与 track_real.py 的关键差异：

  - 像素尺度参数自适应：MIN_AREA / JUMP / Y_GATE 按帧高相对 4K(2160) 缩放；
    刚性下臂距离 LINK_PX 直接从数据自标定（取「蓝/粉各恰好 1 斑」帧的间距中位数），
    因机位/变焦不同会改变它，硬编码 570 只对 1392 成立。
  - HSV 阈值全部走 CLI。IMG_1430 这类「黑包其实是藏青、蓝标在蓝底上」的片子，
    靠提高蓝色明度门限 V>=160 把暗藏青背景滤掉、只留高亮蓝标（实测亮标 V≈240）。
  - 空间门限可设左右/上下边界，挡掉米色墙、释放瞬间入镜的手等粉色假阳性。
  - 可裁时间窗 [t0,t1]（阻尼长拍前 9 分钟多是静止，没必要全跑）。

检测/Viterbi 选轨/鲁棒圆拟合等核心逻辑与 track_real.py 一致。

用法示例：
  # 1392 回归（复现 track_real.py 结果）
  python3 tracking/track_pendulum.py --video ~/Pictures/IMG_1392.MOV --tag 1392g
  # 1430（藏青背景 + 1080p/60fps 阻尼长拍，只跑活跃段）
  python3 tracking/track_pendulum.py --video ~/Pictures/IMG_1430.MOV --tag 1430 \
      --blue-vmin 160 --blue-smin 55 --gate-xmax 1780 --t0 2.5 --t1 200
输出：
  data/tracking/tracked_<tag>.csv   逐帧 像素 + θ1/θ2/w1/w2
  data/tracking/qc_<tag>.png        θ 时序 + 轨迹 + 刚性间距
  data/tracking/overlay_<tag>_*.png 叠加抽帧目视
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

KER = np.ones((5, 5), np.uint8)


def blobs(mask, min_area):
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out.append((a, float(ce[i][0]), float(ce[i][1])))
    return out


def detect_all(hsv, P, min_area):
    """返回 (门限内粉斑候选, 门限内蓝斑候选)，各为 [(area,x,y), ...]。P 是阈值/门限字典。
    m1/m2 都返回全部候选，后续各用一遍「刚性+时序」Viterbi 选轨，不在此处贪心取最大斑。"""
    blue = cv2.inRange(hsv, P["blue_low"], P["blue_high"])
    pink = (cv2.inRange(hsv, P["pink1_low"], P["pink1_high"]) |
            cv2.inRange(hsv, P["pink2_low"], P["pink2_high"]))
    ker = np.ones((3, 3), np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, ker)
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, ker)
    bb = blobs(blue, min_area)
    pp = blobs(pink, min_area)

    # 空间门限（挡墙/手）：x,y 必须落在 [xmin,xmax]×[ymin,ymax]
    def inside(x, y):
        return P["xmin"] <= x <= P["xmax"] and P["ymin"] <= y <= P["ymax"]

    pink_g = [(a, x, y) for (a, x, y) in pp if inside(x, y)]
    blue_g = [(a, x, y) for (a, x, y) in bb if inside(x, y) and y < P["y_gate"]]
    return pink_g, blue_g


# ---- Lab 检测路径（2026-09-06 在 33xx 那批新标记上定稿）----
# 新装置换了标记：关节是深蓝、末端是饱和红。深蓝在 HSV 里 V 偏低时 H 抖得厉害，
# 同一段 IMG_3365 走 HSV 有 185 帧 |Δθ1|>20°/帧（最大 174°）、m2 破 link 405 帧；
# 走 Lab 是 0 帧、最大 16°/帧。老片(14xx)那套粉标记仍走 HSV，两条路径并存。
LAB_L_MIN = 100
LAB_BLUE_HUE = (-120, -30)
LAB_BLUE_CH = 14
LAB_RED_HUE = (-10, 58)      # 上界收到 58°，把手/肤色(hue≈66°)挡在外
LAB_RED_CH = 16
LAB_BASE_CH = 12


def detect_all_lab(bgr, P, min_area):
    """与 detect_all 同签名同返回：(门限内粉/红斑候选, 门限内蓝斑候选)。
    一次连通域，按 Lab 色相角分桶；同样只返回候选，不在此处贪心取最大斑。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[:, :, 0], lab[:, :, 1] - 128, lab[:, :, 2] - 128
    ch = np.hypot(A, B)
    m = ((L >= LAB_L_MIN) & (ch >= LAB_BASE_CH)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lb, st, ce = cv2.connectedComponentsWithStats(m, 8)

    def inside(x, y):
        return P["xmin"] <= x <= P["xmax"] and P["ymin"] <= y <= P["ymax"]

    pink_g, blue_g = [], []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y = float(ce[i][0]), float(ce[i][1])
        if not inside(x, y):
            continue
        sel = lb == i
        hue = float(np.degrees(np.arctan2(np.median(B[sel]), np.median(A[sel]))))
        chm = float(np.median(ch[sel]))
        if LAB_BLUE_HUE[0] < hue < LAB_BLUE_HUE[1] and chm >= LAB_BLUE_CH:
            if y < P["y_gate"]:
                blue_g.append((a, x, y))
        elif LAB_RED_HUE[0] < hue < LAB_RED_HUE[1] and chm >= LAB_RED_CH:
            pink_g.append((a, x, y))
    return pink_g, blue_g


def track_joint(cand_lists, anchor_xy, link_px, lam=1.0):
    """Viterbi 选关节轨迹：发射代价=|到 anchor 距离-link_px|，转移代价=lam*帧间位移。
    m1 以 m2 为 anchor、m2 以 m1 为 anchor，对称使用——刚性下臂约束 + 时序连续性
    联合排除「恰好也落在 link 距离上的假阳性」和「闯入画面的大块干扰（如手）」。"""
    N = len(cand_lists)
    INF = 1e18
    states, emis = [], []
    for i in range(N):
        cs = cand_lists[i]
        if cs:
            st, em = [], []
            for a, x, y in cs:
                if anchor_xy[i] is not None:
                    d = ((x - anchor_xy[i][0])**2 + (y - anchor_xy[i][1])**2) ** 0.5
                    e = abs(d - link_px)
                else:
                    e = 0.0
                st.append((x, y, a)); em.append(e)
            states.append(st); emis.append(em)
        else:
            states.append([None]); emis.append([0.0])

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


def build_undistort_maps(fw, fh, cam_path=None):
    cp = Path(cam_path) if cam_path else CAMERA_PARAMS
    if not cp.exists():
        print(f"  (无 {cp.name}，跳过去畸变)")
        return None
    print(f"  标定文件 {cp.name}")
    d = np.load(cp)
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
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0)))
    res = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    return cx, cy, r, float(np.sqrt(np.mean(res**2)))


def fit_circle_robust(pts, iters=3, k=2.5):
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
    ap.add_argument("--force", action="store_true",
                    help="即使未通过合理性守卫也写盘（默认不通过就拒绝写）")
    ap.add_argument("--detector", choices=["hsv", "lab"], default="hsv",
                    help="hsv=老片(14xx)粉标记那套；lab=33xx 新标记(深蓝+饱和红)，见 detect_all_lab")
    ap.add_argument("--camera", default=None, help="标定 npz 路径；默认 camera_params.npz")
    # HSV 阈值（默认沿用 1392）
    ap.add_argument("--blue-hmin", type=int, default=90)
    ap.add_argument("--blue-hmax", type=int, default=140)
    ap.add_argument("--blue-smin", type=int, default=70)
    ap.add_argument("--blue-vmin", type=int, default=40)
    ap.add_argument("--pink-smin", type=int, default=40)
    ap.add_argument("--pink-vmin", type=int, default=120)
    # 空间门限（像素，默认全画面 + 1392 的 y<1800 蓝门限）
    ap.add_argument("--gate-xmin", type=float, default=0)
    ap.add_argument("--gate-xmax", type=float, default=1e9)
    ap.add_argument("--gate-ymin", type=float, default=0)
    ap.add_argument("--gate-ymax", type=float, default=1e9)
    ap.add_argument("--y-gate-frac", type=float, default=1e9,
                    help="蓝色底部门限，按帧高比例丢弃 y>=frac*H 的蓝斑（默认不启用）")
    # 像素尺度（默认按帧高相对 2160 自适应；可覆盖）
    ap.add_argument("--min-area", type=int, default=0, help="0=按分辨率自适应")
    ap.add_argument("--jump-px", type=float, default=0, help="0=按分辨率自适应")
    ap.add_argument("--link-px", type=float, default=0, help="0=从数据自标定")
    # 时间窗
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    args = ap.parse_args()

    vp = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfull = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = fh / 2160.0
    min_area = args.min_area if args.min_area > 0 else max(int(round(120 * scale**2)), 15)
    jump_px = args.jump_px if args.jump_px > 0 else max(260 * scale, 80)
    y_gate = args.y_gate_frac * fh if args.y_gate_frac < 1e8 else 1e9

    f0 = max(int(round(args.t0 * fps)), 0)
    f1 = min(int(round(args.t1 * fps)), nfull) if args.t1 < 1e8 else nfull
    print(f"加载 {vp.name}: {nfull} 帧 @ {fps:.2f}fps {fw}x{fh}  窗口[{f0},{f1})")
    print(f"  自适应: scale={scale:.3f}  MIN_AREA={min_area}  JUMP={jump_px:.0f}px")

    P = dict(
        blue_low=(args.blue_hmin, args.blue_smin, args.blue_vmin),
        blue_high=(args.blue_hmax, 255, 255),
        pink1_low=(0, args.pink_smin, args.pink_vmin), pink1_high=(15, 255, 255),
        pink2_low=(160, args.pink_smin, args.pink_vmin), pink2_high=(180, 255, 255),
        xmin=args.gate_xmin, xmax=args.gate_xmax,
        ymin=args.gate_ymin, ymax=args.gate_ymax, y_gate=y_gate,
    )

    umaps = None if args.no_undistort else build_undistort_maps(fw, fh, args.camera)

    pink_lists, blue_lists = [], []
    raw_frames = {}
    save_idx = set(int(round(k)) for k in np.linspace(f0, f1 - 1, 6))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if umaps is not None:
            frame = cv2.remap(frame, umaps[0], umaps[1], cv2.INTER_LINEAR)
        if args.detector == "lab":
            pink_g, blue_g = detect_all_lab(frame, P, min_area)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pink_g, blue_g = detect_all(hsv, P, min_area)
        pink_lists.append(pink_g); blue_lists.append(blue_g)
        if fi in save_idx:
            raw_frames[fi - f0] = frame.copy()
        fi += 1
    cap.release()
    N = len(blue_lists)

    # 临时 m2 = 每帧最大粉斑，仅用于（a）给 m1 的 Viterbi 当 anchor、（b）自标定 link。
    # 真正的 m2 在 m1 定下来后用刚性+时序 Viterbi 重选，不被「闯入的大块粉色（手）」带偏。
    m2_prov = [(max(p, key=lambda t: t[0])[1], max(p, key=lambda t: t[0])[2]) if p else None
               for p in pink_lists]

    hit2_raw = np.mean([m is not None for m in m2_prov])
    hit1_raw = np.mean([len(c) > 0 for c in blue_lists])

    # 自标定刚性下臂距离：所有「蓝候选→临时 m2」间距里最密的那一簇就是刚性下臂长。
    # 用滑窗找众数，对「恰好也落在 link 距离上的假阳性」鲁棒（真 m1 几乎帧帧在簇里，
    # 假点散落各处；即便某假点也≈link 距离，只会强化估计、不会带偏长度，Viterbi 再去消歧）。
    if args.link_px > 0:
        link_px = args.link_px
        print(f"  LINK_PX = {link_px:.0f}px（CLI 指定）")
    else:
        ds = []
        for i in range(N):
            if m2_prov[i] is None:
                continue
            mx, my = m2_prov[i]
            for a, x, y in blue_lists[i]:
                ds.append(((x - mx)**2 + (y - my)**2) ** 0.5)
        if len(ds) < 10:
            raise RuntimeError(f"自标定 LINK_PX 失败：候选-m2 间距仅 {len(ds)} 个，请用 --link-px 指定")
        ds = np.array(sorted(ds))
        win = max(30.0 * scale, 12.0)
        best_c, best_l, best_std = -1, 0.0, 0.0
        for d in ds:
            sel = ds[(ds >= d - win) & (ds <= d + win)]
            if len(sel) > best_c:
                best_c, best_l, best_std = len(sel), float(np.median(sel)), float(np.std(sel))
        link_px = best_l
        print(f"  LINK_PX = {link_px:.1f}px（自标定众数，簇内 {best_c}/{len(ds)} 个间距，"
              f"窗±{win:.0f}px，std {best_std:.1f}）")

    # 第一遍：以临时 m2 为 anchor 选 m1（m1 候选干净、转移项主导，临时 m2 的偶发错点不影响）
    m1_path = track_joint(blue_lists, m2_prov, link_px, lam=1.0)
    m1_xy = [(p[0], p[1]) if p else None for p in m1_path]
    # 第二遍：以定下来的 m1 为 anchor、刚性 link 约束重选 m2 —— 闯入的大块粉色（手）离 m1
    # 远（发射代价大）会被否决，Viterbi 锁回真实摆球。这一步替代了「最大粉斑」的贪心选法。
    m2_path = track_joint(pink_lists, m1_xy, link_px, lam=1.0)

    df = pd.DataFrame({
        "frame": np.arange(N) + f0, "t": (np.arange(N) + f0) / fps,
        "x_m1_px": [p[0] if p else np.nan for p in m1_path],
        "y_m1_px": [p[1] if p else np.nan for p in m1_path],
        "x_m2_px": [p[0] if p else np.nan for p in m2_path],
        "y_m2_px": [p[1] if p else np.nan for p in m2_path],
        "area_m1": [p[2] if p else 0 for p in m1_path],
        "area_m2": [p[2] if p else 0 for p in m2_path],
    })

    def jump_outliers(xcol, ycol):
        xv = df[xcol].values.copy(); yv = df[ycol].values.copy()
        bad = np.zeros(N, bool)
        for i in range(N):
            pj = i - 1; nj = i + 1
            dp = np.hypot(xv[i] - xv[pj], yv[i] - yv[pj]) if pj >= 0 and not np.isnan(xv[pj]) else 0
            dn = np.hypot(xv[i] - xv[nj], yv[i] - yv[nj]) if nj < N and not np.isnan(xv[nj]) else 0
            if dp > jump_px and dn > jump_px:
                bad[i] = True
        return bad

    bad1 = jump_outliers("x_m1_px", "y_m1_px")
    df.loc[bad1, ["x_m1_px", "y_m1_px"]] = np.nan
    # link 一致性兜底：m2 重选后仍有「与 m1 间距严重偏离 link」的帧，判为 m2 坏点剔除并插值
    link_all = np.hypot(df.x_m2_px - df.x_m1_px, df.y_m2_px - df.y_m1_px)
    link_tol = max(4.0 * (best_std if args.link_px <= 0 else 0.05 * link_px), 0.12 * link_px)
    bad2 = (np.abs(link_all - link_px) > link_tol).values
    df.loc[bad2, ["x_m2_px", "y_m2_px"]] = np.nan
    n_bad1 = int(bad1.sum()); n_bad2 = int(bad2.sum())

    print(f"  命中率：m2 红/粉 {hit2_raw*100:.1f}%   m1 蓝(门限内有候选) {hit1_raw*100:.1f}%")
    print(f"  双向刚性+时序 Viterbi：剔 m1 限速离群 {n_bad1} 帧、m2 破 link 离群 {n_bad2} 帧"
          f"（link 容差 ±{link_tol:.0f}px，均插值补回）")

    both = df.dropna(subset=["x_m1_px", "x_m2_px"])
    link = np.sqrt((both.x_m2_px - both.x_m1_px)**2 + (both.y_m2_px - both.y_m1_px)**2)
    print(f"  m1-m2 间距：中位 {link.median():.1f}px  std {link.std():.1f}  "
          f"范围 [{link.min():.0f}, {link.max():.0f}]")

    for c in ["x_m1_px", "y_m1_px", "x_m2_px", "y_m2_px"]:
        df[c] = df[c].interpolate(limit_direction="both")

    m1pts = df[["x_m1_px", "y_m1_px"]].dropna().values
    cx, cy, r_px, rms, keep = fit_circle_robust(m1pts)
    print(f"  圆拟合 pivot=({cx:.0f},{cy:.0f})  L1={r_px:.0f}px  "
          f"RMS残差={rms:.1f}px  内点 {keep.sum()}/{len(keep)}")
    ratio = link.median() / r_px
    print(f"  L2/L1 像素比 ≈ {ratio:.2f}（物理设计 L2≈0.20 / L1≈0.25 = 0.80 参照）")

    # ---- 合理性守卫（2026-09-07 补）----
    # 阈值选错时本脚本会【静默出错】而不是报错：假蓝斑混进候选集后，命中率、
    # Δθ 中位这些内建指标反而全绿，但支点会被拟合到画面外、L2/L1 严重偏离设计值。
    # 例：14xx 用默认 --blue-vmin 40 跑，日志显示「命中率 100%、L2/L1=0.78」，
    # 实际 IMG_1432 释放角 111.2°→11.1°。故在写盘前把这三件事变成硬失败。
    bad = []
    if not (0 <= cx <= fw and 0 <= cy <= fh):
        bad.append(f"支点 ({cx:.0f},{cy:.0f}) 落在画面 {fw}x{fh} 之外")
    if rms > 10.0:
        bad.append(f"圆拟合 RMS 残差 {rms:.1f}px > 10px")
    if not (0.60 <= ratio <= 1.00):
        bad.append(f"L2/L1 = {ratio:.2f} 偏离设计值 0.80 超过 ±25%")
    if bad:
        msg = ("追踪结果未通过合理性检查，已拒绝写盘：\n    - " + "\n    - ".join(bad) +
               "\n  多半是颜色阈值不对导致假斑混入候选集（14xx 那批深藏青背景需要"
               "--blue-vmin 160 --blue-smin 55 --gate-xmax 1780，见 tracking/reproduce_29.sh）。"
               "\n  确认无误要强行写盘，加 --force。")
        if not args.force:
            raise SystemExit("✗ " + msg)
        print("  ⚠ " + msg.replace("已拒绝写盘", "但 --force 已指定，仍写盘"))

    th1 = np.arctan2(df.x_m1_px - cx, df.y_m1_px - cy)
    th2 = np.arctan2(df.x_m2_px - df.x_m1_px, df.y_m2_px - df.y_m1_px)
    th1 = np.unwrap(th1.values); th2 = np.unwrap(th2.values)
    dt = 1.0 / fps
    w1 = np.gradient(th1, dt); w2 = np.gradient(th2, dt)
    df["th1"], df["th2"], df["w1"], df["w2"] = th1, th2, w1, w2

    d1 = np.abs(np.diff(np.degrees(th1))); d2 = np.abs(np.diff(np.degrees(th2)))
    print(f"  Δθ1/帧 中位{np.median(d1):.2f}° 95%{np.percentile(d1,95):.1f}° 最大{d1.max():.1f}°  "
          f">20°帧={int((d1>20).sum())}")
    print(f"  Δθ2/帧 中位{np.median(d2):.2f}° 95%{np.percentile(d2,95):.1f}° 最大{d2.max():.1f}°  "
          f">20°帧={int((d2>20).sum())}")
    print(f"  |w1|max {np.abs(w1).max():.1f}  |w2|max {np.abs(w2).max():.1f} rad/s")

    csv = OUT / f"tracked_{args.tag}.csv"
    df.to_csv(csv, index=False)
    print(f"  写出 {csv}")

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
    ax[1, 0].set_title("m1/m2 traj (px) + pivot"); ax[1, 0].invert_yaxis()
    ax[1, 0].set_aspect("equal"); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(both.t.values, link.values, lw=1)
    ax[1, 1].axhline(link_px, color="g", ls="--", lw=1)
    ax[1, 1].set_title("m1-m2 link length (px) — rigidity check"); ax[1, 1].grid(alpha=.3)
    fig.tight_layout()
    qc = OUT / f"qc_{args.tag}.png"
    fig.savefig(qc, dpi=130, facecolor="white"); plt.close(fig)
    print(f"  写出 {qc}")

    for k, frame in sorted(raw_frames.items()):
        vis = frame.copy()
        row = df.iloc[k]
        cv2.circle(vis, (int(cx), int(cy)), max(int(18*scale*2), 8), (0, 255, 0), 3)
        if not np.isnan(row.x_m1_px):
            p1 = (int(row.x_m1_px), int(row.y_m1_px))
            cv2.circle(vis, p1, max(int(26*scale*2), 10), (255, 0, 0), 3)
            cv2.line(vis, (int(cx), int(cy)), p1, (0, 255, 255), 2)
            if not np.isnan(row.x_m2_px):
                p2 = (int(row.x_m2_px), int(row.y_m2_px))
                cv2.circle(vis, p2, max(int(26*scale*2), 10), (0, 0, 255), 3)
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
        small = cv2.resize(vis, (fw // 2, fh // 2))
        cv2.imwrite(str(OUT / f"overlay_{args.tag}_{row.frame:06.0f}.png"), small)
    print(f"  写出 {len(raw_frames)} 张叠加抽帧 overlay_{args.tag}_*.png")


if __name__ == "__main__":
    main()

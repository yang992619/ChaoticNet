"""相机内参 / 畸变标定（棋盘格）。

读一个文件夹里的棋盘格标定照片，检测 9×7 内角点，用 cv2.calibrateCamera
求出相机内参矩阵 mtx + 畸变系数 dist，保存到 .npz，并打印重投影误差。

棋盘规格须与打印件一致：9×7 内角点（10×8 格），每格 20mm。

用法：
  python tools/calibrate_camera.py --images 标定照文件夹
  python tools/calibrate_camera.py --images cal_shots --out camera_params.npz --square 20

输出：
  camera_params.npz（含 mtx, dist, image_size）
  重投影误差（<1.0 像素为好，<0.5 很好）；误差大说明照片角度不够多或有模糊。
"""
import argparse
import glob
import os
import numpy as np
import cv2

NX, NY = 9, 7          # 内角点（横 9 竖 7）
SQUARE_MM = 20.0       # 每格边长 mm，与打印件一致


def main():
    ap = argparse.ArgumentParser(description="棋盘格相机标定")
    ap.add_argument("--images", help="标定照片所在文件夹")
    ap.add_argument("--video", help="标定视频文件（推荐：与拍摆用同一视频模式，避免照片/视频视野不一致）")
    ap.add_argument("--out", default="camera_params.npz", help="输出参数文件")
    ap.add_argument("--square", type=float, default=SQUARE_MM, help="每格边长 mm")
    ap.add_argument("--max-frames", type=int, default=60, help="从视频均匀抽取的候选帧数")
    ap.add_argument("--fix-k3", action="store_true",
                    help="固定 k3=0 用 4 参畸变模型。标定板占画面小、角点条件数差时更稳")
    args = ap.parse_args()

    if not args.images and not args.video:
        print("请用 --images 文件夹 或 --video 视频文件 之一")
        return

    # 棋盘平面世界坐标（z=0），单位 mm
    objp = np.zeros((NX * NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * args.square

    def frame_source():
        """统一帧来源：视频均匀抽帧 或 文件夹图片。yield (标签, BGR图)。"""
        if args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"打不开视频：{args.video}")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            step = max(1, total // max(1, args.max_frames))
            print(f"视频共 {total} 帧，每 {step} 帧抽 1 张做候选")
            idx = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield f"帧{idx:05d}", img
                idx += 1
            cap.release()
        else:
            files = sorted(glob.glob(os.path.join(args.images, "*")))
            if not files:
                print(f"文件夹里没有照片：{args.images}")
            for f in files:
                img = cv2.imread(f)
                if img is not None:
                    yield os.path.basename(f), img

    objpoints, imgpoints = [], []
    w = h = None
    used = 0
    checked = 0
    skipped_size = []      # 尺寸与首张不一致而被拒的
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for label, img in frame_source():
        checked += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape
        # calibrateCamera 只接受一个统一的 imageSize；混入不同尺寸的图会让内参无意义。
        # 2026-09-05 踩过：微信压缩把同一批标定照压成 1708x962 与 2276x1280 两种，
        # 旧代码每轮覆盖 h,w，最后拿末张的尺寸去标定 21 张混尺寸角点，重投影误差报 0.048px
        # （看着极好），过滤成同尺寸重跑实为 0.2185px —— 假数。
        if w is None:
            h, w = gh, gw
        elif (gw, gh) != (w, h):
            skipped_size.append((label, gw, gh))
            continue
        ok, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
        if not ok:
            if args.images:   # 图片模式逐张提示；视频抽帧太多不刷屏
                print(f"  ✗ 未检测到棋盘角点：{label}（废弃）")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1
        print(f"  ✓ {label}")
    print(f"\n检查 {checked} 张候选，检出棋盘 {used} 张")
    if skipped_size:
        print(f"  ⚠ 另有 {len(skipped_size)} 张因尺寸与首张 {w}x{h} 不符被拒："
              f"{', '.join(f'{n}({a}x{b})' for n, a, b in skipped_size[:5])}"
              f"{' ...' if len(skipped_size) > 5 else ''}")
        print("    多半是照片经过了压缩/转发。请用数据线导原图，或改用 --video。")

    if used < 5:
        print(f"\n可用帧只有 {used} 张，太少（建议 ≥10 张不同角度）。标定中止。")
        return

    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, flags=flags)

    # 重投影误差（越小越准），报 RMS 像素。
    # 2026-09-05 修：旧写法 cv2.norm(L2)/len(proj) 是「平方和开根 ÷ 点数」，
    # 比真正的 RMS 小了 sqrt(点数) 倍（9×7=63 个角点 → 小 7.9 倍），
    # 会把一次 0.22px 的标定报成 0.027px，看着极好实则不然。除以 sqrt(N) 才是 RMS。
    total_sq, total_n = 0.0, 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_sq += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) ** 2
        total_n += len(proj)
    reproj_err = float(np.sqrt(total_sq / total_n))

    np.savez(args.out, mtx=mtx, dist=dist, image_size=np.array([w, h]))

    print(f"\n标定完成：用了 {used} 张照片，画面 {w}×{h}")
    print(f"重投影误差 = {reproj_err:.3f} 像素  （<1.0 为好，<0.5 很好）")
    if reproj_err > 1.0:
        print("  ⚠ 误差偏大：多拍些不同角度的照片、确保对焦清晰、棋盘平整无翘曲后重标。")

    # ---- 姿态多样性诊断 ----
    # 低重投影误差 ≠ 标定可信。2026-09-05 踩过：棋盘只占画面 2%、距离几乎不变、
    # 一半以上近似正对，误差看着不错但畸变系数拟到了噪声上，fx 与上一次差了 15%。
    # 畸变只能从画面边缘的弯曲量测出来，棋盘太小就采样不到边缘，参数必然不可信。
    tilts, dists, areas, cells = [], [], [], np.zeros((4, 4), int)
    allpts = []
    for r, t, c in zip(rvecs, tvecs, imgpoints):
        R, _ = cv2.Rodrigues(r)
        tilts.append(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))   # 板法线与光轴夹角
        dists.append(float(np.linalg.norm(t)))
        p = c.reshape(-1, 2)
        allpts.append(p)
        areas.append(cv2.contourArea(cv2.convexHull(p.astype(np.float32))) / (w * h))
        cells[min(3, int(p[:, 1].mean() / h * 4)), min(3, int(p[:, 0].mean() / w * 4))] += 1
    tilts, dists, areas = np.array(tilts), np.array(dists), np.array(areas) * 100
    span = dists.max() / dists.min() if dists.min() > 0 else 1.0

    # 决定畸变能否被约束的，不是单张里棋盘多大，而是【全体角点的并集】有没有铺到画面边缘。
    # 畸变在中心为零、边缘最强；角点只挤在中间时，k1/k2/k3 是在拟合噪声。
    # 小标定板照样可以合格 —— 多走几个位置把边角占满即可。
    P = np.vstack(allpts)
    union_cov = cv2.contourArea(cv2.convexHull(P.astype(np.float32))) / (w * h) * 100
    # 归一化边缘余量：0 = 角点触到画面边，1 = 全挤在中心
    margin = min(P[:, 0].min() / (w / 2), P[:, 1].min() / (h / 2),
                 (w - P[:, 0].max()) / (w / 2), (h - P[:, 1].max()) / (h / 2))
    r_norm = np.hypot((P[:, 0] - w / 2) / (w / 2), (P[:, 1] - h / 2) / (h / 2))

    ok_union = union_cov >= 60
    ok_margin = margin <= 0.15
    ok_tilt = (tilts > 30).sum() >= used * 0.4
    ok_span = span >= 1.6
    ok_cells = (cells == 0).sum() <= 2

    print("\n姿态多样性诊断（这一节比重投影误差更能说明标定可不可信）：")
    print(f"  角点并集覆盖 {union_cov:.0f}% 画面"
          f"   {'合格' if ok_union else '不合格，需 ≥60%（畸变靠边缘弯曲量测，角点挤在中间就测不出）'}")
    print(f"  最外角点余量 距边 {margin * 100:.0f}%（半宽半高归一）"
          f"   {'合格' if ok_margin else '不合格，需 ≤15%，即必须把板子推到画面四边四角'}")
    print(f"  倾斜角       中位 {np.median(tilts):.0f}°   >30° 的 {(tilts > 30).sum()}/{used} 张"
          f"   {'合格' if ok_tilt else '不合格，需四成以上 >30°'}")
    print(f"  远近跨度     最远/最近 = {span:.2f}×"
          f"   {'合格' if ok_span else '不合格，需 ≥1.6×（否则焦距与距离分不开）'}")
    print(f"  十六宫格覆盖 空格 {(cells == 0).sum()}/16"
          f"   {'合格' if ok_cells else '不合格，四角必须拍到'}")
    print(f"  参考：单张棋盘占画面中位 {np.median(areas):.1f}%"
          f"（板子小不要紧，靠多走位置补；<3% 时单张位姿偏弱，建议多录些帧）")
    if not all([ok_union, ok_margin, ok_tilt, ok_span, ok_cells]):
        print("  ⚠ 有项不合格 —— 即便重投影误差很小，本次内参也不可信，建议按上面几条重拍。")
    if np.median(areas) < 5 and not args.fix_k3:
        print("  提示：棋盘占比偏小，畸变参数条件数差。可加 --fix-k3 用 4 参模型"
              "（k3=0），少一个自由度更稳（camera_params_新标定.npz 当初就是这么标的）。")

    print(f"\n内参矩阵 mtx 与畸变系数 dist 已存入 {args.out}")
    print("后续 track.py 可加载此文件对每帧做去畸变（undistort）以提高追踪精度。")


if __name__ == "__main__":
    main()

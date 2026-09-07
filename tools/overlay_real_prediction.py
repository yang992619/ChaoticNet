"""
tools/overlay_real_prediction.py — 真实录像上叠加模型零样本预测

把三模型(ESN/PINN/Hybrid)从实测反演初值出发的闭环预测，逐帧投影回真实视频像素，
叠在原始录像上：白=追踪真值，橙=ESN(纯数据)，绿=PINN，青=Hybrid(物理)。
预测序列直接复用 rc/real_validation_multi（与论文 summary_multi.csv 逐位一致）。

几何：每段视频用 m1 轨迹最小二乘拟合圆得 pivot/L1px，L2px 取链长中位。
角度→像素(竖直向下=0, 右为正, 屏幕y向下)：
  m1 = (cx + L1px*sinθ1, cy + L1px*cosθ1)
  m2 = (m1x + L2px*sinθ2, m1y + L2px*cosθ2)
底图用原始帧(摆居画面中心，畸变<0.5°可忽略，不做去畸变)。

用法：
  python3 tools/overlay_real_prediction.py --tag 1443 --mode sample
  python3 tools/overlay_real_prediction.py --tag 1443 --mode full --slow 2.0
"""
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rc'))
import real_validation_multi as rvm  # noqa: E402

TAU_L = 0.762
FONT_PATH = '/System/Library/Fonts/Hiragino Sans GB.ttc'

# BGR 配色
C_TRUTH = (235, 235, 235)   # 白：追踪真值
C_ESN = (40, 150, 255)      # 橙：ESN 纯数据
C_PINN = (70, 240, 70)      # 绿：PINN
C_HYB = (240, 230, 60)      # 青：Hybrid
C_PIVOT = (160, 160, 160)
MODEL_COLORS = {'ESN': C_ESN, 'PINN': C_PINN, 'Hybrid': C_HYB}


def fit_pivot(df):
    pts = df[['x_m1_px', 'y_m1_px']].dropna().values
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    L2 = float(np.median(np.hypot(df['x_m2_px'] - df['x_m1_px'],
                                  df['y_m2_px'] - df['y_m1_px'])))
    return cx, cy, r, L2


def project(cx, cy, L1, L2, th1, th2):
    m1 = (cx + L1 * np.sin(th1), cy + L1 * np.cos(th1))
    m2 = (m1[0] + L2 * np.sin(th2), m1[1] + L2 * np.cos(th2))
    return m1, m2


def draw_skel(img, cx, cy, m1, m2, color, thick=4, r1=13, r2=13, line=True):
    p0 = (int(round(cx)), int(round(cy)))
    p1 = (int(round(m1[0])), int(round(m1[1])))
    p2 = (int(round(m2[0])), int(round(m2[1])))
    if line:
        cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
    cv2.circle(img, p1, r1, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r2, color, -1, cv2.LINE_AA)
    cv2.circle(img, p1, r1, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.circle(img, p2, r2, (30, 30, 30), 2, cv2.LINE_AA)


def bgr2rgb(c):
    return (c[2], c[1], c[0])


def put_panel(frame_bgr, phase, models_info, tcur, rel_deg, tag, countdown=0.0):
    """中文读数面板。phase: 'observe' 观测阶段 / 'predict' 预测阶段。"""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, 'RGBA')
    f_title = ImageFont.truetype(FONT_PATH, 40)
    f_big = ImageFont.truetype(FONT_PATH, 38)
    f_mid = ImageFont.truetype(FONT_PATH, 32)
    f_sml = ImageFont.truetype(FONT_PATH, 24)

    # 顶部标题条
    d.rectangle([0, 0, img.width, 64], fill=(10, 10, 14, 205))
    d.text((24, 12), f'真实双摆 · 仿真训练「零样本」预测  (IMG_{tag}，释放角 {rel_deg:.0f}°)',
           font=f_title, fill=(245, 245, 245))

    px, py, pw = 24, 84, 600
    if phase == 'observe':
        ph = 150
        d.rectangle([px, py, px + pw, py + ph], fill=(10, 10, 14, 175))
        d.text((px + 18, py + 14), '① 观测阶段', font=f_big, fill=(255, 250, 205))
        d.text((px + 18, py + 66), '从视频反演初值 (θ1,θ2,ω1,ω2)', font=f_mid, fill=(225, 225, 225))
        d.text((px + 18, py + 108), f'预测将在 {countdown:.1f}s 后开始…', font=f_mid, fill=(180, 200, 255))
    else:
        ph = 70 + 56 * len(models_info)
        d.rectangle([px, py, px + pw, py + ph], fill=(10, 10, 14, 175))
        d.text((px + 18, py + 12), f'② 零样本预测   t = +{tcur:4.2f} s', font=f_big,
               fill=(255, 250, 205))
        yy = py + 70
        for name, color, dev, hz, off in models_info:
            rgb = bgr2rgb(color)
            d.ellipse([px + 18, yy + 8, px + 18 + 22, yy + 30], fill=rgb)
            col = rgb if not off else (140, 140, 140)
            d.text((px + 52, yy), name, font=f_mid, fill=col)
            d.text((px + 175, yy), f'偏差 {dev:5.1f}°', font=f_mid, fill=col)
            d.text((px + 365, yy), f'视界 {hz:.2f}s', font=f_mid, fill=col)
            if off:
                d.text((px + 520, yy), '脱靶', font=f_mid, fill=(255, 120, 110))
            yy += 56

    # 底部图例
    d.rectangle([0, img.height - 50, img.width, img.height], fill=(10, 10, 14, 195))
    d.text((24, img.height - 44),
           '白=真值(视频追踪)　橙=ESN纯数据　绿=PINN　青=Hybrid物理　|　偏差>10°判为脱靶 · τ_L≈0.76s',
           font=f_sml, fill=(220, 220, 220))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='1443')
    ap.add_argument('--mode', choices=['sample', 'full'], default='sample')
    ap.add_argument('--slow', type=float, default=2.0, help='慢放倍数')
    ap.add_argument('--lead', type=float, default=4.0)
    ap.add_argument('--pred', type=float, default=4.0)
    ap.add_argument('--models', nargs='+', default=['ESN', 'PINN', 'Hybrid'])
    a = ap.parse_args()
    tag = a.tag

    csv = ROOT / f'data/tracking/tracked_{tag}.csv'
    vid = ROOT / f'实测视频/源视频归档/IMG_{tag}.MOV'
    df = pd.read_csv(csv)
    cx, cy, L1px, L2px = fit_pivot(df)
    print(f'pivot=({cx:.1f},{cy:.1f}) L1px={L1px:.1f} L2px={L2px:.1f}')

    # 预测序列（复用论文管线）
    C = rvm.load_clip(tag, a.lead, a.pred)
    pinn_net, hyb_net = rvm.load_pinn(), rvm.load_hybrid()
    r = rvm.process(C, pinn_net, hyb_net)
    i0, npd, fps = C['lead_n'], C['pred_n'], C['fps']
    rel_deg = C['rel_deg']
    print(f'i0={i0} pred_n={npd} fps={fps:.2f}  '
          + '  '.join(f"{m}视界={r[m]['horizon']:.2f}s" for m in a.models))

    # 预测窗对应的源视频帧号
    frames_win = df['frame'].values[i0:i0 + npd].astype(int)

    cap = cv2.VideoCapture(str(vid))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def render(kk):
        """kk<0 观测阶段；kk>=0 预测帧 k=kk。"""
        ridx = i0 + kk
        fno = int(df['frame'].iloc[ridx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            return None
        # 真值骨架（CSV 原始检测像素）
        row = df.iloc[ridx]
        tm1 = (row['x_m1_px'], row['y_m1_px']); tm2 = (row['x_m2_px'], row['y_m2_px'])
        draw_skel(frame, cx, cy, tm1, tm2, C_TRUTH, thick=3, r1=10, r2=10)
        cv2.circle(frame, (int(cx), int(cy)), 7, C_PIVOT, -1, cv2.LINE_AA)
        if kk < 0:
            return put_panel(frame, 'observe', [], 0.0, rel_deg, tag,
                             countdown=-kk / fps)
        info = []
        for m in a.models:
            th1 = r[m]['th1'][kk]; th2 = r[m]['th2'][kk]
            m1, m2 = project(cx, cy, L1px, L2px, th1, th2)
            dev = r[m]['err'][kk]
            off = dev > rvm.THRESH_DEG
            draw_skel(frame, cx, cy, m1, m2, MODEL_COLORS[m], thick=4,
                      r1=12, r2=12, line=True)
            info.append((m, MODEL_COLORS[m], dev, r[m]['horizon'], off))
        return put_panel(frame, 'predict', info, kk / fps, rel_deg, tag)

    if a.mode == 'sample':
        for k in [0, int(0.5 * fps), int(1.0 * fps), int(2.0 * fps),
                  int(3.0 * fps), min(npd - 1, int(4.0 * fps))]:
            if k >= npd:
                continue
            fr = render(k)
            if fr is not None:
                cv2.imwrite(f'/tmp/overlay_{tag}_k{k:03d}.png', fr)
                print('样张 →', f'/tmp/overlay_{tag}_k{k:03d}.png', f'(t=+{k/fps:.2f}s)')
        cap.release()
        return

    # full：观测引入(1.5s) + 预测窗(pred) + 结尾定格(1.0s)
    lead_n = min(int(1.5 * fps), i0)
    freeze_n = int(1.0 * fps)
    out_raw = f'/tmp/overlay_{tag}_raw.mp4'
    fps_out = fps / a.slow
    vw = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (W, H))
    last = None
    total = lead_n + npd
    for i, kk in enumerate(range(-lead_n, npd)):
        fr = render(kk)
        if fr is None:
            continue
        last = fr
        vw.write(fr)
        if i % 30 == 0:
            print(f'  渲染 {i}/{total}')
    for _ in range(freeze_n):
        if last is not None:
            vw.write(last)
    vw.release()
    cap.release()
    print('原始输出 →', out_raw)


if __name__ == '__main__':
    main()

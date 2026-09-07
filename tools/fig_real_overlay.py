"""
tools/fig_real_overlay.py — 生成论文用「真实录像 × 零样本预测叠加」三联关键帧图

不带演示面板,纯骨架 + 时间标签 + 图例,出版风。素材段 1448(有效集、Hybrid 当家)。
复用 overlay_real_prediction 的几何与预测管线(与 summary_multi 逐位一致)。
输出 data/real_validation/figures/real_overlay_1448.png。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import overlay_real_prediction as ov  # noqa: E402

TAG = '1448'
TIMES = [0.0, 0.9, 1.7]   # 全锁定 / ESN 已脱靶·PINN 临界 / 仅 Hybrid 仍跟


def main():
    df = pd.read_csv(ROOT / f'data/tracking/tracked_{TAG}.csv')
    cx, cy, L1, L2 = ov.fit_pivot(df)
    C = ov.rvm.load_clip(TAG, 4.0, 4.0)
    r = ov.rvm.process(C, ov.rvm.load_pinn(), ov.rvm.load_hybrid())
    i0, fps, rel = C['lead_n'], C['fps'], C['rel_deg']
    cap = cv2.VideoCapture(str(ROOT / f'实测视频/源视频归档/IMG_{TAG}.MOV'))

    x0 = int(max(0, cx - 840)); x1 = int(min(1920, cx + 840))
    y0 = int(max(0, cy - 130)); y1 = int(min(1080, cy + 880))

    def frame_at(tt):
        kk = int(round(tt * fps)); ridx = i0 + kk
        fno = int(df['frame'].iloc[ridx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno); ok, fr = cap.read()
        row = df.iloc[ridx]
        tm1 = (row['x_m1_px'], row['y_m1_px']); tm2 = (row['x_m2_px'], row['y_m2_px'])
        # 白色真值在银灰金属杆上会消失，先画黑色描边再画白线，保证可辨识。
        ov.draw_skel(fr, cx, cy, tm1, tm2, (0, 0, 0), thick=9, r1=16, r2=16)
        ov.draw_skel(fr, cx, cy, tm1, tm2, ov.C_TRUTH, thick=5, r1=12, r2=12)
        for m in ['ESN', 'PINN', 'Hybrid']:
            th1 = r[m]['th1'][kk]; th2 = r[m]['th2'][kk]
            m1, m2 = ov.project(cx, cy, L1, L2, th1, th2)
            ov.draw_skel(fr, cx, cy, m1, m2, ov.MODEL_COLORS[m], thick=4, r1=12, r2=12)
        cv2.circle(fr, (int(cx), int(cy)), 7, ov.C_PIVOT, -1, cv2.LINE_AA)
        return fr[y0:y1, x0:x1]

    panels = [frame_at(t) for t in TIMES]
    cap.release()
    ph, pw = panels[0].shape[:2]
    gap, topbar, legend = 20, 88, 102
    W = pw * 3 + gap * 2
    H = ph + topbar + legend
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    F = ov.FONT_PATH
    fT = ImageFont.truetype(F, 52)
    fL = ImageFont.truetype(F, 44)
    caps = ['t = 0.0 s  三模型全锁定', 't = 0.9 s  ESN 脱靶,物理模型仍跟',
            't = 1.7 s  仅 Hybrid 接近真值']
    for i, (p, cap_t) in enumerate(zip(panels, caps)):
        px = i * (pw + gap)
        canvas.paste(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), (px, topbar))
        d.text((px + 10, 16), cap_t, font=fT, fill=(20, 20, 20))
    items = [((235, 235, 235), '真值 (视频追踪)'), ((255, 150, 40), 'ESN 纯数据'),
             ((70, 240, 70), 'PINN'), ((60, 230, 240), 'Hybrid 物理')]
    lx = 10; ly = topbar + ph + 27
    for col, name in items:
        d.ellipse([lx, ly, lx + 34, ly + 34], fill=col, outline=(0, 0, 0), width=2)
        d.text((lx + 46, ly - 5), name, font=fL, fill=(20, 20, 20))
        lx += 46 + int(d.textlength(name, font=fL)) + 82
    out = ROOT / 'data/real_validation/figures/real_overlay_1448.png'
    canvas.save(out)
    print('saved →', out, canvas.size)


if __name__ == '__main__':
    main()

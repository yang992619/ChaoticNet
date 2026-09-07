#!/usr/bin/env python3
"""
生成三个零件的 DXF 文件（R12 格式，加工厂 CAM 软件通用）

零件：
  P01 摆臂 1: 275×25, R12.5, 两 ⌀5 孔，孔心距 250
  P02 摆臂 2: 225×25, R12.5, 一 ⌀5 孔（上）+ 一 ⌀3.4 孔（下），孔心距 200
  P03 45# 钢配重盘: ⌀30, 中心 ⌀2.5 通孔（攻 M3 全通螺纹）

坐标原点：每件设在外形几何中心。
单位：mm
"""

from pathlib import Path

OUT = Path(__file__).parent / "dxf"
OUT.mkdir(exist_ok=True)


def dxf_line(x1, y1, x2, y2, layer="0"):
    return f"  0\nLINE\n  8\n{layer}\n 10\n{x1:.4f}\n 20\n{y1:.4f}\n 30\n0.0\n 11\n{x2:.4f}\n 21\n{y2:.4f}\n 31\n0.0\n"


def dxf_arc(cx, cy, r, start_deg, end_deg, layer="0"):
    return f"  0\nARC\n  8\n{layer}\n 10\n{cx:.4f}\n 20\n{cy:.4f}\n 30\n0.0\n 40\n{r:.4f}\n 50\n{start_deg:.4f}\n 51\n{end_deg:.4f}\n"


def dxf_circle(cx, cy, r, layer="0"):
    return f"  0\nCIRCLE\n  8\n{layer}\n 10\n{cx:.4f}\n 20\n{cy:.4f}\n 30\n0.0\n 40\n{r:.4f}\n"


def wrap(body: str) -> str:
    return "  0\nSECTION\n  2\nENTITIES\n" + body + "  0\nENDSEC\n  0\nEOF\n"


def rounded_arm(length, width, hole_d_left, hole_d_right):
    """中心在原点的圆角扁条 + 两端各一孔。
    hole_d_*: 孔径 mm（孔位在两端圆心：±(length/2 - width/2)）。
    """
    half_l = length / 2
    half_w = width / 2
    R = half_w  # 圆角半径 = 摆臂宽度的一半（端部是半圆）

    # 端部圆心 X 坐标
    cx_left = -half_l + R
    cx_right = half_l - R

    body = ""
    # 顶边 + 底边
    body += dxf_line(cx_left, half_w, cx_right, half_w)
    body += dxf_line(cx_left, -half_w, cx_right, -half_w)
    # 左半圆（从 90° → 270°，CCW）
    body += dxf_arc(cx_left, 0, R, 90, 270)
    # 右半圆（从 270° → 90°，CCW）
    body += dxf_arc(cx_right, 0, R, 270, 90)
    # 两端孔
    body += dxf_circle(cx_left, 0, hole_d_left / 2)
    body += dxf_circle(cx_right, 0, hole_d_right / 2)
    return wrap(body)


def disc_with_center_hole(outer_d, inner_d):
    body = dxf_circle(0, 0, outer_d / 2)
    body += dxf_circle(0, 0, inner_d / 2)
    return wrap(body)


def rect_plate(width, height, corner_r, holes):
    """中心在原点的圆角矩形 + 任意孔列表。
    holes: [(x, y, diameter), ...] 孔位（中心，⌀）
    """
    hw = width / 2
    hh = height / 2
    R = corner_r
    body = ""
    # 4 条直边（避开圆角）
    body += dxf_line(-hw + R, hh, hw - R, hh)          # top
    body += dxf_line(-hw + R, -hh, hw - R, -hh)        # bottom
    body += dxf_line(-hw, -hh + R, -hw, hh - R)        # left
    body += dxf_line(hw, -hh + R, hw, hh - R)          # right
    # 4 个圆角弧
    body += dxf_arc(hw - R, hh - R, R, 0, 90)          # top-right
    body += dxf_arc(-hw + R, hh - R, R, 90, 180)       # top-left
    body += dxf_arc(-hw + R, -hh + R, R, 180, 270)     # bottom-left
    body += dxf_arc(hw - R, -hh + R, R, 270, 360)      # bottom-right
    # 孔
    for x, y, d in holes:
        body += dxf_circle(x, y, d / 2)
    return wrap(body)


# ============================ 生成三个文件 ============================

# P01 摆臂 1：275 × 25，两端 ⌀10 H7 孔（轴承外圈过盈压入），孔心距 = 275 - 25 = 250 mm = L1
# 两个孔都是 MR105ZZ 轴承座（外径 10mm 与孔配合）
(OUT / "P01_arm_1.dxf").write_text(
    rounded_arm(length=275, width=25, hole_d_left=10.0, hole_d_right=10.0)
)

# P02 摆臂 2：225 × 25，上端 ⌀5 H7（关节轴销 P06 过盈压入），下端 ⌀3.4（M3 中配过孔，评审 4 号建议 3.2→3.4）
# 孔心距 = 225 - 25 = 200 mm = L2
(OUT / "P02_arm_2.dxf").write_text(
    rounded_arm(length=225, width=25, hole_d_left=5.0, hole_d_right=3.4)
)

# P03 45# 钢配重盘：⌀30 外圆 + ⌀2.5 中心通孔（M3 螺纹前的底孔）
(OUT / "P03_brass_weight.dxf").write_text(
    disc_with_center_hole(outer_d=30.0, inner_d=2.5)
)

# P04 主轴座：70 × 40 × 5（厚），R5 圆角；2 个 M5 过孔（⌀5.5，间距 40）+ 1 个 ⌀5 H7 轴孔
# 评审 3 号建议：60→70，M5 间距 30→40，抗弯矩 +33%
# 坐标系：板中心为原点，Y 轴向上（+ 是顶）
#   M5 孔在顶部，距顶 10mm → y = 20 - 10 = +10；x = ±20（间距 40）
#   ⌀5 H7 轴孔在底部，距顶 35mm → y = 20 - 35 = -15；x = 0
(OUT / "P04_pivot_bracket.dxf").write_text(
    rect_plate(
        width=70, height=40, corner_r=5,
        holes=[
            (-20, 10, 5.5),    # M5 安装孔 1
            (20, 10, 5.5),     # M5 安装孔 2
            (0, -15, 5.0),     # ⌀5 H7 轴孔（激光切完厂家二次绞孔到 H7）
        ],
    )
)

print("DXF 生成完成：")
for f in sorted(OUT.glob("*.dxf")):
    print(f"  {f.relative_to(Path(__file__).parent.parent.parent)}  ({f.stat().st_size} bytes)")

#!/usr/bin/env python3
"""
tools/live_demo.py — 答辩交互演示：评委输角度，三摆同框赛跑

评委用滑块设定双摆初始角度，按「放手」，屏幕上三个双摆从同一初值同框赛跑：
  · 真值 (裁判)  : sim/baseline_damped 带实测阻尼的高精度数值解 (RK45)
  · Hybrid (物理): 解析运动方程主干 + 残差网，扛得最久
  · ESN  (纯数据): 储池计算基线，很快崩飞 —— 反衬「物理帮了多少」

三种运行模式（同一脚本）：
  交互  : python3 tools/live_demo.py                 # macosx 弹窗，拖滑块亲手玩
  即时  : python3 tools/live_demo.py --play --a1 110 --a2 100   # 给定角度直接弹窗播
  导出  : python3 tools/live_demo.py --export out.mp4 --a1 110 --a2 100  # 无头出 mp4（铁兜底，嵌 PPT）
  彩蛋  : python3 tools/live_demo.py --butterfly --a1 110 --a2 100       # 蝴蝶效应：差 0.5° 两条真值

公平协议（与论文 lead/pred 一致）：观测窗 Tobs 秒只给 ESN 即时训 W_out；
ESN 与 Hybrid 都从同一个 state[i0] (i0=Tobs*fps) 起跑预测后段，同初值同窗口可比。

诚实声明：demo 的「真值」是高精度数值参考解，不是真实视频；严格真实零样本结果
(29 段实拍 ESN 0.167s / Hybrid 0.450s ≈ 2.7×) 见论文与实拍叠加视频。
"""
import sys
import argparse
import itertools
from pathlib import Path
from collections import deque

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'rc'))
sys.path.insert(0, str(ROOT / 'sim'))

# 摆长（几何，画图用），与 sim/baseline.py 一致
L1, L2 = 0.250, 0.200

# 深色主题配色（matplotlib RGB 0–1）
BG       = '#0d0d12'
C_TRUTH  = (0.90, 0.90, 0.93)
C_HYBRID = (0.22, 0.85, 0.93)   # 青
C_ESN    = (1.00, 0.55, 0.16)   # 橙
C_OFF    = (1.00, 0.30, 0.30)   # 脱靶红
C_TXT    = (0.86, 0.86, 0.90)
C_DIM    = (0.55, 0.55, 0.62)

THRESH_DEG = 10.0   # 视界阈值：角度欧氏偏差首超 10° 即「脱靶」

# 延迟 import 的全局（后端必须在 import pyplot 前定，见 _setup）
plt = FuncAnimation = FFMpegWriter = Slider = Button = TextBox = None
torch = bd = esn_mod = pinn_mod = hybrid_mod = None


def _setup(export):
    """选后端 + 延迟导入。matplotlib.use() 必须在 import pyplot 之前。
    注意：故意不 import real_validation_multi（它顶部硬设 Agg，会毁掉交互窗口），
    所需的 angle_err_deg/horizon_s/run_torch/run_esn 逻辑在本文件内联。"""
    global plt, FuncAnimation, FFMpegWriter, Slider, Button, TextBox
    global torch, bd, esn_mod, pinn_mod, hybrid_mod
    import matplotlib
    matplotlib.use('Agg' if export else 'macosx')
    import matplotlib.pyplot as _plt
    from matplotlib.animation import FuncAnimation as _FA, FFMpegWriter as _FW
    from matplotlib.widgets import Slider as _S, Button as _B, TextBox as _TB
    _plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS',
                                        'PingFang SC', 'Hiragino Sans GB']
    _plt.rcParams['axes.unicode_minus'] = False
    plt, FuncAnimation, FFMpegWriter = _plt, _FA, _FW
    Slider, Button, TextBox = _S, _B, _TB
    import torch as _t
    import baseline_damped as _bd
    import esn as _e
    import pinn as _p
    import hybrid as _h
    torch, bd, esn_mod, pinn_mod, hybrid_mod = _t, _bd, _e, _p, _h


# ---------------- 物理/模型（内联复用 real_validation_multi 的核心逻辑）----------------

def angle_err_deg(p1, p2, t1, t2):
    """两摆角度欧氏偏差（度），用 atan2 处理 ±π 周期。"""
    d1 = np.arctan2(np.sin(p1 - t1), np.cos(p1 - t1))
    d2 = np.arctan2(np.sin(p2 - t2), np.cos(p2 - t2))
    return np.degrees(np.sqrt(d1 ** 2 + d2 ** 2))


def horizon_s(err, fps, thr=THRESH_DEG):
    """有效预测视界（秒）+ 是否右删失（整窗都没超阈 → 视界=窗长，为下界）。"""
    over = err > thr
    if not over.any():
        return len(err) / fps, True
    return int(np.argmax(over)) / fps, False


def load_hybrid():
    net = hybrid_mod.HybridNet(hidden=64, n_layers=3,
                               residual_scale=hybrid_mod.RESIDUAL_SCALE)
    net.load_state_dict(torch.load(ROOT / 'data/hybrid/model.pt',
                                   map_location=hybrid_mod.DEVICE))
    net.to(hybrid_mod.DEVICE).eval()
    return net


def predict_hybrid(net, state, i0, n, dt):
    """从 state[i0] 起 RK4 自回归 n 帧（首帧取真值初值，其后 n-1 步积分）。"""
    s0 = state[i0].astype(np.float32)
    seq = pinn_mod.predict(net, s0, n - 1, dt=dt)
    th1 = np.concatenate([[s0[0]], seq[:, 0]])
    th2 = np.concatenate([[s0[2]], seq[:, 2]])
    return th1, th2


def predict_esn(df, i0, n, seed, n_res=400, sr=0.95, leak=0.3, ridge=1e-3):
    """观测窗 [0:i0] 即时岭回归训 W_out，再从 i0 闭环自回归 n 帧。"""
    U = esn_mod.encode(df)
    Y = np.roll(U, -1, axis=0)
    net = esn_mod.ESN(n_in=6, n_out=6, n_res=n_res, spectral_radius=sr,
                      leak=leak, ridge=ridge, seed=seed)
    washout = min(50, max(i0 // 3, 1))
    _, xf = net.train(U[:i0 - 1], Y[:i0 - 1], washout=washout)
    pred = net.predict(xf, U[i0 - 1], n)
    th1, _, th2, _ = esn_mod.decode(pred)
    return th1, th2


def _esn_pathology(e1, e2, fps):
    """量一条 ESN 闭环轨迹的「非物理丑态」：冻结比例 + 最大单帧跳变(rad)。
    冻结=自回归收敛到不动点（摆停半空，违反能量守恒）；跳变=瞬移闪现。
    二者都是数据匮乏单种子闭环的数值假象，是渲染级 bug 观感，与「ESN 该不该
    输」无关——把它们滤掉是去噪，不是替 ESN 美颜（反而把 ESN 调得更能打）。"""
    if np.isnan(e1).any() or np.isnan(e2).any():
        return 1.0, np.inf
    froz = float(freeze_mask(e1, e2, fps).mean())
    d1 = np.arctan2(np.sin(np.diff(e1)), np.cos(np.diff(e1)))  # 取最短弧，跨 ±π 不误判
    d2 = np.arctan2(np.sin(np.diff(e2)), np.cos(np.diff(e2)))
    jump = float(np.hypot(np.abs(d1), np.abs(d2)).max()) if len(d1) else 0.0
    return froz, jump


def pick_esn_seed(df, i0, n, fps, tt1, tt2, batches=(16, 40),
                  froz_max=0.10, jump_max=1.2):
    """评委可输任意角度，单一写死种子在部分初值会冻结/瞬移（非物理假象，看着像
    程序 bug）。这里在候选种子里挑「非病态的代表性种子」：
      1. 先滤掉冻结>10% / 单帧跳>1.2rad / NaN 的病态种子
         （1.2rad/帧 已实测远高于真值最快摆 ~0.7rad/帧，不会误杀真实快摆）；
      2. 干净池里按 ESN 视界排序取【中位】代表——不取最坏(=不夸大物理优势)、
         也不取最好(=不替纯数据放水)，是诚实的中庸代表；
      3. 逐批扩种子(16→40)：常见角度头 16 个就有干净候选(~0.7s)，只有极端高能
         角度(如 150/180、大角+ω 拉满)16 个全病态时才扩到 40(最慢 ~2.3s)，
         on_go 的「正在计算…」反馈兜住观感；
      4. 万一 40 个全病态(理论极端)，退化为最不病态的一个。
    论文头条 2.7×/7× 是多 seed 中位统计，本函数只影响 demo 的单轨观感，不碰口径。
    抽成模块级函数以便无头穷举验证。返回 (seed, e1, e2, info)。"""
    cand = []
    tried = 0
    for upto in batches:
        for s in range(tried, upto):
            e1, e2 = predict_esn(df, i0, n, s)
            froz, jump = _esn_pathology(e1, e2, fps)
            hz = horizon_s(angle_err_deg(e1, e2, tt1, tt2), fps)[0]
            cand.append((s, e1, e2, froz, jump, hz))
        tried = upto
        if any(c[3] < froz_max and c[4] < jump_max for c in cand):
            break                                  # 已有干净候选，不必再扩
    clean = [c for c in cand if c[3] < froz_max and c[4] < jump_max]
    # 全病态回退：按 (跳变, 冻结) 排序取最不病态的——跳变优先，因为瞬移是「无标注的
    # 肉眼闪现」(最像渲染 bug)，而冻结有诚实的「预测锁死」虚线可视化。宁可交一条会
    # 锁死(被如实标注)的，也不交一条会瞬移(看着像程序坏了)的。
    pool = clean if clean else sorted(cand, key=lambda c: (c[4], c[3]))[:1]
    pool = sorted(pool, key=lambda c: c[5])      # 按 ESN 视界排序取中位
    s, e1, e2, froz, jump, hz = pool[len(pool) // 2]
    info = dict(chosen=s, n_clean=len(clean), scanned=len(cand),
                froz=froz, jump=jump)
    return s, e1, e2, info


DEGEN_MSG = ('此初值在平衡点附近，系统几乎不动\n'
             '（自然下垂 / 刀尖倒立）。混沌要更大角度，试 θ1=120° θ2=100°')


def compute(th1_deg, th2_deg, w1=0.0, w2=0.0, tobs=2.0, tpred=6.0,
            fps=60, seed=-1, hyb_net=None):
    """跑真值 + 两模型，返回逐帧可视数据 + 视界。
    seed<0（默认）→ 自动挑代表性干净种子；seed>=0 → 写死该种子（可复现导出用）。"""
    if hyb_net is None:
        hyb_net = load_hybrid()
    df = bd.simulate(np.deg2rad(th1_deg), np.deg2rad(th2_deg),
                     w1_0=w1, w2_0=w2, t_end=tobs + tpred, fps=fps)
    dt = 1.0 / fps
    i0 = int(round(tobs * fps))
    n = len(df) - i0
    state = df[['th1', 'w1', 'th2', 'w2']].values
    tt1 = df['th1'].values[i0:i0 + n]
    tt2 = df['th2'].values[i0:i0 + n]

    h1, h2 = predict_hybrid(hyb_net, state, i0, n, dt)
    # 平衡点附近(真值前半窗几乎不动 → 稳定下垂或刀尖倒立)：赛跑无意义，跳过选种，
    # 屏上给诚实「平衡点」文案，绝不把「ESN 冻在倒立平衡点恰好不动」误报成「ESN 赢」。
    # 用「前半窗」而非整窗：真混沌初值一放手就动(零点几秒即超 15°)；只有平衡点附近
    # 才会静止数秒。刀尖倒立(±180)数值上能不物理地balance好几秒、末尾才崩，整窗判据会漏。
    truth_exc = angle_err_deg(tt1, tt2,
                              np.full_like(tt1, tt1[0]), np.full_like(tt2, tt2[0]))
    half = max(2, n // 2)
    degenerate = bool(np.max(truth_exc[:half]) < 15.0)
    if seed is not None and seed >= 0:
        e1, e2 = predict_esn(df, i0, n, seed)
        seed_info = dict(chosen=seed, n_clean=None, scanned=1, froz=None, jump=None)
    elif degenerate:
        e1, e2 = predict_esn(df, i0, n, 0); seed = 0
        seed_info = dict(chosen=0, n_clean=None, scanned=1, froz=None, jump=None)
    else:
        seed, e1, e2, seed_info = pick_esn_seed(df, i0, n, fps, tt1, tt2)

    err_h = angle_err_deg(h1, h2, tt1, tt2)
    err_e = angle_err_deg(e1, e2, tt1, tt2)
    hz_h, _ = horizon_s(err_h, fps)
    hz_e, _ = horizon_s(err_e, fps)

    # 播放从 t=0（评委设定的释放角）开始，而不是从观测窗末 state[i0] 起跑——否则
    # 屏上预览停在设定角、放手却从「2 秒混沌甩飞后的位姿」起步，看着像「换了一侧」。
    # 观测窗 [0:i0] 三摆都贴真值（模型在看，本就重合），i0 之后才各走各的预测：
    #   · Hybrid 预测首帧 h1[0]=state[i0]=真值，与观测窗无缝衔接；
    #   · ESN 预测首帧 e1[0] 是它对 i0 的预测（≈真值，含其首步小误差），衔接也顺滑。
    # 误差/锁死/瞬移在观测窗内全填 0/False（此时就是真值），视界数字仍只由预测段算，
    # 口径不变。展示总帧数 N = i0 + n（整条 0→Tobs+Tpred）。
    lead1 = df['th1'].values[:i0]
    lead2 = df['th2'].values[:i0]
    z_obs = np.zeros(i0)
    f_obs = np.zeros(i0, bool)
    N = i0 + n
    Tth1, Tth2 = df['th1'].values[:N], df['th2'].values[:N]   # 真值整条
    H1 = np.concatenate([lead1, h1]); H2 = np.concatenate([lead2, h2])
    E1 = np.concatenate([lead1, e1]); E2 = np.concatenate([lead2, e2])

    frames = {
        'Truth':  dict(th1=Tth1, th2=Tth2, err=np.zeros(N), hz=None, color=C_TRUTH,
                       frozen=np.zeros(N, bool), teleport=np.zeros(N, bool)),
        'Hybrid': dict(th1=H1, th2=H2, err=np.concatenate([z_obs, err_h]), hz=hz_h,
                       color=C_HYBRID,
                       frozen=np.concatenate([f_obs, freeze_mask(h1, h2, fps)]),
                       teleport=np.concatenate([f_obs, teleport_mask(h1, h2, fps)])),
        'ESN':    dict(th1=E1, th2=E2, err=np.concatenate([z_obs, err_e]), hz=hz_e,
                       color=C_ESN,
                       frozen=np.concatenate([f_obs, freeze_mask(e1, e2, fps)]),
                       teleport=np.concatenate([f_obs, teleport_mask(e1, e2, fps)])),
    }
    nan = any(np.isnan(v['th1']).any() or np.isnan(v['th2']).any()
              for v in frames.values())
    meta = dict(n=N, n_pred=n, fps=fps, dt=dt, i0=i0, tobs=tobs, tpred=tpred,
                th1_deg=th1_deg, th2_deg=th2_deg, w1=w1, w2=w2,
                seed=seed, seed_info=seed_info, nan=nan,
                degenerate=degenerate)
    return frames, meta


def pendulum_xy(th1, th2):
    """角度 → 物理坐标（matplotlib y 向上：竖直向下 θ=0 用 -cos）。"""
    m1 = (L1 * np.sin(th1), -L1 * np.cos(th1))
    m2 = (m1[0] + L2 * np.sin(th2), m1[1] - L2 * np.cos(th2))
    return m1, m2


def lerp_angle(a, b, f):
    """两角度按最短弧插值（防 ±π 跨越时反向甩）。相邻高帧率帧 Δθ 很小，安全。"""
    return a + f * np.arctan2(np.sin(b - a), np.cos(b - a))


def freeze_mask(th1, th2, fps, eps=1e-3, hold=0.25):
    """逐帧标出「锁死」：连续 hold 秒内两角几乎不动（< eps rad/帧）。
    ESN 闭环自回归在部分初值会收敛到不动点 → 橙杆静止；这是纯数据模型的真实
    失败模态（不是程序卡死）。检测出来后画面给虚线 + 「预测锁死」标，诚实呈现。
    摆动时单帧位移远大于 eps，连续 hold 秒同时静止只可能是真锁死（过零点瞬时
    速度小但不可能持续 0.25s），故不会误标摆动中的折返点。"""
    th1 = np.asarray(th1, float)
    th2 = np.asarray(th2, float)
    d = np.hypot(np.abs(np.diff(th1, prepend=th1[0])),
                 np.abs(np.diff(th2, prepend=th2[0])))
    still = d < eps
    w = max(2, int(round(hold * fps)))
    mask = np.zeros(len(th1), bool)
    run = 0
    for k in range(len(th1)):
        run = run + 1 if still[k] else 0
        mask[k] = run >= w
    return mask


def teleport_mask(th1, th2, fps, jump_max=1.2):
    """逐帧标出「瞬移发散」：单数据帧两角合位移 ≥ jump_max(rad/帧) → 摆球肉眼一闪
    挪到对侧。这是数据匮乏单种子闭环自回归的真实数值发散(不是渲染 bug)。阈值
    1.2rad/帧 已实测远高于真值最快摆 ~0.7rad/帧，不会误标真实快摆。一旦发散即视为
    全程崩溃 → 标记瞬移帧及其后所有帧，让末帧定格也如实显示「× 预测发散」。"""
    th1 = np.asarray(th1, float)
    th2 = np.asarray(th2, float)
    if np.isnan(th1).any() or np.isnan(th2).any():
        return np.ones(len(th1), bool)
    d1 = np.arctan2(np.sin(np.diff(th1, prepend=th1[0])),
                    np.cos(np.diff(th1, prepend=th1[0])))
    d2 = np.arctan2(np.sin(np.diff(th2, prepend=th2[0])),
                    np.cos(np.diff(th2, prepend=th2[0])))
    jump = np.hypot(np.abs(d1), np.abs(d2))
    mask = jump >= jump_max
    if mask.any():
        mask[int(np.argmax(mask)):] = True   # 发散后保持发散态
    return mask


# ---------------- 画面 ----------------

def _style_pend_ax(ax):
    ax.set_facecolor(BG)
    r = (L1 + L2) * 1.12
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r * 0.55)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot(0, 0, marker='o', ms=7, color=C_DIM, zorder=10)  # 枢轴


def build_scene(ax, interactive):
    """建三摆的 line/trail/文字 artist。返回更新所需句柄。"""
    _style_pend_ax(ax)
    lines, trails, trail_lines, trail_buf = {}, {}, {}, {}
    order = ['Truth', 'ESN', 'Hybrid']          # 画序：真值在底、Hybrid 在顶
    colors = {'Truth': C_TRUTH, 'ESN': C_ESN, 'Hybrid': C_HYBRID}
    widths = {'Truth': 5.0, 'ESN': 3.0, 'Hybrid': 3.0}
    for name in order:
        c = colors[name]
        (tl,) = ax.plot([], [], '-', color=c, lw=1.2, alpha=0.45, zorder=3)
        (ln,) = ax.plot([], [], 'o-', color=c, lw=widths[name],
                        ms=8 if name == 'Truth' else 7,
                        alpha=0.95, zorder=5 + order.index(name))
        lines[name] = ln
        trail_lines[name] = tl
        trail_buf[name] = deque(maxlen=50)
    return dict(lines=lines, trail_lines=trail_lines, trail_buf=trail_buf)


def build_textpanel(fig, x0, interactive):
    """右侧文字面板，返回可动态更新的 Text 句柄。"""
    T = {}
    fig.text(x0, 0.92, '混沌之眼 · 双摆预测对决', color='white',
             fontsize=17, fontweight='bold')
    T['t'] = fig.text(x0, 0.86, 't = +0.00 s', color=C_TXT, fontsize=14,
                      family='monospace')
    fig.text(x0, 0.80, '真值（裁判）', color=C_TRUTH, fontsize=13, fontweight='bold')
    fig.text(x0 + 0.012, 0.765, '高精度数值解 RK45', color=C_DIM, fontsize=10)

    T['hyb'] = fig.text(x0, 0.70, '', color=C_HYBRID, fontsize=13, fontweight='bold')
    T['hyb2'] = fig.text(x0 + 0.012, 0.665, '', color=C_TXT, fontsize=11)
    T['esn'] = fig.text(x0, 0.59, '', color=C_ESN, fontsize=13, fontweight='bold')
    T['esn2'] = fig.text(x0 + 0.012, 0.555, '', color=C_TXT, fontsize=11)

    T['verdict'] = fig.text(x0, 0.45, '', color=C_TXT, fontsize=12)
    # 诚实标注（固定小字，贴底）
    fig.text(x0, 0.13,
             '真值=高精度数值参考解(RK45)，非真实视频。\n'
             '严格真实零样本：29 段实拍\n'
             'ESN 0.167s / Hybrid 0.450s ≈ 2.7×，\n'
             '仿真内 ≈ 7×，详见论文。',
             color=C_DIM, fontsize=9, linespacing=1.5)
    return T


def verdict_text(hz_h, hz_e):
    """据本轮两条视界给【诚实】结论文案。这是项目核心价值观的落点：
      · 大角度真混沌 → Hybrid 显著扛得久 → 亮出「物理把可信窗口推得更远」(主线)；
      · 温和/低能初值系统接近规则、不混沌 → 纯数据也追得上甚至反超 → 如实说明，
        绝不硬凹「物理总赢」。反而点明「物理优势是混沌区现象」，是更强的科学叙事。

    「显著领先」判据 = 绝对差≥0.5s 【或】(赢家视界≥0.2s 非噪声级 且 ≥2× 对方)。
    用「或」而非旧版「且」：深混沌区两条视界都很短(Hybrid≈0.5s/ESN≈0.03s)，绝对
    差恒 <0.5s，旧「且 0.5s」会把 4×~15× 的真物理胜误降级成「旗鼓相当」——既自打
    主线、又和屏上视界数字打架、还随角度增大单调性反转。加「赢家≥0.2s」噪声闸：
    两边都瞬崩(各 1~2 帧)时谁都不算赢，如实说「都很快失准」，不假宣物理胜利。
    比例阈值取 2× 偏保守(宁可少宣称不夸大)，与诚实优先一致。论文头条 2.7×/7× 是
    多 seed 中位统计，与单轮 demo 口径不同，故此处只给定性结论 + 两条绝对视界。"""
    NOISE = 0.2                         # 视界 <0.2s(~12帧) 视作噪声级，不据此宣称谁赢
    nums = f'Hybrid {hz_h:.2f}s  vs  ESN {hz_e:.2f}s'
    phys = (hz_h - hz_e >= 0.5) or (hz_h >= NOISE and hz_h >= 2.0 * hz_e)
    esnw = (hz_e - hz_h >= 0.5) or (hz_e >= NOISE and hz_e >= 2.0 * hz_h)
    if phys and not esnw:
        rtag = f'  ×{hz_h / hz_e:.1f}' if hz_e > 1e-6 else '  （ESN 首帧即崩）'
        return '物理先验把可信窗口推得更远\n' + nums + rtag
    if esnw and not phys:
        return ('此初值能量低、运动接近规则，纯数据也追得上\n'
                + nums + '（物理优势在大角度混沌区）')
    # 旗鼓相当：按两条视界长短给诚实措辞，避免「不混沌」与「大角度混沌区」同框矛盾
    if max(hz_h, hz_e) >= 0.8:
        return ('此初值能量较低、运动接近规则，两者表现相当\n'
                + nums + '（物理优势在大角度混沌区）')
    return '此初值高度混沌，两者都很快失准、暂时相当\n' + nums


def make_renderer(frames, meta, scene, T):
    """返回 render(ph)：把画面渲染到「播放头」ph（浮点数据帧坐标）。
    位置在相邻数据帧间线性插值 → 视觉刷新与仿真帧率解耦，快放慢放都丝滑。
    所有物理已在 compute 预算好，render 只插值 + set_data，零积分。"""
    lines, trail_lines, trail_buf = (scene['lines'], scene['trail_lines'],
                                     scene['trail_buf'])
    fps = meta['fps']
    n = meta['n']
    i0 = meta.get('i0', 0)                     # 观测窗末=预测起跑帧（展示坐标，从 t=0 起算）
    hz = {k: frames[k]['hz'] for k in frames}
    last_idx = {name: -1 for name in lines}   # 尾迹按数据帧采点，与视觉帧率解耦

    def sample(name, ph):
        i = int(ph)
        if i >= n - 1:
            return frames[name]['th1'][n - 1], frames[name]['th2'][n - 1]
        f = ph - i
        a1, a2 = frames[name]['th1'], frames[name]['th2']
        return lerp_angle(a1[i], a1[i + 1], f), lerp_angle(a2[i], a2[i + 1], f)

    def setline(name, ph, k):
        th1, th2 = sample(name, ph)
        m1, m2 = pendulum_xy(th1, th2)
        lines[name].set_data([0, m1[0], m2[0]], [0, m1[1], m2[1]])
        if k > last_idx[name]:                 # 只在跨过新数据帧时落尾迹点
            last_idx[name] = k
            buf = trail_buf[name]
            buf.append(m2)
            arr = np.array(buf)
            trail_lines[name].set_data(arr[:, 0], arr[:, 1])

    def label(name, k):
        dev = frames[name]['err'][k]
        off = dev > THRESH_DEG
        locked = off and bool(frames[name]['frozen'][k])      # 脱靶 + 锁死到不动点
        blown = off and bool(frames[name]['teleport'][k])     # 脱靶 + 瞬移发散（非物理）
        ident = frames[name]['color']
        ln = lines[name]
        # 摆杆始终保持身份色（青=Hybrid / 橙=ESN）。脱靶只用「红圈套住摆球 + 整根变暗」
        # 提示，绝不把整根染红——否则两条都脱靶时会糊成一团红、分不清谁是谁。
        # 锁死叠虚线、瞬移叠点线 → 静止/炸飞的杆一眼读成「模型预测崩溃」的两种失败模态，
        # 不会误以为程序卡死或渲染 bug（瞬移本是数据匮乏闭环的真实数值发散，如实标）。
        ln.set_color(ident)
        if off:
            ln.set_alpha(0.5)
            ln.set_markeredgecolor(C_OFF)
            ln.set_markeredgewidth(2.4)
            ln.set_linestyle((0, (4, 3)) if locked else
                             (0, (1, 2)) if blown else '-')
        else:
            ln.set_alpha(0.95)
            ln.set_markeredgecolor('none')
            ln.set_linestyle('-')
        # 用 U+00D7 乘号（几乎所有字体都含），避免缺字方框
        tag = (('   × 预测锁死' if locked else '   × 预测发散' if blown
                else '   × 脱靶') if off else '')
        return dev, off, tag

    def render(ph):
        k = min(int(round(ph)), n - 1)
        obs = k < i0          # 观测窗：三摆贴真值重合、预测还没起跑（放手即从设定角起步）
        setline('Truth', ph, k)
        for name, key, key2 in [('Hybrid', 'hyb', 'hyb2'),
                                ('ESN', 'esn', 'esn2')]:
            setline(name, ph, k)
            dev, off, tag = label(name, k)
            cn = '物理 Hybrid' if name == 'Hybrid' else '纯数据 ESN'
            if obs:
                # 观测段三摆与真值同步，不打视界/偏差（那是预测段的事），如实说「观测中」
                T[key].set_text(f'{cn}   观测中')
                T[key].set_color(frames[name]['color'])
                T[key2].set_text('与真值同步')
                T[key2].set_color(C_DIM)
            else:
                txt_col = C_OFF if off else frames[name]['color']  # 文字脱靶变红报警（带名字不会混）
                # 退化平衡点：真值整窗几乎不动、赛跑无意义，绝不把「ESN 恰好冻在平衡点
                # → 视界 6s」误读成「ESN 完胜」。屏蔽视界数字，统一显「—（平衡点）」，
                # 结论由 DEGEN_MSG 兜。其余初值照常打两条真实视界。
                hztxt = '—（平衡点）' if meta.get('degenerate') else f'{hz[name]:.2f}s'
                T[key].set_text(f'{cn}   视界 {hztxt}')
                T[key].set_color(txt_col)
                T[key2].set_text(f'偏差 {dev:6.1f}°{tag}')
                T[key2].set_color(txt_col)
        # 计时器 T['t'] 是 monospace（数字不跳动），该字体无中文字形 → 只放 ASCII；
        # 「观测窗 / 预测段」阶段提示由 verdict（黑体中文）和各模型标签（观测中→视界）承载。
        T['t'].set_text(f"t = +{ph / fps:5.2f} s")
        if obs:
            T['verdict'].set_text('三摆从设定角同步释放，模型先观测真值…\n'
                                  '预测从 t = %.1f s 起跑，看谁先脱靶' % (i0 / fps))
        else:
            if k >= n - 1:
                # 结论文案据本轮真实视界自适应（大角度物理赢 / 温和角如实说相当或反超），
                # 诚实优先，绝不硬凹。平衡点附近(赛跑无意义)单独给「平衡点」文案。
                # 统计口径(2.7×/7×)见角落诚实标注。
                if meta.get('degenerate'):
                    T['verdict'].set_text(DEGEN_MSG)
                else:
                    T['verdict'].set_text(verdict_text(hz['Hybrid'], hz['ESN']))
            else:
                T['verdict'].set_text('预测段开跑 · 看谁先脱靶')
        out = [lines[nm] for nm in lines] + [trail_lines[nm] for nm in trail_lines]
        return out + list(T.values())

    return render


def reset_scene(scene, T):
    for n in scene['lines']:
        ln = scene['lines'][n]
        ln.set_data([], [])
        ln.set_alpha(0.95)
        ln.set_markeredgecolor('none')   # 清掉上一轮脱靶留下的红圈
        ln.set_linestyle('-')            # 清掉上一轮锁死留下的虚线
        scene['trail_lines'][n].set_data([], [])
        scene['trail_buf'][n].clear()
    for k in ['hyb', 'hyb2', 'esn', 'esn2', 'verdict']:
        T[k].set_text('')
    T['t'].set_text('t = +0.00 s')


def _bind_slider_textbox(slider, tb, vmin, vmax, nd, on_change, lock):
    """把一个 Slider 和一个 TextBox 双向绑定（评委可拖、也可直接键入精确角度）：
      · 拖滑块  → 同步刷新输入框文字
      · 输入框回车 → 解析→夹到 [vmin,vmax]→写回滑块
    任一侧变化都回调 on_change（刷新三摆预览）。lock['v'] 做重入保护防回环：
    set_val 会同时触发 change/submit，靠这个锁让被动更新的一侧不再反向回调。
    输入非法（空/非数字）→ 静默恢复成滑块当前值的文字，绝不崩。
    抽成模块级函数是为了能无头单测（macosx 交互窗口本身没法无头验）。"""
    fmt = '{:.%df}' % nd

    def fmt_val(v):
        return fmt.format(float(v))

    def on_slide(val):
        if lock['v']:
            return
        lock['v'] = True
        try:
            tb.set_val(fmt_val(val))
        finally:
            lock['v'] = False
        on_change()

    def on_submit(text):
        if lock['v']:
            return
        try:
            v = float(text)
        except (ValueError, TypeError):
            lock['v'] = True
            try:
                tb.set_val(fmt_val(slider.val))   # 非法输入：回滚到当前滑块值
            finally:
                lock['v'] = False
            return
        v = min(max(v, vmin), vmax)               # 越界自动夹到量程
        lock['v'] = True
        try:
            tb.set_val(fmt_val(v))                 # 归一化显示（110→110.0）
            slider.set_val(v)
        finally:
            lock['v'] = False
        on_change()

    slider.on_changed(on_slide)
    tb.on_submit(on_submit)
    return on_slide, on_submit


def _setup_textbox_focus_blink(fig, textboxes, caret=(0.97, 0.97, 1.0),
                               accent=C_HYBRID, idle=(0.23, 0.23, 0.27)):
    """让评委一眼看出「正在编辑哪个输入框」。matplotlib 的 TextBox 默认光标是黑色
    竖线（color='k'），在我们深色框(#22222c)上几乎看不见 —— 这就是「输了数字没反馈」
    的根源。这里：① 把光标改成亮白加粗，点进框即现竖线；② 用计时器让聚焦框的光标
    按 ~0.5s 节拍闪烁（标准文本框观感）；③ 给聚焦框描一圈青色加粗边框。

    边框走 spine（坐标轴边线），不动 facecolor —— 避开 matplotlib 自己用 facecolor
    做 hover 变色的逻辑，二者不打架。光标可见性每次按键时 matplotlib 的 _rendercursor
    会重新置 True，故按键即显、停手才闪。macosx 交互后端专用；导出/无头无事件循环、
    计时器不触发，绝不影响 mp4。返回 timer 句柄（须被引用防 GC）。"""
    for tb in textboxes:
        tb.cursor.set_color(caret)
        tb.cursor.set_linewidth(1.8)

    def _border(tb, on):
        for sp in tb.ax.spines.values():
            sp.set_edgecolor(accent if on else idle)
            sp.set_linewidth(1.8 if on else 0.8)

    for tb in textboxes:
        _border(tb, False)
    last = {'focus': None, 'on': True}

    def tick(_=None):
        focused = next((tb for tb in textboxes
                        if getattr(tb, 'capturekeystrokes', False)), None)
        changed = False
        if focused is not last['focus']:                 # 焦点切换：重描所有框的边
            for tb in textboxes:
                _border(tb, tb is focused)
            last['focus'] = focused
            changed = True
        if focused is not None:                          # 聚焦中：翻转光标可见性 → 闪
            last['on'] = not last['on']
            focused.cursor.set_visible(last['on'])
            changed = True
        else:
            last['on'] = True
        if changed:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

    timer = fig.canvas.new_timer(interval=530)
    timer.add_callback(tick)
    timer.start()
    return timer


# ---------------- 三种模式 ----------------

def _speed_of(speed_ref):
    return speed_ref() if callable(speed_ref) else speed_ref


def drive_animation(fig, render, n, sim_fps, play_fps, speed_ref,
                    export=None, alive=None, freeze_s=1.0, meta_title=None,
                    progress_cb=None, on_finish=None):
    """统一播放驱动：视觉刷新固定在 play_fps（间隔不随速度变），speed 只决定
    每帧推进多少「数据帧」(step = sim_fps/play_fps × speed) → 任意速度都满帧丝滑。
    导出：speed 固定，预生成有限播放头序列；交互/即时：无限计时器 + 实时读 speed_ref。
    progress_cb(ph, n) 每帧回报播放进度（交互模式画进度条用）；on_finish() 到末帧时
    回调一次（交互模式把暂停键复位用）。暂停由外部对 ani.event_source 收发实现，
    停表时 ps['ph'] 原地保留 → 续播无缝，故 tick 自身无需知道暂停状态。"""
    interval = 1000.0 / play_fps

    def step(speed):
        return (sim_fps / play_fps) * max(speed, 1e-3)

    if export:
        s = step(_speed_of(speed_ref))
        phs = []
        ph = 0.0
        while ph < n - 1:
            phs.append(ph)
            ph += s
        phs.append(float(n - 1))
        phs += [float(n - 1)] * int(round(freeze_s * play_fps))   # 末帧定格便于截图
        ani = FuncAnimation(fig, render, frames=phs, interval=interval,
                            blit=False, repeat=False)
        out = Path(export)
        out.parent.mkdir(parents=True, exist_ok=True)
        md = {'title': meta_title} if meta_title else None
        ani.save(str(out), writer=FFMpegWriter(fps=play_fps, bitrate=4000, metadata=md))
        print(f'导出 → {out}  ({len(phs)} 帧 @ {play_fps}fps，约 {len(phs)/play_fps:.1f}s)')
        plt.close(fig)
        return ani

    # 交互/即时：浮点播放头按当前速度推进，到末帧定格并停表
    holder, ps = {}, {'ph': 0.0}

    def tick(_):
        if alive is not None and not alive():
            return []
        ph = min(ps['ph'], float(n - 1))
        arts = render(ph)
        if progress_cb is not None:
            try:
                progress_cb(ph, n)
            except Exception:
                pass
        if ps['ph'] >= n - 1:
            a = holder.get('ani')
            if a is not None:
                try:
                    a.event_source.stop()
                except Exception:
                    pass
            if on_finish is not None and not holder.get('finished'):
                holder['finished'] = True       # 只回调一次，防末帧反复触发
                try:
                    on_finish()
                except Exception:
                    pass
        else:
            ps['ph'] += step(_speed_of(speed_ref))
        return arts

    def seek(ph):
        """跳到任意播放头 ph 并立即渲染一帧（暂停/停表态下手动重绘，不靠计时器）。
        进度条拖动定位用：回拖离开末帧会清掉 finished 标，使其可再次续播到末帧。
        非有限值（指针拖出画布时 transform 会回 nan）直接丢弃，绝不让 nan 毒化播放头
        —— 否则其后每帧 render(nan) 抛错、ps['ph'] 恒为 nan，整轮播放软卡死只能重置。"""
        ph = float(ph)
        if not np.isfinite(ph):
            return None
        ph = min(max(ph, 0.0), float(n - 1))
        ps['ph'] = ph
        arts = render(ph)
        if progress_cb is not None:
            try:
                progress_cb(ph, n)
            except Exception:
                pass
        holder['finished'] = ph >= n - 1
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        return arts

    def seek_frac(frac):
        seek(min(max(float(frac), 0.0), 1.0) * (n - 1))

    ani = FuncAnimation(fig, tick, frames=itertools.count(), interval=interval,
                        blit=False, repeat=False, cache_frame_data=False)
    holder['ani'] = ani
    ani._demo_ctrl = dict(seek=seek, seek_frac=seek_frac, n=n)
    return ani


def run_animation(fig, ax, frames, meta, scene, T, export=None, play_fps=60,
                  alive=None, speed=1.0, progress_cb=None, on_finish=None):
    # 尾迹按 ~0.9s 时间窗采点（随仿真帧率定长度，慢放也不缩短）
    win = max(8, int(round(0.9 * meta['fps'])))
    for name in scene['trail_buf']:
        scene['trail_buf'][name] = deque(scene['trail_buf'][name], maxlen=win)
    render = make_renderer(frames, meta, scene, T)
    return drive_animation(fig, render, meta['n'], meta['fps'], play_fps,
                           speed, export=export, alive=alive,
                           meta_title='混沌之眼双摆预测对决',
                           progress_cb=progress_cb, on_finish=on_finish)


def mode_export(a):
    _setup(export=True)
    frames, meta = compute(a.a1, a.a2, a.w1, a.w2, a.tobs, a.tpred, a.fps, a.seed)
    if meta['nan']:
        print('⚠ 轨迹含 NaN（初值过激/数值发散），换个角度再试')
    fig = plt.figure(figsize=(12, 8.5), facecolor=BG)
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.92])
    scene = build_scene(ax, interactive=False)
    T = build_textpanel(fig, 0.66, interactive=False)
    _print_result(meta, frames)
    run_animation(fig, ax, frames, meta, scene, T, export=a.export,
                  play_fps=a.play_fps, speed=a.speed)


def mode_play(a):
    _setup(export=False)
    frames, meta = compute(a.a1, a.a2, a.w1, a.w2, a.tobs, a.tpred, a.fps, a.seed)
    fig = plt.figure(figsize=(12, 8.5), facecolor=BG)
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.92])
    scene = build_scene(ax, interactive=False)
    T = build_textpanel(fig, 0.66, interactive=False)
    _print_result(meta, frames)
    _ani = run_animation(fig, ax, frames, meta, scene, T, export=None,
                         play_fps=a.play_fps, speed=a.speed)
    plt.show()


def mode_interactive(a):
    _setup(export=False)
    hyb_net = load_hybrid()
    fig = plt.figure(figsize=(12, 8.5), facecolor=BG)
    ax = fig.add_axes([0.03, 0.34, 0.60, 0.63])
    scene = build_scene(ax, interactive=True)
    T = build_textpanel(fig, 0.66, interactive=True)

    def mkslider(y, label, vmin, vmax, vinit, valfmt='%.2f',
                 width=0.48, show_val=True):
        sax = fig.add_axes([0.10, y, width, 0.028])
        sax.set_facecolor('#22222c')
        s = Slider(sax, label, vmin, vmax, valinit=vinit, color=C_DIM, valfmt=valfmt)
        s.label.set_color('white'); s.label.set_fontsize(11)
        s.valtext.set_color('white')
        if not show_val:
            s.valtext.set_visible(False)   # 数值改由右侧输入框显示，避免重复
        return s

    def mktextbox(y, text):
        # 紧贴滑块右侧的精确输入框：评委可直接键入特定角度/角速度
        tax = fig.add_axes([0.485, y - 0.004, 0.075, 0.036])
        tb = TextBox(tax, '', initial=text, color='#22222c',
                     hovercolor='#2c2c38', textalignment='center')
        tb.text_disp.set_color('white'); tb.text_disp.set_fontsize(11)
        return tb

    # 角度/角速度滑块收窄到 0.36，腾出右侧 0.485 起放输入框；数值不再画在滑块尾
    s_th1 = mkslider(0.250, '上摆 θ1 (°)', -180, 180, a.a1,
                     valfmt='%.1f', width=0.36, show_val=False)
    s_th2 = mkslider(0.205, '下摆 θ2 (°)', -180, 180, a.a2,
                     valfmt='%.1f', width=0.36, show_val=False)
    s_w1 = mkslider(0.160, '初角速度 ω1', -6, 6, a.w1,
                    valfmt='%.2f', width=0.36, show_val=False)
    s_w2 = mkslider(0.115, '初角速度 ω2', -6, 6, a.w2,
                    valfmt='%.2f', width=0.36, show_val=False)
    # 播放速度：1×=实时；越小越慢（方便评委看清崩飞细节）。可播放前设、也可播放中实时拖。
    # 它不配输入框（不需要精确键入），保持全宽 + 显示数值。
    s_spd = mkslider(0.060, '播放速度', 0.1, 2.0, a.speed, valfmt='×%.2f')

    # 进度条：底部一条，播放中实时填充；橙色竖标=预测起跑点（观测窗与预测段的分界）。
    # 可拖动：在条上按住左键拖动即「擦洗」定位到任意时刻并定格（讲到 ESN 崩飞那一刻
    # 停住细说）。轴特意做高一点（视觉轨道仍居中 y=0.5），给评委更大的可点区域。
    pax = fig.add_axes([0.10, 0.015, 0.46, 0.034])
    pax.set_xlim(0, 1); pax.set_ylim(0, 1); pax.axis('off')
    pax.plot([0, 1], [0.5, 0.5], lw=6, color='#2a2a34', solid_capstyle='butt')  # 轨道
    (prog_line,) = pax.plot([0, 0], [0.5, 0.5], lw=6, color=C_HYBRID,
                            solid_capstyle='butt')                              # 已播
    (pred_div,) = pax.plot([0, 0], [0.05, 0.95], lw=1.6, color=C_ESN,
                           visible=False)                                       # 预测起跑标
    (playhead,) = pax.plot([0], [0.5], 'o', ms=11, color='white',
                           markeredgecolor=C_HYBRID, markeredgewidth=1.5,
                           zorder=6, visible=False)                            # 拖动手柄
    fig.text(0.045, 0.030, '进度', color='white', fontsize=11)
    prog_txt = fig.text(0.575, 0.030, '', color=C_DIM, fontsize=10,
                        family='monospace')

    t_th1 = mktextbox(0.250, f'{a.a1:.1f}')
    t_th2 = mktextbox(0.205, f'{a.a2:.1f}')
    t_w1 = mktextbox(0.160, f'{a.w1:.2f}')
    t_w2 = mktextbox(0.115, f'{a.w2:.2f}')

    state = {'ani': None, 'token': 0, 'paused': False, 'finished': False}
    scrub = {'active': False}   # 进度条擦洗活跃锁；_stop_current 一并复位防跨轮残留

    def update_progress(ph, n):
        """每帧回调：把播放头 ph 映射成进度条填充 + 手柄位置 + 百分比。"""
        frac = min(max(ph / max(n - 1, 1), 0.0), 1.0)
        prog_line.set_data([0, frac], [0.5, 0.5])
        playhead.set_data([frac], [0.5])
        prog_txt.set_text(f'{frac * 100:3.0f}%')

    def reset_progress():
        """回到「待放手」：进度清零、隐藏预测起跑标与拖动手柄、暂停键复位成「暂停」。"""
        prog_line.set_data([0, 0], [0.5, 0.5])
        playhead.set_data([0], [0.5])
        playhead.set_visible(False)
        prog_txt.set_text('')
        pred_div.set_visible(False)
        state['paused'] = False
        state['finished'] = False
        if 'bpause' in state and state['bpause'] is not None:
            state['bpause'].label.set_text('暂停')

    def _stop_current():
        """停掉当前动画，并让任何残留的定时器 tick 失效。
        macosx 后端 stop() 后可能还会补发一帧，token 自增即可让旧 update 自我跳过。
        同时清掉擦洗活跃锁：万一上一轮拖动的 release 丢了（拖出窗口外松手等），
        放手/重置/改角度时一并复位，杜绝残留锁让 stray motion 劫持新动画的播放头。"""
        if state['ani'] is not None:
            try:
                state['ani'].event_source.stop()
            except Exception:
                pass
            state['ani'] = None
        state['token'] += 1
        scrub['active'] = False

    def refresh_preview():
        """三摆始终在场：按当前初值把三根摆杆画在同一起始姿态（t=0 三者重合，
        放手后才分叉）。开窗即显示、拖滑块/键入角度即时刷新、重置回默认姿态。
        改角度会先停掉在跑的动画再切回预览（播放速度滑块不受影响）。"""
        _stop_current()
        reset_scene(scene, T)
        reset_progress()
        m1, m2 = pendulum_xy(np.deg2rad(s_th1.val), np.deg2rad(s_th2.val))
        for nm in scene['lines']:
            ln = scene['lines'][nm]
            ln.set_data([0, m1[0], m2[0]], [0, m1[1], m2[1]])
            ln.set_alpha(0.95); ln.set_markeredgecolor('none'); ln.set_linestyle('-')
        T['t'].set_text('t = +0.00 s  （待放手）')
        T['verdict'].set_text('三摆已就位于同一初值\n按「放手」开始赛跑')
        fig.canvas.draw_idle()

    def on_go(_evt):
        _stop_current()
        reset_scene(scene, T)
        reset_progress()
        # 自动选种要扫 16 个种子(~0.7s)，先给「计算中」反馈再算，避免点完按钮界面像卡住
        T['verdict'].set_text('放手！正在计算三摆轨迹…')
        try:
            fig.canvas.draw_idle(); fig.canvas.flush_events()
        except Exception:
            pass
        frames, meta = compute(s_th1.val, s_th2.val, s_w1.val, s_w2.val,
                               a.tobs, a.tpred, a.fps, a.seed, hyb_net=hyb_net)
        if meta['nan']:
            T['verdict'].set_text('轨迹发散，换个角度再试'); fig.canvas.draw_idle(); return
        # 进度条上标出预测起跑点（观测窗末）、亮出拖动手柄，并复位暂停键
        pred_div.set_data([meta['i0'] / max(meta['n'] - 1, 1)] * 2, [0.05, 0.95])
        pred_div.set_visible(True)
        playhead.set_visible(True)
        state['paused'] = False; state['finished'] = False
        bpause.label.set_text('暂停')
        my_token = state['token']
        # speed 传一个 lambda：驱动每帧实时读滑块当前值 → 播放中拖动立即变速，
        # 视觉刷新始终 60fps（间隔固定，只改每帧推进的数据帧数），慢放也不卡。
        state['ani'] = run_animation(fig, ax, frames, meta, scene, T,
                                     export=None, play_fps=a.play_fps,
                                     alive=lambda: state['token'] == my_token,
                                     speed=lambda: s_spd.val,
                                     progress_cb=update_progress,
                                     on_finish=_on_finish)
        fig.canvas.draw_idle()

    def _on_finish():
        """到末帧：标记完成、暂停键复位（此后按暂停无意义，on_pause 会自行忽略）。"""
        state['finished'] = True
        state['paused'] = False
        bpause.label.set_text('暂停')
        fig.canvas.draw_idle()

    def _set_paused(paused):
        """停/启动画计时器并同步暂停键文案，播放头原地保留 → 续播无缝。
        暂停键与「拖动进度条擦洗」共用这一处逻辑。"""
        ani = state['ani']
        if ani is None:
            return
        state['paused'] = paused
        try:                              # 优先用公共 API（3.4+）
            ani.pause() if paused else ani.resume()
        except Exception:                 # 老后端兜底：直接收发计时器
            try:
                es = ani.event_source
                es.stop() if paused else es.start()
            except Exception:
                pass
        bpause.label.set_text('继续' if paused else '暂停')
        fig.canvas.draw_idle()

    def on_pause(_evt):
        """暂停/继续：没有在播 or 已播完则忽略（避免把已停的动画又「续」起来）。"""
        if state['ani'] is None or state['finished']:
            return
        _set_paused(not state['paused'])

    def on_reset(_evt):
        """重置：停动画 + 角度滑块回默认（连带刷新输入框与预览），播放速度保留评委设定。"""
        _stop_current()
        for s in (s_th1, s_th2, s_w1, s_w2):
            s.reset()          # 触发绑定回调 → 同步输入框 + refresh_preview
        refresh_preview()      # 兜底，确保最终落到默认姿态

    # 滑块 ↔ 输入框双向绑定，任一侧变化即刷新预览（同一把锁防回环）
    sync = {'v': False}
    _bind_slider_textbox(s_th1, t_th1, -180, 180, 1, refresh_preview, sync)
    _bind_slider_textbox(s_th2, t_th2, -180, 180, 1, refresh_preview, sync)
    _bind_slider_textbox(s_w1, t_w1, -6, 6, 2, refresh_preview, sync)
    _bind_slider_textbox(s_w2, t_w2, -6, 6, 2, refresh_preview, sync)

    # 放手 / 暂停 / 重置 三键一排（各 0.085 宽、间隔 0.01）
    bgo = Button(fig.add_axes([0.660, 0.20, 0.085, 0.06]), '放手',
                 color='#1f6f4f', hovercolor='#2a9d6f')
    bpause = Button(fig.add_axes([0.755, 0.20, 0.085, 0.06]), '暂停',
                    color='#8a6d1f', hovercolor='#b8922a')
    brs = Button(fig.add_axes([0.850, 0.20, 0.085, 0.06]), '重置',
                 color='#444', hovercolor='#666')
    for b in (bgo, bpause, brs):
        b.label.set_color('white'); b.label.set_fontsize(12)
    state['bpause'] = bpause
    bgo.on_clicked(on_go); bpause.on_clicked(on_pause); brs.on_clicked(on_reset)

    # ---- 拖动进度条「擦洗」定位 ----
    # 在进度条上按住左键拖动 → 暂停并跳到该时刻定格；松手停在那一帧，可按「继续」续播。
    # 拖动中用像素坐标反算（横向拖出条的上下边界仍能擦洗）；一旦指针拖出整个画布，
    # event.x 变 None / transform 回 nan，本帧直接跳过不动，等指针回来再续。
    def _seek_to_event(event):
        ani = state['ani']
        ctrl = getattr(ani, '_demo_ctrl', None) if ani is not None else None
        if ctrl is None:
            return
        if event.x is None or event.y is None:     # 指针已在画布外
            return
        fx, _ = pax.transData.inverted().transform((event.x, event.y))
        if not np.isfinite(fx):
            return
        frac = min(max(float(fx), 0.0), 1.0)
        ctrl['seek_frac'](frac)
        # 仅拖到真末帧(frac>=1) 才算已播完——与 seek 内 holder['finished']=ph>=n-1 同口径。
        # 否则「拖到 99.9%」会被提前判完：暂停键标着『暂停』却被 on_pause 当已完成忽略
        # （死键），最后一小段也续播不到。中途一律 finished=False，「继续」可接着播。
        state['finished'] = frac >= 1.0
        bpause.label.set_text('暂停' if state['finished'] else '继续')

    def on_scrub_press(event):
        if event.button != 1 or event.inaxes is not pax:
            return
        if state['ani'] is None:           # 还没放手、无轨迹可擦洗
            return
        if not state['paused']:            # 先停表，避免计时器与拖动抢播放头
            _set_paused(True)
        scrub['active'] = True
        _seek_to_event(event)

    def on_scrub_motion(event):
        if scrub['active']:
            _seek_to_event(event)

    def on_scrub_release(_event):
        scrub['active'] = False

    fig.canvas.mpl_connect('button_press_event', on_scrub_press)
    fig.canvas.mpl_connect('motion_notify_event', on_scrub_motion)
    fig.canvas.mpl_connect('button_release_event', on_scrub_release)

    # 输入框聚焦可视化：亮白闪烁光标 + 选中框描青边（点进哪个框、在编辑都一眼可见）
    focus_timer = _setup_textbox_focus_blink(fig, [t_th1, t_th2, t_w1, t_w2])
    # 句柄挂到 fig 防被 GC（含输入框、同步锁、闪烁计时器、暂停键、擦洗状态）
    fig._demo = (s_th1, s_th2, s_w1, s_w2, s_spd, bgo, bpause, brs,
                 t_th1, t_th2, t_w1, t_w2, sync, state, focus_timer, scrub)
    refresh_preview()          # 开窗即三摆就位
    plt.show()


def mode_butterfly(a):
    """蝴蝶效应彩蛋：仅初值差 0.5°，两条真值轨迹很快面目全非。"""
    _setup(export=bool(a.export))
    fps = a.fps
    d1 = bd.simulate(np.deg2rad(a.a1), np.deg2rad(a.a2), t_end=a.tobs + a.tpred, fps=fps)
    d2 = bd.simulate(np.deg2rad(a.a1 + 0.5), np.deg2rad(a.a2), t_end=a.tobs + a.tpred, fps=fps)
    n = min(len(d1), len(d2))
    fig = plt.figure(figsize=(12, 8.5), facecolor=BG)
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.92])
    _style_pend_ax(ax)
    (t1l,) = ax.plot([], [], '-', color=(0.3, 0.8, 1.0), lw=1.0, alpha=0.4)
    (t2l,) = ax.plot([], [], '-', color=(1.0, 0.5, 0.5), lw=1.0, alpha=0.4)
    (p1,) = ax.plot([], [], 'o-', color=(0.3, 0.8, 1.0), lw=4, ms=8, label='初值 A')
    (p2,) = ax.plot([], [], 'o-', color=(1.0, 0.5, 0.5), lw=4, ms=8, label='初值 A + 0.5°')
    win = max(8, int(round(0.9 * fps)))
    b1, b2 = deque(maxlen=win), deque(maxlen=win)
    fig.text(0.66, 0.9, '蝴蝶效应', color='white', fontsize=18, fontweight='bold')
    fig.text(0.66, 0.82, '两摆初值只差 0.5°\n同一套物理方程\n轨迹很快面目全非',
             color=C_TXT, fontsize=13, linespacing=1.6)
    sep = fig.text(0.66, 0.6, '', color=(1.0, 0.6, 0.6), fontsize=13)
    tt = fig.text(0.66, 0.7, '', color=C_TXT, fontsize=13, family='monospace')
    th1a, th2a = d1['th1'].values, d1['th2'].values
    th1b, th2b = d2['th1'].values, d2['th2'].values
    last = [-1]

    def render(ph):
        i = int(ph); f = ph - i
        j = min(i + 1, n - 1); i = min(i, n - 1)
        k = min(int(round(ph)), n - 1)
        for tha, thb, pl, bl, bf in [(th1a, th2a, p1, t1l, b1),
                                     (th1b, th2b, p2, t2l, b2)]:
            A1 = lerp_angle(tha[i], tha[j], f)
            A2 = lerp_angle(thb[i], thb[j], f)
            m1, m2 = pendulum_xy(A1, A2)
            pl.set_data([0, m1[0], m2[0]], [0, m1[1], m2[1]])
            if k > last[0]:
                bf.append(m2)
            arr = np.array(bf); bl.set_data(arr[:, 0], arr[:, 1])
        last[0] = max(last[0], k)
        e = angle_err_deg(th1a[:k + 1], th2a[:k + 1], th1b[:k + 1], th2b[:k + 1])[-1]
        tt.set_text(f't = +{ph/fps:5.2f}s')
        sep.set_text(f'当前分歧 {e:6.1f}°')
        return [t1l, t2l, p1, p2, tt, sep]

    fig._ani = drive_animation(fig, render, n, fps, a.play_fps, a.speed,
                               export=a.export)
    if not a.export:
        plt.show()
    return fig._ani


def _print_result(meta, frames):
    si = meta.get('seed_info') or {}
    seedtag = f"seed={meta['seed']}"
    if si.get('n_clean') is not None:
        seedtag += f"(自动·干净候选{si['n_clean']}/{si['scanned']}·冻结{si['froz']*100:.0f}%)"
    print(f"初值 θ1={meta['th1_deg']:.0f}° θ2={meta['th2_deg']:.0f}° "
          f"ω1={meta['w1']:.1f} ω2={meta['w2']:.1f}  {seedtag}")
    hz_h, hz_e = frames['Hybrid']['hz'], frames['ESN']['hz']
    if meta.get('degenerate'):
        # 退化平衡点：真值整窗几乎不动、赛跑无意义，绝不打「物理 ×0.0」误导
        # （这行会进控制台录屏 / mp4 元数据，断章取义会变成「物理输了」）。
        print(f"  平衡点附近(真值前半窗 <15°)，赛跑无意义；视界数字略去"
              + ('   ⚠NaN' if meta['nan'] else ''))
    else:
        ratio = hz_h / hz_e if hz_e > 0 else float('inf')
        print(f"  视界  ESN={hz_e:.2f}s  Hybrid={hz_h:.2f}s  → 物理 ×{ratio:.1f}"
              + ('   ⚠NaN' if meta['nan'] else ''))


def build_argparser():
    p = argparse.ArgumentParser(description='混沌之眼双摆预测对决 demo')
    p.add_argument('--play', action='store_true', help='给定角度直接弹窗播（不带滑块）')
    p.add_argument('--export', type=str, default=None, help='导出 mp4 路径（无头）')
    p.add_argument('--butterfly', action='store_true', help='蝴蝶效应彩蛋模式')
    p.add_argument('--a1', type=float, default=70.0, help='上摆初角 θ1 (°)')
    p.add_argument('--a2', type=float, default=90.0, help='下摆初角 θ2 (°)')
    p.add_argument('--w1', type=float, default=0.0, help='上摆初角速度 (rad/s)')
    p.add_argument('--w2', type=float, default=0.0, help='下摆初角速度 (rad/s)')
    p.add_argument('--tobs', type=float, default=2.0, help='观测窗秒（只给 ESN 训练）')
    p.add_argument('--tpred', type=float, default=6.0, help='预测/赛跑秒')
    p.add_argument('--fps', type=int, default=60, help='仿真采样帧率（慢放靠插值，无需调高；过高会诱发 ESN 闭环锁死）')
    p.add_argument('--play-fps', type=int, default=60,
                   help='视觉刷新/导出帧率（固定不随速度变，越高越丝滑）')
    p.add_argument('--speed', type=float, default=0.5,
                   help='播放速度倍率（1×=实时，<1 慢放，>1 快进；交互模式有同名滑块可实时拖）')
    p.add_argument('--seed', type=int, default=-1,
                   help='ESN 随机种子：<0(默认)=自动挑代表性干净种子（防冻结/瞬移）；'
                        '>=0=写死该种子（可复现导出用）')
    return p


def main():
    a = build_argparser().parse_args()
    if a.butterfly:
        mode_butterfly(a)
    elif a.export:
        mode_export(a)
    elif a.play:
        mode_play(a)
    else:
        mode_interactive(a)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
tools/demo_doctor.py — 答辩前一键自检盒 + 口径卡

主讲人（朋友）答辩前 / 彩排前跑这一条命令，确认演示家底全在、跑得动，
并打印一张「口径卡」——所有要背的关键数字，全部从 canonical 唯一来源
（data/canonical_results_B.md）自动抽取，保证嘴上说的和论文/数据一字不差。

链路（自上而下，硬项失败即非零退出 + 红字修复指引）：
  1. 环境体检      Python 依赖 / matplotlib 交互后端 macosx / ffmpeg / 中文字体 / 模型权重
  2. 演示推理冒烟  无头跑一次 live_demo.compute()，验证三模型都能算、视界数字合理
  3. 演示产物清单  开场 mp4 / 战绩墙 PNG / 真实叠加 mp4 / 点子清单 docx 是否都在
  4. 串 smoke.py       数据 / 权重 / tracking 管线
  5. 串 paper_audit.py paper / README / csv 数字一致性 六项
  6. 口径卡        真实 2.7× / 仿真 7× / τ_L / 视界阈 / 物理损失增益（从 canonical 自动抽取）

任何 Mac 都能跑，不开窗、不依赖 GUI。本人不在场时，这是朋友零焦虑彩排的兜底。
用法：python3 tools/demo_doctor.py
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "data" / "canonical_results_B.md"

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
END = "\033[0m"

_hard_fail = []   # 累积硬失败项名，决定退出码


def head(title):
    print(f"\n{BOLD}{title}{END}")


def line(ok, name, fix="", hard=True):
    """ok=True 绿勾；False 时 hard=True 记为硬失败、hard=False 仅黄叹号警告。"""
    if ok:
        print(f"  {OK}  {name}")
    elif hard:
        print(f"  {FAIL}  {name}")
        if fix:
            print(f"      {DIM}修复：{fix}{END}")
        _hard_fail.append(name)
    else:
        print(f"  {WARN}  {name}")
        if fix:
            print(f"      {DIM}{fix}{END}")
    return ok


# ──────────────────────────── 1. 环境体检 ────────────────────────────

def check_env():
    head("1. 环境体检")

    for mod, fix in [("numpy", "pip3 install numpy"),
                     ("scipy", "pip3 install scipy"),
                     ("pandas", "pip3 install pandas"),
                     ("matplotlib", "pip3 install matplotlib"),
                     ("torch", "pip3 install torch")]:
        try:
            __import__(mod)
            line(True, f"依赖 {mod}")
        except ImportError:
            line(False, f"依赖 {mod}", fix)

    # 交互后端：本机只有 macosx 能弹窗（tkinter 没装），故 macosx 是现场交互的唯一指望
    try:
        import matplotlib.backends._macosx  # noqa: F401
        line(True, "交互后端 macosx 可用（现场拖滑块靠它）")
    except Exception:
        line(False, "交互后端 macosx 不可用",
             "现场交互演示跑不了 → 退回放预导出的 mp4（见第 3 节产物清单）", hard=False)
    # tkinter 缺失是已知事实，提示一句不算失败
    try:
        import tkinter  # noqa: F401
        line(True, "TkAgg 后端也在（多一条退路）", hard=False)
    except Exception:
        print(f"  {DIM}·  TkAgg 不可用（本机已知，不影响：交互走 macosx，兜底走 mp4）{END}")

    # ffmpeg：重新导出 mp4 时要用
    ff = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg"
                                    if Path("/opt/homebrew/bin/ffmpeg").exists() else "")
    line(bool(ff), f"ffmpeg{(' → ' + ff) if ff else ''}",
         "brew install ffmpeg（只在需要重新导出 mp4 时才必须）", hard=False)

    # 中文字体：图 / mp4 里的中文不变方框
    try:
        import matplotlib.font_manager as fm
        fonts = []
        for fam in ["Songti SC", "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"]:
            try:
                fm.findfont(fam, fallback_to_default=False)
                fonts.append(fam)
            except Exception:
                pass
        line(bool(fonts), f"中文字体 {fonts[0] if fonts else '(无)'}",
             "装任一中文字体，否则图里中文变方框", hard=not fonts)
    except Exception as e:
        line(False, f"字体检查异常：{e}", hard=False)

    # 模型权重
    for name, p in [("Hybrid 权重", ROOT / "data/hybrid/model.pt"),
                    ("PINN 权重", ROOT / "data/pinn/model.pt")]:
        line(p.exists(), f"{name}  {p.relative_to(ROOT)}",
             "python3 rc/hybrid.py / rc/pinn.py 重训")


# ──────────────────────────── 2. 演示推理冒烟 ────────────────────────────

def check_demo_inference():
    head("2. 演示推理冒烟（无头跑一次 compute，不开窗）")
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import live_demo as D
        D._setup(export=True)            # Agg 无头
        hyb = D.load_hybrid()
        fr, mt = D.compute(70, 90, hyb_net=hyb)   # 主叙事默认角
        hz_e = float(fr["ESN"]["hz"])
        hz_h = float(fr["Hybrid"]["hz"])
        nan = bool(mt.get("nan", False))
        seed = mt.get("seed")

        line(not nan, f"三模型自回归无 NaN（自动 seed={seed}）",
             "live_demo.compute 数值炸了，查 rc/ 权重与 sim 常数")
        sane = (0 < hz_h <= 12) and (0 <= hz_e <= 12)
        line(sane, f"视界数值合理  ESN={hz_e:.2f}s  Hybrid={hz_h:.2f}s")
        # 默认角 Hybrid 应明显扛得更久（主叙事）；个别角 ESN 会追平，故仅警告不硬失败
        line(hz_h >= hz_e, f"默认角 Hybrid({hz_h:.2f}s) ≥ ESN({hz_e:.2f}s)，主叙事成立",
             "默认角叙事翻车了，换默认角或复查 pick_esn_seed", hard=False)
        # 视界阈值常数（单一来源 = live_demo.THRESH_DEG），口径卡复用
        global _THRESH_DEG
        _THRESH_DEG = float(getattr(D, "THRESH_DEG", 10.0))
        line(abs(_THRESH_DEG - 10.0) < 1e-6, f"视界判定阈 = {_THRESH_DEG:.0f}°（与论文一致）",
             hard=False)
    except Exception as e:
        line(False, f"演示推理冒烟抛异常：{e}",
             "先单独跑 python3 tools/live_demo.py --play --a1 70 --a2 90 看报错")


# ──────────────────────────── 3. 演示产物清单 ────────────────────────────

def check_artifacts():
    head("3. 演示产物清单（现场要播 / 要展示的成品）")
    items = [
        ("蝴蝶效应开场 mp4", ROOT / "混沌之眼_蝴蝶效应开场.mp4", True),
        ("预测对决 mp4·默认角", ROOT / "混沌之眼_预测对决_默认70-90.mp4", True),
        ("预测对决 mp4·大角度", ROOT / "混沌之眼_预测对决_大角度120-90.mp4", True),
        ("真实预测叠加 mp4·Hybrid当家", ROOT / "混沌之眼_真实预测叠加_1448_Hybrid当家.mp4", True),
        ("29 段真实战绩墙 PNG", ROOT / "data/real_validation/figures/真实战绩墙_29段.png", True),
        ("演示点子清单 docx", ROOT / "文档/答辩/混沌之眼_演示点子清单.docx", False),
        ("交互演示脚本 live_demo.py", ROOT / "tools/live_demo.py", True),
    ]
    for name, p, hard in items:
        line(p.exists(), f"{name}", f"产物缺失：{p.name}（重跑对应导出脚本补）", hard=hard)


# ──────────────────────────── 4 & 5. 串现成自检脚本 ────────────────────────────

def run_subscript(title, rel, hard=True):
    head(title)
    script = ROOT / rel
    if not script.exists():
        line(False, f"{rel} 不存在", hard=hard)
        return
    try:
        r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300)
        # 原样透出子脚本输出（缩进一格，保留它自己的 ✓/✗ 配色）
        for ln in (r.stdout or "").rstrip("\n").splitlines():
            print(f"  {ln}")
        if r.stderr.strip():
            for ln in r.stderr.rstrip("\n").splitlines()[-6:]:
                print(f"  {DIM}{ln}{END}")
        line(r.returncode == 0, f"{rel} 退出码 {r.returncode}",
             f"按上面 {rel} 的修复指引补", hard=hard)
    except subprocess.TimeoutExpired:
        line(False, f"{rel} 超时（>300s）", hard=hard)
    except Exception as e:
        line(False, f"{rel} 跑挂：{e}", hard=hard)


# ──────────────────────────── 6. 口径卡（从 canonical 自动抽取） ────────────────────────────

def _grab(text, pat, cast=float, default=None):
    m = re.search(pat, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


def cheat_card():
    head("6. 口径卡 · 答辩要背的数字（自动抽取自 data/canonical_results_B.md）")
    if not CANON.exists():
        line(False, "canonical_results_B.md 不存在，口径卡无法生成")
        return
    t = CANON.read_text()

    # —— 真实零样本（29 片）——
    real_esn = _grab(t, r"\|\s*ESN\s*\|\s*([\d.]+)\s*\|\s*\[")
    real_pinn = _grab(t, r"\|\s*PINN（λ=0\.1）\s*\|\s*([\d.]+)")
    real_hyb_h = _grab(t, r"\|\s*Hybrid\s*\|\s*([\d.]+)\s*\|\s*\[0\.450")
    real_ratio = _grab(t, r"Hybrid\s*\|\s*[\d.]+\s*\|\s*\[[^\]]*\]\s*\|\s*([\d.]+)×")
    phys_gain = _grab(t, r"物理损失真实增益[^=]*=\s*[\d./]+\s*=\s*\*\*([\d.]+)×")

    # —— 仿真内 ——
    sim_esn = _grab(t, r"ESN（数据驱动基线）\s*\|\s*([\d.]+)")
    sim_pinn = _grab(t, r"PINN（λ=0\.1，物理损失）\s*\|\s*([\d.]+)")
    sim_mlp = _grab(t, r"纯数据 MLP（λ=0，消融）\s*\|\s*([\d.]+)")
    sim_hyb = _grab(t, r"Hybrid（解析先验\+残差）\s*\|\s*([\d.]+)")
    sim_ratio = (sim_hyb / sim_esn) if (sim_hyb and sim_esn) else None

    tau_L = _grab(t, r"τ_L[=\s]*([\d.]+)s")
    thr = globals().get("_THRESH_DEG", 10.0)

    def s(v, suf="s"):
        return f"{v:.3f}{suf}" if isinstance(v, float) else f"{DIM}?{END}"

    print(f"  {BOLD}真实零样本（仿真训练→真实双摆视频，零微调，29 片中位）{END}")
    print(f"     ESN {s(real_esn)}  ·  PINN {s(real_pinn)}  ·  Hybrid {s(real_hyb_h)}"
          f"   →  物理 ≈ {BOLD}{s(real_ratio, '×')}{END} ESN")
    print(f"     物理损失真实增益 PINN/MLP(λ0) = {BOLD}{s(phys_gain,'×')}{END}（物理损失是「迁移正则」）")
    print(f"  {BOLD}仿真内（per-run 中位）{END}")
    print(f"     ESN {s(sim_esn)}  ·  PINN {s(sim_pinn)}  ·  MLP(λ0) {s(sim_mlp)}  ·  "
          f"Hybrid {s(sim_hyb)}   →  Hybrid ≈ {BOLD}{s(sim_ratio,'×')}{END} ESN")
    print(f"  {BOLD}标尺{END}")
    print(f"     Lyapunov 时标 τ_L = {s(tau_L)}（实测中位）  ·  预测视界判定阈 = {thr:.0f}°（偏差首超即失准）")
    print(f"  {DIM}一句话护城河：仿真世界里物理先验最强(≈7×)；真正卖点是 sim→real 零样本仍领先 ≈2.7×。{END}")
    print(f"  {DIM}真值=逐帧视频追踪点，非模型自仿；演示里的「真值」白线同理（堵自证循环质疑）。{END}")

    # 抽取健全性：方向对不对（不硬失败，提醒 canonical 是否漂移）
    drift = []
    if real_ratio and not (2.3 <= real_ratio <= 3.0):
        drift.append(f"真实比 {real_ratio}× 偏离 ~2.7×")
    if sim_ratio and not (5.5 <= sim_ratio <= 8.5):
        drift.append(f"仿真比 {sim_ratio:.1f}× 偏离 ~7×")
    if drift:
        line(False, "canonical 数字疑似漂移：" + "；".join(drift),
             "口径变了 → 同步更新论文话术再彩排", hard=False)


# ──────────────────────────── 主 ────────────────────────────

def main():
    print(f"{BOLD}══════ 混沌之眼 · 答辩前一键自检盒 ══════{END}")
    print(f"{DIM}跑完确认：环境齐 · 演示能跑 · 产物全 · 数字一致 · 口径背得对{END}")

    check_env()
    check_demo_inference()
    check_artifacts()
    run_subscript("4. smoke.py（数据 / 权重 / tracking 管线）", "tools/smoke.py")
    run_subscript("5. paper_audit.py（paper / README / csv 数字一致性）", "tools/paper_audit.py")
    cheat_card()

    print(f"\n{BOLD}{'═' * 44}{END}")
    if _hard_fail:
        print(f"{FAIL} {BOLD}有 {len(_hard_fail)} 项硬失败，按上面红字修复后再彩排：{END}")
        for nm in _hard_fail:
            print(f"    {FAIL} {nm}")
        return 1
    print(f"{OK} {BOLD}全绿。环境、演示、产物、数字一致性、口径卡全部就绪，可以上场。{END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

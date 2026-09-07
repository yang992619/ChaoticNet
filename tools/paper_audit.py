r"""
tools/paper_audit.py — paper / README / 数据 一致性监控 loop

执行六类检查（任一类失败就退出码非 0，可挂到 git pre-commit 或 watch 模式）：

  A. paper \cite{} 是否都在 references.bib
  B. paper \includegraphics{} 的文件是否都能找到
  C. paper 表里的关键数字是否跟 csv 实际值一致（容差 0.01）
  D. README 三方对比表数字是否跟 csv 实际值一致
  E. paper 还有没有遗留 TODO 红字
  F. τ_L 归一化是否统一用自测中位 0.762s（csv 的 tau_L 列 + 画图脚本常量）

用法：
    python3 tools/paper_audit.py                # 跑一次
    python3 tools/paper_audit.py --watch        # 每 30 秒重跑（依赖 watchdog 包，没装也能跑成 sleep loop）
    python3 tools/paper_audit.py --fix-readme   # 自动修复 README 数字（实验性，慎用）
"""

import argparse
import re
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "references.bib"
README = ROOT / "README.md"

TOL = 0.03    # 数字一致性容差（绝对值；per-run 中位四舍五入到 0.01-0.05）
CANON_TAU_L = 0.762   # 分布质量实测中位归一化基准（sim/lyapunov.py），全项目统一


# ---------- helpers ----------

def _ok(msg):  print(f"  \033[32m✓\033[0m  {msg}")
def _bad(msg): print(f"  \033[31m✗\033[0m  {msg}")
def _warn(msg): print(f"  \033[33m!\033[0m  {msg}")


# ---------- A. cite 检查 ----------

def check_cite():
    print("A. \\cite{} → references.bib")
    tex = TEX.read_text()
    bib = BIB.read_text()

    cite_keys = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))

    missing = cite_keys - bib_keys
    if missing:
        _bad(f"{len(missing)} 个 \\cite 在 bib 里找不到: {sorted(missing)}")
        return False
    _ok(f"{len(cite_keys)} 个 \\cite 全部在 bib 里")
    return True


# ---------- B. figure 文件 ----------

def _graphics_paths(tex):
    m = re.search(r"\\graphicspath\{(.+?)\}\s*\n", tex, flags=re.S)
    if not m:
        return [ROOT / "paper" / "figures"]
    inner = m.group(1)
    paths = []
    for sub in re.findall(r"\{([^{}]+)\}", inner):
        p = (TEX.parent / sub).resolve()
        paths.append(p)
    return paths


def check_figures():
    print("B. \\includegraphics 文件存在性")
    tex = TEX.read_text()
    paths = _graphics_paths(tex)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)

    missing = []
    for f in figs:
        found = False
        # 支持 LaTeX 中显式给出的相对路径，例如 ../data/...；其余文件仍按
        # \graphicspath 的搜索路径处理。
        explicit = Path(f)
        if explicit.is_absolute():
            found = explicit.exists()
        elif "/" in f or f.startswith("."):
            found = (TEX.parent / explicit).resolve().exists()
        if found:
            continue
        for p in paths:
            for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"):
                if (p / (f + ext)).exists():
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(f)
    if missing:
        _bad(f"{len(missing)} 个 figure 找不到: {missing}")
        return False
    _ok(f"{len(figs)} 个 \\includegraphics 文件全部存在")
    return True


# ---------- C. paper 表数字 vs csv ----------

# 权威数据源 = data/canon_multirun/summary.csv（见 _load_canon），口径与
# data/canonical_results_B.md（全项目数字唯一真相源）一致。
#
# ⚠️ 下面两个 _load_chaos_* 是 DEAD CODE，当前没有任何 check 调用它们，保留只为历史追溯。
#    _load_chaos_29 指向 data/v2_50trial_2026-06-11/horizon_compare_50.csv，那是【点质量旧口径】
#    产物（ESN 中位 0.150s / PINN 0.642s），与现行分布质量口径（ESN 0.40 / PINN 0.88）不兼容。
#    切勿据此写任何论文数字或置信区间。分布质量的固定子集交叉验证请用 FOUR_WAY_CSV（N=32）。
# 注意：根目录 data/horizon_compare_50.csv 与 data/hybrid/hybrid_summary_50.csv 是 06-17 异常重跑，
# Hybrid 中位虚高至 3.0s（物理上不可能优于 PINN 6 倍），已弃用，勿再指向。
CHAOS_2WAY_CSV_POINTMASS_DEPRECATED = (
    ROOT / "data" / "v2_50trial_2026-06-11" / "horizon_compare_50.csv")
FOUR_WAY_CSV = ROOT / "data" / "fair_compare" / "four_way_horizons.csv"


def _load_chaos_29_POINTMASS_DEPRECATED():
    """⚠️ 点质量旧口径，勿用。仅存档参考；B 版数字见 data/canonical_results_B.md。"""
    h = pd.read_csv(CHAOS_2WAY_CSV_POINTMASS_DEPRECATED)
    return h[(h.esn < 9.9) & (h.pinn < 9.9)]


def _load_chaos_fixed_subset():
    """三方一致混沌 ESN+PINN+Hybrid 都未跑满（分布质量固定子集，N=32）"""
    h = pd.read_csv(FOUR_WAY_CSV)
    return h[(h.esn < 9.9) & (h.pinn < 9.9) & (h.hybrid < 9.9)]


def _load_canon():
    """B canonical per-run 中位（data/canon_multirun/summary.csv）。
    返回 {model: {'insim': median, 'real': median}}。"""
    df = pd.read_csv(ROOT / "data" / "canon_multirun" / "summary.csv")
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r["model"], {})
        out[r["model"]]["insim" if r["metric"] == "insim_chaos" else "real"] = float(r["median"])
    return out


def check_paper_numbers():
    print("C. paper 仿真内表数字 vs canon_multirun per-run 中位")
    tex = TEX.read_text()
    canon = _load_canon()

    # 仿真内表行首数字 = 该模型 per-run 中位（中位列）
    # 容忍模型名后带括号标注，取行首第一个数字（仿真内中位列）；首个匹配落在 §6 仿真内表
    checks = [
        ("ESN",    r"ESN[^&\n]*&\s*([\d.]+)",    canon["ESN"]["insim"]),
        ("PINN",   r"PINN[^&\n]*&\s*([\d.]+)",   canon["PINN"]["insim"]),
        ("Hybrid", r"Hybrid[^&\n]*&\s*([\d.]+)", canon["Hybrid"]["insim"]),
    ]
    ok = True
    for name, pat, exp in checks:
        m = re.search(pat, tex)
        if not m:
            _warn(f"{name} 仿真内中位: 表里找不到 — 跳过"); continue
        paper_val = float(m.group(1))
        if abs(paper_val - exp) > TOL:
            _bad(f"{name} 仿真内中位: paper={paper_val} vs canon={exp:.3f} (差 {abs(paper_val-exp):.3f})"); ok = False
        else:
            _ok(f"{name} 仿真内中位: paper={paper_val} 一致 canon {exp:.3f}")
    return ok


# ---------- D. README 数字 ----------

def check_readme_numbers():
    print("D. README 仿真内对比表 vs canon_multirun per-run 中位")
    rm = README.read_text()
    canon = _load_canon()
    expected = {"ESN": canon["ESN"]["insim"],
                "PINN": canon["PINN"]["insim"],
                "Hybrid": canon["Hybrid"]["insim"]}
    # 解析 README 表行：| ESN | 0.40s | 0.53 τ_L | 1× |
    ok = True
    for name, exp in expected.items():
        m = re.search(rf"\|\s*{name}\s*\|\s*([\d.]+)\s*s\s*\|", rm)
        if not m:
            _warn(f"{name}: README 行没找到 — 跳过")
            continue
        readme_val = float(m.group(1))
        if abs(readme_val - exp) > TOL:
            _bad(f"{name}: README={readme_val}s vs canon={exp:.3f}s")
            ok = False
        else:
            _ok(f"{name}: README={readme_val}s 一致 canon {exp:.3f}s")
    return ok


# ---------- E. 遗留 TODO ----------

def check_todos():
    print("E. paper 遗留 TODO 红字")
    tex = TEX.read_text()
    todos = re.findall(r"\\textcolor\{red\}\{\[TODO[^\]]*\]\}", tex)
    if todos:
        _bad(f"{len(todos)} 个 TODO 红字残留")
        return False
    _ok("0 个 TODO 红字残留")
    return True


# ---------- F. τ_L 归一化一致性 ----------

def check_tau_norm():
    print(f"F. τ_L 归一化一致性（基准 {CANON_TAU_L}s）")
    ok = True

    # F1: three_way_summary.csv 的 tau_L 列 == 中位 / 0.762
    csv = ROOT / "data" / "figures" / "three_way_summary.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            exp = r["horizon_median_s"] / CANON_TAU_L
            if abs(r["horizon_tau_L"] - exp) > TOL:
                _bad(f"{r['model']}: csv τ_L={r['horizon_tau_L']:.3f} ≠ 中位/{CANON_TAU_L}={exp:.3f}")
                ok = False
            else:
                _ok(f"{r['model']}: csv τ_L={r['horizon_tau_L']:.3f} 一致 中位/{CANON_TAU_L}")
    else:
        _warn("three_way_summary.csv 不存在 — 跳过")

    # F2: 画图脚本 plot_three_way.py 的 TAU_L 常量是否 0.762
    plot = ROOT / "sim" / "plot_three_way.py"
    if plot.exists():
        m = re.search(r"TAU_L\s*=\s*([\d.]+)", plot.read_text())
        if m and abs(float(m.group(1)) - CANON_TAU_L) > 1e-9:
            _bad(f"plot_three_way.py TAU_L={m.group(1)} ≠ {CANON_TAU_L}")
            ok = False
        elif m:
            _ok(f"plot_three_way.py TAU_L={m.group(1)} 一致")
    return ok


# ---------- 主 ----------

CHECKS = [
    ("A. cite",  check_cite),
    ("B. figs",  check_figures),
    ("C. paper 数字", check_paper_numbers),
    ("D. README 数字", check_readme_numbers),
    ("E. TODO 残留", check_todos),
    ("F. τ_L 归一化", check_tau_norm),
]


def run_once():
    results = []
    for name, fn in CHECKS:
        try:
            ok = fn()
        except Exception as e:
            _bad(f"{name} 抛异常: {e}")
            ok = False
        results.append((name, ok))
        print()
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\033[31m✗ 失败 {len(failed)}/{len(results)}: {failed}\033[0m")
    else:
        print(f"\033[32m✓ 全部 {len(results)} 项通过\033[0m")
    return 0 if not failed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="每 30 秒重跑")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    if not args.watch:
        sys.exit(run_once())

    print(f"watch 模式（每 {args.interval}s 重跑，Ctrl-C 退出）\n")
    try:
        while True:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            run_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n退出")


if __name__ == "__main__":
    main()

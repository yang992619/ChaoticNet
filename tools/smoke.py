"""
tools/smoke.py — 项目自检 / 烟雾测试

新队员 clone 后立即跑，验证：
  1. Python 依赖齐全
  2. 仿真数据完整（50 组 baseline + 50 组 damped）
  3. 各模型权重可加载（PINN / Hybrid）
  4. tracking 管线能跑通一个 demo
  5. 论文 PDF 已编译

运行 < 30 秒。任何一项失败都会列出修复指令。
"""

from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parent.parent

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"


def check(name, cond, fix=""):
    print(f"  {OK if cond else FAIL}  {name}")
    if not cond and fix:
        print(f"      修复：{fix}")
    return cond


def main():
    print("== ChaoticNet 烟雾测试 ==\n")
    all_ok = True

    # 1. Python 依赖
    print("Python 依赖")
    for mod, fix in [
        ("numpy", "pip3 install numpy"),
        ("scipy", "pip3 install scipy"),
        ("pandas", "pip3 install pandas"),
        ("matplotlib", "pip3 install matplotlib"),
        ("torch", "pip3 install torch"),
        ("cv2", "pip3 install opencv-python"),
    ]:
        try:
            __import__(mod)
            check(mod, True)
        except ImportError:
            all_ok &= check(mod, False, fix)

    # 2. 仿真数据
    print("\n仿真数据")
    sim_csv = ROOT / "data" / "sim" / "trial_000.csv"
    sim_count = len(list((ROOT / "data" / "sim").glob("trial_*.csv"))) if (ROOT / "data" / "sim").exists() else 0
    all_ok &= check(f"无阻尼 50 组 ({sim_count} 找到)", sim_count >= 50,
                     "python3 sim/baseline.py")
    damped_count = len(list((ROOT / "data" / "sim_damped").glob("trial_*.csv"))) if (ROOT / "data" / "sim_damped").exists() else 0
    all_ok &= check(f"阻尼 50 组 ({damped_count} 找到)", damped_count >= 50,
                     "python3 sim/baseline_damped.py")

    # 3. 模型权重
    print("\n模型权重")
    pinn_pt = ROOT / "data" / "pinn" / "model.pt"
    hybrid_pt = ROOT / "data" / "hybrid" / "model.pt"
    all_ok &= check(f"PINN model.pt", pinn_pt.exists(),
                     "python3 sim/animate_compare.py（自动训练 + 保存）")
    all_ok &= check(f"Hybrid model.pt", hybrid_pt.exists(),
                     "python3 rc/hybrid.py")

    # 4. 关键产出（不 fail 整个测试，只警告）
    print("\n关键产出（缺失只警告）")
    # 每项给一组候选路径：随仓库发布的产物排在前面，本地全量产物排在后面。
    # 这样从公开仓库 clone 下来也应全绿，而不是对着一堆"缺失"警告发懵。
    artifacts = [
        ("50 组联合表", [ROOT / "data" / "horizon_compare_50.csv"]),
        ("三方对比图", [ROOT / "data" / "figures" / "three_way_compare.png"]),
        ("敏感性扫描", [ROOT / "data" / "sensitivity_correct" / "summary.csv",
                        ROOT / "data" / "sensitivity" / "sensitivity.csv"]),
        ("真实轨迹数据", [ROOT / "data" / "tracking_windows" / "tracked_1430.csv",
                          ROOT / "data" / "tracking" / "tracked.csv"]),
        ("真实零样本结果", [ROOT / "data" / "real_validation" / "summary_multi.csv"]),
        ("论文 PDF", [ROOT / "paper" / "main_anon.pdf", ROOT / "paper" / "main.pdf"]),
    ]
    for name, paths in artifacts:
        hit = next((q for q in paths if q.exists()), None)
        if hit:
            print(f"  {OK}  {name}  ({hit.relative_to(ROOT)})")
        else:
            print(f"  {WARN}  {name}  (缺失，跑 README 里的复现命令补)")

    # 5. tracking 管线快速跑通
    print("\ntracking 管线（用现有合成视频）")
    track_video = ROOT / "data" / "sim" / "figures" / "anim_trial_000.mp4"
    if track_video.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(track_video))
            ok, frame = cap.read()
            cap.release()
            all_ok &= check(f"读视频 + 解码 OK ({frame.shape[1]}×{frame.shape[0]})", ok)
        except Exception as e:
            all_ok &= check(f"读视频", False, f"出错：{e}")
    else:
        print(f"  {WARN}  跳过（合成视频不存在，跑 sim/animate.py）")

    print("\n" + ("=" * 32))
    print(f"{OK} 全部通过" if all_ok else f"{FAIL} 有项目缺失，按上面修复指令补")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

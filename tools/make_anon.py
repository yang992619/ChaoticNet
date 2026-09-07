#!/usr/bin/env python3
"""由 paper/main.tex 生成匿名评审版 paper/main_anon.tex。

为什么要有这个脚本：两份 tex 手工同步必然漂移。2026-09-08 的检查发现匿名版
除署名外与正文完全一致——也就是说匿名版里原样留着作者的 GitHub 账号地址
（账号名含姓氏，评委一点即知是谁）与含"指导教师""所在物理系实验室"的致谢。
改法固定下来写成脚本，以后改完 main.tex 跑一次即可，不会再漏。

用法：
    python3 tools/make_anon.py          # 生成 paper/main_anon.tex
    python3 tools/make_anon.py --check  # 只检查是否已同步且无匿名红线，不写盘
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "main.tex"
DST = ROOT / "paper" / "main_anon.tex"

# 匿名红线：这些字样一旦出现在匿名版里就算越线
FORBIDDEN = [
    "崔天源", "高文博", "杨子诚", "郝晨煜",
    "yang992619",
    "河南理工",
]

ANON_URL = (
    r"项目代码、分析窗数据、模型权重、装置 BOM 与复现实验脚本已在公开代码托管平台开源；"
    "\n为遵守匿名评审要求，此处暂隐去仓库地址与作者账号，可经组委会向作者索取，"
    "\n并将在评审结束后随正式版一并给出。安装 \\texttt{requirements.txt} 后运行"
)

ANON_ACK = r"""\section*{致谢}
（为匿名评审需要，致谢内容暂略。）
"""


def anonymize(text: str) -> str:
    # 1) 署名清空
    text, n_author = re.subn(r"\\author\{[^}]*\}", r"\\author{}", text)
    if n_author != 1:
        raise SystemExit(f"预期恰好 1 处 \\author，实际 {n_author} 处")

    # 2) 隐去仓库地址（账号名构成身份泄露）
    old_url = (
        "项目代码、分析窗数据、模型权重、装置 BOM 与复现实验脚本公开于\n"
        "\\url{https://github.com/yang992619/ChaoticNet}；安装 \\texttt{requirements.txt} 后运行"
    )
    if text.count(old_url) != 1:
        raise SystemExit("没找到唯一的仓库地址段落，main.tex 可能已改动，请更新本脚本")
    text = text.replace(old_url, ANON_URL)

    # 3) 致谢整段替换
    ack = re.search(r"\\section\*\{致谢\}.*?(?=\n%\s*=+)", text, re.S)
    if not ack:
        raise SystemExit("没找到致谢段落")
    text = text[: ack.start()] + ANON_ACK + text[ack.end():]

    # 4) 正文里裸出现的仓库名保留（\texttt{ChaoticNet} 不含账号，不构成泄露）
    return text


def check(text: str) -> list:
    return [w for w in FORBIDDEN if w in text]


def main():
    src = SRC.read_text(encoding="utf8")
    out = anonymize(src)
    leaks = check(out)

    if "--check" in sys.argv:
        cur = DST.read_text(encoding="utf8") if DST.exists() else ""
        synced = cur == out
        print(f"匿名版与正文同步: {'是' if synced else '否——需重新生成'}")
        print(f"匿名红线泄露: {leaks if leaks else '无'}")
        sys.exit(0 if synced and not leaks else 1)

    if leaks:
        raise SystemExit(f"生成后仍有匿名红线泄露: {leaks}")
    DST.write_text(out, encoding="utf8")
    print(f"✓ 已生成 {DST.relative_to(ROOT)}（{len(out.splitlines())} 行），匿名红线检查通过")


if __name__ == "__main__":
    main()

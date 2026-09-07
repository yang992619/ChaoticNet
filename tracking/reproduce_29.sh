#!/bin/zsh
# tracking/reproduce_29.sh — 从源视频重建正式 29 段追踪 CSV
#
# 为什么需要这个文件（2026-09-07 补）：
#   data/tracking/ 在 .gitignore 里（245M 不入库），README 与 .gitignore 都写着
#   「可由源视频重跑 tracking 还原」。但还原所需的 CLI 参数此前只有 1430 那一条
#   记在 track_pendulum.py 的 docstring 里，其余 28 段无任何留档——没有 Makefile、
#   没有驱动脚本、data/tracking/ 下也没有任何 log。追踪跑在 2026-06-18，
#   仓库首次提交在 06-21，时间上也不可能被 git 记录。
#
# 为什么不能用默认参数（这是最要紧的一条）：
#   14xx 那批片子背景是深藏青，track_pendulum.py 的 HSV 默认蓝色明度门限
#   --blue-vmin 40 会把背景当成蓝标，每帧蓝候选中位涨到 27~76 个（29 段全中，全库中位 41；
#   逐段实测见 tools/count_blue_candidates.py -> data/tracking_ablation/blue_candidates_vmin40.json。
#   注：若连 --blue-smin 也用脚本默认 70，候选数会低一档，约 18~31 个，别把两套口径混着引）。
#   后果不是报错，是【静默出错】——脚本照样打印「命中率 100%、L2/L1≈0.78」一路绿灯，
#   但 IMG_1448 的释放角会从 66.4° 变成 41.9°，IMG_1432 从 111.2° 变成 11.1°。
#   而 111.2 正是 data/real_validation/summary_multi.csv 里那一行的头条数字。
#   必须抬高 --blue-vmin。好在正确区间很宽：vmin 140~160 / smin 55~60 /
#   gate-xmax 1780~1900 这一片跑出来 Δθ1 中位差 0.010°，远小于论文自报的
#   0.45° 状态估计噪声底。
#
# --t0 的语义：剪掉释放前的静止段与入镜的手。它决定每段的物理零时刻，
#   从而决定 summary_multi.csv 里该行的释放角与初始状态，是一等参数不是随手裁剪。
#   下面的 t0/t1 由 canonical CSV 的 frame 列反推得到，并已用源视频总帧数交叉验证
#   （t1 等于全片长度的段即当初未传 --t1）。
#
# 验证：本脚本重建的 CSV 与 canonical 逐字节相同（md5 一致）。
#
# 用法：
#   ./tracking/reproduce_29.sh                 # 全部 29 段
#   ./tracking/reproduce_29.sh 1432 1448       # 只跑指定段
#   TAG_PREFIX=zz_ ./tracking/reproduce_29.sh  # 写到 tracked_zz_1432.csv，不覆盖 canonical

set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${SRC:-实测视频/源视频归档}"
TAG_PREFIX="${TAG_PREFIX:-}"

# 29 段共用同一套 HSV 阈值——不是每段一套，只有时间窗逐段不同
COMMON=(--detector hsv --blue-vmin 160 --blue-smin 55 --gate-xmax 1780)

# tag  t0  t1        （t1 为 - 表示当初未传 --t1，跑到片尾）
WINDOWS=(
  "1430 2.5 200" "1431 2 194" "1432 2 210" "1434 2 331" "1435 2 285"
  "1436 2 164"   "1437 2 253" "1438 2 194" "1439 2 299" "1440 2 186"
  "1441 2 370"   "1442 2 -"   "1443 2 281" "1444 2 211" "1445 2 301"
  "1446 2 272"
  "1447 - -" "1448 - -" "1449 - -" "1450 - -" "1451 - -" "1452 - -"
  "1453 - -" "1454 - -" "1455 - -" "1456 - -" "1458 - -" "1459 - -"
  "1460 - -"
)

want=("$@")
for w in "${WINDOWS[@]}"; do
  set -- ${=w}
  tag=$1; t0=$2; t1=$3
  if (( ${#want} )) && [[ ! " ${want[*]} " == *" $tag "* ]]; then continue; fi
  args=("${COMMON[@]}")
  [[ "$t0" != "-" ]] && args+=(--t0 "$t0")
  [[ "$t1" != "-" ]] && args+=(--t1 "$t1")
  echo "=== IMG_$tag  ${args[*]} ==="
  python3 tracking/track_pendulum.py \
      --video "$SRC/IMG_$tag.MOV" --tag "${TAG_PREFIX}$tag" "${args[@]}"
done
echo "完成。核对：md5 data/tracking/tracked_<tag>.csv 应与 canonical 一致。"

# ChaoticNet

双摆混沌系统的物理约束神经网络预测 · 物理竞赛项目（截稿 2026-09-15）

研究问题：在本科实验室级精度（初值估计 ±0.5°、60fps 手机相机、手动释放且初值由视频反演）下，对比纯数据驱动 (Reservoir Computing) 和物理约束 NN (PINN) 对混沌轨迹的可信预测窗口。论点：高保真物理建模 + 物理结构能把可信视界扩到 ESN 的数倍（Hybrid 仿真内约 7×、真实零样本约 2.7×）。

> 权威数字以 `data/canonical_results_B.md` 为准（分布质量 B 口径，`tools/demo_doctor.py` 自动核验）。文档/PPT/论文凡与之冲突的旧值均以此源为唯一真相。

---

## 一、目录结构

```
ChaoticNet/
├── README.md                       本文件，项目总入口
├── CHANGELOG.md                    变更日志（最新 v3.1）
│
├── 文档/                           所有中文文档（2026-06-23 起统一归档于此）
│   ├── 报告与计划/                  项目执行计划（14 周 + 5 人分工）、成员贡献说明、
│   │                               人工智能使用声明、实验仪器说明、审校报告、冲刺计划书
│   ├── 操作手册/                    拍摄操作说明书、相机标定教程、视觉追踪交接、实验清单
│   └── 答辩/                       答辩讲稿、逐页讲稿、PPT 大纲、学习清单
│
├── diagrams/                       图纸（理论 + 装置 + 加工）
│   ├── README.md                   3 张主图速览 + 关键设计理由
│   ├── 01_physics_model.svg        双摆理论模型（θ₁/θ₂/L₁/L₂/m₁/m₂）
│   ├── 02_mechanical_front_view.svg 装置正视装配图（含尺寸）
│   ├── 03_camera_setup_top_view.svg 视觉测量俯视布置
│   ├── 04_frame_assembly.svg       框架装配步骤
│   └── parts/                      四件加工图（发加工厂）
│       ├── BOM.md                  物料清单（576-753 元）
│       ├── 组装书.md                两天装配手册（含力矩 + 验收）
│       ├── 轴承.md                  MR105ZZ 轴承规格 + 脱脂步骤
│       ├── 轴销.md                  P05/P06 圆柱销规格（v2 已缩短至 12/15mm）
│       ├── generate_dxf.py         三件铝板 DXF 生成脚本
│       ├── P01_arm_1.{svg,pdf}     摆臂 1（6061-T6 4mm，R12.5）
│       ├── P02_arm_2.{svg,pdf}     摆臂 2（6061-T6 4mm，R12.5）
│       ├── P03_brass_weight.{svg,pdf}  配重盘 v2.1（45# 钢 ⌀30×12mm，66g）
│       ├── P04_pivot_bracket.{svg,pdf} 主轴座（6061-T6 5mm，R10）
│       └── dxf/                    四份 DXF 直接发激光厂
│
├── sim/                            数值仿真（GT 对照）
│   ├── baseline.py                 无阻尼 RK45 解拉格朗日方程，50 组随机初值
│   ├── baseline_distmass.py        分布质量 B 版（头条数据底座），50 组
│   ├── baseline_damped.py          阻尼版，实测拟合 b ≈ 1.66e-5 N·m·s/rad
│   ├── fit_damping.py              能量耗散率法从实测反推粘性 + 二次空气阻力
│   ├── lyapunov.py                 50 组自测最大 Lyapunov 指数 → τ_L
│   ├── visualize.py                4 张图（时间序列/xy 轨迹/相空间/能量）
│   ├── animate.py                  单次仿真摆动 mp4
│   ├── animate_compare.py          三色对比 mp4（GT 白 / ESN 蓝 / PINN 红）
│   ├── plot_horizon_50.py          50 组视界分布四联图
│   └── plot_three_way.py           ESN / PINN / Hybrid 三方对比图
│
├── rc/                             AI 预测模型
│   ├── esn.py                      Echo State Network 基线 (~290 行)
│   ├── pinn.py                     物理约束 NN 基线 (~350 行，含 model.pt 保存)
│   ├── hybrid.py                   解析-NN 混合模型 (~180 行)
│   ├── lnn.py                      Lagrangian NN（Cranmer 2020，训练不稳，归档 future work）
│   ├── sweep_50.py                 50 组 ESN+PINN 评估（无阻尼）
│   ├── sweep_damped.py             50 组 PINN vs Hybrid 评估（阻尼）
│   ├── real_validation_multi.py    29 段真实录像零样本评估
│   ├── error_decomposition.py      真实短视界的三天花板误差分解
│   └── sensitivity.py              ±0.1°/0.5°/1.0° 初值精度敏感性扫描
│
├── tools/                          自动化工具 + loop 监控
│   ├── smoke.py                    项目自检（< 30 秒）
│   ├── acceptance.py               W3 装置验收（τ / 重复性 / 棋盘格）
│   ├── paper_audit.py              paper/README/csv 一致性 loop（A–F 六项检查）
│   ├── demo_doctor.py              头条数字与 canonical_results_B.md 自动核验
│   ├── live_demo.py                答辩交互演示：评委输初值，看三摆赛跑
│   ├── overlay_real_prediction.py  实拍视频上叠加零样本预测轨迹
│   ├── build_canonical_distmass.py 训练并冻结 canonical 权重（分布质量）
│   ├── canon_multirun_distmass.py  多 run 头条统计（仿真内 + 真实）
│   ├── lambda_ablation_distmass.py 物理损失权重消融（分布质量重跑）
│   ├── hp_sweep.py                 PINN 超参/架构 sweep（写 leaderboard）
│   ├── fair_compare.py             公平四方对比（ESN/PINN/Hybrid/λ=0 同 N 同 epoch）
│   ├── multi_seed.py               多 seed 鲁棒性（3 λ × 5 seed）
│   └── data_size_ablation.py       训练数据量 ablation（2/5/10/20/50 trial）
│
├── tracking/                       视觉追踪
│   ├── track_pendulum.py           ★ 正式管线：29 段 canonical CSV 由它产出（HSV / Lab 双检测路径）
│   ├── reproduce_29.sh             由源视频重建 29 段逐帧 CSV（逐段 --t0/--t1，抽验 5 段 md5 一致）
│   └── track.py                    早期版本，仅合成视频验证（0.12° 中位误差），非论文数据来源
│
├── paper/                          论文（36 页 PDF，含 3 节消融附录）
│   ├── main.tex                    中文 LaTeX (ctex) 主文件，9 章 + 摘要 + 目录 + 文献 + 4 附录
│   ├── references.bib              24 篇文献（双摆/ESN/PINN/SciML/视觉/阻尼/气象大模型）
│   ├── texstyles/gb7714/           GB/T 7714—2015 参考文献样式（随项目自带）
│   ├── main.pdf                    编译产物（最新版本，36 页）
│   └── 编译说明.md                  xelatex + biber 编译流程（含 TEXINPUTS 设置）
│
└── data/                           所有数据 + 图
    ├── canonical_results_B.md              ★ 全项目数字唯一真相源（分布质量 B 口径）
    ├── sim_distmass/                       分布质量 B 仿真数据（头条数据底座）
    ├── canon_multirun/                     canonical 多 run 统计（论文头条表来源）
    ├── real_validation/                    29 段真实录像零样本结果 + 图
    ├── ablation_distmass/                  分布质量口径下的消融重跑
    ├── sim/                                无阻尼 baseline 数据
    │   ├── trial_000.csv ~ trial_049.csv   50 组仿真轨迹（30s @ 120fps）
    │   ├── summary.csv                     初值 + 能量漂移汇总
    │   └── figures/                        + v1_baseline 存档 + 三色对比 mp4
    ├── sim_damped/                         阻尼仿真数据（v3 ground truth）
    │   └── trial_000.csv ~ trial_049.csv   50 组阻尼轨迹
    ├── rc/                                 ESN 输出
    │   ├── esn_summary.csv / esn_summary_50.csv
    │   └── figures/
    ├── pinn/                               PINN 输出 + 权重缓存
    │   ├── pinn_summary.csv / pinn_summary_50.csv
    │   ├── train_history.csv / model.pt
    │   └── figures/
    ├── hybrid/                             Hybrid 输出
    │   ├── hybrid_summary_50.csv / model.pt
    │   └── train_history.csv
    ├── hybrid_damped/                      PINN vs Hybrid on 阻尼数据
    │   ├── pinn_summary_damped.csv / hybrid_summary_damped.csv
    │   ├── pinn_damped_model.pt / hybrid_damped_model.pt
    │   └── compare_damped.csv
    ├── sensitivity/                        初值精度敏感性
    │   ├── sensitivity.csv / sensitivity_summary.csv
    │   └── sensitivity_box.png
    ├── tracking/                           视觉追踪逐帧结果（459M，未入库，见 .gitignore）
    │   ├── tracked_<tag>.csv               29 段正式 CSV（74.6M）+ 新片 CSV + overlay 抽帧 PNG
    │   └── tracking_compare.png
    ├── tracking_windows/                   ★ 随仓库发布：29 段分析窗（每段前 600 帧，共 3.1M）
    ├── figures/                            汇总论文图
    │   ├── horizon_compare_50.png          50 组视界 4 联图
    │   ├── three_way_compare.png           三方对比图
    │   └── three_way_summary.csv
    ├── fair_compare/                       公平四方对比（N=32 共同混沌）
    ├── multi_seed/                         多 seed 鲁棒性 results + summary
    ├── data_size/                          数据量 ablation results + curve
    ├── hp_sweep/                           超参 sweep leaderboard
    ├── lnn/                                LNN 训练历史 + 50 组评估（归档）
    ├── acceptance/                         装置验收报告（已完成：标定 0.43px、τ≈130s、阻尼系数并入论文）
    ├── lyapunov.csv                        50 组自测 Lyapunov 指数
    ├── horizon_compare_50.csv              联合表
    └── v2_50trial_2026-06-11/              v2 完整快照（README + 所有图 + 权重）
```

---

## 二、当前进度（截至 2026-09-04）

### ✅ 已完成

| 类别 | 产出 | 状态 |
|---|---|---|
| 硬件设计 | 4 件加工图 SVG + DXF + PDF + BOM + 装配手册 | 可发厂 |
| 物理仿真 | sim/baseline.py 无阻尼版 + sim/baseline_damped.py 阻尼版 v3，各 50 组 | 可用 |
| 仿真可视化 | sim/visualize.py 4 张图 + sim/animate.py 单摆 mp4 | 已存档 |
| ESN 基线 | rc/esn.py，800 储池 | 已存档 |
| PINN 基线 | rc/pinn.py，128×4 MLP + 拉格朗日 loss + 权重缓存 model.pt | 已存档 |
| Hybrid 模型 | rc/hybrid.py，prior(解析 RHS) + 小残差 NN | 已实现 |
| 50 组 sweep | rc/sweep_50.py（无阻尼）+ rc/sweep_damped.py（阻尼） | 已跑 |
| 三方对比 | sim/plot_three_way.py，ESN/PINN/Hybrid 横向 | data/figures/ |
| 三色对比动画 | sim/animate_compare.py，GT 白 / ESN 蓝 / PINN 红 | 已生成 mp4 |
| 视觉追踪 | tracking/track.py，OpenCV HSV + 最小二乘圆拟合，0.12° 中位误差 | 合成视频已验证 |
| 初值敏感性 | rc/sensitivity.py，±0.1°/0.5°/1.0° × K=10 | 已扫描 |
| 消融实验 | λ多seed（3λ×5seed）+ data_size（5 档）+ hp_sweep | 已跑（分布质量重跑），进附录 |
| LNN | rc/lnn.py，Lagrangian NN（训练不稳，中位仅 0.08s） | 归档 future work |
| 一致性监控 | tools/paper_audit.py，paper/README/csv 六项（A–F）自动核对 | loop 可用，6/6 绿 |
| 论文 | paper/main.tex 36 页 PDF（竞赛格式：小四 + 24 磅行距 + 目录 + GB/T 7714 文献），24 篇文献，3 节消融附录，全 TODO 清零 | 已完成，持续精修 |
| 执行计划 | 文档/报告与计划/项目执行计划.md，14 周分阶段 + 5 人分工 | 可发群 |
| 交互演示 | tools/live_demo.py（评委输初值看三摆赛跑）+ tools/demo_doctor.py 一键自检 | 可现场演示 |
| 真实预测叠加 | tools/overlay_real_prediction.py，实拍视频上叠加零样本预测轨迹 | 已出演示视频 |
| 答辩 / 展示物料 | 答辩 PPT（嵌蝴蝶效应开场 + 29 段战绩墙，另有完整演示版）+ A3 深蓝高清海报 | 已出片 |

### 进度（截至 2026-09-04）

| 类别 | 任务 | 状态 |
|---|---|---|
| 硬件 | 加工件 + 淘宝件下单 → 到货 → 装配 | ✅ 已完成 |
| 实测 | 单摆 τ 衰减测试（目标 ≥ 30s） | ✅ 实测 τ≈130s，超标 4× |
| 视觉 | 棋盘格内参标定 + 实物视频跑 tracking | ✅ 标定 0.43px，29 段已追踪 |
| AI | 分布质量重训 PINN/Hybrid（sim→real 零样本，按设计不在真实数据上训练） | ✅ 已完成 |
| 论文 | 实测章节填充 + 消融分布质量重跑 + 竞赛格式改版 | ✅ 36 页，持续精修 |
| AI | Lagrangian NN（训练不稳，中位仅 0.08s） | 🟡 归档 future work |
| 协作 | git 本地版本控制 | ✅ 已建（尚未配置远程仓库，GitHub / Overleaf 待推）|
| 交付 | 匿名投稿版论文 + 作品介绍 + 海报 + 答辩 PPT | ✅ 已成套（截稿前持续精修）|

---

## 三、关键论点（分布质量模型，50 组仿真 + 29 段真实，多 run 诚实复算）

物理结构模型（PINN/Hybrid）显著优于纯数据 ESN，每个模型多次独立训练取 per-run 中位：

| 模型 | 视界 中位 | 归一化 / τ_L | 相对 ESN 提升 |
|---|---|---|---|
| ESN | 0.70s | 0.91 τ_L | 1× |
| PINN | 0.88s | 1.15 τ_L | 1.3× |
| 纯数据 MLP | 1.55s | 2.03 τ_L | 2.2× |
| Hybrid | 2.85s | 3.74 τ_L | 4.1× |

> **2026-09-08 ESN 闭环修正**：上表 ESN 行为修正后的值（旧值 0.40s / 0.53 τ_L，倍数 2.2/3.8/7.0）。
> 旧闭环预测入口把训练末帧输入喂了两遍，使预测窗第 0 帧即带约 1.9° 偏差。修正后 ESN 落在
> 文献报告的"约 1–2 个 Lyapunov 时间"区间内，说明基线未被削弱。根因、逐位等价验证与影响面清单见
> `tools/esn_closedloop_fix.py` 与 `data/esn_closedloop_fix/`。**真实零样本那张表不受影响**
> （真实侧调用本就与训练契约一致）。

仿真内混沌子集（每 run 取 horizon<9.9s）；ESN 多 reservoir seed 中位。真实零样本迁移（29 段）：
ESN 0.167s / 纯数据 0.358s / PINN 0.417s（2.5×）/ Hybrid 0.450s（2.7×）；物理损失真实增益 1.16×。

τ_L = 0.76s（分布质量 50 组扰动轨迹自测最大 Lyapunov 时间中位，sim/lyapunov.py）。
关键结论：sim→real 迁移首要驱动是仿真保真度（分布质量匹配真硬件，点质量 PINN 真实仅 0.27s）；
物理损失是分布迁移下的稳健性正则（仿真内反而无益），独立于 τ_L 绝对值估算。
完整数字来源 data/canonical_results_B.md。

初值敏感性（仿真内混沌子集 49 组，每档 K=20 次独立扰动，修正协议 tools/sensitivity_correct.py）：

| 扰动档 | PINN | Hybrid | 解析真解 |
|---|---|---|---|
| 无扰动 | 1.48s | 3.69s | 10.00s |
| σθ=0.45°（本研究实测） | 1.38s | 3.33s | 4.69s |
| σθ=1.0° | 1.15s | 2.35s | 2.47s |
| 联合（0.45° + 0.09 rad/s） | 1.33s | 2.82s | 3.45s |

读法：解析真解是唯一"纯净"的敏感性曲线（无扰动时逐位复现真值），它从 10s 塌到 3.45s，说明混沌放大真实存在；
而 PINN 只从 1.48 掉到 1.33（−10%），是因为模型自身的逼近误差已经大于这个量级的初值噪声。
所以结论不是"对初值稳健"，而是"当前瓶颈在模型逼近误差，不在初值精度"。
（注：早先版本的扫描协议把扰动逐位抵消了，模型看到的初值误差恒为 0，因此曾错误得出"越扰动视界越长、对初值稳健"。
诊断见 data/sensitivity_correct/old_protocol_diagnosis.txt。）
视觉追踪精度（合成视频，0.12° 中位误差）说明算法本身很准；真实拍摄下的初值状态估计噪声约 0.45°（来自实拍光照/反光/运动模糊），是真实视界的主要可改进来源之一。

### v2.1 消融发现（分布质量 B 口径重跑，2026-06；点质量期归档存于 data/*_pointmass_bak_20260621/）

训练数据量 ablation（PINN λ=1.0，全集 50 trial 评估，3 seed 均值）：

| 训练 trial 数 | 视界均值（3 seed） |
|---|---|
| 2 | 0.31s |
| 5 | 0.70s |
| 10 | 2.00s |
| 20 | 2.60s |
| 50 | 5.52s（注：训练=评估集，含 in-sample 重叠）|

反直觉发现：干净仿真数据上 phys loss 反而拖后腿（分布质量下更显著）。多 seed 验证（5 seed × 3 λ）：

| λ_phys | 视界均值（5 seed） |
|---|---|
| 0.0（纯数据 MLP）| 1.74s |
| 1.0（PINN）| 0.60s |
| 2.0 | 0.64s |

λ=0 在 5/5 个 seed 上都 > λ=1，稳健不是单 seed 巧合。解释：baseline 数据本身完美符合 Lagrangian，data loss 梯度已足够，phys loss 成冗余约束。已被 29 段真实数据证实：物理损失的价值在 sim→real 迁移才显现（真实 λ=0.1 中位 0.42s > λ=0 的 0.36s）。

---

## 四、关键决策已锁定（不要再讨论）

| 决策 | 已锁定值 | 理由 |
|---|---|---|
| 摆长 | L₁=250mm, L₂=200mm | 非对称避退化模态，手机视场可框入 |
| 摆臂 | 6061-T6 铝板 4mm 厚 | 匹配 MR105ZZ 4mm 宽 |
| 轴承 | MR105ZZ ×2（脱脂 + 钟表油） | 目标 τ ≥ 30s（实测 τ≈130s，超标 4×）|
| 配重 | P03 v2.1 45# 碳钢 ⌀30×12mm，66g | 钢比黄铜便宜、刚度足够 |
| 初值精度 | 手动释放，初值从视频反演（估计不确定度 ≈0.5°） | 本科实验室级，本身是论文论证对象 |
| 相机 | 真实 60fps（手机即可）；仿真采样 120fps | 实测最大角频率 4.11 Hz（29 段 41 万帧峰值），60fps 约 15× 过采样，视界按秒算与帧率无关 |
| 视觉标记 | 方形彩色贴片（铰接处蓝、末端浅红，等效直径约 12/16mm）| 实拍实际采用；BOM 里的 3M 反光荧光红 ⌀5mm 圆点属设计方案，29 段未用 |
| 背景 | 摆后方深色织物（画面两侧可见浅色墙面）| 实拍实际采用；BOM 里的哑光黑亚克力背板属设计方案，29 段未用 |

详细论证见 `diagrams/README.md` 节"关键设计理由"。

---

## 五、快速上手

### 环境

```bash
pip3 install --break-system-packages --user -r requirements.txt
```

LaTeX（论文编译）：brew install --cask basictex 然后 `sudo /Library/TeX/texbin/tlmgr install ctex biblatex biber physics cleveref subcaption algorithms enumitem titlesec booktabs setspace`
GB/T 7714—2015 文献样式已随项目放在 `paper/texstyles/gb7714/`，编译时带上 `TEXINPUTS="./texstyles/gb7714//:"` 即可，无需另装。

### 一键复现（从零到 36 页 PDF）

```bash
python3 tools/smoke.py                      # 项目自检 < 30 秒

# 1. 分布质量仿真数据 + Lyapunov（tau_L）
python3 sim/baseline_distmass.py            # 分布质量 50 组轨迹（头条数据底座），~1 分钟
python3 sim/baseline_damped.py              # 阻尼场景（第八节限制，可选 --b 7e-4）
python3 sim/lyapunov.py                     # 实测最大 Lyapunov 指数 -> tau_L≈0.76s

# 2. canonical 头条（仿真内 + 真实零样本，多 run 中位[IQR]）
python3 tools/build_canonical_distmass.py   # 训练并冻结 canonical 权重 data/pinn|hybrid/model.pt
python3 tools/canon_multirun_distmass.py    # 多 run 统计 -> tab:horizon-compare + tab:real-horizon（25 次训练，最耗时的一步）
python3 rc/real_validation_multi.py         # 29 段真实零样本 -> summary_multi.csv（需 data/tracking/*）

# 3. 消融
python3 tools/lambda_ablation_distmass.py   # 物理损失权重 {0,1,2}×5seed -> tab:ablation-lambda
python3 tools/data_size_ablation.py         # 训练数据量 -> tab:ablation-data

# 4. 论文图 + 编 PDF
python3 sim/plot_three_way.py               # 三方对比图
python3 rc/error_decomposition.py           # 三天花板误差分解图
cd paper && TEXINPUTS="./texstyles/gb7714//:" latexmk -xelatex main.tex   # 36 页 PDF
```

全流程 CPU 约 1 小时量级（以多 run 训练为主，机器更慢则更长）。注意 build_canonical_distmass.py 每次都会重新训练，不做权重缓存判断。
旧 λ=1.0 基线脚本 `rc/sweep_50.py` / `sweep_damped.py` / `sensitivity.py` 仍可单独跑做对照，
但**非论文头条来源**；`sweep_50` 权重独立缓存在 `data/pinn/sweep50_model.pt`，不会覆盖 canonical。
真实零样本需 `data/tracking/*.csv`（245M，未入库）；无追踪数据时直接读已入库的
`data/real_validation/summary_multi.csv` 复核头条。

自检 + 一致性监控：

```bash
python3 tools/smoke.py               # 项目自检 < 30 秒
python3 tools/paper_audit.py         # paper/README/csv 六项一致性核对（A–F）
```

### 看 PDF 加工图

```bash
open diagrams/parts/P01_arm_1.pdf    # 任意 P01-P04
```

### 发给加工厂

把 `diagrams/parts/dxf/` 里 4 个 DXF + 对应 PDF 打包发激光切割厂 / 机加工厂。话术见 `diagrams/parts/BOM.md` 节"给加工厂打招呼"。

---

## 六、下一步阅读

1. `文档/报告与计划/项目执行计划.md` —— 14 周完整计划、5 人分工、风险预案、阶段验收
2. `diagrams/parts/BOM.md` —— 采购清单 + 验收
3. `diagrams/parts/组装书.md` —— 两天装配 SOP
4. `diagrams/README.md` —— 关键设计决策背后的物理理由

---

## 七、版本

| 版本 | 日期 | 改动 |
|---|---|---|
| v1.0 | 2026-06-11 | 初版。装置设计冻结，仿真 + ESN + PINN 三个基线落地 |
| v2.0 | 2026-06-11 | 加 Hybrid 模型 + 50 组完整 sweep + 阻尼仿真 + tracking + 初值敏感性 + 9 页论文 PDF |
| v2.1 | 2026-06-12 | loop 模式上线（paper_audit/hp_sweep/fair_compare）+ 多 seed/数据量消融 + LNN 归档 + 14 页论文 PDF |
| v3.0 | 2026-06-21 | 分布质量 B 迁移（点质量→分布质量，主口径切换）+ 真实零样本全管线（29 段实拍，Hybrid 2.7×）+ 工程质量五项闭环 + git 版本控制上线 + 19 页论文 PDF |
| v3.1 | 2026-09-04 | 答辩交付物成套（交互演示 live_demo + 实拍预测叠加 + PPT/海报）+ 文档归入 文档/ + 口径清扫 + 论文竞赛格式改版 19→32 页（目录 / 小四 24 磅 / GB/T 7714 文献，24 篇）|

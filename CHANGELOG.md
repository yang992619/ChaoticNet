# ChaoticNet 变更日志

## v3.1 · 2026-09-04（答辩交付物成套 + 论文竞赛格式改版 19→32 页 + 仓库口径清扫）

汇总 v3.0（2026-06-21 凌晨）之后的全部工作。数字口径未变，仍以 `data/canonical_results_B.md`
为唯一真相源；`tools/paper_audit.py` 6/6 全绿。

### 已入库提交（2026-06-21 深夜 – 06-23）

- 真实录像零样本预测：新增 `tools/overlay_real_prediction.py`（实拍视频上叠加模型预测）与
  `tools/embed_demo_into_pptx.py`（把叠加视频嵌成答辩 deck 的演示页）。
- 答辩交互演示：新增 `tools/live_demo.py`——评委现场输初始角度，看 ESN / PINN / Hybrid 三摆赛跑。
  随后 6 轮体验迭代：脱靶后保身份色可辨 + 重置守卫 + 播放速度滑块；解耦视觉刷新与播放速度
  （全速度段稳定 60fps）；仿真 fps 退回 60 修 ESN 闭环锁死并加锁死诚实标记；角度/角速度滑块旁
  加精确输入框 + 三摆常驻预览；进度条 / 暂停 / 可拖动擦洗；配套 `tools/demo_doctor.py` 一键自检。
- 第三、四轮多智能体审计：修复答辩 PPT 消融页残留的点质量旧值、归档点质量旧 tex；
  严谨性补强 + 可复现性收口 + 许可与清理。
- 口径清扫：入口文档统一到 v3 分布质量口径、收紧唯一真相源；旁支文件残留的旧值同步到 v3，杜绝误读。
- 答辩 PPT：嵌入蝴蝶效应开场动画 + 29 段真实战绩墙、修下标字形，另出「完整演示版」deck。
- 目录治理：散落根目录的文档统一归入 `文档/`（答辩 / 操作手册 / 报告与计划）；
  清理 06-13 废稿素材、早期骨架、截图产物，`build_slides` 旧生成器归档。

### 工作区改动（07–09 月，本条目发布时随同提交）

- **论文 19 页 → 32 页**（`paper/main.pdf`）。主要是竞赛格式规范改版 + 内容补充：
  正文改小四 12pt、行距≈24 磅；新增目录页；参考文献改 GB/T 7714—2015 顺序编码制
  （正文引用改右上标，样式文件随项目放在 `paper/texstyles/gb7714/`，编译需带 `TEXINPUTS`）；
  cleveref 交叉引用中文化（节/图/表/式/算法/附录）；图表标题小一号加粗；页眉去除、页码居中；
  新增「讨论与展望 · 应用前景」小节；新增双摆物理模型示意图 `fig_pendulum_schematic.png`。
- **文献 17 → 24 条**（`paper/references.bib`），新增分布质量摆、物理约束学习综述、PINN 电力应用、
  DeLaN、GraphCast、盘古气象、物理储池计算；正文 24 处引用与 bib 条目一一对应。
- **论文插图统一重制**：`three_way_compare` 改 2+1 版式（原三联横排压得难读）并标注 N=32 混沌子集口径；
  误差分解、多视界-角度、能量扫描、PINN 训练 loss、实拍叠加图的字号与正文协调、dpi 提高、
  改 `bbox_inches="tight"`；`diagrams/01_physics_model.svg` 同步更新。
- `tools/paper_audit.py` 检查 B 支持 tex 中显式写出的相对路径插图（如 `../data/...`）。
- `paper/编译说明.md` 补 `TEXINPUTS` 与 GB/T 7714 样式说明。
- **新增交付物**：匿名投稿版论文（`paper/main_匿名投稿版.pdf`，32 页）、作品介绍与评委意见修订版 PDF、
  A3 深蓝高清海报（`poster/`、`混沌之眼_A3海报_深蓝高清修订版.pptx`）、命题类蓝色样式答辩 PPT，
  以及多轮 PPT 图表放大 / 投影可读性 QC 产物（`ppt_*_2026081x/`、`.ppt_workflow_qc/`、`video_restore_20260815/`）。
- **新增脚本**：`tools/render_pendulum_schematic.py`、`tools/report_figure_qc.py`、
  `tools/fig_butterfly_distmass_large.py`、`tools/fig_error_decomposition_large.py`、
  `tools/fig_real_horizon_large.py`、`tools/make_errdec_slide.py`、
  `tools/build_pptx_blue_competition_style.py`、`sim/plot_insim_canon_4model.py`、
  `sim/replot_pinn_loss.py`、`tracking/replot_qc_1448.py`。
- **新增文档**（`文档/`）：人工智能使用声明、成员贡献说明、实验仪器说明文档、
  论文审校报告_2026-07-14、截稿冲刺计划书_2026-09-04。
- **口径加固**：`tools/bootstrap_sim.py` / `bootstrap_real.py` 的置信区间改用分布质量数据源，
  不再读点质量期的 `data/v2_50trial_2026-06-11/horizon_compare_50.csv`（该文件与旧 README 一并
  标注为仅供存档）；点质量期旧权重 `model_pointmass.pt` / `model_oldparams.pt` 移入
  `archived/old_models/`。
- **清理**：删除根目录 Office 崩溃残留的隐藏临时文件 `.~*.pptx`；README 的页数 / 文献数 /
  检查项数 / 文档路径同步为现状。

## v3.0 · 2026-06-21（分布质量迁移 + 真实零样本全管线 + 工程质量五项闭环）

点质量 A → 分布质量 B 的 v3 级跃迁（6/17–6/21 汇总）：

- 物理模型：全套从点质量切到分布质量 B（M1≈96.5g、M2≈139g、质心距/转动惯量由 BOM 反推），data/sim 即 B；点质量底档存 data/_pointmass_canonical_backup_20260619/。
- 真实零样本：29 段实测双摆 sim→real 全管线跑通，零样本迁移（按设计不在真实数据上训练）。真实中位 ESN 0.167s / PINN 0.417s / Hybrid 0.450s。
- 阻尼拟合：sim/fit_damping.py 能量耗散率法从 16 段反推 b≈1.66e-5（粘性）+ c≈8.72e-6（二次空气阻力，占总耗散≈78%），写入 baseline_damped 默认值。
- 单摆 τ 衰减实测：90° 释放重复两次（552.0 / 550.8s），等效 τ≈130s，超标 4×。
- 消融重跑：tab:ablation-lambda 与 tab:ablation-data 用分布质量重跑替换点质量逐数值（新增 tools/lambda_ablation_distmass.py、data/ablation_distmass/）；λ=0 完胜 λ=1 由 1.5× 增至 2.9×，数据量每翻倍约 2×（50 trial 中位 5.52s≈7.3 τ_L）。
- 论文：14 页 → 19 页 PDF，全 TODO 清零，tools/paper_audit.py 6/6 绿；衰减测试节口径对齐 fit_damping，§sim2real 误差分解三天花板（1.35 / 1.90 / 0.42）。
- 答辩 PPT：改用 ppt-master 生成 23 页瑞士极简（分布质量数字），导出 混沌之眼_答辩PPT_瑞士极简.pptx + Word 版；旧 06-13 网页 PPT（ppt-magazine/ ppt-swiss/，点质量旧值）已被取代。
- 工程质量五项闭环：fit_damping 叙述对齐、验收报告口径统一（标定 0.43px / τ≈130s 已入论文）、A/B 决策落地、消融口径统一、git 版本控制初始化。
- 杂项：新增 requirements.txt；参考_早期骨架/ 归档进 archived/；删除过时的 论文τ补丁_粘贴用.txt（论文已可直接编辑，workaround 作废）。

## v2.1.2 · 2026-06-13（答辩 PPT 双风格 + 网页转 pptx 管线）

新增：

- ppt-magazine/ — 杂志风（电子杂志 × 电子墨水，靛蓝瓷主题）网页 PPT，14 页，guizang-ppt-skill 风格 A
- ppt-swiss/ — 瑞士国际主义风（克莱因蓝 IKB）网页 PPT，13 页，风格 B，过 validate-swiss-deck 校验，用 12 种 S 版式
- tools/shoot_slides.mjs — playwright-core + 系统 Chrome 逐页全屏截图（1920×1080@2x），截图前隐藏 #nav/#hint
- tools/build_pptx_from_shots.py — 截图拼成 16:9 全屏图 pptx
- 产物：混沌之眼_答辩PPT_杂志风.pptx（14 页）、混沌之眼_答辩PPT_瑞士风.pptx（13 页）

说明：两版 PPT 共用同一故事线（0.15→0.64s 逼近 0.81s 天花板、3.4×、λ=0 反直觉、全仿真口径诚实），
4 张结果图（three_way / horizon / sensitivity / datasize）都已嵌入。pptx 为整页图片，改文案需回改
ppt-*/index.html 再重跑 shoot + build 脚本。待明天收集他人意见后迭代。

## v2.1.1 · 2026-06-13（τ_L 归一化修正 + audit 加固）

修正：

- sim/plot_three_way.py 的 TAU_L 从遗留的 0.12 改为自测 0.81，与论文 §2 / README 统一。
  之前 three_way_summary.csv 的 tau_L 列（1.25/5.14/5.42）和 three_way_compare.png 的
  τ_L 参考线用的是旧的 0.12 基准，跟论文表 1（0.19/0.79/0.80）打架。
- 重新生成 csv + png，重编论文 PDF 嵌入新图（仍 14 页）。
- tools/paper_audit.py 新增检查 F：核 csv 的 tau_L 列 = 中位/0.81 + 画图脚本常量为 0.81，
  堵上"只核中位秒数、漏核归一化"的洞（这次靠 chaos-status 肉眼发现）。顺手修 docstring 的 \cite SyntaxWarning。

## v2.1 · 2026-06-12（loop 模式上线 + 反直觉发现归档）

新增：

- tools/paper_audit.py — paper / README / csv 一致性监控 loop，5 项检查全自动
- tools/hp_sweep.py — PINN 超参 / 架构 自动 sweep，12 组完整 search 写出 leaderboard
- tools/fair_compare.py — 公平四方对比（ESN/PINN/Hybrid/λ=0 在同 N 同 epoch 同 seed）
- baseline_damped.py 加 --b/--n/--out CLI 参数（实测 τ 后可直接换 b）
- animate_compare.py 加 --trial/--fps/--slow CLI 参数
- tracking/track.py fit_circle 加 try/except 回退

⚠️ 反直觉发现（已归档 data/fair_compare/，paper 未改）：

在干净 baseline 仿真数据 + 同 N=27 同 epoch=300 同 seed=42 公平对比下：

| 模型 | 视界中位 | vs ESN |
|---|---|---|
| ESN | 0.150s | 1.0× |
| PINN（λ=1.0） | 0.617s | 3.38× |
| Hybrid | 0.650s | 3.56× |
| λ=0 纯数据 MLP | 0.758s | 4.88× ← 最高 |

phys loss 在干净仿真数据上不仅没帮助，还略拖后腿（PINN/lam0 = 0.93×）。
解释：baseline 数据本身完美符合 Lagrangian，data loss MSE 给的梯度已足够；phys loss 是冗余约束。
但 PINN/Hybrid 在 noisy 实测数据上仍可能有用（W4 实测后验证）。

**保留发现，paper 主论点暂不改。**等 W4 实测数据出来 + 终稿阶段再 decide：
- 选项 A：加 λ=0 作消融基线进主表，新故事"phys loss 真正价值在实测 OOD 数据"
- 选项 B：λ=0 只放附录 ablation，保留原 3.4× 主论点

### 后续确认（同日深夜）

multi_seed 5 seed × 3 λ 鲁棒性验证（tools/multi_seed.py）：

| Seed | λ=0 | λ=1 |
|---|---|---|
| 42 | 0.838 | 0.508 |
| 7  | 1.425 | 1.229 |
| 13 | 1.425 | 0.738 |
| 21 | 1.038 | 0.538 |
| 100 | 0.592 | 0.538 |

λ=0 在 5/5 个 seed 上视界都 > λ=1，均值 1.063s vs 0.710s（~1.5× 提升）。
→ v2.1 反直觉发现稳健，不是单 seed 巧合。

LNN 实施尝试（rc/lnn.py）：

简化版 LNN（Tanh MLP 128×4 + cosine lr + grad clip）训练 loss 剧烈震荡（两个数量级间跳变），
最终视界中位仅 0.08s。归档为 future work，paper 已加 LNN 实施挑战章节。
Cranmer 原工作使用 LBFGS / 二阶优化器 / deep ensemble，本简化实现不够。

---

## v2.0 · 2026-06-11（晚间冲刺）

19 项产出，全部在一次会话里完成。

### 新增数据 / 模型

- 阻尼仿真 baseline_damped.py（v3 ground truth，b ≈ 1e-4 N·m·s/rad）
- 50 组完整 sweep（ESN + PINN + Hybrid，三套独立 model.pt）
- 阻尼数据上 PINN vs Hybrid 50 组对比
- 初值精度敏感性扫描（±0.1° / 0.5° / 1.0°，每档 K=10）
- 50 组实测 Lyapunov 指数（中位 λ=1.24/s，τ_L=0.81s）

### 新增工具

- rc/hybrid.py 解析-NN 混合模型（Universal Differential Equation 范式）
- rc/sweep_50.py / sweep_damped.py / sensitivity.py
- sim/animate.py / animate_compare.py（三色对比 mp4）
- sim/plot_horizon_50.py / plot_three_way.py / lyapunov.py
- tracking/track.py（OpenCV HSV + 最小二乘圆拟合，0.12° 中位误差）
- tools/smoke.py（项目自检 < 30 秒）
- tools/acceptance.py（W3 装置到货验收：τ / 重复性 / 棋盘格标定）

### 论文（paper/main.tex）

- 从 5 页骨架扩展到 12 页 PDF
- 中文 + 英文双摘要
- 引言文献综述写完（15 篇文献，覆盖双摆 / RC / PINN / SciML / 视觉 / 阻尼）
- 物理模型 M / b 矩阵显式展开 + Lyapunov 自测节
- 实验装置章节填实（含 R5 圆角 P04 等最新硬件设计）
- 视觉系统章节（含 0.12° 跟踪精度验证）
- AI 模型章节（ESN / PINN / Hybrid 三种 + PINN 训练 algorithm 1 伪代码）
- 结果与讨论（基于 50 组真实统计，3.4× 中位提升）
- Appendix：仓库结构 + 一键复现命令 + 超参数表
- 所有红字 [TODO] 清零

### 项目治理

- .gitignore
- 全 Python 模块 sys.path 兼容（无 cwd 依赖）
- v2_50trial_2026-06-11/ 完整快照（数据 + 图 + 权重）

### 核心数字（论文最终主张）

|  | ESN | PINN | Hybrid |
|--|--|--|--|
| 视界 中位（s） | 0.150 | 0.642 | 0.650 |
| 视界 中位 / τ_L | 0.19 | 0.79 | 0.80 |
| 提升 vs ESN | 1× | 3.4× | 3.6× |

τ_L = 0.81s（50 组自测中位值，独立于文献的 0.12s 估计）。

### 下一会话接着干

W3 装置到货后：

1. 跑 tools/acceptance.py（τ 衰减 + 重复性 + 棋盘格）→ 拿到实测 b（阻尼系数）
2. 用实测 b 重跑 sim/baseline_damped.py
3. tracking/track.py 直接对实物视频（可能要调 HSV 阈值给 3M 反光荧光红）
4. 用实测数据替换或对照 50 组 sweep
5. paper 补"实测 vs 仿真"对比章节

W4 起：

- 论文反复修订（导师 + 队员 2 轮 review）
- 视觉系统标定章节实测数据补
- 阻尼系数实物拟合后重训 Hybrid 看是否真显优势

## v1.0 · 2026-06-11（清晨）

- 装置设计冻结（P04 圆角从 R10 改 R5，配重盘从 13mm 改 12mm 加速加工）
- sim/baseline.py 50 组无阻尼仿真
- rc/esn.py + rc/pinn.py 各 5 组 baseline
- paper/main.tex 5 页骨架
- 项目执行计划 14 周 + 5 人分工

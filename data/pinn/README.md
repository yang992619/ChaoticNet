# data/pinn/ — PINN 权重与训练产物

## 唯一正式权重：model.pt

**`model.pt`（2026-06-19，MD5 `9331eb365f67e5e3646bf9aede67059b`）是本项目 PINN 的唯一正式权重，
对应 B 版「分布质量」（distributed mass）物理参数。论文正文、答辩材料、
data/canonical_results_B.md 中所有 PINN 相关数字，全部来自这份权重。**

加载它的脚本（勿改路径）：

- `rc/real_validation.py` / `rc/real_validation_multi.py` —— 真实数据零样本验证
- `rc/lam0_real_baseline.py` —— λ=1（物理约束）对照分支
- `sim/animate_compare.py` —— 三方对比动画（缓存权重；文件缺失时会即时重训 3–5 分钟）
- `tools/smoke.py` —— 自检时检查该文件存在

## 已移走的旧权重（2026-09-04）

以下两份历史权重原本与 `model.pt` 混放在本目录，现已移至
`archived/old_models/`：

| 文件 | 日期 | 内容 | MD5 |
|---|---|---|---|
| `model_pointmass.pt` | 2026-06-17 | 点质量（point mass）旧版模型，对应已废弃的 3.4× 那套数字 | `40fd16c6144284f729fc9f9d81a7145f` |
| `model_oldparams.pt` | 2026-06-11 | 更早的一版旧参数模型 | `d6e2c01c3a1b4979fd82da36157080b0` |

**移走原因**：三份权重同名前缀、同目录、体积相近（均约 200 KB），
任何脚本若误写成 `torch.load('data/pinn/model_pointmass.pt')`，PyTorch 会**静默加载**
旧模型——不报错、不告警，直接产出错误数字，且从结果上很难察觉。
2026-07-13《项目体检报告》将此列为🟠高优先级风险项。移动前已全仓 grep 确认
**没有任何代码引用这两个文件名**（仅体检报告和计划书等文档提及），因此移动不影响任何脚本。

**这两份旧权重的输出数字一律不得引用。** 它们仅作历史追溯保留。

## 其他文件

- `pinn_summary.csv`（06-11）、`pinn_summary_50.csv`（06-17）—— 训练/评测汇总表
- `train_history.csv` —— 训练损失历史
- `figures/` —— 训练曲线与预测对比图；`figures/v1_baseline_2026-06-11/` 是 v1 旧基线图，另见其内 README

## 规矩

往本目录新增权重前，先想清楚：如果它不该被当成正式权重用，就别放这儿——
放 `archived/old_models/` 并在上表补一行。本目录**只应存在一份 `.pt`**。

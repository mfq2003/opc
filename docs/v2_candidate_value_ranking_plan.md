# v2 候选动作价值排序实施计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-22
- Verification Status: PARTIALLY VERIFIED（WP0–WP2模型族筛选已执行；WP2b及真实Solver未执行）
- Version Label: candidate_value_ranking_plan_v1

## 1. 研究问题

在不让模型直接生成最终 Recipe 的前提下，候选动作价值模型能否在保持坐标搜索质量的同时，
减少真实 OPC Solver 对 `-20/-10/+10/+20 nm` 四个候选动作的评估次数？

模型只负责 Proposal，真实 Solver 保留 Veto 权：只有候选的 L2/EPE/PVB 均不高于固定全零
基线，且 J 严格低于当前 incumbent，动作才允许接受。

## 2. 当前证据基线

- 搜索源：`runs/20260915T014128Z-v2-search-5697a08c/`。
- 范围：M1_test1–6、seed=0、1306 个点、5224 次候选 Solver 调用。
- 坐标搜索结果：五张图改善，平均 J 降幅 15.20%，六图最终回放一致；仍为
  `diagnostic_only`，不是 PPO、跨图泛化或正式验收。
- 2026-09-22 零 Solver 审计：1306 个动作组中，792 组（60.64%）具有 mask、指标和 J
  差异；514 组（39.36%）完全不变；284 组（21.75%）实际改善 incumbent。
- 五张改善版图的全部累计 J 改善 108482 都来自可排序组；M1_test3 的 260 组全部不变。

因此当前证据支持进入离线价值基线，但只支持“既有搜索轨迹上的单步排序问题”。它不能支持
离线模拟完整新策略，也不能证明最终 Recipe 质量。

## 3. 假设与变量

### 主假设

冻结特征与策略后，Top-2 或自适应 Top-k Proposal–Veto 搜索可以在训练版图上保留至少 95%
的完整四动作坐标搜索 J 改善，同时减少至少 40% 的候选 Solver 调用，并保持每项质量护栏。
95%/40% 是本计划提出的门槛，不是项目既有正式 acceptance 标准。

### 自变量

- 候选策略：随机顺序、固定顺序、模型 Top-1、模型 Top-2、自适应 `k∈{0,1,2,4}`。
- 特征版本：基础预测时特征、增加局部邻域连续几何特征。
- 模型：常数/随机基线、ExtraTrees/RandomForest 回归、梯度提升回归；回归有效后再考虑
  Pairwise Ranking 或图模型。

### 因变量

- 离线：Top-1 命中、Top-2 覆盖、Regret、有效动作 Precision/Recall、改善捕获率。
- 在线：候选 Solver 调用数、墙钟时间、最终 L2/EPE/PVB/J、质量保持率、独立回放一致性。

### 控制项

- 固定 FRAG `(corner=16, uniform=32)`、五动作、Golden evaluator、点顺序、seed 和单项护栏。
- 不读取 M1_test7–10 调参；不把 layout、point_id、搜索后指标、最终 Recipe 或接受顺序作为特征。

## 4. 工作包与门槛

### WP0：动作可辨识性审计——已完成

实现：`src/opc_agent/recipe_v2_action_audit.py`。

输出：

- `runs/v2-action-identifiability-001/action-identifiability.json`
- `runs/v2-action-identifiability-001/layout-summary.csv`
- `runs/v2-action-identifiability-001/groups.csv`

结论：整体 `decision_signal=proceed_to_offline_candidate_value_baseline`；M1_test3 保留为
skip/OOD 压力测试，不从六图报告中删除。

### WP1：候选价值数据集——已完成，零 Solver

从 `groups.csv` 与预测时可获得的冻结特征构建“一行一个候选动作”的训练表：

```text
(layout, point, incumbent_state, candidate_action, features)
    -> delta_l2, delta_epe, delta_pvb, delta_j, feasible, beneficial
```

必须由前一组 `incumbent_j_after_group` 重建候选前状态；同一点四个候选必须保持在同一折。
输出需记录源工件 SHA256、特征协议版本、缺失特征原因和泄漏字段黑名单。

门槛已通过：5224 行全部有唯一身份；1306 个四候选组完整；1272个完整视觉特征点与34个
缺视觉特征点均被显式保留；确定性连续几何覆盖全部1306点；目标值和 incumbent 轨迹与
WP0 完全一致。工件为 `runs/v2-candidate-value-dataset-002/`。

### WP2：LOLO 离线价值基线——已完成但未通过质量门槛，零 Solver

按版图 Leave-One-Layout-Out。先训练回归器预测 `delta_j`，再按预测值排序，不直接训练五分类
Recipe。M1_test3 单独报告“是否跳过”，不把四个完全相同的动作强行当作可学习排名。

比较：

1. 随机 Top-1/Top-2；
2. 固定动作顺序；
3. ExtraTrees/RandomForest 回归；
4. 梯度提升回归；
5. 若上述模型确有跨图信号，再加入 Pairwise Ranking。

必须报告每张版图结果、平局感知 Top-k、平均/P95 Regret、最差版图、改善捕获率和估算调用数。
如果模型在五张有信号版图中的多数版图不能优于等预算随机/固定基线，则停止调参，先修正
预测时连续几何特征，不进入真实 Solver 搜索。

实际结果：连续几何将最佳Top-2改善捕获率从81.46%提高到86.65%，调用减少48.70%；优于
随机64.53%和训练折固定顺序78.23%，但未达到95%门槛。M1_test4仅捕获50%，低于随机期望。
完整结果见 `docs/v2_candidate_value_20260922_results.md`。

2026-09-23补充模型族对照：在冻结最佳特征集后，组收益加权ExtraTrees、RandomForest、三种
HistGradientBoosting和两阶段模型均未达到95%；Pairwise Top-2达到95.0093%捕获并减少48.70%
调用，但只高出阈值10.1个J改善量，且M1_test4仅84.46%。该结果属于同一六图LOLO上的
事后模型族筛选，不是冻结外层成绩，不改变WP2b和WP3的先后关系。详见
`docs/v2_candidate_value_model_benchmark_20260923_results.md`。

### WP2b：嵌套LOLO风险回退——下一步，零 Solver

事后Oracle表明，仅需把16个高regret组从Top-2切换为Full，即可达到95.13%捕获且仍减少
48.09%调用；该数值不可部署。下一步在每个外层留出版图内，对其余五图再次执行内层LOLO，
只用内层折外预测训练Top-2 regret风险模型，冻结后在外层图选择 `Top-2/Full`。要求总体和
逐图均报告，且不得用外层真实regret选阈值。

### WP3：训练版图 Proposal–Veto Pilot

只在 WP2b 门槛通过后运行。先用 M1_test4（点数最少）与 M1_test5（已有改善最大）做成本受控
pilot；完整坐标搜索沿用冻结旧工件作为参照，除非代码、配置或 OpenILT 身份不一致。

候选调用预算：

| 策略 | 两图候选上限 | 加两图基线和各两次最终回放 |
| --- | ---: | ---: |
| Top-1 | 350 | 356 |
| Top-2 | 700 | 706 |
| 自适应 Top-k | 1400（最坏） | 1406（最坏） |

若任一最终回放不一致、任一 L2/EPE/PVB 护栏失败，或策略在相同调用预算下被随机/固定顺序
支配，则停止，不进入六图运行。

### WP4：六图冻结策略与保留集

先冻结模型、特征、Top-k/置信度阈值和所有门槛，再在 M1_test1–6 运行一次。理论上 Top-1 为
1306 个候选、Top-2 为2612个候选；额外计入每图基线与两次独立最终回放。

训练版图达到“质量保持率≥95%、候选调用减少≥40%、各项护栏与回放通过”后，才允许使用
M1_test7–8 验证。验证失败只能回到训练设计，不得查看 M1_test9–10 调参；M1_test9–10 只用于
冻结方案的一次最终测试。

## 5. 计划文件

| 文件 | 责任 | 状态 |
| --- | --- | --- |
| `recipe_v2_action_audit.py` | 只读可辨识性审计 | 已实现 |
| `recipe_v2_candidate_value.py` | 构建价值数据集、LOLO 与离线指标 | 已实现 |
| `recipe_v2_guided_search.py` | Proposal–Veto 在线搜索 | WP2 通过后实现 |
| `test_recipe_v2_action_audit.py` | 审计分组与轨迹门禁 | 已实现 |
| `test_recipe_v2_candidate_value.py` | 轨迹重建、标签隔离、几何覆盖与调用计数 | 已实现 |
| `test_recipe_v2_guided_search.py` | Veto、调用计数、回放与失败门禁 | WP2 通过后实现 |

## 6. 主要风险

- 5224 条数据来自每图一条顺序相关的坐标搜索轨迹，不是独立同分布样本。
- 离线 Top-k 只能评价记录状态上的候选排序；新策略改变前序动作后，后续状态会偏离日志。
- M1_test3 说明不同版图可能存在完全不同的动作响应；平均指标不能掩盖零响应版图。
- 六张训练版图不足以支持强统计显著性结论，应以逐图效果、最差情形和物理回放为主。
- Solver Veto 能阻止已测试候选造成护栏退化，但不能防止模型漏掉未测试的真正最优动作。

## 7. 下一执行动作

实现 WP2b 严格嵌套LOLO风险回退。WP3仍被门槛阻断；在风险模型达到95%改善捕获和至少40%
调用减少前，不恢复PPO、不运行Golden或真实Proposal–Veto。

# v2 候选动作价值排序离线结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-22
- Verification Status: VERIFIED（离线确定性复跑一致；真实 Proposal–Veto 未运行）
- Version Label: candidate_value_lolo_v1

## 1. 目的

验证模型能否在既有坐标搜索轨迹的每个状态中，对四个非零候选动作排序，从而减少候选
Solver 调用。模型只预测动作的有效改善率；真实部署仍需 Solver 检查 L2/EPE/PVB 护栏并
决定接受或拒绝。

本报告只评价日志状态上的单步候选排序。模型一旦改变前序动作，后续状态会偏离日志，因此
本结果不能替代真实搜索或最终 Recipe Golden 回放。

## 2. 输入和数据集

- 搜索源：`runs/20260915T014128Z-v2-search-5697a08c/`
- 视觉特征：`runs/v2-vision-dataset-004/training.xlsx`
- 确定性几何：`runs/v2-vision-dataset-004/manifest.json`
- 候选数据：`runs/v2-candidate-value-dataset-002/candidate-values.csv`
- 数据清单：`runs/v2-candidate-value-dataset-002/dataset-manifest.json`

数据集包含 1306 个点、5224 条候选。1272 点有完整 28 个视觉布尔特征，34 点缺少视觉
特征；确定性几何覆盖全部 1306 点。共有 419 条候选满足真实护栏且改善当时 incumbent。
模型输入不包含版图名、point_id、epe_id、最终动作标签、搜索顺序或候选运行后的指标。

新增确定性连续特征包括分段长度、到两端点距离、角点类型、邻近 EPE 点距离/密度、同方向
与反法线邻点，以及沿法线前后 128nm 走廊内的点数和距离。绝对 x/y 坐标被排除。

## 3. 协议

- 六图 Leave-One-Layout-Out，不随机拆分同图或同一点的四候选。
- 模型：`ExtraTreesRegressor`，300棵树，`min_samples_leaf=2`，`max_features=sqrt`，seed=0。
- 目标：真实护栏下的 `effective_gain / incumbent_j`；非改善或护栏失败候选目标为0。
- 缺完整视觉特征的点不强制预测，退回四动作完整搜索并计入调用预算。
- 对照：随机Top-k精确期望，以及只用训练版图平均收益确定的固定动作顺序。
- Top-k找到的候选仍按真实Solver veto语义计算捕获改善；未测试动作不计入结果。

## 4. 汇总结果

| 特征集 | Top-1改善捕获率 | Top-1调用减少 | Top-2改善捕获率 | Top-2调用减少 |
| --- | ---: | ---: | ---: | ---: |
| 28布尔特征+动作 | 65.00% | 73.05% | 80.65% | 48.70% |
| 28布尔特征+动作+当前状态 | 70.39% | 73.05% | 81.46% | 48.70% |
| 确定性连续几何+动作 | 76.74% | 75.00% | 85.91% | 50.00% |
| 确定性连续几何+动作+当前状态 | 73.06% | 75.00% | 83.67% | 50.00% |
| 28布尔+连续几何+动作+当前状态 | **76.38%** | 73.05% | **86.65%** | 48.70% |

最佳Top-2捕获 93997/108482 的日志内改善。相同调用口径下，训练折固定动作顺序捕获
78.23%，随机Top-2精确期望捕获64.53%。连续几何提供了真实增益，但仍未达到计划中的
95%质量保持门槛。

## 5. 最佳Top-2逐图结果

| 留出版图 | Oracle改善 | 模型捕获率 | 固定顺序 | 随机期望 |
| --- | ---: | ---: | ---: | ---: |
| M1_test1 | 23740 | 78% | 41% | 67% |
| M1_test2 | 15600 | 95% | 71% | 68% |
| M1_test3 | 0 | 不适用 | 不适用 | 不适用 |
| M1_test4 | 5578 | 50% | 50% | 56% |
| M1_test5 | 36243 | 90% | 98% | 64% |
| M1_test6 | 27321 | 92% | 94% | 63% |

M1_test4 低于随机期望，M1_test5/6 仍低于简单固定顺序。总体提升不能掩盖这些跨图失败。

## 6. 自适应回退上界

使用留出数据的真实 regret 事后选择回退组，是不可部署的 Oracle 上界：最佳模型从Top-2
开始，只需将16组改为四动作完整搜索，即可达到95.13%改善捕获率，同时仍减少48.09%的
候选调用。

这不属于模型成绩。它只说明问题可能集中在少量高 regret 点，下一步值得训练一个严格嵌套
LOLO的风险/不确定性模型来决定 `Top-2` 或 `Full`。如果可部署模型不能接近该上界，则应停止
动作排序路线，而不是直接启动真实 Solver。

## 7. 决策

WP1 数据构建通过。WP2 证明候选排序信号明显优于随机，但固定Top-1/Top-2没有达到
`改善捕获率≥95%且调用减少≥40%`的预设门槛，因此 WP3 真实 Proposal–Veto pilot 暂停。

下一步只做零 Solver 的 WP2b：在每个外层留出版图内，再对其余训练版图做内层LOLO，生成
无泄漏的Top-2 regret 标签和置信特征，训练风险模型选择 `Top-2/Full`。禁止直接用外层留出
结果选阈值。

## 8. 验证

- 离线评估环境：Python 3.12.14、NumPy 2.3.5、scikit-learn 1.7.2。
- 主结果：`runs/v2-candidate-value-lolo-003/metrics.json`
- 折外预测：`runs/v2-candidate-value-lolo-003/out-of-fold-predictions.csv`
- 完整复跑：`runs/v2-candidate-value-lolo-004-repro/`
- 两次 `metrics.json` SHA256 均为
  `4A380D82F72AAC3BE520BA8EF5F25956906C29D72BA292C88455D4B59CBBF0F6`。
- 两次折外预测 CSV SHA256 均为
  `6F65C9F665994BF992AA35DDB2BCE90B47AD4F6FA6ABE86C458BE1B492DB80F3`。
- 本轮 Solver 调用为0，未使用 M1_test7–10，未运行 PPO 或 Golden。

## 9. 2026-09-23模型族补跑

后续在同一数据、特征和六图LOLO口径下补跑了加权ExtraTrees、RandomForest、三种
HistGradientBoosting、两阶段模型和Pairwise Ranking。Pairwise Top-2达到95.0093%捕获并减少
48.70%调用，是唯一越过95%/40%筛选线的模型；但通过余量只有10.1个J改善量，M1_test4仍为
84.46%，且模型族选择使用了同一组LOLO结果，所以没有据此启动真实Solver。完整结果见
`docs/v2_candidate_value_model_benchmark_20260923_results.md`。

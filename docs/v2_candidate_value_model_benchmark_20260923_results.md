# v2 候选价值模型族 LOLO 对照结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + reproducibility check
- Origin Date: 2026-09-23
- Verification Status: VERIFIED（同环境同配置确定性复跑一致；模型选择与真实 Solver 未验证）
- Version Label: candidate_value_model_benchmark_v1

## 1. 目的与边界

在不调用 Solver、不改变数据、特征或缺失值策略的条件下，补齐原计划中尚未执行的候选价值
模型族对照。实验固定六图 Leave-One-Layout-Out、`vision_plus_geometry_action_state` 特征、
缺视觉特征点四动作 Full 回退以及 Top-1/Top-2 计费口径，只改变模型。

本轮使用 M1_test1–6 的既有顺序坐标搜索日志，不读取 M1_test7–10。结果只能说明日志状态上的
单步排序能力。模型族是在同一组 LOLO 结果上事后比较的；即使某模型越过筛选线，也不能把它
当作冻结部署模型、WP2b 外层测试成绩或最终 Recipe 验收。

## 2. 固定模型

1. 原始 `ExtraTreesRegressor`；
2. 训练折内组 oracle gain 加权的 `ExtraTreesRegressor`；
3. `RandomForestRegressor`；
4. `HistGradientBoostingRegressor`，分别使用 squared、absolute、quantile=0.9 loss；
5. 两阶段 `P(beneficial) × conditional_gain`；
6. 组内 Pairwise Ranking：跳过真实收益平局，对非平局动作对生成正反差分样本，并按训练折
   收益差加权，再把候选对其他三个动作的预测胜率相加作为排序分数。

森林固定300棵树，boosting固定200轮，随机种子为0。加权、Pairwise 标签和缩放分位数都只从
当前训练版图计算，留出版图仅用于一次折外评价。

## 3. 汇总结果

| 模型 | Top-1捕获 | Top-2捕获 | Top-2调用减少 | 95%/40%筛选线 |
| --- | ---: | ---: | ---: | --- |
| ExtraTrees原始 | 76.38% | 86.65% | 48.70% | 未通过 |
| ExtraTrees组收益加权 | 76.59% | 86.58% | 48.70% | 未通过 |
| RandomForest | 73.88% | 83.53% | 48.70% | 未通过 |
| HistGradientBoosting squared | 72.99% | 81.85% | 48.70% | 未通过 |
| HistGradientBoosting absolute | 15.63% | 46.56% | 48.70% | 未通过 |
| HistGradientBoosting quantile90 | 15.63% | 46.56% | 48.70% | 未通过 |
| 两阶段 beneficial×gain | 74.35% | 87.10% | 48.70% | 未通过 |
| Pairwise gain difference | 68.62% | **95.01%** | **48.70%** | **筛选通过** |

Pairwise Top-2 捕获 `103068/108482=95.0093%`，只比95%阈值高 `10.1` 个J改善量；这是很窄的
通过余量。其 Top-1 反而低于原始 ExtraTrees，说明当前优势限定在“挑两个动作”的目标上，
不能泛化成绝对价值回归更准确。

## 4. Pairwise逐图结果

| 留出版图 | Oracle改善 | Top-2捕获 | 固定顺序 | 随机期望 | 调用减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 23740 | 92.74% | 41.18% | 67.17% | 48.14% |
| M1_test2 | 15600 | 96.97% | 70.94% | 68.22% | 48.56% |
| M1_test3 | 0 | 不适用 | 不适用 | 不适用 | 48.65% |
| M1_test4 | 5578 | 84.46% | 50.29% | 56.13% | 50.00% |
| M1_test5 | 36243 | 98.25% | 98.25% | 63.89% | 48.17% |
| M1_test6 | 27321 | 93.72% | 93.72% | 62.70% | 49.39% |

M1_test4 已由原始 ExtraTrees 的约50%提高到84.46%，但仍是最差版图；M1_test1和M1_test6也
没有达到95%。聚合通过不能掩盖逐图风险。M1_test3 的 Oracle 改善为0，因此其100%捕获不应
解释为模型成功。

## 5. 决策

Pairwise 证明组内排序目标明显比本轮绝对收益回归更适合 Top-2，但本轮不解锁 WP3：

- 通过余量只有10.1，容易受模型版本、参数或数据扰动影响；
- 模型族是在同一六图 LOLO 上事后选择，仍有模型选择偏差；
- M1_test4 只有84.46%；
- 离线日志仍是顺序相关的单条轨迹，策略改变后续状态时必须由真实 Solver 验证。

下一步仍是零 Solver：把 Pairwise 作为候选 base ranker，进入严格嵌套 LOLO。内层负责选择
模型与任何阈值，外层版图只用于一次评价；不得直接把本轮 Pairwise 结果当作冻结成绩。

## 6. 工件与复现

- 实现：`src/opc_agent/recipe_v2_candidate_value_models.py`
- 主结果：`runs/v2-candidate-value-model-benchmark-001/`
- 确定性复跑：`runs/v2-candidate-value-model-benchmark-002-repro/`
- `metrics.json` 两次 SHA256：
  `283AC3D69307DB30BAB87E4376625C79D76CC27255A93B9001D0F7A9175CFC22`
- `out-of-fold-predictions.csv` 两次 SHA256：
  `A6AAD7A4259C08DD56E93DF004645E8161BDA486EAFB6F5ECFF1C81F05F7752B`
- 环境：Python 3.12.14、NumPy 2.3.5、scikit-learn 1.7.2。
- Solver调用：0；PPO、Golden和M1_test7–10均未运行。

运行命令：

```powershell
$env:PYTHONPATH='src;runs/_tree_runtime_py312;runs/_tree_support_py312'
& 'C:/Users/雷神/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B `
  -m opc_agent.recipe_v2_candidate_value_models `
  --dataset runs/v2-candidate-value-dataset-002/candidate-values.csv `
  --manifest runs/v2-candidate-value-dataset-002/dataset-manifest.json `
  --output runs/v2-candidate-value-model-benchmark-001 `
  --n-estimators 300 `
  --max-iter 200
```

# OPC Agent：点级 Recipe PPO 主线

## 2026-09-23：M1_test7–10 启发式诊断与十图 PVB/EPE N/EPE D 汇总

`v2-search` 默认行为仍严格锁定 `M1_test1–6`。只有同时提供完整评估版图列表和显式授权参数，
才允许在 `M1_test7–10` 上运行同一套 `coordinate-only-resumable-v4-evaluation-scope` 坐标搜索：

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
python -B -m opc_agent.cli v2-search \
  --config configs/recipe_ppo_v2.yaml \
  --layouts M1_test7 M1_test8 M1_test9 M1_test10 \
  --allow-validation-test-diagnostic
```

协议继续固定 FRAG `(corner=16nm, uniform=32nm)`、EPE 五动作
`[-20,-10,0,+10,+20]nm`、`seed=0` 和一次完整坐标扫描。候选只有在 L2、OpenILT 15 坐标
Golden EPE、PVB 均不高于全零基线且 `J=L2+100×EPE+PVB` 严格下降时才接受；每图最后必须
独立完整回放一致。新工件额外保存 `baseline-mask.png`、`baseline-printed.png` 及哈希，并记录
真实 Golden evaluator 的逐图身份。评估图搜索使用了每张图自己的 solver 反馈，所以结果始终是
`diagnostic_only`，不得解释为共享模型对未见版图的泛化表现或正式 acceptance。
新搜索还会在 `result.json` 中保存 target/final mask/final printed PNG 的 SHA-256；
十图汇总在存在该新字段时必须校验一致，旧六图无该字段时保留兼容但不声称已做哈希校验。

新四图完成并下载后，用已有六图运行和新四图运行生成统一十图报告：

```bash
PYTHONPATH=src python -B -m opc_agent.recipe_v2_ten_layout_summary \
  --search-runs \
    runs/20260915T014128Z-v2-search-5697a08c \
    runs/<M1_test7-10的新v2-search运行目录> \
  --output-dir runs/<新的十图汇总目录> \
  --seed 0 \
  --tolerance-nm 1
```

汇总入口要求两个运行的核心 OpenILT、solver、FRAG、动作、几何和 Golden 全局协议一致，且
M1_test1–10 恰好各出现一次。旧六图没有基线 printed 图，因此会各独立回放一次全零 Recipe，
先核对历史 L2/15坐标EPE/PVB/J 完全一致，再把新基线图写入汇总目录；不会修改旧运行。
若新四图的 `result.json` 已声明基线或最终 PNG，任一图缺失或哈希不一致都会立即停止，
不会用重放掩盖下载不完整。
EPE N/D 使用 v2 `dissect(16,32)` 的固定 segment 中点：EPE N 是距离严格大于 1nm 的点数，
EPE D 只累加这些点到最终印刷前景边界的完整最近欧氏距离。1nm EPE N/D 只是后处理指标，
不属于原搜索护栏，结果可能改善也可能变差。

输出 `ten-layout-epe-summary.json` 和 `ten-layout-epe-summary.csv`，包含逐图基线→最终 PVB、
EPE N、EPE D、绝对/相对变化、十图等权宏平均 `ALL_MEAN`，以及全部采样点上的
违规总数、距离总和和违规率 `ALL_TOTAL`。
不能只报告平均计数而省略每图采样点数和微观分母。当前本地缺少锁定 OpenILT/CUDA，
这里只完成代码与 CPU 测试，真实 test7–10 搜索和十图数值必须在云端执行后登记。

## 2026-09-23：按 Recipe v2 冻结点集离线补算 1 nm EPE N/D

新增 `src/opc_agent/sampled_epe_metrics.py`，用于在不重跑 GPU 或 Solver 的前提下，从坐标搜索
工件的 `target.png`、`final-mask.png`、`final-printed.png` 和 `result.json` 补算 1 nm EPE N/EPE D。
采样点严格沿用本次搜索的 Recipe v2 冻结协议：从 `config.snapshot.yaml` 读取全局 FRAG
`corner=16 nm`、`uniform=32 nm` 和几何适配器，调用锁定 OpenILT 的 `polygon.dissect` 重建
segment，并取每个 segment 中点作为固定 EPE 采样点。v2 没有独立的“FRAG 采样点”；FRAG
在这里是生成 EPE 点的全局分段参数，不能回退到 `simpleopc-recipe-point-v1` 的 96/8/±40 nm
打点规则。重建点 ID 必须与 `result.json` 的完整动作点和搜索顺序集合逐一相等，并要求
`final_replay_equal=true`，否则立即失败。该新指标不能与 OpenILT `EPE_CONSTRAINT=15` 的
Golden violation count 混用。

EPE N 统计距离严格大于 `1 nm` 的采样点数；EPE D 只累加这些违规点到最终印刷前景边界的
完整最近欧氏距离，不累加未违规点，也不把 `distance-1 nm` 当作 EPE D。云端运行命令：

```powershell
$env:PYTHONPATH='src'
python -B -m opc_agent.sampled_epe_metrics `
  --run-dir runs/20260915T014128Z-v2-search-5697a08c `
  --output-dir runs/20260915T014128Z-v2-search-5697a08c/sampled-epe-1nm `
  --seed 0 `
  --tolerance-nm 1
```

Linux 云端使用等价命令：

```bash
PYTHONPATH=src python -B -m opc_agent.sampled_epe_metrics \
  --run-dir runs/20260915T014128Z-v2-search-5697a08c \
  --output-dir runs/20260915T014128Z-v2-search-5697a08c/sampled-epe-1nm \
  --seed 0 \
  --tolerance-nm 1
```

成功后生成 `sampled-epe-metrics.json`（逐图输入/GLP/Recipe 哈希、旧 15 nm Golden EPE、新
EPE N/D、全运行汇总和逐点距离）、`sampled-epe-summary.csv`（逐图及 ALL 汇总表）及
`sampled-epe-points.csv`（逐点审计表）。程序还会读取 `search.layout_parents` 并与实际版图目录
逐一核对；部分下载、额外版图、任一最终工件缺失、图像尺寸不一致、GLP 重建 target 不一致、
点 ID 漂移或最终回放不一致都会立即失败。若云端 OpenILT/ICCAD13 不在配置快照记录的相对路径，
可显式追加 `--openilt-dir` 和 `--iccad13-dir`，这两个参数只改数据路径，不改冻结统计协议。
原始运行目录只读，结果写入单独输出目录。

## 2026-09-23：候选价值模型族LOLO补跑完成

新增 `src/opc_agent/recipe_v2_candidate_value_models.py`，在不改变六图LOLO、最佳冻结特征集、
缺视觉特征点Full回退和Top-k计费口径的前提下，对比原始/组收益加权ExtraTrees、RandomForest、
三种HistGradientBoosting、两阶段beneficial×gain和组内Pairwise Ranking。实验不调用Solver，
不使用M1_test7–10。

原始ExtraTrees的Top-2结果精确复现为86.65%改善捕获、48.70%调用减少；加权ExtraTrees、
RandomForest和三种boosting均未提高。两阶段模型小幅提高到87.10%。Pairwise Top-2达到
`103068/108482=95.0093%`改善捕获，同时减少48.70%调用，越过预设95%/40%筛选线；但只高出
阈值10.1个J改善量，M1_test4仍只有84.46%，而且模型族是在同一组LOLO上事后比较。因此该
结果仍为`diagnostic_only`，不解锁真实Proposal–Veto或Golden。

主工件为`runs/v2-candidate-value-model-benchmark-001/`，独立复跑为
`runs/v2-candidate-value-model-benchmark-002-repro/`；两次metrics和折外预测SHA256分别完全
一致。下一步仍为零Solver的严格嵌套LOLO，把Pairwise视为候选base ranker而不是冻结模型。
详细结果和复现命令见
[候选价值模型族LOLO结果](docs/v2_candidate_value_model_benchmark_20260923_results.md)。

## 2026-09-22 最新：候选价值LOLO完成，真实Solver pilot暂缓

新增 `src/opc_agent/recipe_v2_candidate_value.py`，把六图坐标搜索的1306个点展开为5224条
候选价值记录，并按版图留一训练固定参数 ExtraTrees 回归器。输入不使用版图名、point_id、
最终动作标签、搜索顺序或候选运行后指标；34个缺完整视觉特征点退回四动作完整搜索，不靠
缺数据虚增节省率。确定性连续几何由既有 `manifest.json` 重建，覆盖全部1306点，不调用
OpenILT。

最佳固定Top-2采用“28布尔+连续几何+动作+当前归一化状态”，在减少48.70%候选调用时，
捕获既有轨迹86.65%的改善；训练折固定动作顺序为78.23%，随机Top-2精确期望为64.53%。
Top-1减少73.05%调用，但只捕获76.38%。M1_test4的Top-2捕获仅50%，低于随机期望56%，
因此未达到预设的“改善捕获≥95%、调用减少≥40%”门槛，不启动真实Proposal–Veto或Golden。

事后Oracle上界显示：若能正确识别16个高regret组并从Top-2回退Full，可达到95.13%捕获且
仍减少48.09%调用；该上界使用留出真值，不是模型成绩。下一步是严格嵌套LOLO的风险回退
模型，禁止用外层留出结果调阈值。主工件为 `runs/v2-candidate-value-dataset-002/` 和
`runs/v2-candidate-value-lolo-003/`，确定性复跑文件哈希完全一致。详细结果见
[候选动作价值排序离线结果](docs/v2_candidate_value_20260922_results.md)。

## 2026-09-22 最新：候选动作可辨识性审计完成

新增只读入口 `src/opc_agent/recipe_v2_action_audit.py`，从既有六图坐标搜索工件重建每个点的
四候选动作组，不调用 OpenILT、不训练模型。运行命令：

```powershell
$env:PYTHONPATH='src'
python -B -m opc_agent.recipe_v2_action_audit `
  --source-run runs/20260915T014128Z-v2-search-5697a08c `
  --output runs/v2-action-identifiability-001
```

审计覆盖 1306 个点、5224 条候选记录：792 组（60.64%）的 mask、L2/EPE/PVB 和 J 均存在
候选差异，514 组（39.36%）完全不变；284 组（21.75%）在当时 incumbent 状态下产生实际
改善。五张改善版图的累计 J 改善 108482 全部来自可排序组，因此当前证据不支持停止整条
EPE 点移动路线，可以进入离线候选价值回归/排序基线。M1_test3 的 260 组全部不变且无改善，
保留为 skip/OOD 压力测试，不能据此推断联合动作无效。

工件位于 `runs/v2-action-identifiability-001/`：`action-identifiability.json` 为总表，
`layout-summary.csv` 为逐图摘要，`groups.csv` 为逐点动作组。状态固定为 `diagnostic_only`、
`accepted=false`、`solver_calls=0`。完整下一阶段、门槛和 Solver 预算见
[候选动作价值排序实施计划](docs/v2_candidate_value_ranking_plan.md)。当前先构建动作价值数据集并做
六图留一离线基线；离线模型通过等预算随机/固定顺序对照前，不恢复 PPO、不消耗
M1_test7–10。

## 2026-09-21 最新：标准 0.5 阈值的二分类→四分类两阶段结果

前一轮把 80% move precision 作为硬门槛，导致全部点回退 stay，不能表示用户要求的
“先二分类，再四分类”实际串联表现。现已用固定概率阈值 `0.5` 重跑六图外层留一：
二分类先判断 stay/move，判为 move 的点再交给四分类森林预测 `-20/-10/+10/+20 nm`。
缺特征的 34 点仍保守回退 stay。

```powershell
$env:PYTHONPATH='src;runs/_tree_runtime_py312;runs/_tree_support_py312'
& 'C:/Users/雷神/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m opc_agent.recipe_v2_vision_gate build `
  --input runs/v2-vision-dataset-004/training.xlsx `
  --source-recipes docs/v2_search_20260915_recipes.json `
  --output runs/v2-vision-gate-004-oof-fixed05-001 `
  --gate-threshold 0.5
```

二分类门控在完整 1306 点上的结果（含 34 点缺特征回退）：

| TP | FP | FN | TN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 73 | 270 | 949 | 16.09% | 4.93% | 7.55% |

四分类档位模型已独立训练，并只在 279 个有特征的真实移动点上做留一版图评估：

| Accuracy | Macro-F1（四类） | Balanced accuracy |
| ---: | ---: | ---: |
| 38.71% | 32.27% | 33.08% |

两阶段串联后，完整 Recipe 共预测 87 个 move，其中只有 14 个真 move，73 个为误移动；
动作分布为 `-20/-10/0/+10/+20 nm = 24/18/1219/20/25`。五分类端到端指标为：

| Accuracy | Macro-F1（五类） | Balanced accuracy |
| ---: | ---: | ---: |
| 73.20% | 20.35% | 21.56% |

完整工件位于 `runs/v2-vision-gate-004-oof-fixed05-001/`。这组数值证明四分类模型确实已训练，
但当前主要瓶颈是前级二分类：它只找回 14/284 个真移动点，而且额外误移动 73 点。
因此该 Recipe 仍为 `diagnostic_only`，不能根据分类准确率宣称优化。

这一组 87 个非零动作的 Recipe 现在值得做 Golden 回放；但本地仍缺少配置指定的
`third_party/OpenILT` 锁定提交 checkout，因此尚未产生新的 L2/EPE/PVB/J。应在准备好的云端
环境中执行本节下方的 `replay` 命令，共计 18 次 solver，再决定是否进入特征改造。

## 2026-09-21 敏感性对照：80% precision 硬门槛的全 stay 回退

新增 `src/opc_agent/recipe_v2_vision_gate.py`，将全点预测拆成两阶段：先用二分随机森林
判断 stay/move，只有通过门控的点才进入既有四分类移动档位森林。外层按版图
留一产生折外完整 Recipe；每个外层折的门控阈值只从其余训练版图的内层留一
概率中选择，不读取留出版图标签调阈值。缺失特征和低置信度点均回退 stay。

```powershell
$env:PYTHONPATH='src;runs/_tree_runtime_py312;runs/_tree_support_py312'
& 'C:/Users/雷神/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m opc_agent.recipe_v2_vision_gate build `
  --input runs/v2-vision-dataset-004/training.xlsx `
  --source-recipes docs/v2_search_20260915_recipes.json `
  --output runs/v2-vision-gate-004-oof-001 `
  --minimum-move-precision 0.8
```

实际结果是有信息的负结果：六个外层折的内层概率都找不到达到 80% move precision
的非空阈值，因此安全回退为全 stay。完整 1306 点中 1272 点有特征，34 点缺特征回退
stay；预测 move 数为 0，因此 284 个真实移动点全部漏检。全量内层折外阈值曲线的最高
move precision 仅约 27.42%（precision=156/569，recall=55.91%），说明失败不是单纯由 80%
阈值过严导致；当前 28 个布尔特征不足以形成高精度的跨图移动选择器。

完整工件位于 `runs/v2-vision-gate-004-oof-001/`：`metrics.json` 保留每折内层阈值和外层指标，
`threshold-curve.csv` 保留精度-召回权衡，`out_of_fold_predictions.csv` 保留逐点概率，
`predicted-recipes.json` 恰好覆盖 1306 点，`two_stage_models.joblib` 保留全数据模型及折外选定阈值。
该运行始终为 `diagnostic_only`、`accepted=false`。

暂不运行 Golden 回放：当前预测 Recipe 等于全零基线，重放不会提供新的模型质量证据；
且本地缺少配置指定的 `third_party/OpenILT` 锁定提交 checkout。不将现有
`third_party/OpenILT-main/OpenILT-main` 强行改名或修改配置后冒充正式回放。云端环境准备好后可执行：

```bash
python -B -m opc_agent.recipe_v2_vision_gate replay \
  --config configs/recipe_ppo_v2.yaml \
  --predicted-recipes runs/v2-vision-gate-004-oof-fixed05-001/predicted-recipes.json \
  --output runs/<new-run-id>-v2-vision-gate-fixed05-golden-replay
```

`replay` 严格检查六张训练版图、全量 point_id、五动作表和训练禁用状态；每图计费
1 次全零基线和 2 次独立预测 Recipe 回放，六图共 18 次 solver。只有两次回放完全一致，
且每图 L2/EPE/PVB 不差于全零基线、平均 J 严格下降，才值得进入独立 validation；回放结果
仍不会自动写成 accepted。

当前停止继续搜索随机森林参数，下一步应先增加新版图预测时也能确定获取的连续几何特征，
例如段长、到凹/凸角距离、四方向多边形间距和局部密度。特征协议冻结后再重跑同一嵌套门控诊断；
在高精度门控出现非空移动前，不恢复长 PPO，也不对 M1_test7–8 消耗标注/求解资源。

本轮新增 3 项两阶段测试，并与现有视觉树/森林测试合计 34 项通过。单元测试使用 Fake episode，
不代表真实 OpenILT/Golden 已运行。

## 2026-09-17 最新：全部点五分类随机森林对照

沿用移动点参数搜索得到的随机森林参数，在同一份 `training.xlsx` 上重新纳入 `result=0`：共 1272 条、28 个布尔特征，类别 `-2/-1/0/+1/+2` 分别为 `84/12/993/76/107`。使用 500 棵树、`max_depth=None`、`min_samples_leaf=1`、`max_features=sqrt`、`class_weight=balanced_subsample`、`seed=0`，按六张版图进行留一验证。命令未传 `--drop-stay`：

```powershell
$env:PYTHONPATH='src;runs/_tree_runtime_py312;runs/_tree_support_py312'
& 'C:/Users/雷神/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m opc_agent.recipe_v2_vision_tree --input runs/v2-vision-dataset-004/training.xlsx --output runs/v2-vision-forest-004-all-points-tuned-001 --model-type forest --n-estimators 500 --max-depth none --min-samples-leaf 1 --max-features sqrt --class-weight balanced_subsample
```

折外五分类结果为 accuracy=47.33%、macro-F1=0.2289、balanced accuracy=27.45%；全预测 `0` 的基线分别为 78.07%、0.1754、20.00%。森林虽然提高了类别均衡指标，但误报严重：993 个真实 stay 点仅 539 个预测为 0，454 个被误判为移动；279 个真实移动点中 163 个被识别为某个非零类别。由此得到二阶段视角下的移动检测 recall=58.42%、precision=26.42%。因此该模型目前不能直接生成全点 Recipe；较高的五分类 macro-F1 也不能抵消大量假移动带来的 solver 风险。完整产物位于 `runs/v2-vision-forest-004-all-points-tuned-001/`，仍为 `diagnostic_only`，未做独立测试版图或 Golden solver 回放。这里沿用移动点上选出的参数，仅用于隔离“加入 stay 类”的影响，并非五分类参数最优性结论。

本项目复现论文 *Intelligent OPC Engineer Assistant* 的两阶段思路。当前工作已从 PPO 链路验证转向 v2 坐标搜索及其 Recipe 的探索性决策树蒸馏：调整原始 target 上的 EPE 测量点，项目侧 recipe-aware OPC solver 在内部优化 mask，FRAG 暂时固定。PPO 续训暂停；Qwen v4 工作簿已经下载，新增独立五分类 Excel 决策树入口；搜索或决策树结果不能替代 PPO 验收。最新工作交接以文首 2026-09-17 记录为准。

## 2026-09-17 最新：移动点随机森林参数敏感性实验

在完全相同的 279 条非零样本和五折版图留一划分上完成固定网格：max_depth=
`5/8/12/None`、min_samples_leaf=`1/2/5/10`、max_features=`sqrt/0.5/1.0`、
class_weight=`balanced/balanced_subsample/None`，共 `4×4×3×3=144` 组；每组 500 棵树，
bootstrap=True、seed=0、n_jobs=-1。没有随机拆点。结果位于
`runs/v2-vision-forest-search-004-move-only-001/` 的 `grid_results.csv` 和 `search.json`。

```bash
export PYTHONPATH=src
python -B -m opc_agent.recipe_v2_vision_forest_search \
  --input runs/v2-vision-dataset-004/training.xlsx \
  --output runs/v2-vision-forest-search-004-move-only-001 --n-estimators 500
```

按合并五折 Macro-F1 排名的最佳参数为 max_depth=None、min_samples_leaf=1、
max_features=sqrt、class_weight=balanced_subsample；accuracy=38.71%、macro-F1=0.3227、
balanced accuracy=33.08%，最差单版图 macro-F1=0.2594。该组合相对原固定森林的
28.67%/0.2649/31.80% 有提升，也略高于多数类对照 accuracy=38.35%，但同一批折同时用于
比较参数和报告最佳分数，存在选择偏差，不能当作独立泛化成绩。

已用最佳参数在 `runs/v2-vision-forest-004-move-only-tuned-001/` 生成最终 500 树森林、
forest.json/joblib、折外预测、报告和重要性。最终全样本拟合森林训练 macro-F1=0.6503；
其 Gini 前五为 near_concave_corner（9.70%）、at_long_path_side（9.29%）、
near_convex_corner（7.49%）、near_horizontal_edge（7.47%）、face_convex_corner（6.30%）。

另行统计完全相同的 28 位特征向量：279 点只有 119 种向量，39 种向量内部存在多个标签，
覆盖 185 点（66.31%）。若每种精确特征组合只能输出其多数标签，训练 accuracy 上限约 72.76%。
明细保存在 `feature_label_conflicts.json`。这证明当前问题不仅是原森林参数欠拟合，也存在
明显的特征到动作一对多冲突；该上限只针对确定性地按完整特征向量映射类别，不是所有模型的
理论上限。训练模块与搜索冲突统计合计 31 项测试通过，源 Excel 未修改。

## 2026-09-17 历史对照：移动点随机森林固定参数

在同一入口新增 `--model-type forest --n-estimators 300`，默认仍为 tree。随机森林使用与
上一版四分类树完全相同的 Excel SHA256、279 个非零样本、28 个 0/1 特征、五折按版图留一划分，
不补入 stay 或失败点，不搜索超参数。固定 300 棵树、max_depth=5、min_samples_leaf=10、
class_weight=balanced、random_state=0、bootstrap=True、max_features=sqrt、n_jobs=1。
沿用下节本地隔离 CPU 环境，未调用 API/GPU。

```bash
export PYTHONPATH=src
python -B -m opc_agent.recipe_v2_vision_tree \
  --input runs/v2-vision-dataset-004/training.xlsx \
  --output runs/v2-vision-forest-004-move-only-001 \
  --drop-stay --model-type forest --n-estimators 300
```

已完成运行 `runs/v2-vision-forest-004-move-only-001/`。五折合并的验证指标：

| 模型 | Accuracy | Macro-F1（四类） | Balanced accuracy |
| --- | ---: | ---: | ---: |
| 决策树 | 30.47% | 0.2725 | 32.11% |
| 随机森林 | 28.67% | 0.2649 | 31.80% |
| 各折训练集多数类对照 | 38.35% | 0.1386 | 25.00% |

本组固定参数下随机森林没有改善决策树表现，不能据此排除其他参数或证明特征本身无效。
Gini 前五为 type_V（11.02%）、face_convex_corner（7.74%）、near_ver_dir_has_polygon（7.32%）、
on_horizontal_edge（6.58%）、near_horizontal_edge（6.49%）。置换重要性仍按各留出版图计算并保留负值，
不能把训练模型内部的 Gini 排名当作泛化收益保证。

森林产物包括 `random_forest.joblib`、`forest.json`、`metrics.json`、`report.md`、
`feature_importance.csv/png`、`training_binary.csv` 和 `out_of_fold_predictions.csv`。
不将某一子树当作整个森林，不导出单树规则图。forest.json 中子树内部类别编码恢复为真实
移动类别，按各树的类别概率平均推理；`predict_forest` 与 sklearn 的预测已核对。
训练模块测试命令 `python -B -m pytest -q tests/test_recipe_v2_vision_tree.py` 共 28 项通过，
包含随机森林来源保持、折间隔离、无零类别、JSON 推理及模型重载检查。
原始 Excel 哈希未变，旧模型结果保持不变；独立测试版图与 Golden 回放仍未执行。

## 2026-09-17：删除不移动点后的四分类树

按用户要求，训练入口新增 `--drop-stay`：先移除 `result=0` 的行，再在保留样本中删除含
空值的特征列并将 true/false 转为 1/0。原始 Excel 与上一版五分类模型不改动。
本次从 1272 条样本中排除 993 条 stay，保留 279 条、28 个特征；类别
`-2/-1/+1/+2` 样本数为 `84/12/76/107`。M1_test3 全部是 stay，过滤后没有样本，
其余五张版图参与留一版图验证，不将没有样本的 test3 计作验证折。

```bash
export PYTHONPATH=src
python -B -m opc_agent.recipe_v2_vision_tree \
  --input runs/v2-vision-dataset-004/training.xlsx \
  --output runs/v2-vision-tree-004-move-only-001 --drop-stay
```

本地实际运行环境为 Windows CPU、Python 3.12.14、NumPy 2.3.5、scikit-learn 1.7.2，
解释器为 `C:\Users\雷神\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。
依赖从本项目 `runs/_tree_runtime_py312` 和 `runs/_tree_support_py312` 加载，不使用 AutoDL、
项目 `.venv` 或 Anaconda；没有新增 API 费用或 GPU 运算。Windows 复现命令：

```powershell
$env:PYTHONPATH='src;runs/_tree_runtime_py312;runs/_tree_support_py312'
& 'C:/Users/雷神/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m opc_agent.recipe_v2_vision_tree --input runs/v2-vision-dataset-004/training.xlsx --output runs/v2-vision-tree-004-move-only-002 --drop-stay
```

输出目录须为空。树参数保持深度 5、叶节点至少 10 条、类别加权 balanced、seed=0。
评估和置换重要性只使用固定四类；对照改为每折仅根据训练版图选取的多数移动类别，
不读取留出版图标签选对照类别。本轮五折对照均选择 +2。
四分类验证 accuracy=30.47%、macro-F1=0.2725、balanced accuracy=32.11%；
多数类对照分别为 38.35%、0.1386、25.00%。不能将这组指标直接与包含 stay 的五分类结果比较，
因为保留人群与类别集合已改变。

本轮目录 `runs/v2-vision-tree-004-move-only-001` 保存模型、0/1 数据、树图、报告和新的重要性排名。
Gini 前五为 far_vertical_edge（20.00%）、type_V（17.46%）、face_convex_corner（17.15%）、
at_short_path_side（12.37%）、near_hor_dir_has_polygon（9.37%）。metrics.json 记录被移除点 ID、
空样本版图、准确的运行环境及类别集合；tree.json/joblib 标记 `requires_external_move_selection=true`。
这棵树只对已知需要移动的点预测移动档位，不能单独决定是否移动，也不能直接组装全点 Recipe。
新增过滤顺序、全 stay 拒绝、无样本版图排除和四分类导出测试，训练模块 26 项测试通过。

## 2026-09-17 历史对照：包含 stay 的五分类决策树

当前输入是 `runs/v2-vision-dataset-004/training.xlsx` 的 `samples` 表，实际含 1272 条样本、
28 个布尔特征，来自六张训练版图；工作簿未纳入的 34 个失败点不补入训练。
样本类别 `-2/-1/0/+1/+2` 分别为 `84/12/993/76/107`，标签含义仍为法向 `-20/-10/0/+10/+20 nm`。
原始 Excel 保持不变，不依赖尚未完整下载的图片或 annotations。

入口 `src/opc_agent/recipe_v2_vision_tree.py` 使用标准库只读解析 OOXML，避免额外 Excel 读取依赖。
只允许既有 28 列特征；每列出现空白、null、None、NaN 时整列删除，true/false 严格转成 1/0。
本次 28 列均无空值，无需删列。`face_jog/on_jog_long_edge/on_jog_short_edge/at_long_path_end`
四列恒为 false，保留并报告零重要性。epe_id 和 provenance 只供追溯与分组，不能进入 X。
检查编号唯一、来源一一对应、result 与 normal_offset_nm 相符，并记录输入文件 SHA256。

固定 `DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=0)`。
参数预先固定，不根据验证结果搜索。先做六张训练版图内部的留一版图交叉验证，再用全部
1272 条样本拟合最终共享树；没有使用独立 validation/test。报告 accuracy、固定五类 macro-F1、
balanced accuracy、逐类 precision/recall/F1、混淆矩阵，并与全 stay 对照比较。

```bash
export PYTHONPATH=src
python -B -m opc_agent.recipe_v2_vision_tree \
  --input runs/v2-vision-dataset-004/training.xlsx \
  --output runs/v2-vision-tree-004-001
```

输出目录必须为空。`--max-depth`、`--min-samples-leaf` 可修改树限制；修改参数属于另一次实验，
应使用新目录。默认生成图；无 Matplotlib 的云端环境可用 `--no-plots` 只训练并导出数值结果。
现有项目依赖已包含 scikit-learn；本地 Anaconda 存在 sklearn/NumPy 二进制冲突，本轮使用
独立 Python 3.12 和 `runs/_tree_runtime_py312`、`runs/_tree_support_py312` 隔离依赖，不修改原环境。
复现实验以 metrics.json 中 Python/NumPy/sklearn 版本为准；joblib 应用相同版本加载，
跨环境可使用纯数据 `tree.json` 和模块中的 `predict_tree`（已与 sklearn 预测核对）。

产物：`decision_tree.joblib`（模型及特征顺序）、`tree.json`（可执行节点）、`tree_rules.txt`、
`training_binary.csv`（0/1 数据）、`out_of_fold_predictions.csv`、`metrics.json`、`report.md`、
`feature_importance.csv`、`feature_importance.png`、`decision_tree.svg/png`。
重要性 CSV 按最终全样本树的 Gini 重要性降序；同时提供六个留出版图上、每列 10 次置换的
macro-F1 下降均值/标准差（版图等权、seed=0、负值不截断）。它衡量该模型的预测依赖，
不是因果效应，相关类型/方向列会分摊或替代重要性。

测试：`python -B -m pytest -q tests/test_recipe_v2_vision_tree.py`。
该入口保持 `coordinate_search_not_ppo`、`diagnostic_only`；不接入旧九分类 PPO 教师门禁，
不调用付费 API/OPC，不把分类成绩当作完整 Recipe 的 Golden 回放结果。

本次已完成运行 `runs/v2-vision-tree-004-001/`，最终树深 5、25 个叶节点。
训练集 accuracy=34.28%、macro-F1=0.2416；六图留一验证 accuracy=29.40%、macro-F1=0.1742、
balanced accuracy=27.51%。全 stay 对照分别为 78.07%、0.1754、20.00%。类别加权提高了
少数类召回，但总体 macro-F1 尚未超过 stay 对照，不能据此宣称蒸馏成功。
Gini 排名前五为 face_convex_corner（19.11%）、far_vertical_edge（15.21%）、type_V（12.19%）、
near_hor_dir_has_polygon（11.05%）、near_concave_corner（10.99%）。置换重要性同时保存在 CSV，
两种排名含义不同；目前不自动按排名删除特征。21 项新增测试通过，源 Excel 哈希未变，
joblib 重新加载和 tree.json 推理均与 sklearn 预测一致。尚未进行独立评估图或 Golden 回放。

## 2026-09-16 历史交接：六图搜索完成，Recipe 已导出，决策树当时尚未训练

### 当前结论与证据位置

六张训练版图 M1_test1–6、seed=0、仅坐标搜索已全部完成。五张图的 J 降低，所有图最终 L2/EPE/PVB 均不高于各自全零基线，六图工件均记录最终独立回放一致。由此确认当前 EPE 偏移动作空间在多张训练图上存在可复现的改善方案；不是 PPO 已学会优化，也不是共享模型的跨图泛化证明。状态仍为 `diagnostic_only`、`accepted=false`。

- 最新完整 run：`runs/20260915T014128Z-v2-search-5697a08c/`。
- 总表：该目录的 `recipe-v2-search.json`；配置与环境记录为 `config.snapshot.yaml`、`metadata.json`。
- 逐图结果：`<layout>/seed-0/coordinate/result.json`；完整候选日志：同目录 `candidates.jsonl`。
- 逐图图片：同目录 `search-curve.png`、`target.png`、`final-mask.png`、`final-printed.png`。
- 输入样例：`ppo-input-examples/<layout>/manifest.json`，共六份 manifest、24 张 PNG、24 份 NPZ；它们只是样例，不是全部点训练特征。
- 完整标签导出：[docs/v2_search_20260915_recipes.json](docs/v2_search_20260915_recipes.json)；补充报告：[docs/v2_search_20260915_results.md](docs/v2_search_20260915_results.md)。本节已包含交接所需核心信息，无需依赖补充报告理解进度。

### 冻结实验协议

固定 FRAG `(corner=16, uniform=32)`，五动作索引 `0/1/2/3/4` 对应 EPE 法向绝对偏移 `-20/-10/0/+10/+20 nm`，索引 2 为 stay。每图从全零 Recipe 起步，seed=0 确定点访问顺序；逐点比较其余四动作，每个候选从同一当前最优 Recipe 出发执行完整 solver。仅当 L2/EPE/PVB 分别不高于固定全零基线且 J 严格下降时接受；平局保持原值。一轮扫描后独立回放最终完整 Recipe。评价为固定 Golden，`J=L2+100×EPE+PVB`。未训练 PPO，未搜索 FRAG，未使用 validation/test；随机搜索已从运行入口取消。

### 六图完整指标

以下为“全零基线 → 最终坐标搜索”，越低越好。

| 版图 | L2 | Golden EPE | PVB | J | J 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 68302 → 53862 | 28 → 14 | 65748 → 57848 | 136850 → 113110 | 17.35% |
| M1_test2 | 46111 → 35577 | 8 → 3 | 59177 → 54611 | 106088 → 90488 | 14.70% |
| M1_test3 | 160846 → 160846 | 125 → 125 | 32646 → 32646 | 205992 → 205992 | 0.00% |
| M1_test4 | 23591 → 18059 | 6 → 6 | 28317 → 28271 | 52508 → 46930 | 10.62% |
| M1_test5 | 66295 → 37935 | 20 → 1 | 66823 → 60840 | 135118 → 98875 | 26.82% |
| M1_test6 | 63808 → 39847 | 23 → 0 | 59739 → 58679 | 125847 → 98526 | 21.71% |

六图 J 降幅算术平均为 15.20%，包含未改善的 test3。test4 的 EPE 是持平，不是下降；test6 的 EPE=0 仅对应当前 Golden 评价口径，不代表所有工艺条件无缺陷。

### 搜索成本、异常与验证范围

| 版图 | 点数 | 候选求解数 | 接受的点更新数 | 候选不同 mask 数 | 搜索记录耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 242 | 968 | 66 | 258 | 6130.76 |
| M1_test2 | 208 | 832 | 44 | 236 | 4618.67 |
| M1_test3 | 260 | 1040 | 0 | 1 | 7080.18 |
| M1_test4 | 104 | 416 | 22 | 88 | 1268.74 |
| M1_test5 | 246 | 984 | 85 | 283 | 6374.04 |
| M1_test6 | 246 | 984 | 67 | 249 | 6232.18 |

总计 1306 点、5224 次候选，加六次基线与六次独立回放为 5236 次 solver 调用。实际 `resume_from=null`、复用候选为零：本次是从头运行，不是续跑。各臂记录耗时合计约 8.81 小时，不含全部初始化开销，不能当作精确端到端时间。

M1_test3 的 1040 个候选只有一个 mask 哈希，全部为 `no_strict_improvement`，没有护栏拒绝。因此该轮单点动作未改变最终 mask；不能推断联合动作无效，也不能把其全 stay 当成唯一正确解。保留此图，不悄悄从报告或训练集删除。

已用本地 JSON 解析核对：六图候选数与预算一致、编号连续、最优 J 不增加、最终单项护栏满足、逐图 best 与总表一致。六图 `final_replay_equal=true`，工件记录 OpenILT tracked diff 干净。导出文件的源总表 SHA256、全部点 ID、动作索引、偏移和 Recipe 哈希已与源工件核对一致。未现场重跑 CUDA；输入图片本次核对数量，未重新核验所有图像内容和哈希。

### 已提取的 Recipe 与决策树交接

`docs/v2_search_20260915_recipes.json` 含六图完整 1306 点标签，每点包含 `point_id/action_index/normal_offset_nm`，并保留固定 FRAG、源总表 SHA256、每图 Recipe/mask 身份、Golden 字段及基线/最终指标。源 Recipe 哈希标识原协议的 canonical Recipe，不是导出 JSON 文件本身的哈希。原始 runs 未修改。

| 版图 | -20 nm | -10 nm | stay | +10 nm | +20 nm |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 30 | 4 | 176 | 14 | 18 |
| M1_test2 | 9 | 0 | 164 | 17 | 18 |
| M1_test3 | 0 | 0 | 260 | 0 | 0 |
| M1_test4 | 6 | 3 | 82 | 6 | 7 |
| M1_test5 | 23 | 1 | 161 | 22 | 39 |
| M1_test6 | 16 | 4 | 179 | 20 | 27 |

下一步是探索性训练一棵跨图共享的 v2 五分类 EPE 决策树，而不是每图各训一棵。当前仅完成标签提取，尚未实现或运行新的 v2 决策树训练。旧决策树入口采用旧动作语义和 accepted 教师门禁，不能直接混用或绕过门禁；新教师应明确标为 `coordinate_search_not_ppo`、`diagnostic_only`。

后续待办按顺序如下：

1. 按源配置重建全部点并逐一匹配 point_id，导出所有点预测时可获得的几何/冻结全零状态特征。现有每图四个样例不足以训练完整模型；不需要重跑坐标搜索。若使用 baseline mask/printed 特征，新图仍需一次基线求解。
2. 固定特征协议，禁止将点编号、版图名称、最终 mask、最终质量、搜索接受次序或动作标签作为输入捷径。最终标签是联合 Recipe 的组成部分，不是每点独立的全局最优动作。
3. 训练浅层共享树并约束叶节点样本数；按版图分组验证，不随机拆同图点后宣称泛化。stay 占 1022/1306=78.25%，必须报告类别表现并比较全 stay 对照，不能只看 accuracy。
4. 将树预测组装成完整 Recipe，调用原 solver/Golden 比较 L2/EPE/PVB/J 与成本；分类准确率不能代替求解质量。保持 validation/test 划分，不从评估图搜索结果反向生成训练标签。
5. test3 的处理须显式说明；若做排除它的敏感性分析，也保留包含六图的结果。单 seed 不支持搜索顺序稳定性结论，六张训练图上的搜索改善不等于未知版图泛化。

无需为确认已有收益重复六图搜索。尚未启动新训练、未修改 OpenILT、未提交 Git。下方 v1、三 seed、搜索待验证及决策树暂缓等内容为历史实现或阶段记录，与本节冲突时以本节为准。

## 2026-09-16 Qwen 双图特征数据集（v4 已实现，全量待执行）

### 当前 v4：几何确定字段、语义复核和 1306 点全量入口

v3 云端 12 点试标为 11 ok、1 failed，但抽查发现 start/end 和邻近角点可能误判。
v4（`appendix-a2-hybrid-v4`）使用项目现有 EPEControlPoint 分段和端点凹凸类型，
由程序生成八列：`type_CV/CH/H/V`、`on_horizontal_edge/on_vertical_edge`、
`on_start_corner_seg/on_end_corner_seg`。起止端按顺时针重排，当前分段端点实际连接原始角点
才属于对应角点段；短边两端均为角点时两个字段均可为 true。这是项目明确的混合标注协议，
不是声称论文使用了相同的程序规则。其余 20 列仍由视觉模型判断，near/far 不增设数值阈值。

manifest 保存 `geometry_features/corner_evidence/geometry_evidence`；Prompt 只提供几何字段，
不提供搜索动作、版图名称或最终质量。模型原值保存在 `model_features`，合并后的值在
`parsed.features`，`feature_sources` 标明 geometry/vision。固定字段不一致，或分段端点
为凸/凹角而模型没有将对应 near-corner 标为 true，均保存 `review_reasons` 并进入
`needs_review`，即使全部字段都是布尔值也不进入训练表。只有分段端点没有角点时，
仍由模型判断附近是否另有角点；不会自动填 false，也不会把所有可见角点都认作 near。

v4 图1为 512×584，图2为 1024×584：原有 512 高图形区保持不变，底部增加独立图例。
局部图的 `SEG START/SEG END` 表示蓝色分段箭头两端，不等同于原始边角点。
红圈=当前 EPE 点，蓝箭头=顺时针分段方向，绿箭头=外法向正方向，橙框=裁剪窗口，黑色=target。
绿色箭头不表示预测动作。图2仍为左 CONTEXT、右 FULL TARGET；灰色为未知区或概览留白。

上传这三个代码文件后，在已有 OpenILT 和密钥环境变量的云端项目根目录直接执行全量：

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
python -B -m opc_agent.recipe_v2_vision run-all --dataset runs/v2-vision-dataset-004 --timeout-seconds 300
```

`run-all` 没有 manifest 时调用既有 prepare，从冻结搜索源重建全部 1306 点，不调用 solver。
已有 manifest 时按当前版本续跑；不覆盖旧版 001/002/003。首轮每点最多一次 API 请求，
总上限为 manifest 点数，不做同轮重试，单点失败继续下一个；400/401/403/404 立即停止。
全部所选图片和缓存身份在首次 API 请求前检查。每点立即保存，正常结束或 Ctrl+C 后自动
导出已完成 JSON/Excel 和 `annotation-summary.json`。准备阶段中断、manifest 尚未生成时，
输出非空目录仍拒绝覆盖，应使用新目录。300 秒是 SDK 网络超时，不是严格墙钟上限。

重复同一 run-all 命令会跳过 ok 和 needs_review，只请求 missing/failed；若要重试待复核项，
使用 annotate 的 `--retry-uncertain`。改变图片/提示词/模型需要新目录，不混用旧缓存。
按此前约 18 秒/点粗估，1306 点约 6.5 小时；空流和超时会延长，实际费用以平台计量为准。

输出包括所有 PNG、原始响应、八列几何来源、复核原因、`training.json`、`training.xlsx` 和
`annotation-summary.json`（各状态、各版图、未知特征、复核原因、每个失败点最后一次错误类型）。
成功和失败响应均保存可用的流计数；不记录思考正文、密钥或 SDK 异常全文。仅外层唯一编号
严格匹配且内部特征完整时在线规范化 JSON 包装；空对象仍 failed。导出会从原始响应重新
计算合并和复核结论，拒绝被改写的 parsed/status，不会将纯布尔 needs_review 误收入训练集。

本地本轮检查只有 27 个 v3 图片文件，无 SILICONFLOW_API_KEY 环境变量、无配置所需 OpenILT
目录，未调用真实 API，也未完成 1306 点标注。CPU 合成测试不代表视觉准确率；ok 仅表示
通过格式及当前语义规则，需保留人工抽查。全量结果的 complete 和质量尚待云端运行确认。

### 历史 v3：顺时针修正、全图概览和离线格式恢复

六图首批 60 点的云端结果为 41 ok、15 needs_review、4 failed；ok 仅指结构校验通过，不是视觉准确性验收。现已核对下载工件，独立离线格式恢复得到 43 ok、16 needs_review、1 failed（819 空对象）。报告 `runs/v2-vision-dataset-002-format-recovery.json` 保留源响应、源文件哈希、旧 Prompt 版本和规范化操作，原 annotations/training 文件未改动。旧 start/end 特征仍保留旧语义，不能混进 v3 训练集。

v3 统一蓝色箭头为屏幕坐标（右 x、下 y）顺时针方向：根据外法线 n 构造切线 (-ny,nx)，行进时黑色实体在右侧，独立于原始顶点顺序；只调整显示，不改变 point_id、原始几何或搜索动作。之前 v2 的“原始遍历方向”与词典“顺时针”不一致，由本节修正。

图1仍为 512×512 局部细节，图2为 1024×512 双面板：左侧上下文放大，右侧完整 target 等比例概览；均标同一测量点，右侧橙框指示上下文范围。near/far 在左侧上下文范围内作定性判断；全图用于路径连续性/长短判断。这是项目协议，不是论文数值阈值。jog 必须看到连续台阶折转，单个直角不算；被截断或过小无法判断仍返回 null。分段箭头端点不自动等同于原始边角点。

以下命令记录历史 v3 试标流程；当前代码执行请使用上方 v4 run-all 和新目录：

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
python -B -m opc_agent.recipe_v2_vision prepare --output runs/v2-vision-dataset-003
python -B -m opc_agent.recipe_v2_vision annotate \
  --dataset runs/v2-vision-dataset-003 \
  --point-ids 0,242,450,457,710,711,715,817,819,821,1060,1068 \
  --limit 12 --retries 0 --timeout-seconds 300
python -B -m opc_agent.recipe_v2_vision export --dataset runs/v2-vision-dataset-003
```

这 12 点覆盖六图的完整、未知、格式错误与空响应样例，每次最多 12 个请求；既有完成项跳过。先对照截图审核实际特征，不仅看 ok 数量。新图片/Prompt 需新目录，旧图片 manifest 缺少 v3 visual_protocol 时拒绝标注，不能复用旧图硬套新 Prompt。裁图仍从完整源 run 重建，不重新执行搜索；真实 v3 标注待云端验证。

独立格式恢复（无 API、无需下载全部图片）：

```bash
python -B -m opc_agent.recipe_v2_vision recover \
  --dataset runs/v2-vision-dataset-002 \
  --output runs/v2-vision-dataset-002-format-recovery.json
```

仅允许外层唯一键严格匹配该点编号、内部是完整特征字典或完整规范响应；空对象、错编号、额外字段仍拒绝。布尔/未知原值不变，缺少的未知列表只从已有 null 派生。报告必须放在源目录外且不得覆盖已有文件。恢复不会更新旧训练 JSON/Excel，也不会消除语义不确定性。

当前允许使用硅基流动 Qwen 做独立的探索性混合标注。下方“MLLM 暂缓”属于历史状态，以本节为准。本入口不训练决策树、不改变旧 accepted 教师门禁。

### 云端文件与准备

在包含 src/docs/runs 的 opc_agent 项目根目录运行。上传修改的代码、`docs/v2_search_20260915_recipes.json`，以及源目录 `runs/20260915T014128Z-v2-search-5697a08c/` 的 `recipe-v2-search.json`、`config.snapshot.yaml`、六图 `<layout>/seed-0/coordinate/result.json`。不需要 candidates.jsonl。保留原先锁定且 tracked diff 干净的 OpenILT clone 和 GLP。

沿用现有云端环境与安装方式 `python -m pip install -e .`。项目固定 `openai==1.35.7`，现显式锁定其兼容传递依赖 `httpx==0.27.2`；旧环境需更新此依赖。HTTPX 0.28 移除了 `proxies` 参数，与该旧版 SDK 默认客户端构造不兼容。若出现 `unexpected keyword argument 'proxies'`，在已激活的项目虚拟环境执行 `python -m pip install "httpx==0.27.2"`，再执行 `python -m pip check`。该错误发生在客户端初始化，尚未发出 API 请求。

可用假密钥离线验证构造（不请求网络、不验证真实密钥）：`python -B -c "from openai import OpenAI; c=OpenAI(api_key='offline-check', base_url='https://api.siliconflow.cn/v1'); c.close(); print('client init OK')"`。通过后继续 annotate，不需要重建图片。Excel 由标准库 OOXML 写出，无需 Office/openpyxl。裁图不调用 OPC/PPO，不创建 CUDA solver，但上游几何模块可能导入已有 OpenILT 依赖。

### 1. 裁出全部点，不需要密钥

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
python -m opc_agent.recipe_v2_vision prepare \
  --labels docs/v2_search_20260915_recipes.json \
  --source-run runs/20260915T014128Z-v2-search-5697a08c \
  --output runs/v2-vision-dataset-004
```

核对源总表 SHA256、完整 point_id、源动作、Recipe 哈希、最终回放标记；从源快照重建原始 target 和分段，不重新搜索。预期 1306 点、2612 张 PNG。默认局部窗口 128×128、上下文窗口 512×512（栅格坐标范围）；v4 输出尺寸及图例见上节。这是项目输入设计，不是论文超参数。用 `--local-window`、`--context-window` 调整，在试标后冻结。

如果云端路径不同，在 prepare 命令后添加 `--openilt-dir /你的路径/OpenILT --iccad13-dir /你的路径/OpenILT/benchmark/ICCAD2013`，只覆盖数据路径，不修改源分段配置。输出目录须为空；裁图中断后使用新目录重建。

图片黑色为 target，白色为背景，灰色为版图外未知区域。红圈中心是测量点，v3 蓝色箭头为顺时针方向，绿箭头为外法线。保持 raster 右 +x、下 +y。没有最终 mask 或搜索动作信息；无法判断时保留未知，start/end 使用 v3 顺时针协议。

### 2. 密钥与模型在哪里设置

只通过进程环境变量 `SILICONFLOW_API_KEY` 注入密钥，不写 Python/YAML/README/.env。Bash 隐藏输入（粘贴后回车）：

```bash
read -rsp 'SiliconFlow API key: ' SILICONFLOW_API_KEY
export SILICONFLOW_API_KEY
printf '\n'
```

默认模型 `Qwen/Qwen3.8-27B`，端点 `https://api.siliconflow.cn/v1`。该模型在硅基流动模型页标为原生视觉语言模型；本入口显式使用两张 `detail=high` 图片、关闭思考模式、将最终 JSON 限制为 1024 token，并采用流式接收以避免长时间无首包。需要修改模型或端点时使用 annotate 的 `--model`、`--base-url`。本入口不读取旧 `configs/paper_repro.yaml` 的 qwen 模型字段。账户权限、多图和 JSON 模式仍需云端确认，失败不会静默换模型。

### 3. 试标、续跑与导出

```bash
python -m opc_agent.recipe_v2_vision annotate \
  --dataset runs/v2-vision-dataset-004 --limit 60
python -m opc_agent.recipe_v2_vision export \
  --dataset runs/v2-vision-dataset-004
```

每点一次请求两张图片，模型只判断 features。按版图轮转取样，正常前 60 点每图 10 点；不保证几何类别覆盖，需人工检查图片和标注。

`--limit` 是本次 API 请求总上限，包含重试，不是保证成功的样本数。默认 60。每点最多重试 2 次（`--retries 0..5`），默认 SDK 网络超时 300 秒（`--timeout-seconds` 可调整，必须是有限正数），串行调用、SDK 自动重试关闭。该设置约束网络阶段等待，并非整个请求的严格墙钟上限。每次请求记录 elapsed_seconds 和 timeout_seconds，终端显示开始及结束耗时；实际成本以试标/平台计量为准。认证、权限、模型、参数错误（400/401/403/404）立即停止；其他失败记录后可续跑。

如果双图请求在原来的 120 秒设置下超时，先测试一个点，不重试：

```bash
python -B -m opc_agent.recipe_v2_vision annotate \
  --dataset runs/v2-vision-dataset-004 \
  --limit 1 --retries 0 --timeout-seconds 300
```

超时和耗时属于传输诊断，不参与缓存身份；调整超时可继续使用原截图和成功缓存。失败项会重新尝试。此命令会发送最多一次模型请求，是否成功及实际耗时仍需云端确认。

确认小批质量后处理剩余点：

```bash
python -m opc_agent.recipe_v2_vision annotate \
  --dataset runs/v2-vision-dataset-004 --limit 1306 --retries 0
python -m opc_agent.recipe_v2_vision export \
  --dataset runs/v2-vision-dataset-004
```

每点响应立即原子保存；重复同一命令跳过完成项，重新尝试失败项。`needs_review` 默认跳过，添加 `--retry-uncertain` 可重试未知项，不保证消除歧义。必要时重复有限预算命令；检查导出的 complete/review，不把请求数当完成数。Ctrl+C 可中断，已完成点保留；在途请求可能已经计费，无法保证恰好一次调用。

缓存身份包含图片、模型、端点和完整提示词；变更后拒绝复用，须新建数据集目录。暂不提供人工改写已缓存响应的入口。

### 提示词、接口与产物

- `src/opc_agent/recipe_v2_vision_prompt.py`：完整中文 prompt 和 28 个布尔字段。附录 25 条目中的 types 展开为 CV/CH/H/V 四列。near/far、long/short 保留视觉定性含义，不计算阈值；原文 `far_ver_dir_has_polygon` 的解释写 no、名称写 has，本协议显式按 has 肯定含义处理。v2 Prompt 为 start/end 角点段补充了原始边遍历箭头，避免模型从无方向静态图猜测该字段。
- `src/opc_agent/qwen.py::SiliconFlowQwen.extract_point_features(image_paths, prompt)`：复用客户端，一次两张 Base64 图片，返回原始文本。默认网络超时 300 秒，构造函数支持 timeout_seconds，重试由入口控制。
- `src/opc_agent/recipe_v2_vision.py`：prepare、annotate、export、recover、run-all 五个子命令。
- `manifest.json`：全点来源、point_id、坐标、动作及图像/GLP/源配置/标签哈希；内容哈希防止无意修改后复用缓存。
- `feature_dictionary.json`：制作时的特征词典；`annotations/*.json`：每点完整 prompt、原始响应、模型、请求身份、失败类型。不记录密钥和 SDK 异常全文。流式无正文时仅记录安全诊断计数（流块数量、正文/思考字段字符数、结束原因），用于区分模型只输出思考字段和服务端空流。
- `training.json`：samples 数组，每点 `epe_id/features/result`，元数据含教师来源、完整性和数量。
- `training.xlsx`：samples（布尔特征，最后一列 result）、features（定义）、provenance（版图/point_id/Recipe）、review（缺失/失败/待复核）、summary。表头冻结、可筛选；与 JSON 从同一份合格记录生成。

`result=-2/-1/0/1/2` 对应 `-20/-10/0/+10/+20 nm`，正方向是该点外法线。标签由程序从搜索结果合并，不由模型猜测，不是旧九分类动作。模型返回 true/false/null，未知字段必须列在 uncertain_features；漏字段、字符串布尔拒绝，v4 几何/语义矛盾保留为 needs_review。最终训练 samples 仅含完整布尔且无复核原因的记录；未解决项保留在 review。complete=false 不能宣称完成 1306 点数据集，API 成功不代表特征正确。

后续树训练只取 features 为 X、result 为 y，按 provenance 的父版图分组；禁止把编号/来源/坐标放入 X。当前是 coordinate_search_not_ppo、diagnostic_only，未训练树、未验证分类质量和 Golden 回放。

### 测试和未验证范围

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q tests/test_recipe_v2_vision.py tests/test_recipe_v2_geometry_contract.py
```

本轮以上命令 48 项测试通过，覆盖源点匹配、动作映射、顺时针角点八列、几何冲突排除、
响应包装规范化、图片预检、认证停止、流式中断计数和关闭、全量入口续跑及中断后导出、
缓存防篡改和 JSON/Excel 一致性。局部/上下文合成图已渲染检查图例；无真实 API 调用。
本地 Pydantic V2 发出两条项目既有 V1 用法弃用警告。真实 1306 点重建、Qwen 标注准确率、
云端依赖和计费尚待验证；没有声称全量运行已经启动或完成。

## 历史 v1 实现与证据边界（当前 v2 进度见文首）

- 当前主线版本为 `simpleopc-recipe-point-v1`，产物标签为 `ppo-recipe-point-v1`。
- OpenILT 固定提交为 `dabb97c6ca3dfd159362e48273c436444c77353b`。运行时只读导入其 GLP、polygon、光刻和 EPE 函数；若提交不一致或 tracked 文件有改动，训练会停止。PPO 产物只写入本项目 `runs/`。
- 当前代码已实现单步九分类 Recipe 环境、64×64 CNN observation、共享 PPO、EPE/FRAG 双类 Recipe、内部 mask OPC 和训练后质量审核。CPU 协议测试不等于真实 CUDA/OpenILT 成功；必须先完成云端 smoke。
- 论文没有公开完整分段算法、PPO 超参数和内部 solver 细节。`96nm` 初始 fragment、`8nm` 最小 fragment、内部 `8/4nm` mask 步长等都属于版本化实现选择，不得写成论文原始超参数。
- Qwen 可以用于后续 MLLM 阶段，不要求 GPT-4o；当前 PPO 训练不调用任何大模型 API，也没有实现 DQN。
- 历史 `simpleopc-multistep-v3` 仍保留用于回归，但已退出论文主线：它直接移动 mask 边段、每个 episode 用四步累计到 `±40nm`，且没有 FRAG。历史诊断见 [`docs/simpleopc_v3_m1_test1_diagnostic_20260825.md`](docs/simpleopc_v3_m1_test1_diagnostic_20260825.md)。

## 下一代局部 EPE + 全局 FRAG Recipe 整改（共享小训练已完成，暂停 PPO，转向启发式搜索）

### 2026-09-14 当前决定与实验登记

2026-09-16 更新：六图 seed=0 坐标搜索 `20260915T014128Z-v2-search-5697a08c` 已全部完成，5/6 图改善，J 降幅分别为 17.35%/14.70%/0%/10.62%/26.82%/21.71%，最终三项均不恶化、六图回放一致。详细协议、数值、调用预算、动作分布与解释边界见 [六图搜索结果登记](docs/v2_search_20260915_results.md)。已导出 [六图完整 Recipe 标签](docs/v2_search_20260915_recipes.json)，共 1306 点，仅为 diagnostic_only 坐标搜索教师候选，非 PPO/accepted 教师。允许下一阶段探索 v2 决策树，但不改变旧版教师验收门禁；全部点的输入特征尚需按冻结配置重建，现有 24 个可视化样例不等于完整训练集。当前未训练决策树。下方搜索未验证或决策树暂缓表述属于此前阶段记录，以此更新为准。

六图历史决定：暂停当时的云端运行，取消后续随机搜索，仅保留六张训练版图 M1_test1–6 的全零基线与单种子坐标搜索，配置 `search.seeds: [0]`，入口拒绝其他种子设置以避免误用旧预算。下方多种子、两方法预算与无续跑说明为更早历史版本。已完成六图运行使用 `coordinate-only-resumable-v3`；2026-09-23 起当前入口升级为文首记录的 v4，但不改变六图历史工件。该六图全新运行预算为 5224 次候选 + 6 次基线 + 6 次回放 = 5236 次调用。旧随机与 seed=1/2 工件保留但不再安排运行，底层随机分支仅供历史回归。

从已下载的旧 run 续跑时仅读取 seed=0；M1_test1 已完成的 968 个坐标候选可复用，其余五图共 4256 个候选，连同六图基线与回放预计新增 4268 次调用。按此前 M1_test1 的约 6.37 秒/次粗估约 7.6 小时（全新约 9.3 小时），版图差异会影响耗时，不是保证。seed=1 未完成日志不会继续，也不会删除。单 seed 用于六图探索，不支持搜索顺序稳定性结论。

已核对 `runs/20260914T085650Z-v2-search-f6e6ddbc`：M1_test1、seed=0 基线 L2=68302、EPE=28、PVB=65748、J=136850；坐标搜索 968 个候选后 L2=53862、EPE=14、PVB=57848、J=113110，J 下降 17.35%，三个单项均改善，最终独立回放一致。同 seed 随机 J=127553、seed=1 随机 J=129241，均回放一致。本地 seed=1 坐标已有 360 个候选，incumbent J=122416，尚无最终回放。证据证明该图 seed=0 存在可复现改善，不代表 PPO 已学会优化、六图泛化或正式 accepted。

续跑：将配置 `search.resume_from` 从默认 null 改为旧目录，例如 `runs/20260914T085650Z-v2-search-f6e6ddbc`，再运行原 `v2-search --config configs/recipe_ppo_v2.yaml` 命令。无需现在启动。先停止旧进程并完整下载，勿从正在写入的目录迁移。新 run 只读导入坐标日志，要求 search 之外配置完全一致，逐候选核对动作、点顺序和编号；仅复用完整候选组，尾部未提交完整组重新计算，中间损坏拒绝。已完成臂跳过全部候选，但仍独立回放一次；旧工件不覆盖。历史候选数与本次实际调用分别记账。约每 30 秒在候选组边界打印进度、J 和本臂 ETA（不是全局 ETA）。每图四组输入可视化保留。

启发式搜索入口现已扩展到全部六张训练版图 M1_test1–6：`python -B -m opc_agent.cli v2-search --config configs/recipe_ppo_v2.yaml`。配置 `search.layout_parents` 指定六图，seeds 为 0/1/2；`search.budget_policy=one_full_coordinate_sweep` 按实际点数计算每方法每图每 seed 的预算（五动作时为点数×4）。不再接受固定 `candidate_budget=984`。同图随机与坐标搜索使用相同预算，validation/test 不参与。运行前后执行 `git -C third_party/OpenILT diff HEAD --exit-code --`，运行前先执行 `python -B -m pytest -q -ra`。当前入口尚待云端真实验证。

按已有冻结点数，六图每臂候选预算依次为 `968/832/1040/416/984/984`。两方法×三 seed 共 36 臂，合计 31344 次候选求解，另有每臂一次基线和一次独立回放，预计总计 31416 次 solver 调用；实际预算以运行时点数为准。工件协议为 `coordinate-six-layout-single-sweep-v2`，每臂记录实际候选预算，总表记录版图点数和调用总数。每图仍保存四组输入样例，六图合计 24 组。

护栏冻结为每项 L2/EPE/PVB 均不高于该图全零基线，同时 J 必须严格低于当前最优值；相同 J 保持旧 Recipe。点顺序按 seed 从排序后的完整点集打乱，各点比较当前动作以外全部四动作后统一提交，候选不从其他候选 mask 热启动。首轮只做一次完整扫描，预算不足四次时不开始新点；不使用结果缓存。此搜索护栏独立于正式 acceptance，结果保持 diagnostic_only。

工件位于 `runs/<run_id>/`：总表 `recipe-v2-search.json`；逐臂 `<layout>/seed-<seed>/<method>/result.json` 与逐候选刷盘的 `candidates.jsonl`（包含完整动作、偏移、指标、哈希、接受原因和调用数）；逐臂 `search-curve.png`、`target.png`、`final-mask.png`、`final-printed.png`；每图四组 `ppo-input-examples/<layout>/` PNG/NPZ。该命令不启动 PPO。中断时保留已有臂和候选日志，当前未实现自动续跑。总表只在全部臂完成且回放一致后生成。后文“搜索尚未实现”属于此前规划记录，以本段为准。

共享 terminal 小训练 `20260914T075849Z-v2-ppo-small-train-39fe2774` 已完成。一个模型共同训练 M1_test5/6，五次更新、4920 timestep、每图十个 episode；`execution_pass=true`，所有 final replay 一致，`accepted=false`。完整运行为 70 次 solver 调用、453.21 秒。工件路径为 `runs/20260914T075849Z-v2-ppo-small-train-39fe2774/recipe-v2-ppo-small-train.json`。

| 版图 | 零偏移 J | 随机完整 Recipe 中位数 J | 随机最好 J | 最终确定性 PPO J |
| --- | ---: | ---: | ---: | ---: |
| M1_test5 | 135118 | 124701.5 | 116754 | 135118 |
| M1_test6 | 125847 | 113312 | 104976 | 125847 |

每图十个随机对照都降低 J；最终 PPO 在两图均输出全零偏移，未优于基线。更新后 Critic 归一化 RMSE 为 `0.505/0.356/0.241/0.178/0.224`；最终动作概率仍接近均匀，不能把全 stay 的 argmax 输出直接解释为高置信度概率塌缩。随机对照的 J 改善不等于全部单项指标改善，也不证明 PPO 优于随机。5 个 checkpoint、最终模型和 8 组输入 PNG/NPZ 的文件哈希已核对；NPZ 均为有限 float32 的 `5×128×128 + 12`。这些是同图、单 seed 诊断结果。

用户决定暂停 PPO 续训，下一阶段采用预算受限的离散坐标搜索诊断。固定 FRAG `(16,32)` 和 EPE 五动作 `[-20,-10,0,10,20]nm`、Golden evaluator、conflict-stay 规则。由全零 Recipe 起步，固定其他点，在当前点比较其余四个动作；每个候选均运行完整 solver，选择满足单项质量护栏且 J 严格降低的候选，随后基于已更新 Recipe 继续搜索。预算用尽或完整一轮无改善时停止，最终完整 Recipe 再独立求解验证一致性。

搜索对照为全零偏移、均匀随机完整 Recipe、离散坐标搜索。随机与坐标搜索使用相同候选 solver 预算；基线、最终回放、缓存命中和总调用数分别记账，并为最终回放预留预算。多张训练版图分别搜索和报告，不读取 validation/test。保存每次候选的点 ID、动作、完整 Recipe 身份、L2/EPE/PVB/J、接受/拒绝原因及累计调用数；继续保留输入和结果可视化。

搜索实现前仍需冻结版图列表、每图预算、点访问顺序/seed、单项护栏及其参照（零偏移基线或当前 incumbent）、不完整点扫描的处理规则。当前 `acceptance` 中的 null 不代表无限宽松护栏。搜索代码尚未实现或运行；以下 PPO 命令及配置启用项保留为历史复现入口，不表示当前要求续跑。本文当前决定优先于下方旧阶段的“下一步”描述。

下一代目标已修正为：共享 EPE 策略逐点读取局部 patch，并为每个 EPE 点分别输出沿自身法线的位移 `delta_i`；FRAG 不再逐点移动，只为每张版图选择两个全局参数 `lenCorner/lenUniform`；mask segment 的真实位移仍由 SimpleOPC solver 内部完成。ICCAD2013/GLP 主语义参考是配置所指 OpenILT 根目录下的 [`pyilt/simpleopc.py`](third_party/OpenILT-main/OpenILT-main/pyilt/simpleopc.py)；`opc/simpleopc.py` 只是旧 GDS/PatchSim 对照。上游 `checkEPE(distance=16)` 只作为原始控制 probe 对照，不能代替逐点 EPE Recipe；其锁定云端源码身份已由本次工件核对，本地下载镜像仍只用于源码审查。

v2 当前采用 PPO-first：分别训练“每个 EPE 动作后重跑 solver、获取条件边际 reward”的 dense PPO，以及“先收集完整 `delta_1...delta_N`、只在终点运行一次 solver”的 terminal PPO；确定性批量输出完整 Recipe 后的一次 solver 回放，是唯一部署一致的 validation/test 口径。两种协议的 Actor 都读取同一个冻结零偏移基准状态，best-prefix 只作诊断。FRAG-PPO 在 EPE 协议比较后训练。反事实教师、坐标下降、GNN/Transformer、轻量选择器和低置信度回退全部延期，不作为当前 PPO 开始训练的前置条件。完整语义、Golden EPE 防作弊边界、测试与云端门槛见 [`docs/recipe_ppo_global_parameters_v2_design.md`](docs/recipe_ppo_global_parameters_v2_design.md)。

现有 `simpleopc-recipe-point-v1` 及其 accepted 记录仍按旧协议保留，但其 EPE 沿切向、FRAG 逐点切向的动作语义只能作为历史诊断；在 v2 逐点法向 contract、固定 Golden 评价和新 validation 全部通过前，禁止将 v1 Recipe 用作决策树教师。

### v2 当前代码落点与阻断门槛

2026-09-02 已按并行版本推进 Phase 1–3，不原地改写 v1：

- `src/opc_agent/recipe_v2_contract.py`：独立 v2 版本、逐点 `base_xy/normal_xy/normal_offset_nm/moved_xy`、与 FRAG 类型分离的 `GoldenPointSet`、全局 FRAG 和严格 solver/Golden 数据协议；
- `src/opc_agent/recipe_v2.py`：法向 crossing/probe、动作别名和越界检查、全局 `dissect` adapter、冻结零偏移 observation、原子提交的 dense/terminal Fake-solver episode、final-only payload；
- `src/opc_agent/recipe_ppo_v2.py`：普通 Gymnasium/SB3 `Discrete(K)` 薄适配；构造时硬检查所有动作合法；
- `src/opc_agent/recipe_v2_runner.py`：dense/terminal 完全独立的单版图 CUDA PPO 数值 smoke，以及恰好三次更新的稳定性 pilot；保存逐次训练诊断、模型、确定性完整 Recipe 回放和 final replay；
- `src/opc_agent/recipe_v2_small_train.py`：M1_test5/6 两个独立环境向同一个 terminal PPO Actor-Critic 提供平衡 rollout；保存随机完整 Recipe 对照、更新前后 Critic/return/advantage/动作分布、逐更新 checkpoint、两图确定性回放和每图 PPO 输入样例；
- `src/opc_agent/recipe_v2_openilt.py`：只读锁定 OpenILT 的 v2 solver、正式 `evaluation.py::epecheck` Golden evaluator、全点 probe 几何扫描，以及从基线任一内部轮次的 active、无冲突点中按原始边/法线/角点特征分层抽样的真实 sensitivity 编排；
- `configs/recipe_ppo_v2.yaml`：`shared_terminal_small_training_rollout_shape_fix_cloud_rerun_pending_long_training_disabled` 配置；只显式开放受控的双版图 terminal 小训练子入口，`training.enabled=false` 继续禁止通用长训练，dense 仍因三更新 Critic RMSE 门禁失败而阻断；
- `tests/test_recipe_v2.py`、`tests/test_recipe_v2_geometry_contract.py`、`tests/test_recipe_v2_openilt.py`：不依赖 CUDA/OpenILT 的几何、协议、预检编排、哈希、失败回滚和数值测试。

用户于 2026-09-02 提供的云端终端输出已显示：OpenILT 提交为锁定值、tracked diff 为空、v2 专项测试及全量 pytest 均通过；随后上传的 preflight metadata 记录 Python `3.8.10`、Torch `2.0.1+cu118`，但未记录 GPU 型号。这些证据解除本次环境/依赖入口门槛，不代表 PPO 已训练或收敛。

首次真实 `v2-preflight` 在默认零偏移 Recipe 的 `polygon-7-edge-2-segment-2` 发现 inner 缺印与 outer 多印同时发生，当时的严格 solver 因单一法向方向不唯一而按设计停止；OpenILT 运行后 tracked diff 仍为空。两轮真实工件证明零冲突门禁在既定 train split 上不可达，因此当前正式协议改为 `both-sides-conflict-stay-v1`：完整记录冲突，并令该点当轮 stay。锁定源码的 `checkEPE` 对 inner 与 outer 分两次赋值，双侧同时违规时后写入的 outer 方向会覆盖 inner 方向；这是上游实现顺序，不是物理唯一性的证明，所以项目不采用任一覆盖优先级。

第二次云端 diagnostic run `20260902T031959Z-v2-preflight-f4a12f8a` 的三份工件已读取并校验。M1_test1 共 242 个 EPE 点；baseline 为 `L2=64498, EPE=28, PVB=73766, weighted_loss=141064`。六个候选 probe 中，`24nm` 恰好让 `[-20,-10,0,10,20]nm` 对全部 242 点几何合法；`16nm` 只支持 `[-10,0,10]nm`，其余 probe 也没有支持完整九动作。原抽样直接取前两个几何合法点：第一个点的 `±10nm` 均与 baseline 完全相同，第二个点仅 `-10nm` 改变 mask，loss 降至 `139417`（单次观察改善 `1647`，约 `1.1676%`）；这只能证明至少一个动作链路有真实响应，不能当成 PPO 改善。两处双侧冲突各在 5 次 solver 调用的内部第 6 轮重复出现，共 10 条记录，所以该 run 正确保持 `pass=false`、`training_enabled=false`。运行前后 OpenILT 都是 `dabb97c6...` 且 tracked diff 干净。

以下是首轮 preflight 当时的历史判断，不再代表当前阻断状态：六张 train 版图的首轮候选协议完成时，`d_probe=24nm` 对五动作 `[-20,-10,0,10,20]nm` 在六图全点几何扫描中都合法，但仅 M1_test6 通过当时的两点 sensitivity 门禁；M1_test1/2/3 还观测到双侧控制冲突。后续分层复检、conflict-stay、128 episode、Golden contract 和 CUDA smoke 已补齐这些入口证据。
对应正式证据目录为 `20260902T063309Z-v2-preflight-bf6caaf8`、`20260902T064148Z-v2-preflight-731c6e4d`、`20260902T064242Z-v2-preflight-15c4764f`、`20260902T064347Z-v2-preflight-94160101`、`20260902T064419Z-v2-preflight-4c441c6f` 和 `20260902T064520Z-v2-preflight-db3c9d11`，均位于 `runs/`。

| 版图 | `24nm` 全点五动作合法 | baseline 冲突（occurrence / 唯一点） | 响应点 | 当时 `pass` | 抽样动作最好 / 最差 $\Delta J$ |
| --- | --- | ---: | ---: | --- | ---: |
| M1_test1 | 是 | 1 / 1 | 1/2 | false | -29 / +509 |
| M1_test2 | 是 | 1 / 1 | 1/2 | false | -519 / 0 |
| M1_test3 | 是 | 22 / 16 | 0/2 | false | 0 / 0 |
| M1_test4 | 是 | 0 / 0 | 0/2 | false | 0 / 0 |
| M1_test5 | 是 | 0 / 0 | 1/2 | false | -81 / 0 |
| M1_test6 | 是 | 0 / 0 | 2/2 | true | -658 / +749 |

上表的 $\Delta J=J_{action}-J_{baseline}$，负值为该次单点观测改善。M1_test3 的 `22` 是基线内部轮次中的冲突出现次数，`16` 才是唯一 point_id 数。首轮实现对 eligible 列表直接取 `[:2]`，样本多为 `polygon-0/edge-0` 的相邻 segment；所以合计 `5/12` 只是有偏诊断计数，禁止当作总体响应率。

第二轮 `baseline-any-step-active-nonconflict-edge-stratified-v2` 已完成：每图固定抽 8 点、测试 32 个单点动作，六图均完成 33 次 solver 调用，OpenILT tracked diff 保持干净。

| 版图 | 响应点 | 改变 mask+Golden 的动作 | baseline 冲突 occurrence / 唯一点 | 33 次调用冲突 occurrence / 唯一点 | 最好 / 最差 $\Delta J$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 5/8 | 10/32 | 1 / 1 | 44 / 4 | -715 / +551 |
| M1_test2 | 4/8 | 10/32 | 1 / 1 | 38 / 5 | -773 / +54 |
| M1_test3 | 0/8 | 0/32 | 22 / 16 | 716 / 25 | 0 / 0 |
| M1_test4 | 4/8 | 8/32 | 0 / 0 | 8 / 2 | 0 / +572 |
| M1_test5 | 4/8 | 7/32 | 0 / 0 | 0 / 0 | -3802 / 0 |
| M1_test6 | 6/8 | 15/32 | 0 / 0 | 0 / 0 | -1627 / +4466 |

结果证明五动作链路在 M1_test1/2/4/5/6 上有真实响应，但也否定了“只过滤 baseline 冲突点就能保证训练安全”：M1_test4 的 baseline 零冲突，单点动作仍诱发了冲突。“抽样 active 点必须 100% 响应”也过强；离散 mask 更新中存在无效和等价动作，不能与整个动作链路无效混为一谈。

因此不再重复同一 preflight，也不依据当前 48 点样本删减 `±10/±20nm` 动作。下一实现步骤是冻结保守的 `both-sides-conflict -> stay`：不伪造 inner/outer 优先级，冲突点当轮不移动，但仍保留在 Golden EPE/L2/PVB 评价和轨迹中。随后才接入 dense/terminal PPO 的最小 CUDA smoke；M1_test3 保留在 train split，不因单点无响应而悄然删除。

控制与验收仍严格拆开：候选 `epe_probe_distance_nm=24` 只属于内部控制。正式 Golden EPE 来自锁定 OpenILT 的 `pyilt/evaluation.py::epecheck`；已上传工件把全局稳定的 `EPE_CONSTRAINT=15` 与 source SHA256 `cc2c1119...f3d8c` 核对成功，配置已记录这两个值。`sampling_state_sha256`、`coordinate_system_sha256` 和 `evaluator_contract_sha256` 含版图/target 身份，不能拿 M1_test1 的值冒充全局常量，仍须逐版图记录。Golden evaluator 的 `nm_per_coordinate` 与 raster 坐标系统必须和 solver 完全一致，否则 episode 在构造期失败。

当前几何 adapter 还显式冻结了 `raster_mapping_version=db-coordinate-equals-raster-pixel-v1`、`raster_scale=1`、`raster_offset_xy=[0,0]` 和 `normal_probe_coordinate=2`：即数据库坐标与 target raster 像素一一对应。其他缩放或原点会提前失败，必须先实现坐标变换和测试；这不是已支持任意 DBU/raster 映射的声明。`minimum_fragment_rule_version=min-corner-uniform-coordinate-v1` 只是当前复刻并版本化的上游最短段检查，不等同于已经冻结独立 MRC 规则。

当前 CPU/Fake solver payload 的 canonical final 动作保存在按 `point_id` 排序的 `epe_points[].normal_offset_nm`，canonical final 指标保存在 `raw_metrics`，其实际来源由 `raw_metrics_source` 标明。diagnostic payload 的 `accepted_metrics_source=null`，`required_accepted_metrics_source=batched_final_replay` 只声明未来正式验收必须采用的口径；当 `status=diagnostic_only`、`full_recipe_replay_sha256=null` 时，不代表已经完成 batched replay 或 accepted。v2 的 seed 数、改善比例和 L2/EPE/PVB guardrail 当前也全部未冻结，`configs/recipe_ppo_v2.yaml` 保持为 `null`；README 后文的 `1%` 等门槛仅属于已验收 v1，不能套用到 v2。

当前 v2 observation 固定为 `128×128` 单尺度输入。在 `nm_per_coordinate=1`时，`64×64` 半径约 32nm，小于候选动作和 probe 的最远控制位置 `20+24=44nm`；`128×128` 既包含原中心区域，又覆盖完整控制上下文，因此 v2 不再重复运行 64 smoke。底层 64 版本仅保留为历史回归兼容，不是训练候选。多尺度保持 `not_frozen`，不作为当前 Phase 4 前置条件。

当前可运行的 CPU 门槛为：

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q tests/test_recipe_v2.py -ra
python -m pytest -q tests/test_recipe_v2_geometry_contract.py -ra
python -m pytest -q tests/test_recipe_v2_openilt.py -ra
```

这些命令只验证 CPU contract/Fake solver/预检编排，本身不代表真实 OpenILT、CUDA smoke 或 validation 通过；真实 128 episode smoke 与 terminal PPO smoke 的独立云端证据见下文。v2 已接入受限的单图、单 rollout PPO smoke，但长训练 workflow 仍被配置硬禁用，正式实验产物只能写入 `runs/<run_id>/`。

云端 M1_test1 conflict-stay 复检 `20260902T075719Z-v2-preflight-85d91420` 已完成：

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -m opc_agent.cli v2-preflight --config configs/recipe_ppo_v2.yaml --layout M1_test1
git -C third_party/OpenILT diff HEAD --exit-code --
```

工件显示 `policy=both-sides-conflict-stay-v1`、`training_compatible=true`、`resolved_by_frozen_policy=true`、`solver_calls=33` 且 OpenILT tracked diff 干净；与上一份 M1_test1 工件的版图、分段、baseline、Golden 和 sensitivity 结果完全一致。这解除了正式 solver 遇冲突直接崩溃的阻断，但不代表 PPO 已训练。

只使用 `128×128` observation 的完整 episode smoke 已在云端通过。M1_test4 run `20260902T083239Z-v2-episode-smoke-9b75f6f9` 对 dense/terminal 各重复两次，使用同一完整动作表，并强制 sequential final 与 batched final replay 一致：

```bash
python -m opc_agent.cli v2-episode-smoke \
  --config configs/recipe_ppo_v2.yaml \
  --layout M1_test4
git -C third_party/OpenILT diff HEAD --exit-code --
```

已下载工件与终端结果一致：`patch_size=128`、4 个完整 episode、218 次 solver 调用、`pass=true`、`cross_protocol_final_equal=true`、`repeat_baseline_equal=true`，四个 variant 均满足 `final_replay_equal=true` 且 `reward_telescoping_error=0`；运行后 OpenILT tracked diff 干净。该证据解除 128 observation 与完整 episode 数值协议门禁，但该入口仍不训练 PPO，也不能证明 PPO 收敛。

v2 现在强制每个 smoke/后续训练 run 按 `geometry-diverse-by-point-id-v1` 固定选取少量样例，抽样不依赖 reward 或最终结果。每个样例直接读取实际 Actor 的冻结输入，保存一张六面板 PNG（五个原始通道加解释性 overlay）和一个精确的 `image[5,128,128] + vector[12]` NPZ；`manifest.json` 记录 point ID、基准坐标、法线、observation/file SHA256。产物与运行 JSON 一起位于：

```text
runs/<run_id>/ppo-input-examples/
├── manifest.json
├── example-00-...-5ch.png
├── example-00-...-input.npz
└── ...
```

已有的 episode smoke 早于该导出功能，没必要为四张汇报图重复 218 次 solver。使用轻量入口只做一次冻结基线求解并生成同源样例：

```bash
python -m opc_agent.cli v2-input-examples \
  --config configs/recipe_ppo_v2.yaml \
  --layout M1_test4
git -C third_party/OpenILT diff HEAD --exit-code --
```

命令写入 `runs/<run_id>/recipe-v2-input-examples.json` 与 `ppo-input-examples/`。PNG 仅用于汇报和人工审计；NPZ/observation 哈希用于证明它来自真实输入；二者都不参与 reward、训练或 accepted 判定。

云端 run `20260902T091312Z-v2-input-examples-a460fefd` 已完成并下载核验：M1_test4 共 104 点，只调用 1 次 baseline solver，按固定策略保存 4 个样例。manifest、4 个 PNG、4 个 NPZ 和重建的逐点 observation 哈希全部一致；每个 NPZ 均只含有限 `float32 image[5,128,128]` 与 `vector[12]`，四个样例覆盖 `(-1,0)/(0,1)/(1,0)/(0,-1)` 四种法线，PNG 逐张人工检查通过。该门禁已关闭，但仍不代表 PPO 已训练。

六张 train 版图的 `sampling_state_sha256`、`coordinate_system_sha256`、`evaluator_contract_sha256` 也已从每图两次独立 preflight 交叉核对，两次结果逐图一致，现已写入 `golden.layout_contracts`。PPO runner 构造训练/回放环境时必须按 `layout_parent` 匹配，缺失或错配会立即失败；普通 diagnostic preflight 仍允许在尚未冻结的新 validation/test 版图上采集 contract。

首个 terminal PPO run `20260914T014413Z-v2-ppo-smoke-fe22a4d2` 已在 RTX 4090 上完成 104 timestep、一次更新、模型保存及确定性完整 Recipe 回放。该 v1 工件暴露了近常量 return 的病态 explained variance 被误保留的问题，因此 runner 将工件协议升级为 v2：当 return 方差小于等于 `1e-12` 时将 explained variance 写为 `null`，并记录实际 `return_variance/return_std` 及原因。

刷新后的正式 terminal smoke 工件为 `20260914T015912Z-v2-ppo-smoke-aa5e798c`。云端全量 pytest 通过，运行前后 OpenILT tracked diff 均为空；下载工件的模型、manifest、4 个 PNG/NPZ 哈希全部一致。`smoke_version=v2`，`return_variance=1.6086e-16`、`return_std=1.2683e-8`、`explained_variance=null`、`numeric_pass=true`、`final_replay_equal=true`。更新前后策略哈希不同，且新旧两次 run 的初始/训练后策略、动作表、最终 Recipe、mask、Golden 指标和 baseline 哈希完全一致，证明诊断修补没有改变训练行为。terminal PPO CUDA smoke 门禁据此关闭。

该单 rollout 的 final weighted loss仍从 `52508` 变为 `54100`，EPE 从 `6` 变为 `7`、PVB 从 `28317` 变为 `31380`；它只证明训练主链路、数值记录和回放可运行，不证明 PPO 质量改善、收敛或跨图泛化。

dense PPO smoke 工件 `20260914T021120Z-v2-ppo-smoke-cce530d6` 也已完成并下载核验：模型、manifest、4 个 PNG/NPZ 哈希全部一致，104 timestep、一次更新、`numeric_pass=true`、`final_replay_equal=true`，运行后 OpenILT tracked diff 干净。训练与评价合计调用 211 次 solver，墙钟 `568.25s`；terminal 对应 5 次和 `19.57s`，dense 的调用量为 `42.2×`、墙钟约 `29.0×`。dense 与 terminal 从相同初始策略出发，训练后策略和动作表哈希不同，但最终 mask 与 Golden 指标完全相同，仍为 `L2=22020, EPE=7, PVB=31380, J=54100`；两者都不是改善证据。

dense 的 `return_std=0.01419`、`approx_kl=2.36e-5`、`clip_fraction=0` 且所有张量有限，但 `value_target_normalized_rmse=10.77`，首次越过设计中的候选警戒值 `10`；规则要求连续三次超限才停止，不能根据一次 smoke 判定 Critic 失败。为此新增严格受限的 `v2-ppo-pilot`：只在 M1_test4、seed 0 上连续运行恰好三个完整 rollout，逐次保存数值与策略哈希，最后只做一次确定性 full-Recipe replay。该 pilot 仍为 `diagnostic_only`、`accepted=false`、`long_training_enabled=false`，并继续保存 4 组 PPO 输入样例。

terminal 三更新 pilot 工件 `20260914T023940Z-v2-ppo-pilot-2a54368f` 已完成并下载核验：三次更新累计 312 timestep，逐次策略哈希不同，模型、manifest、4 个 PNG/NPZ 哈希均一致；4 组 NPZ 均为有限 `float32` 的 `5×128×128` image 与 12 维 vector。三次归一化 value RMSE 为 `9.59 / 502.80 / 5.45`，KL 分别约为 `2.44e-5 / 4.54e-6 / 5.44e-6`，clip fraction 均为 `0`；只有第二次 RMSE 超限，最大连续超限次数为 1，因此按冻结的“连续三次才失败”规则 `stability_pass=true`。训练和评价共调用 7 次 solver，墙钟 `27.13s`，final replay 一致，运行后 OpenILT tracked diff 干净。

该 pilot 的确定性策略对 104 个点全部选择动作类 4，即 `+20nm`，`dominant_action_fraction=1.0`；final `J` 从 `52508` 恶化到 `58454`（`L2:23591→31500`、`EPE:6→13`、`PVB:28317→25654`）。这不违反当前“连续三次数值超限”门禁，但属于明确的动作集中与质量退化红旗：terminal 子门禁只证明连续训练入口和回放稳定，不能证明 Critic 有效、策略未塌缩或质量改善。是否正式判定动作塌缩仍须不同局部点最优动作证据和后续多 seed/validation guardrail；现在不得据此开启长训练。

dense 三更新 pilot 工件 `20260914T030101Z-v2-ppo-pilot-0cdce0b9` 已完成并下载核验。模型、manifest、4 个 PNG/NPZ 哈希均一致，NPZ 仍为有限 `float32` 的 `5×128×128 + 12`；三次策略哈希互异，final replay 一致，419 次 solver 账目闭合，墙钟 `1125.11s`，运行前后 OpenILT tracked diff 为空。三次归一化 value RMSE 为 `10.77 / 156.87 / 13.28`，连续三次均超过阈值 10，明确触发 `value-target-normalized-rmse-high-for-three-updates`，因此 `stability_pass=false`、`pass=false`。

dense 最终动作计数为 `[-20,-10,0,+10,+20]=[1,0,0,0,103]`，确定性 argmax 占比 `0.9904`；但每次 entropy loss 的绝对值仍接近 `ln(5)`，说明随机策略分布接近均匀，现阶段只能称确定性输出高度集中，不能单凭 argmax 计数断言概率策略已塌缩。dense 与 terminal 从相同初始策略开始，训练后模型和 Recipe 不同，却得到相同 final mask 及 `J=58454`（`L2=31500, EPE=13, PVB=25654`），相对 baseline 的 `J` 恶化 `5946`，约 `11.32%`。当前阻断原因是 Critic 相对 return 尺度连续失配，同时伴随确定性质量退化；不得调高阈值来把失败改成通过，也不得继续长训练。

### 双版图单共享模型 terminal 受控小训练

用户已明确选择不在 M1_test5/6 上分别训练两个模型，而是让两张图的环境共同更新一个共享策略。新增的 `v2-ppo-small-train` 只允许配置冻结的 `[M1_test5, M1_test6]` 和 terminal 协议：两图均有 246 个 EPE 点；每次更新每图运行两个完整 episode，`n_steps=492/env`，两个环境组成 984-step rollout buffer；共五次更新，因此每图 10 个训练 episode、总计 4920 timestep。Actor 仍只读取冻结的 `5×128×128 + 12` 输入，不加入 layout ID。

该入口同时保存：每图 10 个固定随机完整 Recipe 对照、未训练策略回放、每次更新后两图的确定性完整 Recipe 回放、每次 checkpoint、更新前/后 Critic 对同批 return target 的误差、advantage 正负比例、动作概率/熵/argmax margin、三类 solver 调用账目，以及每图 4 个 PNG+NPZ 输入样例（共 8 个）。连续三次更新后 Critic 归一化 RMSE、KL 或 clip 超限会提前停止；任一 final replay 不一致也立即停止。

这只是有限预算的可学习性诊断，不是长训练。即使命令最终输出 `execution_pass=true`，也只表示预算完成、数值门与回放一致性通过；工件仍固定为 `diagnostic_only`、`quality_accepted=false`、`accepted=false`、`long_training_enabled=false`。dense 小训练仍未开放。

首次云端执行前的全量 pytest 与 OpenILT clean check 均通过，两个环境也已实际完成首个 984-step CUDA rollout；随后诊断层因仍假设 `returns.shape=(n_steps,n_envs)` 而停止。SB3 在 `model.learn()` 内部取 minibatch 时会把 `returns/values/actions/advantages` 按环境优先展平为 `(n_steps*n_envs,1)`，但 `rewards` 仍可能保留二维形状。该问题属于工件诊断解析缺陷，不是 PPO 数值失败，也没有形成完整训练 JSON。当前修复依据 rollout buffer 自带的 `buffer_size` 与 `n_envs` 同时兼容更新前二维和更新后 env-major 两种形式，并新增对应回归测试；训练版图、共享模型、超参数和预算均未改变。

云端在同步本次代码后执行：

```bash
python -B -m pytest -q -ra

git -C third_party/OpenILT diff HEAD --exit-code --

python -B -m opc_agent.cli v2-ppo-small-train \
  --config configs/recipe_ppo_v2.yaml \
  --layouts M1_test5 M1_test6 \
  --protocol terminal

git -C third_party/OpenILT diff HEAD --exit-code --
```

dense/terminal v2 smoke 与两个三更新 pilot 均已结束；dense pilot 按预设规则失败。现在不再重复 pilot，也不直接开启通用长训练，而是先执行上面的共享 terminal 小训练，用五次更新中记录的 Critic/return/advantage 和逐图质量轨迹判断是否存在可学习信号。首轮仍保持 reward scale `1e-5`，不得借此提高阈值或生成 accepted。

## 当前已实现的 v1 主线架构

```text
原始 ICCAD13 target GLP
        |
        +-- 在原始 target 边上建立 EPE 测量点和 FRAG 分段点
        |
        +-- 每个点提取 5×64×64 局部图像 + 14 维辅助向量（含点坐标）
        |       5 通道：target / 当前最佳 mask / printed / EPE marker / FRAG marker
        |
共享 CNN-PPO（只在 6 个 train 父版图训练）
        |
        +-- 一个环境 step 只处理一个 recipe 点
        +-- Discrete(9) 直接选择 -40,-30,...,40nm 的绝对位置偏移
        +-- 每个点在一次确定性回放中恰好访问一次，不再分四步累计
        |
RecipeAwareOPCSolver
        +-- EPE 偏移：沿 target 边切向移动测量位置
        +-- FRAG 偏移：沿 target 边切向移动分段位置并重新分段
        +-- mask 位移：仅在 solver 内按 EPE 方向迭代
        +-- 评价：始终使用原始 target 的固定边界评价点
        |
validation 质量门槛
        +-- rejected：停止，不训练树
        +-- accepted：PPO 主线验收完成，才允许规划 MLLM/双树
```

这里的 `64×64` 是像素尺寸，不是无条件等同于 `64nm×64nm`。默认 `nm_per_coordinate=1.0`、`openilt_scale=1` 时，一个坐标像素对应 `1nm`，此时局部视野才约为 `64nm×64nm`。

### PPO 动作与 episode

- 动作空间固定为 `Discrete(9)`，类别代表值固定为 `[-40,-30,-20,-10,0,10,20,30,40]nm`。
- 动作含义是“当前 recipe 点相对初始位置的绝对切向偏移”，不是 mask 位移，也不是 `10nm` 增量。
- 一个 episode 遍历当前版图的全部 EPE/FRAG 点一次；训练时可打乱点顺序，确定性验收时使用固定顺序。
- 每个 action 后都会用完整 Recipe 重新运行内部 OPC solver，并返回 `-(L2+100×EPE+PVB)`。训练和验收都绑定原始损失版本 `paper-weighted-sum-raw-v1`，不再除以每个 episode 的初始损失。
- PPO 模型在全部 train clip 之间共享；validation/test 只做确定性回放，不参与参数更新。

### Recipe-aware solver

`src/opc_agent/recipe_ppo.py` 在项目侧封装 solver，不修改 OpenILT：

1. 从原始 GLP 多边形建立 target，不从 OPC 后 mask 重新打点。
2. 默认均匀分段生成 EPE fragment 中点和 FRAG 内部切点。
3. 应用完整 EPE/FRAG Recipe 后重建分段。
4. mask 从 target 起步，只在 solver 内按 recipe EPE 点测得的误差方向移动 fragment。
5. 保存内部历史最佳 mask；最终 L2/EPE/PVB 始终以原始 target 和固定 target 边界计算，防止 PPO 通过移动测量点逃避评价。

每次 `solve` 默认包含 1 次初始仿真和 8 次内部 mask 更新仿真，因此点级训练的计算量显著高于历史 v3。粗略下界为 `PPO timestep × 9` 次光刻仿真，另有 episode reset、十图确定性回放和三种子开销。`--smoke` 的 256 timestep 只验证接口与产物，可以小于完整 episode；它不证明收敛。

## 主要文件

- `src/opc_agent/recipe_contract.py`：无 GPU 依赖的版本化数据协议。
- `src/opc_agent/recipe_ppo.py`：点级环境、EPE/FRAG 几何和只读 OpenILT solver。
- `src/opc_agent/recipe_ppo_runner.py`：5 通道 CNN、共享 PPO、默认 Recipe 与确定性回放导出。
- `src/opc_agent/recipe_ppo_quality.py`：训练后只读审核与 validation 门槛。
- `src/opc_agent/recipe_point_visualization.py`：当前主线训练前十图与训练后逐 seed 打点对比图。
- `src/opc_agent/recipe_v2_contract.py`、`recipe_v2.py`、`recipe_ppo_v2.py`、`recipe_v2_openilt.py`、`recipe_v2_runner.py`、`recipe_v2_visualization.py`：已通过 preflight、128 episode smoke、dense/terminal 单 rollout PPO smoke 及 terminal 三更新 pilot；dense 三更新 pilot 已因 Critic RMSE 连续超限而失败，当前进入原因诊断，仍不属于已验收 v1，也不代表长训练或收敛。
- `src/opc_agent/workflow.py`：`train-oracle` 主线编排。
- `configs/paper_repro.yaml`：动作、图像、solver、切分与验收配置。
- `configs/recipe_ppo_v2.yaml`：v2 独立预检配置；当前明确禁用训练。
- `src/opc_agent/simpleopc*.py`：历史 v3 兼容代码，不是当前论文主线。

## 云端环境

目标环境为 Ubuntu、CUDA GPU、Python 3.8。上传本项目后，在仓库根目录执行：

```bash
cd ~/autodl-tmp/opc_agent
bash scripts/cloud/bootstrap.sh
source .venv/bin/activate

export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

pytest -q -ra
```

启动脚本会克隆并校验固定 OpenILT 提交、安装 `stable-baselines3==2.0.0`、`gymnasium==0.28.1` 和 CUDA PyTorch。项目不会读取或创建 `.env`，也不会把密钥写入产物。

训练前必须确认 OpenILT 的 tracked 源码未变：

```bash
git -C third_party/OpenILT rev-parse HEAD
git -C third_party/OpenILT diff HEAD --exit-code --
```

第一条必须输出 `dabb97c6ca3dfd159362e48273c436444c77353b`，第二条退出码必须为 `0`。普通未跟踪文件不参与该检查，但建议不要在 OpenILT 目录保存实验结果。

`PYTHONDONTWRITEBYTECODE=1` 是当前云端流程的必要只读保护。固定 OpenILT 提交本身跟踪了若干
`__pycache__/*.pyc`；没有该变量时，Python 导入可能重写这些文件并令 post-run diff 返回 `1`。

## 先复现官方 SimpleOPC 十图基线

在 PPO 训练前，先原样运行固定 OpenILT 提交中的 `pyilt/simpleopc.py`：

```bash
python -m opc_agent.cli baseline --config configs/paper_repro.yaml
git -C third_party/OpenILT diff HEAD --exit-code --
```

该入口不修改官方脚本的 `STEPS=8`、`STEPSIZE=8`、`DECAY=4`、`MAXDIST=24`、
`lenCorner=16`、`lenUniform=32`、全图同步 fragment 移动和最终 Step 7 图片选择。为保持
OpenILT clone 只读，项目会在本次 `runs/<run_id>/official-simpleopc/` 下建立只读符号链接工作目录，
官方硬编码的 `tmp/SimpleOPC_*.png` 因而写入本项目运行目录，不会写回上游 clone。

成功运行必须产生：

```text
runs/<run_id>/
├── openilt-baseline.log
├── openilt-revision.txt
├── official-simpleopc.metrics.json
└── official-simpleopc/tmp/
    ├── SimpleOPC_target1.png
    ├── SimpleOPC_mask1.png
    ├── SimpleOPC_resist1.png
    └── ...                               # 共十图三类、30 张 PNG
```

`official-simpleopc.metrics.json` 严格要求十图各有 `Initialized` 和 `Step 0..7`，并记录 PNG
SHA256、总运行时间及每轮 L2/PVBand/总 EPE。`official_final` 始终保留上游实际保存的 Step 7；
`common_objective_best` 只按本项目统一的 `L2+100×EPE+PVB` 离线标出最佳轮，不能冒充官方输出。
上游脚本关闭 shot counting，因此 `Shot: -1` 表示未测量，不得解释为负的或零次曝光。

### 2026-08-26 官方 SimpleOPC 参考基线

云端运行 `20260826T093351Z-baseline-04c6f654` 已在固定 OpenILT 提交
`dabb97c6ca3dfd159362e48273c436444c77353b` 上完成，进程退出码为 `0`，十张版图均运行至
Step 7，Shell 记录的总墙钟时间为 `22 s`。原始日志和结构化指标分别归档在
`runs/20260826T093351Z-baseline-04c6f654/openilt-baseline.log` 与
`runs/20260826T093351Z-baseline-04c6f654/official-simpleopc.metrics.json`。该时间只适合在相同云端
环境下作粗略效率参考；当前记录未包含 GPU 型号和峰值显存，不能用于跨机器性能结论。

下表将 L2、PVBand 和 EPE 分开报告。“共同最佳轮”是运行结束后按
`L2+100×EPE+PVBand` 对每张版图回看选出的离线最优轮；“最终值”才是官方脚本固定保存的 Step 7。
因此，共同最佳值属于 oracle 参考，不能当成无需回看的正式测试结果。

| 版图 | 数据集 | 共同最佳轮 | L2（初始 / 共同最佳 / 最终） | PVBand（初始 / 共同最佳 / 最终） | EPE（初始 / 共同最佳 / 最终） | 加权损失（初始 / 共同最佳 / 最终） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 训练 | 6 | 116184 / 49289 / 54355 | 45874 / 82588 / 93675 | 86 / 8 / 16 | 170658 / 132677 / 149630 |
| M1_test2 | 训练 | 2 | 117802 / 43215 / 40349 | 37035 / 58726 / 78214 | 84 / 11 / 12 | 163237 / 103041 / 119763 |
| M1_test3 | 训练 | 0 | 160846 / 160846 / 100706 | 32646 / 32646 / 150400 | 125 / 125 / 51 | 205992 / 205992 / 256206 |
| M1_test4 | 训练 | 3 | 84037 / 21437 / 21318 | 101 / 30530 / 33130 | 64 / 4 / 4 | 90538 / 52367 / 54848 |
| M1_test5 | 训练 | 4 | 117516 / 51221 / 51015 | 59189 / 69005 / 69674 | 71 / 3 / 2 | 183805 / 120526 / 120889 |
| M1_test6 | 训练 | 4 | 110523 / 48688 / 48494 | 50684 / 65664 / 66471 | 66 / 1 / 1 | 167807 / 114452 / 115065 |
| M1_test7 | 验证 | 4 | 103219 / 35538 / 35538 | 54316 / 55064 / 55064 | 71 / 0 / 0 | 164635 / 90602 / 90602 |
| M1_test8 | 验证 | 4 | 55012 / 19539 / 19539 | 19084 / 26513 / 26513 | 37 / 2 / 2 | 77796 / 46252 / 46252 |
| M1_test9 | 测试 | 7 | 120212 / 58730 / 58730 | 60797 / 79976 / 79976 | 66 / 3 / 3 | 187609 / 139006 / 139006 |
| M1_test10 | 测试 | 2 | 41291 / 23177 / 22123 | 15039 / 18988 / 21217 | 26 / 0 / 1 | 58930 / 42165 / 43440 |

十图平均及相对初始值的变化如下；正的“下降”表示该项变好，负值表示该项变差：

| 结果口径 | L2 平均值（下降） | PVBand 平均值（下降） | EPE 平均值（下降） | 加权损失平均值（下降） |
| --- | ---: | ---: | ---: | ---: |
| 初始 | 102664.20 | 37476.50 | 69.60 | 147100.70 |
| 官方最终 Step 7 | 45216.70（55.96%） | 67433.40（-79.94%） | 9.20（86.78%） | 113570.10（22.79%） |
| 逐图共同最佳轮（oracle） | 51168.00（50.16%） | 51970.00（-38.67%） | 15.70（77.44%） | 104708.00（28.82%） |
| 各指标各自最佳轮（oracle） | 44404.00（56.75%） | 36296.20（3.15%） | 7.60（89.08%） | 不适用 |

“各指标各自最佳轮”中的三个最小值可能来自不同轮次、不同 mask，只用于观察算法能力上限，禁止把
三项拼成一个实际存在的结果。该基线的主要现象是：官方 Step 7 在十图上都降低了 L2 和 EPE，
但十图 PVBand 都高于初始值；其中 M1_test3 的加权损失从 `205992` 恶化到 `256206`。因此后续 PPO
比较必须同时单列 L2、PVBand、EPE 和统一加权损失，既比较固定最终轮，也可补充标注清楚的 oracle
最优轮，不能只用 EPE 改善宣称整体优于 SimpleOPC。

官方脚本还依赖可导入的 KLayout Python 包（`import klayout.db as pya`）。当前
`requirements-lock.txt` 尚未锁定该运行时依赖；正式复现实验前应记录实际 KLayout 版本并补齐依赖锁定，
否则新环境可能在进入算法前因 `ModuleNotFoundError: klayout` 失败。

## v1：训练前生成十张 Recipe 点总览

在完整训练前先运行当前主线专用可视化：

```bash
python -m opc_agent.recipe_point_visualization \
  --config configs/paper_repro.yaml

git -C third_party/OpenILT diff HEAD --exit-code --
```

该命令复用训练主线的同一 solver 构造配置，为固定 `6/2/2` 切分的十张 ICCAD2013 版图生成默认
Recipe 点，但不会调用 `solver.solve`、不会训练 PPO、不会写入 `runs/`。产物统一写入：

```text
outputs/recipe_point_visualization/before-training/
├── M1_test1-points.png
├── ...
├── M1_test10-points.png
└── visualization-summary.json
```

每张图使用深灰 target 边界、红色圆形 EPE 点、蓝色方形 FRAG 点和逐点编号；颜色之外同时使用
不同形状，便于灰度检查。摘要保存每图点数、EPE/FRAG 数量、动作分布、PNG SHA256、OpenILT
提交和 post-run 只读状态。继续训练前必须满足：

- `clip_count == 10`，并且十张 PNG 均存在且非空；
- 每张图 `epe_point_count > 0`、`frag_point_count > 0`；
- 训练前所有图 `nonzero_displacement_count == 0`；
- `openilt_mutation == "none"`，外部 OpenILT diff 退出码仍为 `0`；
- 人工检查点位于 target 边上，没有明显聚集、越界或编号错位。

这里生成的是整张版图的打点总览，不是每个点的 `5×64×64` PPO 输入 patch。后者数量较多，
当前没有默认全部落盘，避免把汇报总览和模型输入诊断混为一谈。

## v1：训练前配置检查

```bash
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/paper_repro.yaml").read_text(encoding="utf-8"))
oracle = config["oracle"]
simpleopc = config["simpleopc"]

assert oracle["environment"] == "simpleopc-recipe-point-v1"
assert oracle["reward_mode"] == "paper_raw"
assert simpleopc["local_patch_size"] == 64
assert "step_sizes_nm" not in simpleopc
assert simpleopc["displacement_classes_nm"] == [-40, -30, -20, -10, 0, 10, 20, 30, 40]
print("Recipe PPO 配置检查通过")
PY
```

配置关键项：

- `data.train_parents/validation_parents/test_parents`：父版图级 `6/2/2` 固定切分。
- `oracle.ppo_n_steps`、`ppo_batch_size`、`learning_rate`：共享 PPO 参数。
- `simpleopc.local_patch_size`：当前必须为 `64`。
- `simpleopc.displacement_classes_nm`：当前必须是完整九类。
- `simpleopc.base_fragment_length_nm`、`min_fragment_length_nm`：默认分段与 FRAG 可行动作边界。
- `simpleopc.inner_step_sizes_nm`：只控制 solver 内部 mask OPC，绝不是 PPO 四步位移。
- `ppo_acceptance`：只在 validation 上决定 accepted/rejected；test 不参与选模。

## v1：先运行只读 preflight

preflight 会为首个 train clip 建立真实 Recipe 点、运行一次默认内部 solver、生成 observation 和点数，但不会构造或训练 PPO：

```bash
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --preflight
```

成功后查看输出运行 ID 对应的 `stage-result.json`。只有以下条件全部满足，才继续 smoke：

- `mode == preflight` 且 `preflight.training_started == false`
- `preflight.image_shape == [5,64,64]`、`preflight.vector_shape == [14]`
- `preflight.reset.epe_point_count > 0`、`frag_point_count > 0`
- `git -C third_party/OpenILT diff HEAD --exit-code --` 返回 `0`

## v1：再运行 smoke

```bash
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --smoke
```

命令成功结束会输出运行 ID，例如 `20260825T...-train-oracle-...`。随后只读检查：

```bash
RUN_ID='<替换为上一步输出的运行ID>'

python -m opc_agent.recipe_ppo_quality \
  --run-dir "runs/$RUN_ID" \
  --config configs/paper_repro.yaml

git -C third_party/OpenILT diff HEAD --exit-code --
```

smoke 的质量状态必须是 `diagnostic_only`，不是 `accepted`。还应检查 `runs/$RUN_ID/stage-result.json`：

- `environment == simpleopc-recipe-point-v1`
- `action_semantics == one_absolute_nine_class_decision_per_recipe_point`
- `patch_shape == [5,64,64]`
- `epe_point_count > 0` 且 `frag_point_count > 0`
- model、metadata、default Recipe 和 PPO Recipe 文件均存在且哈希一致
- OpenILT diff 退出码仍为 `0`

任一条件失败都不要启动完整训练。

## v1：完整共享 PPO 训练与验收

smoke 通过后再启动长任务：

```bash
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml
```

当前 `total_timesteps=10000` 表示每个 seed 的共享策略总步数，不是每个 clip 各 10000 步；三个 seed 合计 30000 PPO timestep。训练只使用 6 个 train 父版图，但每个 seed 训练后会对 train/validation/test 十图做一次完整确定性回放并导出 Recipe。

训练结束后执行：

```bash
RUN_ID='<替换为完整训练输出的运行ID>'

python -m opc_agent.recipe_ppo_quality \
  --run-dir "runs/$RUN_ID" \
  --config configs/paper_repro.yaml

git -C third_party/OpenILT diff HEAD --exit-code --
```

质量报告会验证：

- 模型/版图哈希和全部版本号一致；
- 确定性回放对每个 EPE/FRAG 点恰好决策一次；
- 所有位移精确属于九类且没有量化误差；
- 轨迹使用原始 `L2+100×EPE+PVB`；
- validation 最佳 seed 不差于默认零位移 Recipe，且至少改善 1%；
- 三个 seed 的损失变异系数不高于 `0.20`；
- EPE 和 FRAG 各自的最大单一动作占比不超过 `0.95`。

任何正式门槛失败都会写出 `rejected` 并返回退出码 `2`。只有 `accepted` 才能把这一阶段作为 PPO 教师；test 指标只报告，不参与 accepted 决定。

## v1：已完成运行登记（2026-08-26）

当前 Recipe 点级 PPO 主线已完成从接口预检、CUDA smoke 到正式三种子训练的逐级验收。正式运行
产物均保存在 `runs/<run_id>/`；本节只登记运行身份、结论和证据边界，不能替代运行目录中的原始
JSON、模型与 Recipe。

| 运行 ID | 模式 | 状态 | 说明 |
| --- | --- | --- | --- |
| `20260825T131840Z-train-oracle-41af618f` | preflight | complete | 首次真实 EPE/FRAG 点、solver 与 observation 检查；运行后外部 Git 检查发现 OpenILT tracked `.pyc` 被 Python 重写，因此不作为最终只读 preflight。 |
| `20260825T132235Z-train-oracle-63010c34` | preflight | complete | 设置 `PYTHONDONTWRITEBYTECODE=1` 后重跑；M1_test1 包含 58 个 EPE 点、21 个 FRAG 点，图像为 `[5,64,64]`、向量为 `[14]`，运行后 OpenILT diff 为 `0`。 |
| `20260825T132516Z-train-oracle-293d9938` | smoke | diagnostic_only | CUDA 上完成 256 timestep，耗时 187 秒；模型、metadata、单图 Recipe 和质量报告均成功生成，质量审核与 OpenILT diff 退出码均为 `0`。 |
| `20260826T015433Z-train-oracle-2b7aa7f3` | full | accepted | 三个共享模型、十张 ICCAD2013 版图和三种子确定性回放完成；validation-only 正式质量门槛通过。 |

### 正式 PPO 结果

本次正式结果目录为：

```text
runs/20260826T015433Z-train-oracle-2b7aa7f3/
```

质量报告为 `accepted`，审核退出码为 `0`，`tree_training_allowed=true`，报告 SHA256 为：

```text
9f9d8378387e32a7a09c4f234bffc7eb938fe3c5c36e52742cce435f62c24c21
```

训练后 OpenILT 仍位于固定提交 `dabb97c6ca3dfd159362e48273c436444c77353b`，tracked diff
退出码为 `0`。各 split 的最佳相对改善为：

- train 六图平均约 `2.23%`；M1_test3（约 `0.889%`）和 M1_test5（约 `0.783%`）低于
  单图 `1%` 门槛，但损失仍优于默认 Recipe；
- validation：M1_test7 约 `2.405%`、M1_test8 约 `5.636%`，两图均通过并决定本次
  `accepted`；
- test：M1_test9 约 `1.114%`、M1_test10 约 `1.072%`，只用于报告，不参与选模。

三个 seed 的损失变异系数均远低于 `0.20` 门槛，当前观察最大值约为 `0.01186`。Validation
没有触发 `0.95` 动作塌缩门槛，但类别明显集中：EPE 只出现 `-40/0/+30nm = 9/15/70`，
FRAG 只出现 `-40/0/+30nm = 3/6/45`，其余六类没有出现。`+30nm` 占 EPE 约 `74.47%`、
FRAG 约 `83.33%`。

因此，本次 `accepted` 只证明当前 Recipe PPO 通过既定 validation 门槛。质量报告中的
`tree_training_allowed=true` 是协议许可，不表示旧 `ppo_recipe_labels`/`build-recipe` 已兼容新的
`ppo-recipe-point-v1`。在进入决策树前，仍须审计全 split、全 seed 的类别不平衡并完成新标签适配。
不得把本次结果表述为论文整体复现完成、跨数据集泛化已证明或决策树阶段已经完成。

### 训练后生成逐 seed 前后对比图

smoke 或完整训练已经写出 `stage-result.json` 和 PPO Recipe 后，可以从运行证据重新生成可视化：

```bash
RUN_ID='<替换为 smoke 或完整训练运行ID>'

python -m opc_agent.recipe_point_visualization \
  --config configs/paper_repro.yaml \
  --run-dir "runs/$RUN_ID"

git -C third_party/OpenILT diff HEAD --exit-code --
```

输出位于 `outputs/recipe_point_visualization/after-training/<run_id>/seed-<n>/`。每张图保留默认点，
并以金色切向箭头连接到 PPO 点；摘要同时绑定源 Recipe 路径和 SHA256。smoke 只产生首个 clip、首个
seed 的诊断图；完整三种子运行应产生 `3×10=30` 张图。图像用于检查动作塌缩、FRAG 异常聚集和
位移方向，不参与 `accepted/rejected` 计算，也不能替代 `ppo-quality.json`。

## v1：训练产物

`runs/<run_id>/` 至少包含：

```text
stage-result.json
stage-progress.json                      # 长任务关键节点增量写入
ppo-quality.json                       # 运行质量审核后生成
models/shared-recipe-seed-<n>.zip
models/shared-recipe-seed-<n>.metadata.json
clips/<clip_id>/default-recipe.json
recipes/<clip_id>-seed-<n>.recipe.json
```

Recipe JSON 同时包含 EPE 与 FRAG 标签、逐点坐标、九分类位移、完整轨迹、内部 solver 轨迹、模型哈希、版图哈希和 OpenILT 提交。`stage-progress.json` 会在模型和每个 Recipe 保存后原子更新；训练中断时先只读检查它和现有 ZIP/JSON，不要因为终端日志不完整就直接重跑。当前代码提供中断取证，但尚未实现从半个 seed 自动续训。

可视化 PNG 属于可从 GLP 和 Recipe 重建的派生产物，因此统一保存在 `outputs/`，不进入上述
`runs/<run_id>/` 训练证据目录。

## MLLM、Qwen 与决策树暂缓

当前不执行 `ppo_recipe_labels` 或 `build-recipe`。这两个历史入口仍面向旧 `ppo-simpleopc-multistep-v3` 产物，尚未迁移到新的 `ppo-recipe-point-v1` 双类标签；强行使用会混淆旧 mask 段标签和新 Recipe 点标签。

后续阶段可以选择硅基流动的 Qwen，而非强制 GPT-4o。届时需要另行实现并验证语义特征发现、图像/文本 prompt、原始响应与哈希追溯、自改进特征池和最终 Recipe 语法适配。在 PPO `accepted` 前，这些工作不进入当前训练路径。

## 已知未验证项与风险

- 本地工作区没有配置中的干净 `third_party/OpenILT` clone，也没有 CUDA；新增十图可视化的真实 GLP 集成仍需上传后在云端执行并人工检查。
- v2 的首轮 `16nm probe + ±40nm 九类` 已被真实 M1_test1 几何证据否决；当前 `24nm probe + 五动作` 已通过六张 train 版图全点几何复检，冲突冻结为 `both-sides-conflict-stay-v1`，128 observation、dense/terminal 完整 episode smoke、PPO 输入汇报样例和六图 Golden contract 均已通过或冻结。dense/terminal 单 rollout PPO v2 数值 smoke与 terminal 三更新 pilot 已通过；dense 三更新 pilot 因 Critic RMSE 连续三次超限而失败，原因诊断完成前不得启动长训练。
- 云端 smoke `20260825T132516Z-train-oracle-293d9938` 已验证 CUDA、M1_test1 的 58 个 EPE/21 个 FRAG 点、CNN 保存加载、Recipe 导出和质量审核；正式 full 运行 `20260826T015433Z-train-oracle-2b7aa7f3` 已进一步通过三 seed、十图 validation-only 门槛。
- 当前 solver 每个点都重新执行完整内部 OPC，计算量较高；本次 full 已完成，但单次运行不能支持对其他硬件、参数或版图集合的耗时承诺。
- 点级/全局级动作、分段长度、内部 mask 步长和原始奖励均应作为后续消融变量；当前选择是可追溯实现，不是论文未公开细节的事实声明。
- 正式 validation 动作虽未超过 `0.95` 塌缩阈值，但只覆盖 `-40/0/+30nm` 三类；类别不平衡、训练后十图可视化人工检查和新 Recipe 点标签到决策树的适配仍未完成。

## 历史兼容代码

以下模块和数据仅供回归，不得混入当前主线结论：

- `simpleopc-multistep-v3`、`simpleopc_quality.py`：旧四步直接 mask 边段 PPO。
- `oracle-weighted-loss-*`、`oracle_batch_labels.py`、`ppo_evaluation.py`：旧候选点九动作 Oracle/PPO。
- `point_sampling.py`、`point_visualization.py`：历史 v2/v3 mask 动作点审计。
- `workflow.oracle_candidate_index` 和旧候选 NPZ：历史读取兼容。

所有新实验报告必须明确区分：CPU 协议测试、云端 smoke、同图诊断、validation accepted 和 test 报告，不能把其中任一层级替代另一层级。

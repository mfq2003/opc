# OPC Agent：论文功能复现与双颗粒度闭环

本项目复现 OPC Agent 的可验证功能链路，并实现“快速预测—难例精算—数据回流”的双颗粒度闭环。v1 只处理公开的 ICCAD13 数据；不接入商业 OPC、私有 PDK 或 NVDLA 的未公开划分。

## 当前状态与边界

- OpenILT 固定为提交 `dabb97c6ca3dfd159362e48273c436444c77353b`，适配层不修改上游源码。
- OpenILT 的公开文档说明其在 Python 3.8 / PyTorch 1.10 开发，也测试过 PyTorch 2.0；本项目云端环境锁定 Python 3.8、PyTorch 2.0.1 + CUDA 11.8。OpenILT 基线指标按其 `Basic`、`EPEChecker` 的定义计算。
- 论文的完整 PPO 超参数、提示词、特征池和 NVDLA 切分未公开。因此验收报告只能声明“功能与趋势复现”，不得声称数值完全复现。
- `SILICONFLOW_API_KEY` 只能从进程环境变量读取。项目不会读取、生成或提交 `.env`。
- 已在云端 RTX 4080 SUPER、固定 OpenILT 提交上验证十父版图多 clip Oracle 与双树 Recipe 覆盖流程：十个缓存均达到 36/36，40 行标签覆盖固定 6/2/2；双树与 9 条 Recipe 成功导出，但小样本 EPE/FRAG 九分类 macro-F1 仅为 0.0952/0.1296，属于管线通过、精度未达标。
- 审计发现旧 `raster-fragment-v1` 方块平移在 13/40 个点产生损失平局，并由 `argmin` 对类别 0 形成顺序偏置。新数据改用 `raster-boundary-strip-v2`，九动作必须生成九张不同掩模；平局标签采用最小绝对位移并保留歧义、损失间隔和候选碰撞证据。v2 已完成云端 `M1_test1`/`M1_test7` 两父版图 pilot：72/72 个动作指标完整，合并 8 行标签，歧义与候选碰撞均为 0；当前已增加点位、法线和九动作的 CPU 可视化审计。
- v3 EPE 第一阶段已完成正式十图云端回放：`adaptive-edge-segment-sampler-v3` 从 base mask 生成 1437 个可移动边段，十图可视化审计 `1437/1437` 通过、核算覆盖率为 1；运行 `20260823T030224Z-train-oracle-8b0f0ba5` 完成三种子、每种子每 clip 10000 timestep 和 12933 个九动作指标，合并标签为 train/validation/test `937/205/295`、歧义 4、碰撞 0。正式 `ppo_evaluation` 显示三个 seed 的准确率为 `0.226862/0.233125/0.282533`、九分类 macro-F1 为 `0.145187/0.180078/0.163935`、平均加权损失遗憾为 `1057.812109/1121.748086/983.407098`；均优于随机、固定 0 nm 和多数类基线的命中率与损失遗憾，但分类质量和种子稳定性仍弱。该评估是逐 clip 同分布拟合检查，不是跨版图泛化。v3 仍未实现 FRAG 的嵌套分段优化，不能声称双树或完整论文复现完成。

## 架构

```text
ICCAD13 GLP / 图像 -> LayoutClip + SQLite 元数据 -> OpenILTEngine (精算真值)
                                                -> 几何特征 + Qwen 离线标签
                                                -> EPE / FRAG 决策树 + JSON Recipe
在线样本 -> 决策树概率 -> 高置信度：快速 Recipe
                       -> 低置信度 / OOD / Recipe 无效：PPO + OpenILT 精算 -> SQLite 回流
```

`src/opc_agent/models.py` 定义版本化数据模型；`storage.py` 只保存路径、哈希和结构化结果，大图与模型留在 `runs/<run_id>/`；`engine.py` 是 OpenILT 的只读适配层；`routing.py` 负责阈值选择和精算路由。

## 云端部署

目标环境为 Ubuntu 22.04、24 GB GPU、32 GB RAM、80 GB 以上磁盘。将本目录复制到云端后执行：

```bash
cd opc_agent
bash scripts/cloud/bootstrap.sh
source .venv/bin/activate
pytest -q
```

启动脚本会克隆并校验固定的 OpenILT 提交、安装 CUDA 11.8 的 PyTorch、其余锁定依赖和 adaptive-boxes。它不会安装系统包，也不会创建或读取 `.env`。若服务器没有 `python3.8`，请由服务器管理员安装后再运行；脚本不会自行改动系统 Python。

## 配置与命令

配置均在 `configs/`：

- `paper_repro.yaml`：ICCAD13 数据、6/2/2 父版图划分及论文复现参数。
- `closed_loop.yaml`：置信度扫描范围与闭环质量约束。
- `cloud_24gb.yaml`：24 GB 云端资源和固定 OpenILT 版本。

```bash
python -m opc_agent.cli prepare-data --config configs/paper_repro.yaml
python -m opc_agent.cli baseline --config configs/paper_repro.yaml
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml
python -m opc_agent.cli build-recipe --config configs/paper_repro.yaml
python -m opc_agent.cli evaluate --config configs/paper_repro.yaml
python -m opc_agent.cli run-loop --config configs/closed_loop.yaml
python -m opc_agent.cli report --run-id <run_id>
pytest -q
```

`prepare-data` 会验证十个 `M1_test*.glp` 是否都存在，并按父版图固定切分，保证切片不会跨集合。`baseline` 调用上游 `pyilt/simpleilt.py`，将原始日志、提交号、配置快照和依赖信息放入唯一的 `runs/<run_id>/`。`train-oracle`、`build-recipe` 和 `run-loop` 已接入真实阶段实现，前置文件缺失会显式失败；`evaluate` 与 Qwen 在线探针仍保持未实现边界，不会产生伪结果。

上游基线完成后，使用以下命令离线归档既有日志，不会重跑 GPU：python -m opc_agent.baseline_results --run-id <run_id>。它会生成 openilt-baseline.metrics.json，并把十个父版图的 L2、PVB、总 EPE、Shot 与耗时写入 SQLite。上游日志没有 EPE N/EPE D，归档器会明确保留该缺口而不会推断数值。

## Qwen 与费用记录

默认模型为 `Qwen/Qwen3.6-35B-A3B`，OpenAI 兼容端点是 `https://api.siliconflow.cn/v1`。先执行图片能力探针，再用于离线特征标签和 Recipe 归纳；在线快速路径绝不调用 API。所有响应须经 Pydantic 校验；无效 JSON 会记录原始响应的 SHA-256 和失败原因，不能进入训练。探针的 macro-F1 低于 0.80 时，确定性几何特征成为权威标签。

## 指标、可复现性与故障处理

指标包括 L2、PVB、EPE N、EPE D 和运行时间。闭环阈值只在 0.50–0.99 中扫描，选择满足相对全精算 EPE D 退化不超过 5% 的最低精算调用率；没有可行阈值会明确报告失败。所有运行保存随机种子、配置、依赖版本、OpenILT 提交、模型、日志和费用。SQLite 使用幂等写入，运行中断后可从现有记录续跑，不会自动重跑整个实验。

完成 GPU 实验后，使用 `report` 生成 `reports/reproduction.md` 和 `reports/closed_loop.md`，其中必须写明与论文表 1 的差异、未验证项和失败实验。

## 候选点可视化审计

候选索引生成后、启动 GPU Oracle 前，应先在 CPU 上检查点位、外法线和九动作。该工具只读取已经通过哈希校验的
`candidate-index.json`、manifest、metadata 和紧凑 NPZ，不重新采样，不调用 OpenILT、PPO、GPU 或 API：

```bash
python -m opc_agent.point_visualization \
  --index data/processed/candidates_pilot_v2/candidate-index.json \
  --output-dir outputs/point_visualization/pilot_v2 \
  --overview --local-patches --action-grids
```

未指定三个图片开关时默认全部生成。每个 clip 输出 `overview.png`、逐点 `*-local.png` 和 `*-actions.png`；根目录的
`visualization-summary.json` 自动检查动作点是否位于其参考边界、法线是否由前景指向背景、base mask 支持窗口是否包含
边界、九动作是否唯一、负/正位移是否分别收缩/扩张以及修改是否局限在动作附近。v2 的点参考 target；v3 的可移动
边段参考真正被修改的 base mask。总览图中红色圆点为 EPE、蓝色
方块为 FRAG、绿色箭头为外法线、黄色轮廓为 base mask；动作图中绿色为新增像素、红色为删除像素、青色为 target
轮廓。派生文件允许相同内容幂等执行，但拒绝覆盖已有不同图片；改变数据或渲染方式时应使用新的输出目录。

v3 总览额外用橙线标出完整可移动边段，红点表示该边段中心的 EPE 任务动作锚点；真正的 EPE 数值由 OpenILT
仿真评价，不等同于红点坐标。摘要额外给出 `segments`、`passed_segments`、`segment_on_action_boundary`、
`segment_normal_points_outward`、`boundary_reference`、训练覆盖率 `minimum_axis_boundary_coverage` 和核算覆盖率
`minimum_accounted_axis_boundary_coverage`；整段边界、法线、九动作唯一性、
面积单调性或修改范围任一失败时，该段都不会通过审计。

pilot 必须达到 `passed_points == points`，并由人工确认点位分布、法线和动作变化后，才能对正式十图索引运行相同审计：

```bash
python -m opc_agent.point_visualization \
  --index data/processed/candidates_v2/candidate-index.json \
  --output-dir outputs/point_visualization/formal_v2
```

## v3 自适应边段第一阶段

v3 借鉴“每个可移动边段中心一个点”和“整段沿法线移动”，但不照搬只适用于方形 via 的固定四边/四点。
处理流程是：从 base mask 提取最大水平/垂直动作边界段；长段优先保留两端的转角短段，再均衡切分中部；每个
最终子段只放一个中心动作锚点；一个动作会修改该子段的完整跨度。target 只参与 OpenILT 损失评价，不用于定义
被移动的 mask 几何。默认参数为目标最长 128 px、最短 32 px、转角段 64 px。
长度关系必须满足 `min <= corner <= target`。原始边段短于 `min` 时不作为独立九动作训练样本；达到长度要求但因
法线内侧过薄、外侧相邻图形等原因无法产生九张不同掩模时也不进入训练。两类排除都会写入 manifest/metadata 的
`sampling_audit` 和 `sampling_exclusions`，记录原因、坐标和边界长度。训练覆盖率可以低于 1，但“可训练边界 + 已解释
排除边界”的核算覆盖率必须为 1；因此不会把不可行动作送入 PPO，也不会静默删边。最大位移可能越界或内部边界核算
不守恒仍会显式失败。

先在两个父版图上生成 v3 pilot（必须使用全新的输出目录，不能覆盖 v2）：

```bash
python -m opc_agent.point_sampling \
  --config configs/paper_repro.yaml \
  --image-dir third_party/OpenILT/tmp \
  --output-dir data/processed/candidates_pilot_v3 \
  --parents M1_test1 M1_test7 \
  --sampler-version adaptive-edge-segment-sampler-v3 \
  --support-radius 8 --scale-nm-per-pixel 1.0 \
  --v3-target-segment-length 128 \
  --v3-min-segment-length 32 \
  --v3-corner-segment-length 64

python -m opc_agent.point_visualization \
  --index data/processed/candidates_pilot_v3/candidate-index.json \
  --output-dir outputs/point_visualization/pilot_v3 \
  --overview

python -c 'import json; p="outputs/point_visualization/pilot_v3/visualization-summary.json"; d=json.load(open(p)); print("clips =", d["clips"]); print("segments =", d["segments"]); print("excluded =", d["excluded_segments"]); print("passed =", d["passed_segments"]); print("reference =", d["boundary_references"]); print("trainable_coverage =", d["minimum_axis_boundary_coverage"]); print("accounted_coverage =", d["minimum_accounted_axis_boundary_coverage"]); print("failed =", d["failed_points"])'
```

只有 `failed == []`、`passed == segments`、`boundary_references == ["base_mask"]` 且
`minimum_accounted_axis_boundary_coverage == 1.0`，并人工审查训练覆盖率、排除原因及橙色边段，才能把同一命令扩展到
十个父版图并启动云端 Oracle。训练覆盖率不要求机械达到 1，因为短于 32 px 或固定九动作发生物理饱和的边段不是
当前动作空间内的有效训练样本；其排除必须完整记录。
v3 第一阶段只生成 EPE：索引中的 `frag_points` 合法值为 0，
`--max-epe`/`--max-frag` 是 v2 参数，不会截断 v3 的正式边段覆盖。当前代码已经定义了 PPO 可用的状态和九分类动作，
但本地生成与可视化本身不是强化学习；只有云端 `train-oracle` 真正运行 PPO 时才进入强化学习阶段。

暂时不要为 v3 人为添加 FRAG 点。仅移动“分割点”而不改变掩模不会产生不同的 OpenILT 结果；后续 FRAG 必须定义为
候选分段方案，并在每个方案内部完成边段动作优化后再比较总损失。完成这一层嵌套 Oracle 前，v3 可以生成 EPE 标签和
训练 EPE 树，但不能用于宣称 EPE/FRAG 双树闭环完成。

## PPO 策略质量评估

完成九动作缓存与标签合并后，必须评估 PPO 确定性动作，不能把模型文件存在或 timestep 达标等同于策略收敛。
评估器会复核候选索引、运行摘要、模型元数据、缓存和标签身份，在 CPU 上逐 clip、逐 seed 推理；它不调用
OpenILT，也不重新训练：

```bash
python -m opc_agent.ppo_evaluation \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --labels data/processed/point_training_formal_v3_audited.json \
  --config configs/paper_repro.yaml \
  --output outputs/ppo_evaluation/formal_v3_audited.json
```

输出同时报告规范标签准确率、允许歧义最优集合的 `optimal_set_accuracy`、全部九类
`macro_f1_all_nine_classes`、位移绝对误差、加权损失遗憾、动作分布、固定 0 nm、规范标签多数类和均匀随机基线。
有意义的 PPO 至少应在最优集合命中率和损失遗憾上同时优于随机、零位移与多数类基线；否则应先修复奖励尺度、状态特征或
训练方式。当前每个模型都在自身 clip 上训练和评估，因此这是同 clip 拟合检查，不是跨版图泛化证明；train、validation、
test 划分用于后续决策树，不能把 validation/test clip 的 PPO 指标误称为未见版图测试精度。

正式输出为 `outputs/ppo_evaluation/formal_v3_audited.json`，实测汇总如下：

| 策略 | accuracy | 九分类 macro-F1 | mean weighted-loss regret |
| --- | ---: | ---: | ---: |
| PPO seed 0 | 0.226862 | 0.145187 | 1057.812109 |
| PPO seed 1 | 0.233125 | 0.180078 | 1121.748086 |
| PPO seed 2 | 0.282533 | 0.163935 | 983.407098 |
| 固定 0 nm | 0.173278 | 不适用 | 1482.963814 |
| 多数类动作 5（+10 nm） | 0.210856 | 不适用 | 1834.085595 |
| 均匀随机期望 | 0.111111 | 不适用 | 3100.934431 |

三个 PPO seed 均通过“命中率和损失遗憾同时优于三类基线”的最低有效性门槛；其中 seed 2 的准确率最高且平均遗憾最低，
seed 1 的 macro-F1 最高。由于最佳准确率仍只有 0.282533、最佳 macro-F1 只有 0.180078，且不同 seed 差异明显，当前结论
只能是“PPO 学到了优于基线的同 clip 策略信号”，不能写成可靠九分类器或跨版图泛化成功。下一阶段应复用现有 Oracle 缓存，
让共享策略只在 6 个 train 父版图训练、在 validation 选择模型并最终只在 test 评估；无需为此重新运行 12933 个 OpenILT 候选。

## GPU Oracle 与闭环工作流

完整单点流程和故障处理见 `docs/workflow_commands.md`；多点、多父版图命令见 `docs/multi_clip_workflow.md`。v2 pilot 命令保留用于
回归双任务旧链路；当前 v3 正式 EPE 数据必须显式传入 audited 索引，不能依赖 `paper_repro.yaml` 中仍为 v2 双任务链路保留的默认路径：

```bash
# 已完成的 v3 正式 Oracle；重跑会创建新的长任务 run，不用于日常检查
export OMP_NUM_THREADS=1
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --candidate-index data/processed/candidates_formal_v3_audited/candidate-index.json

# 已完成的正式标签合并
python -m opc_agent.oracle_batch_labels \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_formal_v3_audited.json

# 已完成的逐 clip PPO 质量评估；该命令只读缓存并在 CPU 推理
python -m opc_agent.ppo_evaluation \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --labels data/processed/point_training_formal_v3_audited.json \
  --config configs/paper_repro.yaml \
  --output outputs/ppo_evaluation/formal_v3_audited.json
```

候选数据采用 `compact-point-geometry-v1`：磁盘只保存目标、基准掩模和每个点的几何参数，OpenILT 评价动作时才即时生成一个候选。`axis-boundary-sampler-v2` 从正交栅格边界确定性生成 EPE/FRAG 点和外法线；`raster-boundary-strip-v2` 沿法线扩张或收缩局部边界条带，并在写入候选数据前强制验证九动作唯一。旧 `raster-fragment-v1` 仅用于读取历史数据，不再生成正式标签。上述规则属于论文未公开细节的版本化兼容适配，不声称逐数值等同于论文。

`paper_repro.yaml` 的 `workflow.oracle_candidate_index` 与 `workflow.point_training_dataset` 暂时保留 v2 双任务默认路径，避免把
EPE-only v3 数据误送入要求 EPE/FRAG 同时存在的 `build-recipe`。因此 v3 Oracle、标签合并和 PPO 评估必须像上面一样显式传入
`candidates_formal_v3_audited`、正式 run 和 `point_training_formal_v3_audited.json`。`--candidate-index` 与 `--smoke` 合用时会遍历
索引全部 clip，但只使用种子 0 和 256 timestep，适合先估算时间和显存。

多 clip 标签通过 `opc_agent.oracle_batch_labels` 合并，期间会校验索引身份、候选 NPZ 哈希、运行索引版本和九动作缓存完整性。pilot 只有 train/validation，不能用于训练决策树；正式训练集必须同时包含 6/2/2 的 train、validation、test 以及 EPE/FRAG 两类任务。

决策树固定训练 EPE/FRAG 两个模型，测试 macro-F1 始终按全部九类计算，并导出无需 pickle 的 JSON 树及确定性 Recipe。阈值扫描只接受 validation 数据；无可行阈值时必须报告 `no_feasible_threshold`。Qwen 图片能力探针、Recipe 文本解释和统一 `evaluate` 仍未验证或未实现，不能宣称闭环已经完成。

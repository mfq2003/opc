# OPC Agent：论文功能复现与双颗粒度闭环

本项目复现 OPC Agent 的可验证功能链路。当前主线使用 OpenILT 的 SimpleOPC 作为多步物理环境：先训练并验收 PPO，随后只能用 accepted PPO 的最优 Recipe 训练决策树。项目只处理公开的 ICCAD13 数据；不接入商业 OPC、私有 PDK 或 NVDLA 的未公开划分。

## 当前状态与边界

- OpenILT 固定为提交 `dabb97c6ca3dfd159362e48273c436444c77353b`。OpenILT 可以在云端重新 `git clone`；本项目只导入其 GLP、polygon、光刻和 EPE 函数，不修改或要求上传任何上游源码。运行时会拒绝已跟踪文件存在本地改动的 OpenILT 工作树（不受普通未跟踪文件影响）；多步 PPO 的模型、轨迹和 Recipe 全部写入本项目 `runs/`。
- OpenILT 的公开文档说明其在 Python 3.8 / PyTorch 1.10 开发，也测试过 PyTorch 2.0；本项目云端环境锁定 Python 3.8、PyTorch 2.0.1 + CUDA 11.8。OpenILT 基线指标按其 `Basic`、`EPEChecker` 的定义计算。
- 论文的完整 PPO 超参数、提示词、特征池和 NVDLA 切分未公开。因此验收报告只能声明“功能与趋势复现”，不得声称数值完全复现。
- `SILICONFLOW_API_KEY` 只能从进程环境变量读取。项目不会读取、生成或提交 `.env`。
- `simpleopc-multistep-v3` 已实现固定分段下的 EPE 多步 PPO：每轮 PPO 同时给全部边段选择 inward/stay/outward，批量更新掩模后只调用一次光刻仿真。论文只明确动作范围为 `±40nm`，并说明最终 Recipe 使用 9 类，没有公开 PPO 单步位移、迭代数和分类边界；本项目把该范围解释为相对初始边段的累计偏移边界，并采用 4 个 `10nm` 增量，使 `[-40,-30,-20,-10,0,10,20,30,40]nm` 九个等距代表值都能精确到达。累计边界解释和 `10nm × 4` 均是公开细节缺失后的实现选择，不得表述为论文原始超参数。
- v3 统一训练与质量验收损失：先计算论文形式的原始加权和 `Lraw=L2+100×EPE+PVB`，再用 `Lraw/Lraw_initial` 作为 PPO 数值尺度。整体除以初始总损失不会改变三项相对权重；Recipe、轨迹、stage 和质量报告均绑定 `paper-weighted-sum-initial-normalized-v1`，旧损失版本会被拒绝。
- 云端 v2 诊断暴露的“训练逐指标归一化、验收原始加权和”目标冲突已由 v3 修复；v2 产物只作为诊断证据，不能进入 v3 质量验收或决策树标签。v3 已在固定 OpenILT 提交和真实 CUDA 环境完成 256 timestep 冒烟及 `M1_test1/seed0` 的 10000 timestep 诊断：PPO 相对初始掩模改善 `14.1968%`，最佳 Recipe 为 `-10/0/+10nm` 三类非坍缩动作，但原始加权损失仍比启发式高 `0.3550%`。该单图单种子结果不能替代 validation、多种子或正式 accepted 报告；完整证据见 [`docs/simpleopc_v3_m1_test1_diagnostic_20260825.md`](docs/simpleopc_v3_m1_test1_diagnostic_20260825.md)。
- 决策树入口现在只接受 `ppo-simpleopc-*` 标签、`accepted` 质量状态和质量报告哈希。旧九动作 Oracle 数据仍可读取和审计，但不能再作为当前主线的决策树正式标签。
- FRAG 外层重新分段 PPO 尚未实现。当前 EPE-only PPO 标签可以导出供审计，但 `build-recipe` 会拒绝缺少 FRAG 的数据，因此不能宣称双树或完整论文复现完成。
- 历史上已在云端 RTX 4080 SUPER 验证旧单步候选管线和双树导出，但其 EPE/FRAG macro-F1 仅为 0.0952/0.1296；这些结果只保留为回归证据，不再代表当前方法。
- 审计发现旧 `raster-fragment-v1` 方块平移在 13/40 个点产生损失平局，并由 `argmin` 对类别 0 形成顺序偏置。新数据改用 `raster-boundary-strip-v2`，九动作必须生成九张不同掩模；平局标签采用最小绝对位移并保留歧义、损失间隔和候选碰撞证据。v2 已完成云端 `M1_test1`/`M1_test7` 两父版图 pilot：72/72 个动作指标完整，合并 8 行标签，歧义与候选碰撞均为 0；当前已增加点位、法线和九动作的 CPU 可视化审计。
- 历史 v3 EPE 单步候选实验完成过正式十图云端回放：1437 个边段审计通过，三个 seed 的最佳准确率仅 `0.282533`、最佳九分类 macro-F1 仅 `0.180078`。该结果解释了为什么当前主线已改为 SimpleOPC 多步 PPO；它不是新方法的性能证据。

## 架构

```text
ICCAD13 GLP -> 只读 OpenILT SimpleOPC -> 固定分段 + 初始 EPE 状态
                                      -> 多步 EPE PPO（持续更新同一掩模）
                                      -> 冻结 PPO 确定性回放 + 历史最优 Recipe
                                      -> validation 质量门槛
                                         ├─ rejected：停止，禁止训练树
                                         └─ accepted：导出 PPO EPE 标签
未来：FRAG 外层重新分段 PPO -> 内层 EPE PPO -> accepted EPE/FRAG 标签
                                              -> EPE / FRAG 决策树
```

`src/opc_agent/simpleopc.py` 是只读 OpenILT 后端与多步环境；`simpleopc_runner.py` 训练 PPO 并导出模型绑定 Recipe；`simpleopc_quality.py` 只用 validation 决定是否 accepted；`ppo_recipe_labels.py` 把选中的 PPO Recipe 转为 EPE 树标签。`engine.py` 仅保留上游 SimpleOPC 基线入口。

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
# 可选：仅核对上游 SimpleOPC 脚本，不是多步 PPO 前置条件，且会在 OpenILT/tmp 写图片
python -m opc_agent.cli baseline --config configs/paper_repro.yaml
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml --smoke
# 冒烟通过后才启动完整十图、三种子任务；这是长时间 GPU 作业
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml
# 完整任务结束后，将下面的 <run_id> 换成 train-oracle 输出
python -m opc_agent.simpleopc_quality --run-dir runs/<run_id> --config configs/paper_repro.yaml
python -m opc_agent.ppo_recipe_labels \
  --stage runs/<run_id>/stage-result.json \
  --quality runs/<run_id>/ppo-quality.json \
  --output data/processed/ppo_simpleopc_epe_training.json
# FRAG 嵌套 PPO 未完成前不要执行 build-recipe；代码也会显式拒绝 EPE-only 数据
python -m opc_agent.cli evaluate --config configs/paper_repro.yaml
python -m opc_agent.cli run-loop --config configs/closed_loop.yaml
python -m opc_agent.cli report --run-id <run_id>
pytest -q
```

`prepare-data` 会验证十个 `M1_test*.glp` 是否都存在，并按父版图固定切分。`baseline` 调用上游 `pyilt/simpleopc.py`；该上游脚本可能在 OpenILT 的 `tmp/` 生成未跟踪图片，但不会修改跟踪源码。当前主线 `train-oracle` 不执行该脚本，而是直接导入只读函数，所有输出写入本项目 `runs/<run_id>/`。`--smoke` 只跑首个 train 父版图、首个种子和 256 timestep；完整命令按 6/2/2 十个父版图和三个种子运行。

`simpleopc_quality` 只读既有模型和 Recipe，不调用 GPU。它会逐步重算轨迹的原始加权和与整体归一化值，确认训练和验收共用同一损失版本后，才在 validation 上检查：每个 clip 的最佳 PPO 不差于 SimpleOPC 启发式、相对初始掩模至少改善 1%、三种子损失变异系数不高于 0.20，且单一动作类别占比不超过 0.95。任何条件失败都输出 `rejected` 并以退出码 2 结束。test 指标只报告，不参与 accepted 决定。

`opc_agent.baseline_results` 只兼容历史 SimpleILT 日志，不能解析当前上游 SimpleOPC 的逐轮输出；它仅为旧工件保留。当前主线的可比较基线由 `train-oracle` 为每个 GLP 生成 `simpleopc-heuristic.json`，与 PPO 使用完全相同的分段、步长和指标。

## Qwen 与费用记录

默认模型为 `Qwen/Qwen3.6-35B-A3B`，OpenAI 兼容端点是 `https://api.siliconflow.cn/v1`。先执行图片能力探针，再用于离线特征标签和 Recipe 归纳；在线快速路径绝不调用 API。所有响应须经 Pydantic 校验；无效 JSON 会记录原始响应的 SHA-256 和失败原因，不能进入训练。探针的 macro-F1 低于 0.80 时，确定性几何特征成为权威标签。

## 指标、可复现性与故障处理

指标包括 L2、PVB、EPE N、EPE D 和运行时间。闭环阈值只在 0.50–0.99 中扫描，选择满足相对全精算 EPE D 退化不超过 5% 的最低精算调用率；没有可行阈值会明确报告失败。所有运行保存随机种子、配置、依赖版本、OpenILT 提交、模型、日志和费用。SQLite 使用幂等写入，运行中断后可从现有记录续跑，不会自动重跑整个实验。

完成 GPU 实验后，使用 `report` 生成 `reports/reproduction.md` 和 `reports/closed_loop.md`，其中必须写明与论文表 1 的差异、未验证项和失败实验。

## 历史候选点可视化审计

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

## 历史 v3 自适应边段单步实验

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

## 历史单步 PPO 策略质量评估

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

## 历史单步 Oracle 工件读取

下面两个命令只读取已经存在的历史工件，用于复核旧实验；默认 `paper_repro.yaml` 已切换到
`simpleopc-multistep-v3`，不允许再带 `--candidate-index` 启动旧单步训练。不要把下面输出转换成当前决策树标签：

```bash
# 只读合并历史九动作标签
python -m opc_agent.oracle_batch_labels \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_formal_v3_audited.json

# 只读复核历史单步 PPO
python -m opc_agent.ppo_evaluation \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --labels data/processed/point_training_formal_v3_audited.json \
  --config configs/paper_repro.yaml \
  --output outputs/ppo_evaluation/formal_v3_audited.json
```

候选数据采用 `compact-point-geometry-v1`：磁盘只保存目标、基准掩模和每个点的几何参数，OpenILT 评价动作时才即时生成一个候选。`axis-boundary-sampler-v2` 从正交栅格边界确定性生成 EPE/FRAG 点和外法线；`raster-boundary-strip-v2` 沿法线扩张或收缩局部边界条带，并在写入候选数据前强制验证九动作唯一。旧 `raster-fragment-v1` 仅用于读取历史数据，不再生成正式标签。上述规则属于论文未公开细节的版本化兼容适配，不声称逐数值等同于论文。

`workflow.oracle_candidate_index` 和旧候选 NPZ 仅为历史读取兼容保留。当前
`workflow.point_training_dataset` 指向尚待 EPE+FRAG PPO 完成后生成的新数据；旧 `oracle-weighted-loss-*`
标签即使手动传给 `build-recipe` 也会被拒绝。

旧多 clip 标签仍可通过 `opc_agent.oracle_batch_labels` 合并以复核哈希和缓存完整性，但不能用于当前主线训练决策树。

最终决策树仍固定训练 EPE/FRAG 两个模型，测试 macro-F1 按全部九类计算；但输入必须同时具有 accepted PPO 标签、质量报告哈希和 6/2/2 父版图划分。Qwen 图片能力探针、FRAG 嵌套 PPO、Recipe 端到端仿真和统一 `evaluate` 尚未验证或未实现，不能宣称闭环完成。

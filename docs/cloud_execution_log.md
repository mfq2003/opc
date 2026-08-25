# 云端执行记录与命令说明

本文档记录 AutoDL 云端执行的命令、结果与下一步命令。每次新增云端操作时，应补充命令用途、前置条件、影响范围、成功标志和异常处理；不得记录密码、私钥、API 密钥或其他凭据。

## 已完成：固定 OpenILT

```bash
mkdir -p third_party
git clone https://github.com/OpenOPC/OpenILT.git third_party/OpenILT
git -C third_party/OpenILT fetch --depth 1 origin dabb97c6ca3dfd159362e48273c436444c77353b
git -C third_party/OpenILT checkout --detach dabb97c6ca3dfd159362e48273c436444c77353b
git -C third_party/OpenILT rev-parse HEAD
```

- 用途：下载公开 OpenILT 源码，并固定到可复现的提交。
- 前置条件：当前目录是项目根目录，云端网络可访问 GitHub。
- 影响：新增 `third_party/OpenILT/`；不修改上游源码。
- 成功标志：最后一条输出完整提交 `dabb97c6ca3dfd159362e48273c436444c77353b`。
- 已验证结果：成功，`HEAD is now at dabb97c Subtle changes`。

## 已完成：虚拟环境与 GPU 依赖

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install --no-cache-dir --find-links https://mirrors.aliyun.com/pytorch-wheels/cu118/ "torch==2.0.1+cu118" "torchvision==0.15.2+cu118"
python -m pip install -r requirements-lock.txt
python -m pip install -e third_party/OpenILT/thirdparty/adaptive-boxes
python -m pip install --no-build-isolation -e .
```

- 用途：创建隔离 Python 3.8 环境，安装 CUDA 11.8 版 PyTorch、项目锁定依赖、adaptive-boxes 和本项目。
- 前置条件：已进入项目根目录；激活后提示符应包含 `(.venv)`。
- 影响：仅向 `.venv/` 写入 Python 包；不会下载训练数据或调用 API。
- 成功标志：`python -m opc_agent.cli --help` 能显示子命令。
- 异常处理：若阿里云镜像的隔离构建无法解析 `setuptools>=68`，使用 `python -m pip install --no-build-isolation -e .`。
- 已验证结果：Torch 为 `2.0.1+cu118`，Torch CUDA 为 `11.8`，CUDA 可用，OPC Agent 版本为 `0.1.0`。

## 已完成：测试与数据准备

```bash
pytest -q
python -m opc_agent.cli prepare-data --config configs/paper_repro.yaml
find runs -maxdepth 2 -type f | sort
```

- 用途：运行单元测试；验证十个 ICCAD13 GLP 文件、计算哈希并写入 SQLite 元数据。
- 前置条件：虚拟环境已激活，OpenILT 已固定到目标提交。
- 影响：生成可忽略的 `.pytest_cache/`；写入 `runs/<run_id>/` 与 `runs/opc_agent.sqlite3`。
- 成功标志：pytest 输出 `14 passed`；`runs/` 下存在 `prepare-data.json`。
- 已验证结果：14 项测试全部通过；数据准备运行编号为 `20260724T083641Z-prepare-data-dc440699`。

## 已完成：OpenILT 十图基线与离线归档

```bash
screen -S opc-baseline
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate
python -m pip install -r third_party/OpenILT/requirements_pip.txt
python -m opc_agent.cli baseline --config configs/paper_repro.yaml
```

- 用途：在固定 OpenILT 提交上运行 ICCAD13 十图 SimpleILT 基线，保存原始日志与版本记录。
- 前置条件：GPU 可用，`.venv` 已激活，数据准备成功；建议数据盘保留至少 10 GB 空间。
- 影响：使用 GPU，向 `third_party/OpenILT/tmp/` 与新的 `runs/<run_id>/` 写入掩膜图和日志；不会下载额外数据集。
- 成功标志：命令输出新运行编号，`runs/<run_id>/openilt-baseline.log` 和 `openilt-revision.txt` 存在。
- 运行维护：`Ctrl+A` 后按 `D` 可脱离 screen；使用 `screen -r opc-baseline` 回到会话。异常时使用 `Ctrl+C` 停止，不删除已有运行目录。


### 已验证结果

- 基线运行编号：`20260724T085216Z-baseline-aceb4da8`。
- 十个 Testcase 与 `[Result]` 汇总均存在，OpenILT 提交为 `dabb97c6ca3dfd159362e48273c436444c77353b`。
- 平均指标：L2 `35860`、PVBand `48080`、总 EPE `6.6`、Shot `356.9`、SolveTime `1.12 s`。
- 已执行 `python -m opc_agent.baseline_results --run-id 20260724T085216Z-baseline-aceb4da8`，归档器不重跑 GPU，并已生成 JSON 与 SQLite 的十条逐图记录。
- 总 EPE 不能替代 EPE N/EPE D；该缺口保留到后续逐点评估适配阶段解决。

## 本地已完成：GPU 工作流代码准备

```powershell
$env:Path='F:\anaconda3;F:\anaconda3\DLLs;F:\anaconda3\Library\bin;' + $env:Path
$env:PYTHONPATH='src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m compileall -q src tests
python -m pytest -ra
```

- 用途：在不安装本地依赖的前提下，对新增 Schema、EPE、紧凑候选、Oracle 标签、闭环路由和源码语法做验证。
- 前置条件：使用机器上既有 Python；不执行 `pip install`。
- 影响：只生成 Python/pytest 缓存，不运行 GPU、OpenILT 或网络。
- 已验证结果：新增工作流合入前为 `27 passed, 2 skipped`；两个 skip 分别来自本地未安装 Gymnasium、本地 NumPy/scikit-learn 二进制不兼容。随后紧凑候选与 Oracle 标签定向测试为 `4 passed`，源码 compileall 通过。
- 云端要求：上传后必须重新执行 `pytest -ra`；云端锁定依赖完整，GPU Oracle 和决策树测试不应沿用本地 skip。
- 未验证项：真实 OpenILT GPU Oracle API、PPO 模型输出、完整九动作缓存、树 macro-F1 和闭环结果均需在云端运行后确认。

下一步云端命令及逐条解释统一见 `docs/workflow_commands.md`。

## 已完成：单点 GPU Oracle 冒烟

```bash
pytest -q tests/test_oracle_runner.py
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml --smoke
```

- 运行编号：`20260727T073802Z-train-oracle-f2b25737`。
- 已验证环境：RTX 4080 SUPER，约 31.48 GB 显存；PyTorch 2.0.1+cu118，CUDA 可用。
- 已验证结果：Oracle GPU 入口成功，PPO 环境启动，OpenILT 九动作指标完整性为 `9/9`，退出状态为 0。
- 首次兼容故障：`pyilt.evaluation` 导入阶段从项目根目录读取 `./config/lithosimple.txt` 失败；修复为在导入 evaluation 前切换至 OpenILT 根目录，并在初始化后恢复 cwd。对应回归测试云端 `3 passed`。
- 首次 timestep 偏差：配置请求 256，但 SB3 默认 `n_steps=2048`，实际收集 2048。该冒烟结果仍用于功能验证；后续代码会选择整除请求总步数的 rollout/batch，并分别记录请求值和实际值。


## 已完成：精确 256 timestep 的单点 GPU Oracle

```bash
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml --smoke
python -m opc_agent.oracle_labels \
  --dataset data/processed/oracle_smoke.npz \
  --metadata data/processed/oracle_smoke.metadata.json \
  --cache runs/20260727T074829Z-train-oracle-760ec135/oracle-metrics.cache.json \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_smoke_v2.json
```

- 运行编号：`20260727T074829Z-train-oracle-760ec135`。
- 已验证结果：请求和实际 timestep 均为 256，`n_steps=256`、`batch_size=64`、种子 0、设备 CUDA，PPO 耗时约 2.33 秒。
- 候选数据哈希：`399e928c1adbd7175faa8cd9478075b0a2486e012d4d9a975f9d6c20d24638c8`。
- 模型与标签：模型 ZIP 已生成；`point_training_smoke_v2.json` 已由完整 9/9 缓存生成。
- 科学边界：这仍是单 EPE 点，仅证明 OpenILT/PPO/标签链路可运行，不能训练或验收 EPE/FRAG 双树。

## 本地已完成：多点与多 clip 批量路径

- 新增确定性正交边界采样、EPE/FRAG 点上限、6/2/2 父版图防泄漏、候选索引哈希、多 clip 独立缓存/模型和标签合并。
- 新增链路定向测试为 `7 passed`；本地测试使用合成图和替身后端，未调用 GPU 或网络。
- 下一步真实云端动作不是十图长任务，而是按 `docs/multi_clip_workflow.md` 运行两个父版图、最多 72 个不同 OpenILT 候选评价的小批量试跑。
## 已完成：两父版图多 clip GPU 小批量与标签合并

```bash
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --candidate-index data/processed/candidates_pilot_v1/candidate-index.json \
  --smoke

python -m opc_agent.oracle_batch_labels \
  --index data/processed/candidates_pilot_v1/candidate-index.json \
  --oracle-run runs/20260727T130201Z-train-oracle-dee8899e \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_pilot_v1.json
```

- 云端回归测试：相关 9 项测试全部通过。
- 候选索引：`M1_test1` 与 `M1_test7`，共 2 clips、4 EPE 点和 4 FRAG 点。
- GPU 运行编号：`20260727T130201Z-train-oracle-dee8899e`，退出状态 0。
- PPO：两个 clip 均使用 CUDA、种子 0、精确 256 timestep；指标缓存均达到 36/36。
- 标签结果：共 8 行，EPE/FRAG 各 4 行，train/validation 各 4 行，test 为 0；输出 SHA-256 为 `8606ef5d1f21cfc0a0e622a257490c2d41bb9fa47981f34dd21fff66995071d9`。
- 存储：训练 JSON 与摘要各占 4 KB。
- 环境提示：运行时出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，但未影响退出状态和指标完整性；后续任务在当前 shell 显式使用 `export OMP_NUM_THREADS=1`。
- 科学边界：该 pilot 没有 test 父版图，不能用于决策树 macro-F1 或论文验收。
## 已完成：十父版图小步数 GPU 覆盖与 6/2/2 标签

- 候选索引：`data/processed/candidates_smoke10_v1/candidate-index.json`，10 clips、20 EPE 点、20 FRAG 点，占用 604 KB。
- GPU 运行编号：`20260727T130817Z-train-oracle-551a36cc`，Python 退出状态 0。
- PPO 与缓存：种子仅为 0，每个模型精确 256 timestep；十个 clip 的缓存均为 36/36，十个模型文件均存在。
- 精算规模：40 点 × 9 动作，共 360 个不同候选指标完成。
- 合并训练集：`data/processed/point_training_smoke10_v1.json`，共 40 行；train 24、validation 8、test 8；EPE/FRAG 各 20。
- 标签输出 SHA-256：`b0def71103a5d68de953a70d2aa419a9c28ac989340105f0f052820aa957d895`。
- 科学边界：该数据用于验证十图、双任务、6/2/2 和后续树训练接口；每父版图仅 4 点且只有种子 0，macro-F1 不得作为论文复现或 v1 验收值。
## 已完成：十图小样本双树与 Recipe 导出

- 运行编号：`20260727T131549Z-build-recipe-fd4c35e6`。
- 输入：`data/processed/point_training_smoke10_v1.json`，40 行、固定 6/2/2、EPE/FRAG 各 20。
- 输出：`epe.tree.json`、`frag.tree.json`、`recipe.raw.json` 均生成，各约 4 KB。
- EPE：7 个树节点，九分类 macro-F1 `0.09523809523809523`。
- FRAG：9 个树节点，九分类 macro-F1 `0.12962962962962962`。
- Recipe：共 9 条动作规则；Qwen 解释状态为 `not_run`。
- 结论：端到端树训练与 JSON Recipe 导出管线通过，但精度远低于 0.75 验收线。当前只有每父版图 2 个 EPE 和 2 个 FRAG 点，必须先检查类别分布和逐条预测，再决定扩大采样或修改特征；不得把该分数写成论文复现结果。
## 已定位并修复：v1 候选动作塌缩与标签顺序偏置

- v1 标签审计：40 点中 EPE 7 点、FRAG 6 点存在最优损失平局，共 13/40；改用最小绝对位移平局策略后共有 12 个标签变化。
- 典型证据：多个点的 `-40/-30/-20/-10 nm` 四个动作损失完全相同，而次优损失仍有 10 到 634 的明显差值。
- 根因一：`raster-fragment-v1` 清空并平移整个方形块；向内平移时目标区域已经是前景，`maximum` 合并无效，不同负位移可能生成同一候选。
- 根因二：`np.argmin` 在平局时固定返回最小动作编号，把候选塌缩进一步表现为类别 0（-40 nm）偏置。
- 修复：新增 `raster-boundary-strip-v2`，按外法线扩张或向内收缩局部条带；v2 写入 NPZ 并由 Oracle 按版本分派，旧 v1 数据仍按旧实现读取。
- 质量门：v2 每个点写入紧凑候选前必须生成九张不同二值掩模；否则在 OpenILT 前显式失败。GPU 缓存新增 `mask_sha256`。
- 标签 v2：平局选择绝对位移最小的等优动作，同时记录 `optimal_classes`、`ambiguous`、`loss_margin` 和 `candidate_collision`。
- 本地验证：全量 `39 passed`；Pydantic 2 弃用提示来自本地非锁定环境，云端项目仍锁定 Pydantic 1.10.23。
- 下一步：上传 v2 文件后仅重跑两父版图、每图 2 EPE + 2 FRAG 的小批量；使用新的 `candidates_pilot_v2` 和标签 v2 路径，绝不覆盖 v1 故障证据。

## 已完成：v3 十图 EPE Oracle 与正式标签

- 候选与审计：`data/processed/candidates_formal_v3_audited/candidate-index.json` 共 10 clips、1437 个 EPE 边段，`1437/1437` 通过；1096 个排除边段均有短边或九动作碰撞原因，最低训练边界覆盖率约 0.73775，核算覆盖率为 1。
- GPU 运行编号：`20260823T030224Z-train-oracle-8b0f0ba5`；模式为 full，种子 `[0,1,2]`，每 seed 每 clip 10000 timestep，十个 clip 均生成三个模型。
- OpenILT 指标：1437 点 × 9 动作，共 12933 个候选指标完整；缓存与正式索引版本通过标签合并校验。
- 标签文件：`data/processed/point_training_formal_v3_audited.json`，SHA-256 为 `e6aa72a5b2115a8a0a5f6993294d8b4a031a67650dfbcc66905a524d35d4b47a`；train/validation/test 为 `937/205/295`，EPE 1437、FRAG 0、歧义 4、候选碰撞 0。
- 标签分布：九个位移类均在全量数据出现，但 validation 不含 -40/-30 nm，而 train 的 -40 nm 有 139 条，存在明显父版图类别漂移。
- PPO 评估输出：`outputs/ppo_evaluation/formal_v3_audited.json`，身份校验通过，覆盖 10 clips、1437 行、3 seeds；seed 0/1/2 的 accuracy 分别为 `0.226862/0.233125/0.282533`，九分类 macro-F1 为 `0.145187/0.180078/0.163935`，平均加权损失遗憾为 `1057.812109/1121.748086/983.407098`。
- 基线：固定 0 nm 的 accuracy/regret 为 `0.173278/1482.963814`；多数类动作 5（+10 nm）为 `0.210856/1834.085595`；均匀随机期望为 `0.111111/3100.934431`。三个 PPO seed 的命中率和损失遗憾均同时优于三类基线，确认学到有效但较弱的策略信号。
- 评估判断：seed 2 的准确率最高、遗憾最低，seed 1 的 macro-F1 最高；最佳 macro-F1 仍只有 `0.180078`，且种子结果差异明显，不能据此声称稳定九分类策略。原日志的高熵、低 KL 与低 explained variance 警告仍有效，但不再写成“尚未评估”。
- 科学边界：模型按 clip 独立训练并在同一 clip 上评估，因此这不是跨版图泛化；该结果只覆盖 v3 EPE，FRAG 嵌套分段动作、共享跨版图策略、双树、跨版图闭环和完整论文复现仍未完成。
- 后续路线：保留本次 run 与 12933 个指标作为只读证据，优先从现有缓存构造只在 6 个 train 父版图训练的共享策略，用 validation 选型、test 最终评估；该步骤不需要重新执行全部 OpenILT 候选。

## 已完成：SimpleOPC 多步 PPO v3 云端冒烟与单模型诊断

- 环境：`simpleopc-multistep-v3`；损失版本为 `paper-weighted-sum-initial-normalized-v1`，训练和验收统一使用 `L2+100×EPE+PVB` 后整体除以初始加权和。
- OpenILT：固定提交 `dabb97c6ca3dfd159362e48273c436444c77353b`，训练前后 tracked diff 退出码均为 0。
- 云端回归：v3 配置、公式和完整 pytest 均达到 100%，未观察到测试失败。
- 256 timestep 冒烟：run `20260825T030855Z-train-oracle-7363bc90`；PPO 每一步均劣于初始状态，历史最佳回退到 step 0 和 `0nm×242`；该结果只证明链路可执行。
- 10000 timestep 诊断：run `20260825T031554Z-train-oracle-ff3b8ee9`；实际 10000 timestep，耗时 `1787.6209s`，模型 ZIP、Recipe、metadata、heuristic 和 stage 均存在且通过 JSON/模型哈希检查。
- PPO 最佳：step 1，`L2/EPE/PVB=96275/58/44355`，原始加权损失 `146430`，归一化损失 `0.85803185`，相对初始改善 `14.1968%`。
- 启发式最佳：step 2，`L2/EPE/PVB=64890/25/78522`，原始加权损失 `145912`，归一化损失 `0.85499654`。
- 差距：PPO 比启发式高 `518`，相对高 `0.3550%`；未通过配置中“不差于启发式”的严格门槛。
- 动作：`-10/0/+10nm=73/79/90`，最大类别占比 `0.3719`，没有动作坍缩。三类是因为最佳点在第一步，不代表九个位移类别不可达。
- 科学边界：当前是 train split 的单 clip、单 seed smoke 模式，不能运行正式 quality accepted 判定，也不能据此训练决策树。
- 汇报材料与 GPU 迁移分析：`docs/simpleopc_v3_m1_test1_diagnostic_20260825.md`。

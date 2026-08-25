# SimpleOPC 多步 PPO 云端工作流

本文记录当前主线：只读使用云端 `git clone` 的 OpenILT，先训练并验收多步 PPO，再由 accepted PPO
Recipe 生成决策树标签。命令默认在 `~/autodl-tmp/opc_agent` 执行，不读取或记录任何凭据。

## 0. 上传范围与 OpenILT 边界

只上传本项目中变更的 `src/`、`tests/`、`configs/`、`README.md` 和 `docs/`。不要上传本机
`third_party/OpenILT-main`，也不要把文件复制进云端 `third_party/OpenILT`。

新环境通过 Python 导入以下上游只读能力：GLP 解析、`utils.polygon.dissect/segs2poly/poly2img`、
`pylitho` 和 `pyilt.evaluation`。模型、轨迹、启发式基线和 Recipe 均写到本项目 `runs/`。

上传后执行：

```bash
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest -q
git -C third_party/OpenILT rev-parse HEAD
git -C third_party/OpenILT diff HEAD --exit-code --
```

成功条件：

- pytest 无失败；云端安装了 Gymnasium，`tests/test_simpleopc.py` 不应 skip。
- OpenILT HEAD 为 `dabb97c6ca3dfd159362e48273c436444c77353b`。
- `git diff HEAD --exit-code --` 返回 0，证明已跟踪源码没有被修改。

## 1. 核验 ICCAD13 GLP

```bash
python -m opc_agent.cli prepare-data --config configs/paper_repro.yaml
```

成功条件：命令输出新的 run id，且十个
`third_party/OpenILT/benchmark/ICCAD2013/M1_test*.glp` 全部存在。父版图划分固定为 6/2/2。

## 2. SimpleOPC 多步 PPO 冒烟

冒烟只跑 `M1_test1`、seed 0、256 timestep，但每个 episode 仍完整执行 4 次掩模状态转移：

```bash
export OMP_NUM_THREADS=1
python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --smoke
```

每轮动作空间为每个固定边段各一个 `inward/stay/outward`；所有边段先批量更新，再执行一次 OpenILT
光刻仿真。4 轮均使用 10nm 增量，累积位移限制在 ±40nm，因此历史最优位移只能落在
`[-40,-30,-20,-10,0,10,20,30,40]nm` 九个等距代表值上，不再做近似量化。论文正文只明确
`±40nm` 范围和最终使用 9 类，并未公开 10nm 步长、4 步 episode 或分类边界；这些是本项目为了
让九个等距类别可达而采用的适配参数，不是论文披露值。

v3 的训练与验收共用同一损失：`Lraw=L2+100×EPE+PVB`，PPO 使用 `Lraw/Lraw_initial` 仅做整体
数值缩放。不得分别除以各指标初值，否则会改写论文的相对权重。所有 v3 轨迹同时保存 `loss`、
`raw_weighted_loss` 和 `loss_version=paper-weighted-sum-initial-normalized-v1`；质量审核会逐步复算。

输出位于新的 `runs/<smoke_run_id>/`：

- `models/M1_test1-seed-0.zip`：PPO 模型；
- `models/M1_test1-seed-0.metadata.json`：模型、版图和 OpenILT 来源；
- `models/M1_test1-seed-0.recipe.json`：冻结策略完整回放及历史最优 Recipe；
- `clips/M1_test1/simpleopc-heuristic.json`：相同步长下的 SimpleOPC EPE 启发式；
- `stage-result.json`：必须显示 `environment=simpleopc-multistep-v3`、`mode=smoke`、
  `openilt_mutation=none` 和正确的 `loss_version`。

停止条件：CUDA/OpenILT 异常、法向无法唯一确定、显存溢出、Recipe/模型哈希不一致或任一测试失败。
冒烟只证明可执行，不能生成 accepted 报告或决策树标签。

## 3. 完整十图三种子训练

只有第 2 步成功并人工查看 Recipe 轨迹后才执行：

```bash
screen -S simpleopc-ppo
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate
export OMP_NUM_THREADS=1
python -m opc_agent.cli train-oracle --config configs/paper_repro.yaml
```

该命令按 train/validation/test 十个父版图和 seed 0/1/2 分别训练，每个模型默认 10000 timestep。
这是长时间 GPU 作业；不要静默减少版图、种子、episode 步数或 timestep。另一个终端使用
`watch -n 2 nvidia-smi` 监控。任务失败时保留原 run，不手工拼接或覆盖产物。

历史 `simpleopc-multistep-v2` 模型使用逐指标归一化损失，与质量门槛目标不一致，不能续跑、合并或
转换为 v3 标签。上传 v3 文件后必须生成新的 run id，并先重跑 smoke 与单 clip/seed 10000 timestep 诊断。

## 4. 只读 PPO 质量验收

把 `<full_run_id>` 替换为完整训练输出：

```bash
python -m opc_agent.simpleopc_quality \
  --run-dir runs/<full_run_id> \
  --config configs/paper_repro.yaml
```

该命令不调用 GPU。只有 validation 决定状态：

- 每个 validation clip 的最佳 PPO seed 不差于 SimpleOPC 启发式；
- 相对初始掩模至少改善 1%；
- 三种子终局损失变异系数不高于 0.20；
- validation 单一动作类别占比不超过 0.95。

成功输出 `runs/<full_run_id>/ppo-quality.json` 和 `accepted`；若输出 `rejected`，退出码为 2，必须回到
PPO 状态、奖励或训练参数，禁止继续训练树。test 只报告，不参与门槛选择。

## 5. 从 accepted PPO 生成 EPE 标签

```bash
python -m opc_agent.ppo_recipe_labels \
  --stage runs/<full_run_id>/stage-result.json \
  --quality runs/<full_run_id>/ppo-quality.json \
  --output data/processed/ppo_simpleopc_epe_training.json
```

输出标签保留 PPO 模型哈希、质量报告哈希、精确累积位移、最近九分类及量化误差。转换器只读取质量
报告为每个 clip 选择的 PPO Recipe，不读取旧九动作 Oracle 最优类。

## 6. 当前 FRAG 与决策树边界

SimpleOPC 当前只在初始化时固定分段；只移动 FRAG 端点而不重新分段不会改变掩模。下一阶段必须实现：

```text
外层 FRAG PPO 选择分割点/分段方案
        -> 内层 accepted EPE PPO 完成多步边段优化
        -> 用最终 L2/EPE/PVBand 评价 FRAG 动作
```

在该层完成前，`ppo_simpleopc_epe_training.json` 仅供审计。默认配置中的 `build-recipe` 会同时检查：

- 标签版本来自 PPO SimpleOPC；
- PPO 质量状态为 accepted；
- 质量报告哈希存在；
- 数据同时包含 EPE 和 FRAG。

缺少任一条件都会显式失败，因此当前不要执行 `build-recipe`。

## 7. 历史单步工件

旧候选索引、九动作缓存、`oracle_batch_labels` 和 `ppo_evaluation` 仍保留用于复核既有实验，不再是当前
训练入口。`docs/multi_clip_workflow.md` 记录的是历史单步流程；默认配置下带 `--candidate-index` 会被明确拒绝。

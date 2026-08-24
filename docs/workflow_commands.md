# OPC Agent 工作流命令说明

本文记录从已完成 OpenILT 基线继续到 GPU Oracle、决策树和闭环阈值扫描的命令。每一步都说明用途、前置条件、写入内容、成功标志和失败处理。命令默认在云端项目根目录 `~/autodl-tmp/opc_agent` 执行；不记录任何密码、私钥或 API 密钥。

## 0. 上传新代码后刷新项目安装

```bash
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
pytest -ra
```

- 用途：让虚拟环境读取刚上传的源码，并运行包括 GPU 工作流在内的测试。
- 前置条件：`.venv` 和锁定依赖已安装；本次不需要重新下载 PyTorch 或 OpenILT。
- 写入：editable 安装元数据和 pytest 缓存，不生成实验结果。
- 成功标志：pytest 无失败；云端应实际运行 `test_oracle_runner.py` 与 `test_recipe_tree.py`，不能出现本地环境对应的 skip。
- 失败处理：若 editable 构建隔离失败，只使用上述 `--no-build-isolation`；不要盲目升级 pip 或重装 2.27GB 的 PyTorch。

## 1. 准备一个点的候选清单

先找出实际存在的目标图和基准掩模，不猜文件名：

```bash
find third_party/OpenILT/tmp -maxdepth 1 -type f -printf '%f\n' | sort
mkdir -p data/manifests data/processed
nano data/manifests/oracle_smoke.json
```

最小清单格式如下；`target_path` 和 `base_mask_path` 必须改成上一步实际找到的文件，`x/y` 必须位于图像内：

```json
{
  "schema_version": "1.0",
  "adapter_version": "raster-boundary-strip-v2",
  "clip_id": "M1_test1-smoke",
  "parent_layout": "M1_test1",
  "split": "train",
  "target_path": "third_party/OpenILT/tmp/请替换为实际目标图.png",
  "base_mask_path": "third_party/OpenILT/tmp/请替换为实际掩模图.png",
  "scale_nm_per_pixel": 1.0,
  "points": [
    {
      "point_id": "epe-p0",
      "task_type": "EPE",
      "x": 512,
      "y": 512,
      "normal_x": 1,
      "normal_y": 0,
      "support_radius_px": 8
    }
  ]
}
```

- 用途：明确一个点的位置、任务类型、轴向法线和局部移动范围。
- 前置条件：已有公开 ICCAD13 的目标图和 SimpleILT 掩模图。
- 写入：仅新增一个很小的 JSON。
- 成功标志：JSON 能被下一步 Pydantic 校验。
- 失败处理：法线只能是 `(1,0)`、`(-1,0)`、`(0,1)`、`(0,-1)`；点越界会显式失败。
- 复现边界：`raster-boundary-strip-v2` 是论文未公开点移动代码的兼容适配；它按法线扩张/收缩边界条带并要求九动作唯一，不宣称逐像素等同论文实现。

## 2. 生成紧凑候选数据

```bash
python -m opc_agent.candidate_masks \
  --manifest data/manifests/oracle_smoke.json \
  --output data/processed/oracle_smoke.npz

du -h data/processed/oracle_smoke.npz data/processed/oracle_smoke.metadata.json

python - <<'PY'
import numpy as np
p = np.load("data/processed/oracle_smoke.npz", allow_pickle=False)
print({name: p[name].shape for name in p.files})
PY
```

- 用途：提取十维点状态，并保存“基准掩模 + 点几何”，九个动作在评价时即时生成。
- 前置条件：第 1 步 JSON 中两张图可读取且尺寸一致。
- 写入：`oracle_smoke.npz` 和 `oracle_smoke.metadata.json`。
- 成功标志：NPZ 字段含 `observations`、`target`、`base_mask`、`point_geometry`、`scale_nm_per_pixel`；不会为每个点复制九张全尺寸图。
- 失败处理：输出已存在时命令拒绝覆盖，应换新版本文件名；不要删除旧实验来强行复用路径。
- 磁盘说明：正式格式不保存 `[点数,9,高,宽]` 大数组，避免 2048 图随点数增长到几十 GB。

## 3. GPU 单点 PPO + OpenILT 冒烟

建议在 `screen` 中执行：

```bash
screen -S opc-oracle-smoke
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate

python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --smoke
```

另开一个 SSH 终端监控：

```bash
watch -n 2 nvidia-smi
```

- 用途：种子 0 运行 256 个 PPO timestep；每个动作通过 CUDA OpenILT `LithoSim`、`Basic` 和 `EPEChecker` 得到 `L2 + 100×EPE + PVB` 奖励，之后补齐该点全部九动作指标。
- 前置条件：GPU 模式、CUDA 可用、固定 OpenILT 提交正确、第 2 步输出路径与 `paper_repro.yaml` 一致。
- 写入：新的 `runs/<run_id>/`，其中包括模型 ZIP、模型元数据、`oracle-metrics.cache.json`、配置快照和 `stage-result.json`。
- 成功标志：命令打印新 run id；日志出现 `Oracle 指标完整性：9/9`；`stage-result.json` 存在。
- 中断恢复：指标缓存逐动作原子写入；重新执行会产生新 run，不会覆盖旧 run。若需要复用中断缓存，应先保留原 run 并在后续恢复功能完成后再操作，目前不要手工合并缓存。
- OOM 处理：先停止任务，记录错误和 `nvidia-smi`；不要自动重跑完整实验。紧凑格式一次只生成一个候选，不应把全部候选同时放入 GPU。

## 4. 把完整九动作指标转换为树标签

将 `<oracle_run_id>` 替换为第 3 步输出：

```bash
python -m opc_agent.oracle_labels \
  --dataset data/processed/oracle_smoke.npz \
  --metadata data/processed/oracle_smoke.metadata.json \
  --cache runs/<oracle_run_id>/oracle-metrics.cache.json \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_smoke.json
```

- 用途：为每个点选择加权损失最低的真实动作类别，并生成 geometry-v1 表格。
- 前置条件：九个动作都已由 OpenILT 评价。
- 写入：版本化点级训练 JSON。
- 成功标志：输出存在，且每行 `displacement_class` 在 0–8。
- 失败处理：任一动作缺指标、缓存与 NPZ 哈希不一致或输出版本冲突都会显式失败。
- 注意：单点 smoke 不能训练或验收决策树；正式树数据必须包含 EPE/FRAG、train/test 多父版图样本。

## 5. 多点、多父版图 Oracle

确定性边界采样、两个父版图的小批量 GPU 试跑、标签合并以及十图正式命令，统一记录在
`docs/multi_clip_workflow.md`。v2 pilot 与 v3 十图三种子长任务均已完成；当前正式 v3 运行是
`20260823T030224Z-train-oracle-8b0f0ba5`，包含 1437 个 EPE 边段和 12933 个完整九动作指标。v3 命令必须显式传入
`data/processed/candidates_formal_v3_audited/candidate-index.json`，不能依赖仍为 v2 双任务链路保留的配置默认路径。

### 5.1 评估正式 PPO

```bash
python -m opc_agent.ppo_evaluation \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --labels data/processed/point_training_formal_v3_audited.json \
  --config configs/paper_repro.yaml \
  --output outputs/ppo_evaluation/formal_v3_audited.json
```

- 已验证输出：10 clips、1437 rows、3 seeds。
- seed 0/1/2 accuracy：`0.226862/0.233125/0.282533`；九分类 macro-F1：
  `0.145187/0.180078/0.163935`；平均加权损失遗憾：`1057.812109/1121.748086/983.407098`。
- 固定 0 nm、多数类动作和均匀随机基线 accuracy 分别为 `0.173278/0.210856/0.111111`，遗憾分别为
  `1482.963814/1834.085595/3100.934431`。三个 PPO seed 均同时优于三者，但总体分类质量仍弱。
- 该命令只读既有模型、缓存和标签，在 CPU 推理，不调用 OpenILT。当前模型按 clip 独立训练，指标只能证明同 clip
  拟合优于基线，不能证明跨版图泛化。

## 6. 训练 EPE/FRAG 双树并导出 Recipe

本节仍属于 v2 双任务链路。先将各 clip 的点级 JSON 合并为 `data/processed/point_training_v2.json`，并保证同一父版图不跨
split；配置中的路径保留指向该文件。v3 正式标签只有 EPE 1437、FRAG 0，不得用它执行下面的双树命令：

```bash
python -m opc_agent.cli build-recipe --config configs/paper_repro.yaml
```

- 用途：只用 train 拟合两棵独立树，在 test 上按全部九类计算 macro-F1，并导出可直接遍历的 JSON 节点和根到叶 Recipe。
- 前置条件：数据同时包含 EPE/FRAG，且两类任务均有 train/test 样本。
- 写入：新 `runs/<run_id>/models/` 下的 `epe.tree.json`、`frag.tree.json`、`recipe.raw.json` 和 `stage-result.json`。
- 成功标志：两棵树和 Recipe 均通过 Pydantic 校验；指标不会因训练集中缺类而隐藏，macro-F1 固定按九类计算。
- 当前边界：这是确定性原始 Recipe；Qwen 文本解释/视觉标注尚未执行，`stage-result.json` 会记录 `not_run`。

## 7. 闭环阈值扫描

准备 `data/processed/validation_outcomes_v1.json`，每条必须含 validation 样本的置信度、快速 EPE D、全精算 EPE D、OOD 和 Recipe 校验状态，然后执行：

```bash
python -m opc_agent.cli run-loop --config configs/closed_loop.yaml
```

- 用途：扫描 0.50–0.99，选择“相对全精算 EPE D 退化不超过 5%”下精算率最低的阈值。
- 前置条件：只允许 validation 数据，禁止使用 test 调阈值；快速与全精算 EPE D 都必须来自真实评价。
- 写入：`runs/<run_id>/closed-loop.thresholds.json` 和 `stage-result.json`。
- 成功标志：状态为 `feasible` 并给出 selected；是否满足精算率 ≤20% 单独记录。
- 失败处理：没有可行阈值时状态为 `no_feasible_threshold`，不得人工改结果。

## 当前仍未实现或未验证

- v3 十图 EPE 候选、`1437/1437` 可视化审计、三种子长任务、12933 个 OpenILT 指标、1437 行正式标签和逐 clip PPO
  评估均已完成；不再列为“待运行”。PPO 优于三类基线，但最佳九分类 macro-F1 仅 `0.180078`。
- 尚未实现只在 train 父版图训练、在 validation 选型并在 test 最终评估的共享跨版图 PPO；当前逐 clip PPO 结果不能作为
  泛化精度。
- v3 FRAG 嵌套分段动作尚未实现，因此 v3 双树和完整闭环未完成；v2 小样本双树只保留为旧管线证据。
- v1 十父版图小步数链路已完成，但审计发现方块平移动作塌缩和 argmin 顺序偏置；这些结果保留为故障证据，不作为正式标签。
- Qwen 图片能力探针、特征 macro-F1、Recipe 文本解释和 API 费用记录尚未执行。
- evaluate 统一命令仍保持显式未实现，直到快速 Recipe 应用后的真实 OpenILT 评价数据契约完成。

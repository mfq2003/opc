# 多点、多父版图 Oracle 工作流

本文记录从已完成的单点 GPU 冒烟扩展到多 clip 候选、独立精算缓存和合并训练集的命令。命令默认在
`~/autodl-tmp/opc_agent` 执行，不记录凭据，也不会自动启动十图长任务。

## 1. 上传后验证代码

```bash
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
pytest -q \
  tests/test_point_sampling.py \
  tests/test_oracle_batch_labels.py \
  tests/test_workflow_batch.py \
  tests/test_oracle_runner.py
```

- 用途：验证确定性点采样、多 clip 索引、标签合并和既有 GPU Oracle 回归测试。
- 影响：只刷新 editable 安装和测试缓存，不重新下载 PyTorch，不运行正式实验。
- 成功标志：上述测试全部通过；任何失败都应先停止，不进入 GPU 试跑。

## 2. 生成两个父版图的小型候选索引

先确认文件名：

```bash
find third_party/OpenILT/tmp -maxdepth 1 -type f -printf '%f\n' | sort
```

必须能找到 `SimpleOPC_target1.png`、`SimpleOPC_mask1.png`、`SimpleOPC_target7.png` 和
`SimpleOPC_mask7.png`，然后执行：

```bash
python -m opc_agent.point_sampling \
  --config configs/paper_repro.yaml \
  --image-dir third_party/OpenILT/tmp \
  --output-dir data/processed/candidates_pilot_v2 \
  --parents M1_test1 M1_test7 \
  --epe-spacing 128 \
  --frag-spacing 64 \
  --support-radius 8 \
  --max-epe 2 \
  --max-frag 2 \
  --scale-nm-per-pixel 1.0

cat data/processed/candidates_pilot_v2/candidate-index.json
du -sh data/processed/candidates_pilot_v2
```

- 用途：自动提取正交栅格边界和外法线，每图最多生成 2 个 EPE 点和 2 个 FRAG 点。
- 几何版本：新索引必须显示 `axis-boundary-sampler-v2`，每个 manifest/NPZ 必须显示 `raster-boundary-strip-v2`；生成端会要求每点九个位移对应九张不同掩模。
- 写入：每个 clip 的 manifest、紧凑 NPZ、元数据和带哈希的 `candidate-index.json`。
- 磁盘：不保存 `[点数,9,高,宽]` 数组，只保存目标、基准掩模和点几何，规模很小。
- 成功标志：理想输出为 `clips=2 epe=4 frag=4`；边界筛选不足会显式失败，不会伪造点。
- 比例边界：`1.0 nm/pixel` 暂与 `openilt_scale: 1` 对齐，但不是论文公开的已验证物理比例。若确认比例不同，
  必须换输出目录和数据版本，不能覆盖本索引。

## 3. 两个 clip 的真实 GPU 小步数试跑

进入 GPU 前先执行只读 CPU 可视化审计：

```bash
python -m opc_agent.point_visualization \
  --index data/processed/candidates_pilot_v2/candidate-index.json \
  --output-dir outputs/point_visualization/pilot_v2 \
  --overview --local-patches --action-grids

cat outputs/point_visualization/pilot_v2/visualization-summary.json
```

- 总览图：红色圆点为 EPE、蓝色方块为 FRAG、绿色箭头为外法线、黄色轮廓为 base mask。
- 九动作图：绿色为新增像素、红色为删除像素、青色为 target 轮廓。
- 自动门禁：`passed_points` 必须等于 `points`，即全部点位、法线、支持窗口、动作唯一性、面积单调性和修改局部性通过。
- 人工门禁：逐点确认 target 与 base mask 对齐、箭头由前景指向背景、负位移收缩、正位移扩张且不修改无关图形。
- 影响：只读取候选文件并写 PNG/JSON，不重新采样，不调用 GPU、OpenILT、PPO 或 API。

可视化自动或人工检查失败时应停止，不得进入下面的真实 GPU 试跑。

建议在 `screen` 中执行：

```bash
screen -S opc-oracle-pilot
cd ~/autodl-tmp/opc_agent
source .venv/bin/activate

python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --candidate-index data/processed/candidates_pilot_v2/candidate-index.json \
  --smoke
```

- 用途：遍历两个 clip；每个 clip 仅用种子 0 训练 256 timestep，并补齐每点九动作 OpenILT 指标。
- 最大精算量：`2×(2+2)×9=72` 个不同候选评价；缓存命中不会重复精算。
- 写入：`runs/<run_id>/clips/<clip_id>/oracle-metrics.cache.json` 与
  `runs/<run_id>/models/<clip_id>-seed-0.zip`，不同 clip 不共用缓存。
- 成功标志：两个 clip 均打印指标完整性；若各 4 点则分别为 `36/36`，最后输出 run id。
- 监控：另开 SSH 执行 `watch -n 2 nvidia-smi`。OOM 或 OpenILT 失败时停止并保留 run，不自动重跑全实验。
- 恢复边界：动作缓存会原子写入，但统一 CLI 每次启动仍创建新 run；自动指定旧 run 恢复尚未实现。

## 4. 合并多 clip Oracle 标签

将 `<pilot_run_id>` 替换为上一条命令最后输出的编号：

```bash
python -m opc_agent.oracle_batch_labels \
  --index data/processed/candidates_pilot_v2/candidate-index.json \
  --oracle-run runs/<pilot_run_id> \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_pilot_v2.json

cat data/processed/point_training_pilot_v2.summary.json
```

- 用途：校验索引哈希、NPZ 哈希和九动作缓存完整性，再合并点级训练集。
- 成功标志：理想摘要为 8 行，EPE/FRAG 各 4 行，包含 train/validation；摘要同时记录 `ambiguous_rows` 与 `candidate_collision_rows`。以实际采样点数为准。
- 重要边界：pilot 没有 test 父版图，不能用于 `build-recipe` 或报告 macro-F1。

## 5. 已完成：v3 十父版图正式候选与三种子 Oracle

v2 pilot 证明了双任务旧链路可运行，但正式 EPE 第一阶段已经切换为 v3 的“边段中心锚点 + 整段沿法线移动”。已验证索引为
`data/processed/candidates_formal_v3_audited/candidate-index.json`：10 clips、1437 个可训练 EPE 边段、1096 个有明确原因的排除边段，
自动审计 `1437/1437` 通过，最低训练边界覆盖率约 `0.7377468`，核算覆盖率为 `1.0`。

正式生成参数如下。输出目录存在时不得覆盖；如需重采样，必须使用新的版本化目录：

```bash
python -m opc_agent.point_sampling \
  --config configs/paper_repro.yaml \
  --image-dir third_party/OpenILT/tmp \
  --output-dir data/processed/candidates_formal_v3_audited \
  --sampler-version adaptive-edge-segment-sampler-v3 \
  --support-radius 8 \
  --scale-nm-per-pixel 1.0 \
  --v3-target-segment-length 128 \
  --v3-min-segment-length 32 \
  --v3-corner-segment-length 64

python -m opc_agent.point_visualization \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --output-dir outputs/point_visualization/formal_v3_audited \
  --overview
```

已完成的长任务命令与运行编号为：

```bash
export OMP_NUM_THREADS=1

python -m opc_agent.cli train-oracle \
  --config configs/paper_repro.yaml \
  --candidate-index data/processed/candidates_formal_v3_audited/candidate-index.json

# 已完成运行：20260823T030224Z-train-oracle-8b0f0ba5
```

- 模式为 full，PPO seeds 为 `[0,1,2]`，每 seed 每 clip 为 10000 timestep。
- 10 个 clip 均生成 3 个模型；1437 点 × 9 动作，共 12933 个 OpenILT 指标完整。
- `paper_repro.yaml` 仍保留 v2 双任务默认索引，v3 必须显式传 `--candidate-index`，避免 EPE-only 数据被误认为双树输入。

正式标签已经合并：

```bash
python -m opc_agent.oracle_batch_labels \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --config configs/paper_repro.yaml \
  --output data/processed/point_training_formal_v3_audited.json
```

实测为 1437 行，train/validation/test 为 `937/205/295`，EPE 1437、FRAG 0、歧义 4、候选碰撞 0。v3 第一阶段的
`FRAG=0` 是已声明设计边界，不是数据丢失；该文件不能直接送入当前要求 EPE/FRAG 同时存在的双树训练。

## 6. 已完成：PPO 模型相对 Oracle 真值的质量审计

```bash
python -m opc_agent.ppo_evaluation \
  --index data/processed/candidates_formal_v3_audited/candidate-index.json \
  --oracle-run runs/20260823T030224Z-train-oracle-8b0f0ba5 \
  --labels data/processed/point_training_formal_v3_audited.json \
  --config configs/paper_repro.yaml \
  --output outputs/ppo_evaluation/formal_v3_audited.json
```

- 输入身份：候选索引哈希、NPZ 哈希、运行索引、模型 seed 元数据、候选来源哈希、缓存完整性和 1437 行标签全部匹配。
- PPO seed 0/1/2：accuracy 为 `0.226862/0.233125/0.282533`，九分类 macro-F1 为
  `0.145187/0.180078/0.163935`，平均加权损失遗憾为 `1057.812109/1121.748086/983.407098`。
- 固定 0 nm：accuracy `0.173278`、遗憾 `1482.963814`；多数类动作 5（+10 nm）：accuracy `0.210856`、遗憾
  `1834.085595`；均匀随机期望：accuracy `0.111111`、遗憾 `3100.934431`。
- 三个 seed 均在命中率和损失遗憾上同时优于三类基线，说明 PPO 学到了有效信号；但最佳 macro-F1 仅 `0.180078`，且
  种子差异明显，策略仍弱。
- 模型按 clip 独立训练并在同一 clip 上评估，因此这是同 clip 拟合检查，不是跨版图泛化。下一阶段应复用现有缓存，训练
  只使用 6 个 train 父版图的共享策略，以 validation 选型，并只在最后使用 test；无需重跑 12933 个 OpenILT 候选。

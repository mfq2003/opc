# Recipe PPO：逐点 EPE 法向参数与全局 FRAG 参数设计（v2 提案）

> 六图搜索更新：用户已确认扩展至全部训练版图 M1_test1–6。当前 `v2-search` 使用三 seed、两种搜索，每图每臂候选预算由实际点数×4 得到，保证完整一轮坐标扫描且随机搜索同预算；废弃固定 984 和固定 246 点限制。按既有点数预计共 31344 次候选、31416 次总求解、36 臂和 24 组输入样例。配置为 `search.budget_policy=one_full_coordinate_sweep`，工件版本 `coordinate-six-layout-single-sweep-v2`；下方两图搜索预算是历史记录，由本条覆盖。质量护栏、固定 FRAG、独立回放和 validation/test 隔离规则不变。

> 搜索实施更新：已实现 `v2-search`，冻结 M1_test5/6、三 seed、每方法每图每 seed 984 候选、单轮坐标扫描、单项不超过零偏移基线及严格下降 J。总预算含基线和回放为 11832 次 solver 调用。工件、命令、配置和中断行为见 README 当前登记；历史“待冻结/未实现”描述由本条覆盖。CPU Fake solver 测试通过，真实云端搜索尚未运行。

> 2026-09-14 最新路线覆盖说明：共享 terminal 小训练 `20260914T075849Z-v2-ppo-small-train-39fe2774` 已完整结束（4920 timestep、五次更新、每图十个 episode、execution_pass=true、accepted=false）。最终两图均全 stay，J 与零偏移基线相同；两图各十个随机 Recipe 均降低 J，尚未证明 PPO 优于随机。用户明确暂停 PPO 续训，转向固定 FRAG `(16,32)`、五动作 EPE 离散坐标搜索。下文“坐标下降延期”“下一步重跑/续训 PPO”等是历史路线，本条优先。详细结果与新路线见 README 的“2026-09-14 当前决定与实验登记”。

> 下一开发任务：实现预算受限、多训练版图的全零/随机/坐标搜索诊断。坐标搜索基于当前完整 Recipe 逐点比较其余四动作，按冻结的单项护栏和严格更低 J 接受；完整一轮无改善或预算耗尽即停，预留最终独立回放。随机与坐标搜索匹配候选求解预算，基线/回放另列并纳入总成本；记录全部候选、接受原因、点与动作、指标、哈希和调用数。版图列表、预算、访问顺序/seed、护栏参照与阈值、预算不足以完成四动作扫描时的规则待冻结。搜索尚未实现或运行；仅找到同图好 Recipe 不构成跨图泛化或正式 accepted。

> 状态：设计已按“PPO 优先、两种 EPE 回报协议都验证”修订；Phase 1–3 的 CPU contract、几何、Fake solver、Gym adapter、只读 OpenILT solver/Golden evaluator 和独立 preflight CLI 已实现。六张 train 版图的两轮真实 v2 preflight 已审阅；`24nm + 五动作` 通过六图几何扫描，分层抽样证明五图有真实动作响应，`both-sides-conflict-stay-v1` 已通过 M1_test1 真实复检。128-only dense/terminal 完整 episode smoke、PPO 输入样例、六图 Golden contract、两个协议的单 rollout PPO CUDA smoke 及 terminal 三更新 pilot 均已通过或冻结。dense 三更新 pilot 已因 Critic RMSE 连续三次超限而失败。M1_test5/6 双环境、单共享 terminal 模型的首次云端小训练完成首个 984-step rollout 后，暴露诊断层不兼容 SB3 更新后 env-major 展平 buffer；代码和回归测试已修复，等待云端从新 run 重跑。dense 小训练、通用长训练和正式验收仍未启用。
> 日期：2026-09-02（首次设计为 2026-09-01）。
> 当前文件名为兼容既有链接而保留；“global parameters”只再指两个 FRAG 参数，不代表 EPE 是全局参数。
> 环境名 `simpleopc-recipe-local-epe-global-frag-v2` 已用于独立 v2 contract；当前真实 solver 只允许由 `v2-preflight` 生成 `diagnostic_only` 工件，在动作/probe、逐版图 Golden contract、训练入口和云端证据完成前不能宣称端到端支持。
> 当前阶段不实施反事实教师、坐标下降、GNN/Transformer 或低置信度回退；这些只保留为 PPO 结果不理想时的后续路线。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-01
- Verification Status: SHARED_TERMINAL_ROLLOUT_SHAPE_FIX_CLOUD_RERUN_PENDING
- Version Label: recipe_ppo_local_epe_global_frag_v2_shared_small_train_v1

### 2026-09-01 实施起点与新增阻断证据

- 新增独立的 `recipe_v2_contract.py`、`recipe_v2.py`、`recipe_ppo_v2.py` 和 CPU 测试，未原地修改已验收的 v1 contract/environment/runner；
- 已实现逐点法向 `q=p+delta*n`、整数单位校验、probe 越界/target-validity、动作栅格别名、全局 FRAG dissect adapter、绑定物理尺度/target/法线语义的拓扑与分段实例哈希、冻结零偏移 observation、失败时原子回滚的 dense/terminal episode 核心、与 FRAG 控制点类型分离的 GoldenPointSet/evaluator contract 和 final-only payload；
- 严格按本设计的“inner 在 target 内、outer 在 target 外”检查时，孤立直边上的 `d_probe=16nm` 只能容纳绝对值小于该窗口的 crossing 位移；首轮九类表中的 `±20/±30/±40nm` 因而不是有效控制动作。普通 `stable-baselines3.PPO` 又不会消费 action mask，所以首轮配置必须阻断训练，不能静默退化；
- 配置指定的 `third_party/OpenILT` 本地不存在；本地下载镜像没有 `.git`，只能按文件 SHA256 做源码审查，不能把外层项目提交误写成 OpenILT 镜像提交；
- 本地镜像显示正式 Golden EPE 走 `pyilt/evaluation.py::epecheck`，其 `EPE_CONSTRAINT=15`，而 `16` 是 `pyilt/simpleopc.py::checkEPE` 的控制 probe。该条记录描述当时尚待云端复核的状态；现已由 M1_test1 锁定工件确认 source/constraint，但产物仍禁止写 `golden_epe_distance_nm=16`。

### 2026-09-02 真实预检入口

- 新增项目侧只读 `OpenILTV2Solver` 和 `OpenILTGoldenEvaluator`，复用锁定 OpenILT 的 GLP、`polygon.dissect`、LithoSim、`boundaries/epecheck`，不修改上游源码；
- 新增 `python -m opc_agent.cli v2-preflight --config configs/recipe_ppo_v2.yaml --layout M1_test1`，先全点扫描候选 probe 的动作合法率，再从基线任一内部轮次的 active、无冲突点中抽样并运行配置指定的非零动作；
- 预检固定写入 `runs/<run_id>/recipe-v2-preflight.json`，并在运行前后核对提交与 tracked diff；结果始终为 `diagnostic_only`，不会自动修改配置或启用训练；
- 云端 v2 专项和全量 pytest、OpenILT 提交/干净性及 CUDA 可用性已由用户确认；完整 M1_test1 diagnostic 工件已审阅，但 action/probe 仍只是候选，Golden 仅冻结跨版图稳定的 source/constraint，版图相关哈希仍未冻结。
- 首次真实运行在默认零偏移 Recipe 的 `polygon-7-edge-2-segment-2` 观测到 inner 缺印与 outer 多印同时发生；严格 solver 正确拒绝把它强行映射为单一移动方向，运行后 OpenILT tracked diff 仍为空；
- 为继续收集而非掩盖该证据，最初 preflight 单独使用过 diagnostic-only record-and-stay。两轮六图证据后，正式协议冻结为 `both-sides-conflict-stay-v1`：诊断与正式 solver 共用同一路径，逐轮记录冲突并令该点 stay，不伪造 inner/outer 优先级。
- 第二次云端 run `20260902T031959Z-v2-preflight-f4a12f8a` 的三份工件已完整审阅。M1_test1 有 242 个 EPE 点；baseline 为 `L2=64498, EPE=28, PVB=73766, J=141064`。`probe=24nm` 对五动作 `[-20,-10,0,10,20]nm` 的 242 个点全部几何合法，而没有候选 probe 支持完整九动作。原先按存储顺序取前两个合法点，其中一个对 `±10nm` 完全无响应；另一个仅 `-10nm` 改变 mask，并得到 `J=139417`。该抽样不能证明 inactive 点上的协议失败，因此后续改为 `baseline-any-step-active-nonconflict-v1`。
- 同一工件记录两个唯一冲突点、共 10 次 occurrence：二者都在 baseline 与四个单点候选的内部第 6 轮重复出现。锁定 `pyilt/simpleopc.py` 的 inner/outer 赋值顺序会让 outer 覆盖 inner，但该实现顺序不证明物理方向唯一；正式训练继续硬失败。下一轮先检验 `probe=24nm` 是否自然消除冲突，不提前采用覆盖优先级。
- Golden 的 `EPE_CONSTRAINT=15` 与 source SHA256 `cc2c111993491f9f0123e0bbad5e9815acb849e0f9d36b3ccefe0e12589f3d8c` 已由锁定云端工件核对。sampling、coordinate-system、contract 三个哈希包含版图 target/边界身份，必须逐版图记录，不能把 M1_test1 的值写成全局常量。

### 2026-09-02 六张 train 版图首轮结果与抽样修正

| 版图 | EPE 点 / baseline active | `24nm` 全点合法 | baseline 冲突 occurrence / 唯一点 | 响应点 | `pass` | 最好 / 最差 $\Delta J$ |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| M1_test1 | 242 / 197 | 是 | 1 / 1 | 1/2 | false | -29 / +509 |
| M1_test2 | 208 / 179 | 是 | 1 / 1 | 1/2 | false | -519 / 0 |
| M1_test3 | 260 / 258 | 是 | 22 / 16 | 0/2 | false | 0 / 0 |
| M1_test4 | 104 / 104 | 是 | 0 / 0 | 0/2 | false | 0 / 0 |
| M1_test5 | 246 / 120 | 是 | 0 / 0 | 1/2 | false | -81 / 0 |
| M1_test6 | 246 / 118 | 是 | 0 / 0 | 2/2 | true | -658 / +749 |

- 这六份工件都记录 OpenILT tracked diff 干净，配置 snapshot 与当时候选配置一致；`probe=24nm` 是唯一在六图上都支持当前五动作的候选，`32nm` 只在 M1_test2/4/5/6 通过。
- 首轮抽样在 eligible 列表上直接取 `[:2]`，多数选中同一原始边的相邻 segment。因此 `5/12` 仅是有偏样本中的诊断计数，不是响应率估计，也不能用于删减动作。
- M1_test3 的 baseline `control_conflict_count=22` 是内部轮次中的 occurrence 数；`sensitivity_summary.baseline_conflict_point_count=16` 是唯一 point_id 数。全部 9 次 solver 调用合计记录 192 次、17 个唯一冲突点，这不是可忽略的单点噪声。
- 下一诊断版本固定为 `baseline-any-step-active-nonconflict-edge-stratified-v2`：每图预先选 8 点，先覆盖不同 `(polygon, source_edge)`，再按法线和角点类型稳定轮询，不根据 sensitivity 结果二次挑样；所以不会因“找到有响应点就停”而高估响应性。
- 响应判定改为把每个非零动作的 `(mask_sha256, Golden metrics)` 与零偏移 baseline 直接比较，并保存动作等价类。这修复了“所有非零动作彼此同结果、但与 baseline 不同”时被误判为无响应的语义漏洞。
- 新诊断每图需 `1 + 8*4 = 33` 次 solver 调用。它仍是 `diagnostic_only`，不解决冲突；在分层工件证明可控点覆盖之前，不选择 outer-overwrites-inner、不删减动作、不开始 PPO。

### 2026-09-02 六张 train 版图分层复检

| 版图 | 响应点 | 改变 mask+Golden 的动作 | baseline 冲突 occurrence / 唯一点 | 33 次调用冲突 occurrence / 唯一点 | 最好 / 最差 $\Delta J$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_test1 | 5/8 | 10/32 | 1 / 1 | 44 / 4 | -715 / +551 |
| M1_test2 | 4/8 | 10/32 | 1 / 1 | 38 / 5 | -773 / +54 |
| M1_test3 | 0/8 | 0/32 | 22 / 16 | 716 / 25 | 0 / 0 |
| M1_test4 | 4/8 | 8/32 | 0 / 0 | 8 / 2 | 0 / +572 |
| M1_test5 | 4/8 | 7/32 | 0 / 0 | 0 / 0 | -3802 / 0 |
| M1_test6 | 6/8 | 15/32 | 0 / 0 | 0 / 0 | -1627 / +4466 |

- 六份工件均为新抽样策略、8 点、32 个单点动作结果和 33 次 solver 调用；`probe=24nm` 全点合法、`training_enabled=false`、OpenILT tracked diff 干净。
- M1_test1/2/4/5/6 共同证明局部 Recipe 能改变真实 mask 和 Golden 结果；M1_test3 的八个单点候选均无响应，且 33 次调用中冲突累计 716 次、25 个唯一点。该结果不足以否定多点完整 Recipe 的联合响应，也不允许把 M1_test3 从既定 train split 删除。
- M1_test4 的 baseline 无冲突，但部分 `+10/+20nm` 动作会诱发冲突；因此“只排除 baseline conflict point”不是训练安全策略。
- “抽样 active 点必须 100% 响应”与“六图必须零冲突”的原始门禁已被真实数据证明不可达，不再继续重复同一扫描。下一实现采用显式版本化的 `both-sides-conflict -> stay`：冲突点当轮 mask 位移为零，冲突记录和固定 Golden 惩罚仍保留。该策略不声称内/外任一方物理优先，也不会像上游赋值顺序那样默默覆盖方向。
- 当前不按 48 点样本删减 `±10/±20nm`：两组幅度在不同点上既有别名，也有独立响应。先保留五动作表完成 dense/terminal 最小 smoke，再用 final full-Recipe replay 比较。

### 2026-09-02 128-only 完整 episode smoke

- 云端 run `20260902T083239Z-v2-episode-smoke-9b75f6f9` 在 M1_test4 完成 dense/terminal 各两次完整 episode，共 4 个 variant、218 次 solver 调用；
- 工件记录 `patch_size=128`、`pass=true`、`cross_protocol_final_equal=true`、`repeat_baseline_equal=true`，四个 variant 的 `final_replay_equal=true`、`reward_telescoping_error=0`，运行后 OpenILT tracked diff 干净；
- 该证据解除 128 observation 与 dense/terminal 完整 Recipe 数值一致性门禁，不代表 PPO 已训练或收敛；下一实现是 PPO runner，同时保留每个版图的固定 PPO 输入汇报样例。

### 2026-09-02 PPO 输入样例与逐版图 Golden 冻结

- 云端 run `20260902T091312Z-v2-input-examples-a460fefd` 对 M1_test4 只运行一次 baseline solver，从真实冻结 Actor cache 保存 4 组 `PNG + NPZ`；manifest、文件与重建 observation 哈希全部一致，四个轴向法线均被覆盖；
- 六张 train 版图各自两次 preflight 中的 `sampling_state_sha256`、`coordinate_system_sha256`、`evaluator_contract_sha256` 逐图一致，现写入 `golden.layout_contracts`；运行时必须按版图匹配，不能把 M1_test1 的身份当全局常量；
- 输入样例和 Golden contract 门禁均已关闭；dense/terminal 独立单 rollout PPO smoke 均已在 RTX 4090 上通过。dense 工件记录 211 次 solver 调用和 `568.25s` 墙钟，terminal 为 5 次和 `19.57s`；两个臂训练后策略/Recipe 不同，但最终 mask 与 Golden 指标相同，均令 `J:52508->54100`，所以不能宣称改善。dense 的归一化 value RMSE 首次达到 `10.77`，下一步按既定“连续三次”规则运行恰好三次更新的稳定性 pilot，不直接开启长训练。

### 2026-09-14 terminal 三更新稳定性 pilot

- 云端 run `20260914T023940Z-v2-ppo-pilot-2a54368f` 在 M1_test4、seed 0 上连续完成三次 terminal PPO 更新，共 312 timestep；三条策略哈希互异，最终保存模型哈希与工件一致，final replay 严格相等，运行前后 OpenILT tracked diff 均为空；
- 三次归一化 value RMSE 为 `9.59 / 502.80 / 5.45`，第二次单独越过阈值 `10`，最大连续超限次数只有 1；KL 均小于 `2.5e-5`、clip fraction 均为 0，所有记录有限，因此按冻结的连续三次规则 `stability_pass=true`；
- 训练与评价共 7 次 solver、墙钟 `27.13s`；4 组 PPO 输入样例的 PNG/NPZ 哈希一致，NPZ 均为有限 `float32` 的 `5×128×128 + 12`；
- 最终动作全部为类 4（`+20nm`），`dominant_action_fraction=1.0`，Golden `J:52508->58454`、`L2:23591->31500`、`EPE:6->13`、`PVB:28317->25654`。因此 terminal pilot 只关闭执行/连续数值超限子门禁；单次巨大 RMSE、动作集中和质量退化必须保留为后续风险，不能称为 Critic 有效、质量改善、收敛或 accepted。当时的下一门禁是 dense 三更新 pilot，现已完成且失败。

### 2026-09-14 dense 三更新稳定性 pilot

- 云端 run `20260914T030101Z-v2-ppo-pilot-0cdce0b9` 在 M1_test4、seed 0 上连续完成三次 dense PPO 更新，共 312 timestep；模型、manifest、4 个 PNG/NPZ 哈希一致，final replay 相等，419 次 solver 账目闭合，墙钟 `1125.11s`，OpenILT tracked diff 为空；
- 三次归一化 value RMSE 为 `10.77 / 156.87 / 13.28`，连续三次均超过阈值 10，触发既定 `value-target-normalized-rmse-high-for-three-updates`，故 `stability_pass=false`、`pass=false`。KL 始终低于 `2.4e-5`、clip fraction 均为 0、张量均有限，不能把失败归因于 NaN、过大 KL 或裁剪饱和；
- 确定性动作计数为 `[-20,-10,0,+10,+20]=[1,0,0,0,103]`，argmax 占比 `0.9904`；同时 entropy 仍接近五分类最大熵，因此这是确定性输出集中红旗，而非已证实的概率策略塌缩；
- dense 与 terminal 使用相同初始策略，训练后模型和 Recipe 不同，却产生相同 final mask 与 `J=58454`，相对 baseline 恶化 `5946`（`11.32%`）。这要求保留 Critic 的 value/return/advantage 尺度诊断；不得提高阈值、重复相同 pilot 或直接启动通用长训练。用户随后选择先运行不改变首轮 reward scale 的 terminal 受控小训练，用更多完整 Recipe 样本判断是否存在学习趋势，而不是把三次更新直接解释为方法失败。

### 2026-09-14 双版图单共享模型 terminal 小训练实现

- M1_test5 与 M1_test6 不各自训练模型。两个独立 `LocalEPEEpisode`/Gym 环境以固定顺序写入同一个 PPO rollout buffer，只更新一个共享 Actor-Critic；Actor observation 不加入 layout ID，避免模型记忆版图标签；
- 两张图均为 246 个 EPE 点。每次 PPO 更新每个环境完整运行 2 个 episode，因此 `n_steps=492/env`、共享 buffer 为 984 timestep；五次更新后每图 10 个训练 episode、总计 20 episode/4920 timestep；`batch_size=123` 将共享 buffer 恰好分为 8 个 minibatch，`n_epochs=1`，其余首轮超参数不变；
- 每图分别保留零偏移基线、未训练确定性策略、10 个固定随机完整 Recipe 对照，以及每次更新后的确定性完整 Recipe 回放。随机样本不参与梯度、checkpoint 选择或提前停止；
- 每次更新保存 checkpoint，并记录同批 rollout 的更新前/后 value 对 return target 的 RMSE、reward/return/advantage 分布及正负比例、采样动作与确定性 argmax 计数、平均动作概率、熵、argmax margin；更新后 Critic 归一化 RMSE、KL、clip 连续三次超限或 final replay 不一致时提前停止；
- 每个版图固定保存 4 组真实 Actor 输入的 PNG+NPZ，共 8 组，抽样不依赖 reward 或训练结果，供汇报复核；
- 该入口只回答“一个共享 terminal 策略在有限预算内是否显示可学习趋势”。即使执行门通过，也始终输出 `diagnostic_only`、`quality_accepted=false`、`accepted=false`、`long_training_enabled=false`；dense 小训练和 FRAG-PPO 不随之自动开放。
- 首次云端尝试前全量 pytest 与 OpenILT tracked diff 均通过，两个环境实际完成了 984 timestep 的首个 CUDA rollout；训练更新后，SB3 已将 `returns/values/actions/advantages` 从 `step×env` 按 env-major 展平，旧诊断仍读取第二维作为环境数，因而在写 checkpoint/完整 JSON 前显式失败。这是诊断工件解析错误，不是 Critic 门禁、PPO 数值失败或质量结论。修复后依据 `rollout_buffer.buffer_size/n_envs` 恢复 `step×env`，observation 则统一成保留通道维的 env-major batch，并用更新前二维、更新后展平两种形状做回归测试；训练预算和算法参数保持不变。

## 0. 一句话定稿

v2 当前先使用两个独立训练、不同输入粒度的 PPO 策略：EPE-PPO 对每个 EPE 点读取该点附近的局部 patch，并输出该点独立的法向位移；FRAG-PPO 读取整张版图或全局特征，只为该版图输出一次两个全局分段参数：

$$
R(x)=\{\delta_1,\delta_2,\ldots,\delta_N,
L_{\mathrm{corner}},L_{\mathrm{uniform}}\}
$$

- $\delta_i$：第 $i$ 个 EPE 点沿自身局部法线的位移；不同点可以不同；
- $L_{\mathrm{corner}}$：传给 `polygon.dissect` 的全局拐角分段长度；
- $L_{\mathrm{uniform}}$：传给 `polygon.dissect` 的全局普通直边分段长度；
- mask segment 的真实法向移动继续由 SimpleOPC solver 内部完成，不属于 PPO 的直接动作。

EPE-PPO 共享的是策略权重 $\theta_{\mathrm{EPE}}$，不是 EPE 位移值。对两个局部 patch $o_i\neq o_j$，同一个局部策略可以输出 $\delta_i\neq\delta_j$：

$$
\delta_i\sim\pi_{\theta_{\mathrm{EPE}}}(\cdot\mid o_i,g_i),\qquad
\delta_j\sim\pi_{\theta_{\mathrm{EPE}}}(\cdot\mid o_j,g_j)
$$

其中 $g_i$ 是该点的法线、边方向、角点距离等局部几何信息。

EPE-PPO 必须实现并比较两个训练协议，但最终部署和正式验收只有一个统一口径：

| 协议 | solver 调用 | reward | 当前定位 |
| --- | --- | --- | --- |
| `ppo_dense_sequential` | 每个点动作后，以当前完整 Recipe 重跑 solver | 相邻完整 Recipe 的 Golden loss 改善量 | 首先实现；提供稠密条件边际信号 |
| `ppo_terminal_full_recipe` | 先收集全图所有点动作，只在 episode 终点运行一次 solver | 完整 Recipe 相对默认 Recipe 的终局改善 | 必须实现的对照；与部署一致 |
| `batched_final_replay` | 确定性批量推理全部点后，运行一次 solver | 不训练，只计算固定 Golden 指标 | 唯一正式验收与部署口径 |

逐点训练中的中间最优前缀只能用于诊断，不能作为最终 Recipe、validation 结果或 test 结果。反事实教师和 GNN 路线暂缓，不阻塞当前 PPO 实验。

## 1. 区分全局 probe distance 与逐点 EPE 法向位移

原始 `pyilt/simpleopc.py` 的 `checkEPE(distance=16)` 使用一个全图共享的控制探针距离。当前 v2 学习的则是每个局部 crossing 的法向平移。二者都是可做消融的 Recipe 变量，但几何作用不同，不能互相冒充：

1. 改变全局 `distance` 会让内外探针相对基准点向相反方向移动，改变控制窗口宽度；
2. 改变局部 $\delta_i$ 会让内外探针一起沿法线平移，改变该位置的期望 crossing；
3. 把 $\delta_i$ 填进 `distance`、把全部 $\delta_i$ 取平均，都会产生错误动作语义。

如果 observation 是以单个 EPE 点为中心的局部 patch，而 action 却只输出整图唯一的 `distance`，输入粒度和动作粒度不一致：角点、线端、密集区和长直边即使有不同局部图形，也会被强制使用同一个值，局部策略失去主要意义。

与上游固定参数完全一致的全局基线仍保留为：

$$
R_{\mathrm{source}}(x)=
\{d_{\mathrm{control}},L_{\mathrm{corner}},L_{\mathrm{uniform}}\}
$$

它只作为 source-faithful 基线和敏感性消融，不是当前逐点 EPE-PPO 的主动作。当前 v2 的研究对象明确为：

$$
R_{\mathrm{local}}(x)=
\{\delta_1,\ldots,\delta_N,L_{\mathrm{corner}},L_{\mathrm{uniform}}\}
$$

因此后续文档不得再把“学习原始固定 `distance`”和“学习逐点 $\delta_i$”描述成同一个实验。

## 2. v2 的变量所有权

| 变量 | 粒度 | 由谁产生 | 是否直接移动 mask |
| --- | --- | --- | --- |
| PPO 权重 $\theta_{\mathrm{EPE}}$ | 全训练集共享 | EPE PPO | 否 |
| PPO 权重 $\theta_{\mathrm{FRAG}}$ | 全训练集共享 | FRAG PPO | 否 |
| $\delta_i$ | 每个 EPE 点一个值 | EPE PPO 的局部策略 | 否，只移动局部控制点/目标 crossing |
| $L_{\mathrm{corner}}$ | 每张版图一个值 | 当前阶段由 FRAG-PPO 产生 | 否，只改变分段 |
| $L_{\mathrm{uniform}}$ | 每张版图一个值 | 当前阶段由 FRAG-PPO 产生 | 否，只改变分段 |
| `hmoves/vmoves` | 每轮、每个 segment | SimpleOPC | 给出内部移动方向 |
| mask segment 位移 | 每轮、每个 segment | SimpleOPC | 是 |
| $d_{\mathrm{probe}}$ | 全协议固定 | 控制点检查协议 | 否，只定义控制探针窗口 |
| Golden evaluator 版本/来源/约束 | 全实验固定 | Golden 验收协议 | 否，只评分；不得用控制 probe 的 16 冒充 |

必须明确区分：

- “一个模型服务所有点”不等于“所有点输出同一个参数”；
- “EPE 点沿法线移动”不等于“PPO 直接移动 mask”；
- “每点一个 $\delta_i$”不等于“每点训练一套独立模型”。

两套 PPO 的 contract 为：

| 策略 | observation | 单次 action | 决策次数 |
| --- | --- | --- | --- |
| EPE-PPO $\pi_{\theta_{\mathrm{EPE}}}$ | 当前 EPE 点附近的局部 patch + 点几何 | 当前点的一个 $\delta_i$ | 每个 EPE 点一次 |
| FRAG-PPO $\pi_{\theta_{\mathrm{FRAG}}}$ | 整张版图/全局几何摘要 | 一对 $(L_{\mathrm{corner}},L_{\mathrm{uniform}})$ | 每张版图一次 |

两者不能共享 observation，也不能把 EPE 点 patch 拼给 FRAG-PPO。即使底层将来复用某些图像编码模块，也必须保留独立的策略头、动作协议、模型身份、训练轨迹和验收结果。

## 3. EPE 点的几何语义

### 3.1 基准点、法线与动作

对第 $i$ 个 EPE 基准点，保存：

- 原始 target 边界上的基准坐标 $p_i$；
- 与拥有它的 segment 绑定的单位法线 $n_i$；
- 稳定点 ID、polygon/segment ID 和局部几何；
- PPO 输出的离散或连续法向位移 $\delta_i$。

移动后的 Recipe 控制点为：

$$
q_i=p_i+\delta_i n_i
$$

水平边的 $n_i$ 为竖直方向，垂直边的 $n_i$ 为水平方向；凹角、凸角和边方向必须通过几何测试确认内外符号，禁止只按坐标增减猜测。

首轮预检沿用九类动作作为候选：

$$
\delta_i\in\{-40,-30,-20,-10,0,10,20,30,40\}\ \mathrm{nm}
$$

这组数值来自当前项目协议，不应写成论文公开的完整实现细节。真实 M1_test1 几何扫描已经否决“任一候选 probe + 完整九动作”作为普通 PPO 的直接入口。基于该图证据，下一轮只复检以下候选，不代表已经冻结：

$$
d_{\mathrm{probe}}=24\ \mathrm{nm},\qquad
\delta_i\in\{-20,-10,0,10,20\}\ \mathrm{nm}
$$

六图已通过全点几何合法扫描，且五图的分层 active 样本有真实响应。实测否定了“每个抽样点都响应且所有版图零冲突”这一不可达门禁；当前入口改为五动作全点几何合法、动作链路有真实响应、冲突 stay 语义冻结，然后只允许最小 CUDA smoke。

### 3.2 移动控制点后的检查窗口

不能把 `tangent` 机械替换成 `normal` 后继续沿用所有旧逻辑。为了避免移动后的点两侧同时落在 target 内或同时落在 target 外，v2 把 $q_i$ 定义为局部期望 crossing，并围绕它使用固定探针半径 $d_{\mathrm{probe}}$：

$$
c_i^{\mathrm{in}}=q_i-d_{\mathrm{probe}}n_i,\qquad
c_i^{\mathrm{out}}=q_i+d_{\mathrm{probe}}n_i
$$

约定 $n_i$ 指向 target 外侧时：

- `in` 探针期望打印为 1；若缺失，solver 为该 segment 产生向外修正；
- `out` 探针期望打印为 0；若多印，solver 为该 segment 产生向内修正；
- 两侧均满足时，该 segment 本轮不移动。

这里 $d_{\mathrm{probe}}$ 是固定检查窗口，不是 PPO 动作。上游原始值 16 已作为首轮对照；当前 24 只是证据驱动的复检候选，必须经过六张训练版图的单位、有效性与响应验证后才能冻结。

“移动 crossing + 固定 probe window”是为当前项目提出的可测试适配方案，不是论文已经公开的内部实现事实。若水平边、垂直边、凹凸角测试或真实 OpenILT sensitivity 不能证明它稳定产生正确方向，开发必须停在 preflight，不能直接进入 PPO 训练。

### 3.3 实际 mask 移动仍归 solver

PPO 只产生 $\delta_i$。SimpleOPC 根据每个控制点的违规情况生成 $s_i\in\{-1,0,+1\}$，再使用原始步长逻辑移动真实 mask segment：

$$
\Delta m_i=s_i\cdot\mathrm{STEPSIZE}\cdot n_i
$$

首版不改变 `STEPS`、`STEPSIZE`、`DECAY`、`MAXDIST` 和最大累计位移边界。这样可以把“Recipe 学习”和“真实 mask 优化”分开归因。

## 4. 两个 FRAG 参数保持全局

FRAG 不为每个切点学习独立 offset。对每张版图只选择：

$$
F(x)=\{L_{\mathrm{corner}},L_{\mathrm{uniform}}\}
$$

`utils/polygon.py::dissect` 据此决定：

- 角部 segment 的目标长度；
- 普通直边 segment 的目标长度；
- 派生的 FRAG 切点数量和位置。

切点是全局规则作用于具体几何后的结果，不是 PPO 逐点动作。修改 $L_{\mathrm{corner}}$ 或 $L_{\mathrm{uniform}}$ 只能通过 `dissect` 改变分段，禁止直接写 mask 坐标。

不同 FRAG 参数会改变 segment 数量，从而改变 EPE 基准点集合。推荐固定数据流为：

1. 先为整张版图选择 $(L_{\mathrm{corner}},L_{\mathrm{uniform}})$；
2. 调用 `dissect` 生成稳定 segment；
3. 在每个 segment 上建立 EPE 基准点 $p_i$ 和法线 $n_i$；
4. 对每个 EPE 点裁剪局部 observation 并产生 $\delta_i$；
5. 用完整 Recipe 运行 solver。

因此 FRAG 参数必须在一个 episode 内保持不变，不能在访问 EPE 点的过程中重新分段，否则点 ID、访问顺序和已有动作会失效。

## 5. Observation 与策略结构

### 5.1 EPE 局部策略

EPE Actor 继续采用以当前基准点 $p_i$ 为中心的局部输入，而不是整图唯一输入。为了让逐点训练和全图批量回放共享同一策略，Actor observation 必须来自同一个冻结基准状态：

1. 当前版图的 FRAG 参数确定后，以 `all δ_i=0` 运行并缓存一次基准 solver；
2. 冻结该基准的 target、mask、printed、segment 和 EPE 点集合；
3. 所有点的 Actor patch 都从该基准状态裁剪，不读取逐点前缀产生的动态 mask、printed 或 episode best；
4. 逐点重跑 solver 得到的当前指标只用于 reward、Critic 和诊断；
5. patch 始终以原始 $p_i$ 为中心，不能随 $\delta_i$ 移动观察窗口。

当前 PPO-first 阶段先使用 CNN + 几何向量，不实现 GNN/Transformer。EPE 图像输入的首选通道为：

- target 或 target signed-distance/边界；
- 零偏移基准 mask；
- 零偏移基准 nominal printed；
- 当前原始基准点 $p_i$ 的固定 marker；动作候选位置只进入独立 preflight/可视化，不把尚未选择的 action 结果写入 Actor patch；
- 当前 segment 与法线标记。

EPE 与 FRAG 已拆成独立策略，因此 EPE/FRAG task one-hot 和 FRAG marker 不再占用 EPE Actor 输入。若 PVB 是训练目标，process max/min 或 band map 是否替换其中一个通道，必须作为版本化 observation 消融，不能在运行中静默改变。

当前预检配置冻结 `nm_per_coordinate=1.0`、`raster_mapping_version=db-coordinate-equals-raster-pixel-v1`、`raster_scale=1`、`raster_offset_xy=(0,0)`。候选最大动作 20nm 加 probe 24nm 后，最远控制采样位置距 $p_i$ 可达约 44nm；`64×64` 半径只有约 32nm，无法覆盖完整控制上下文。因此 v2 正式协议直接冻结 `128×128` 单尺度输入：它包含原中心 64 区域，且半径约 64nm，可覆盖 44nm 控制范围。底层 64 版本只保留用于历史回归和版本隔离测试，不再运行云端 smoke，也不作为训练候选。多尺度输入保持 `not_frozen`，若后续启用，必须另行版本化 observation、配置和验收记录。

$$
R_{\mathrm{view}}\ge |\delta|_{\max}+d_{\mathrm{probe}}+R_{\mathrm{optical}}
$$

正式输入为 `128×128` 单尺度 CNN；不再为上下文覆盖不足的 `64×64` 重复消耗真实 OpenILT 调用，也不能把尚未实现的多尺度写成已验证能力。

为保证汇报图与实际策略输入一致，每个 smoke 和后续训练 run 必须从已经 reset 的同一 `FrozenObservationCache` 固定抽取少量点，保存五通道 PNG 汇总图与精确 `image/vector` NPZ。抽样只依据 point ID、原始边、法线和角点类型，不依据 reward、动作响应或最终结果，避免挑选性展示；manifest 必须绑定 observation cache、baseline state、逐点 observation 和文件 SHA256。可视化只用于人工审计和汇报，不能进入 reward 或 accepted 判定。

策略学习：

$$
\pi_{\theta_{\mathrm{EPE}}}
(a_i\mid o_i,g_i,L_{\mathrm{corner}},L_{\mathrm{uniform}})
$$

动作 $a_i$ 映射到该点自己的 $\delta_i$。不同点即使属于同一张版图，也允许输出不同类别。

同一个局部 Actor 可以按点顺序调用，也可以把同一版图的 $N$ 个 patch 堆成 batch，一次前向得到 $N$ 行 logits。完整 Recipe 的因子化策略写为：

$$
\pi_{\theta_{\mathrm{EPE}}}(\boldsymbol{\delta}\mid O)
=\prod_{i=1}^{N}\pi_{\theta_{\mathrm{EPE}}}(\delta_i\mid o_i,g_i,L_{\mathrm{corner}},L_{\mathrm{uniform}})
$$

这不是固定长度的 $N$ 维 action head；每一行仍绑定唯一 `point_id`，因此可以处理不同版图和不同 FRAG 参数产生的可变 $N$。当前阶段用标准逐点 `Discrete(9)` 接口分别实现稠密 reward 和终局 reward，不得声称现有 v1 环境已支持该协议。

Actor 与 Critic 采用非对称信息边界：

- Actor 只读取部署时可获得的冻结局部输入和局部几何；
- Critic 可以额外读取当前全局 L2/EPE/PVB、$J/J_0$、已决定点比例、动作直方图、当前 partial Recipe 摘要、FRAG 参数和整图低分辨率摘要；
- 当前前缀的动态 mask/printed 不得回灌给 Actor，但可以进入 Critic；
- 若最小 smoke 暂时仍使用局部 Critic，只能标记为 `diagnostic_only`，不得把低 explained variance 解释成局部 EPE 没有物理规律。

Actor 几何向量优先保留 segment 长度、角点/line-end 相对距离、凹凸角类型、最近邻边距离或 pitch、基准 signed EPE error、法线/切向和 FRAG 上下文；删除绝对 `x/y`、episode progress、当前点已有位移、接近常量的上下边界、当前全图 loss 和 best-prefix 字段。非法候选必须由真正消费 mask 的策略实现排除；普通 `stable-baselines3.PPO` 不会因环境提供布尔 mask 自动屏蔽 logits，因此若继续使用普通 PPO，就必须在 preflight 保证全部动作对全部训练点都合法。仅返回 mask 但训练器不消费时，训练必须失败。

### 5.2 FRAG 全局策略

FRAG-PPO 的输入粒度与 EPE-PPO 不同，必须读取整图或全局几何摘要，例如：

- 下采样 target 和边界图；
- 角点、线端、凹角和凸角密度；
- 水平/垂直边长度直方图；
- 默认分段下的 segment 数和长度分布；
- 版图尺寸、缩放比例和边界余量。

FRAG-PPO 学习：

$$
\pi_{\theta_{\mathrm{FRAG}}}
(a_F\mid O_x,G_x)
$$

其中 $O_x$ 是整图输入，$G_x$ 是全局几何统计，$a_F$ 映射到一对合法的 $(L_{\mathrm{corner}},L_{\mathrm{uniform}})$。它每张版图只决策一次，不接收“当前 EPE 点”或点访问顺序。

因为 FRAG 只有一个全局动作，它在数学上接近一步 contextual bandit。当前阶段后续仍采用 FRAG-PPO，但训练顺序必须是先比较并冻结 EPE dense/terminal 协议，再训练 FRAG-PPO。FRAG 动作必须用合法参数对表定义单个 `Discrete(K)`，不能让两个独立 action head 任意组合出非法参数。合法性扫描负责栅格对齐、零长/交叉/越界、最小 segment 和 topology hash 去别名；若 MRC checker 已实现并冻结，则 MRC 作为硬约束，否则只能标记为未验证。这不等于当前启用穷举教师或轻量选择器。

不能把局部 EPE head 和全局 FRAG head 强行做成同一个 action。更可靠的分阶段训练路线是：

1. 先在默认 FRAG $(16,32)$ 下分别训练和比较两种 EPE-PPO reward 协议；
2. 冻结 EPE-PPO；
3. 训练 FRAG-PPO：每个全局动作先重新分段，再调用冻结的 EPE-PPO 为全部 EPE 点决策，最后由完整 solver/Golden evaluator 给 FRAG reward；
4. 冻结 FRAG-PPO，在它产生的分段分布上微调一次 EPE-PPO；
5. 若需要第二轮，仍采用交替冻结，禁止在同一批 transition 中同时更新两套策略。

这样既保持两套 PPO 分开训练，又处理了 FRAG 会改变 EPE 点数量和位置的问题。第一轮 EPE 只在默认 $(16,32)$ 上训练，不能直接当成对所有 FRAG 参数都有效的正式策略。共同最佳参数、逐图穷举 Oracle 和轻量 selector 可以后续作为诊断/替代基线，但不作为当前 PPO 开始训练的前置条件。

## 6. 推荐数据流

### 6.1 按阶段区分的公共前处理

Phase 1–5 尚未训练 FRAG-PPO，必须固定默认 FRAG $(16,32)$，先完成两种 EPE 协议：

```text
整张原始 target
        |
        v
固定 lenCorner=16 / lenUniform=32
        |
        v
dissect(target)
        |
        +-- segments
        +-- EPE 基准点 p_i / 法线 n_i
                        |
                        v
        运行并缓存 all δ_i=0 的基准 solver
                        |
                        v
        从同一基准状态裁剪全部局部 patch
```

只有 Phase 4/5 的 final replay 已比较并冻结 EPE 协议后，Phase 6 才允许把 FRAG-PPO 接到前面：

```text
整张原始 target
        |
        +-- 整图/全局特征 --> FRAG-PPO πθ_FRAG
                                     |
                                     +-- lenCorner
                                     +-- lenUniform
                                              |
                                              v
                                      dissect(target)
                                              |
                                              +-- segments
                                              +-- EPE 基准点 / 法线
                                                         |
                                                         v
                                              冻结 EPE-PPO 产生完整 Recipe
```

基准 solver 的调用可以缓存，但必须计入和报告计算成本，不能把它描述成零成本。FRAG 改变后，segments、EPE 点、基准状态和所有缓存必须整体失效并重建。上图描述执行依赖，不改变“先 EPE、后 FRAG”的训练顺序。

### 6.2 协议 A：逐点即时 solver 的 PPO 训练

```text
冻结基准 Actor observation + 固定点 schedule
        |
        v
共享 EPE-PPO 为点 i 输出 δ_i
        |
        v
写入当前完整 Recipe（只增加/替换当前点动作）
        |
        v
重跑一次完整 SimpleOPC solver
        |
        v
固定 Golden evaluator 得到 J_i
        |
        v
r_i = s × (J_{i-1} - J_i)
```

该协议用于产生稠密条件边际 reward。它测得的是 $\Delta J_i(a_i\mid\text{当前前缀})$，不是与点顺序无关的单点真实贡献。训练时可以随机化 schedule 以平均顺序偏差，但必须记录 `point_order_sha256`；轨迹从当前实际 Recipe 前进，不允许回退到 episode 历史最佳前缀。

### 6.3 协议 B：完整 Recipe 终局 PPO 与批量回放

```text
从同一冻结基准状态取得全部 EPE observation
        |
        v
共享 EPE-PPO 逐点收集或批量输出 δ_1...δ_N
        |
        v
组装所有 point_id 都已决定的完整 Recipe
        |
        v
只对该候选运行一次完整 SimpleOPC solver
        |
        v
固定 Golden evaluator
        |
        +-- PPO 训练：只返回终局 reward
        +-- 确定性回放：产生唯一 accepted 候选指标
```

终局 PPO 训练不要求固定长度的 `MultiDiscrete(N)`。环境仍可逐点接收 `Discrete(9)` 动作，中间只保存动作并返回零 reward，在最后一个点才调用 solver。确定性部署可把所有 patch 批量前向，以减少网络调用；两种执行必须绑定相同 `point_id` 并生成同一完整 Recipe。

这里的“只运行一次 solver”指每个候选完整 Recipe 只调用一次 solver；solver 内部仍按冻结协议执行多轮光刻仿真和 mask 法向更新。

### 6.4 训练与正式评价边界

两种 EPE-PPO 协议必须从相同初始化分别训练，使用相同动作表、FRAG、数据 split、seed 集合、Golden evaluator 和 PPO 超参数。只有两个独立实验都稳定后，才允许增加第三个 PPO-only 消融：用逐点即时 Actor warm start 完整 Recipe 终局 PPO。该 warm start 不能替代两个独立实验臂。

正式 validation/test 始终使用 6.3 的确定性完整 Recipe 批量回放。逐点轨迹中的 best-prefix、最大单步改善和中间 solver 结果只能作为诊断。

这里是执行时的数据依赖，不表示两套 PPO 联合训练：FRAG-PPO 先输出一次全局参数，EPE-PPO 再对派生出的每个点输出局部动作。训练时必须冻结另一套策略并记录其模型哈希。

用户指定的 OpenILT 原始目录保持只读。逐点控制点适配、Golden 评价和产物输出全部放在项目侧，不能为了方便训练直接修改上游 `simpleopc.py`。

## 7. 原始 SimpleOPC 与 v2 的边界

当前 ICCAD2013/GLP 主线的上游语义参考是 `<openilt_dir>/pyilt/simpleopc.py`。它固定了：

| 原始设置 | 固定值 | v2 处理 |
| --- | ---: | --- |
| `lenCorner` | 16 | 首个 EPE 修正阶段保持固定；之后才允许作为全局 FRAG 参数 |
| `lenUniform` | 32 | 首个 EPE 修正阶段保持固定；之后才允许作为全局 FRAG 参数 |
| `checkEPE(distance=16)` | 16 | 保留为默认 probe/基线依据，不作为全局 PPO 动作 |
| `STEPS/STEPSIZE/DECAY/MAXDIST` | 8/8/4/24 | 首版不学习 |

原函数 `checkEPE(segments, ..., distance=16)` 只接受 segment 和一个统一 distance，不能直接表达任意 $q_i=p_i+\delta_i n_i$。因此 v2 必须在项目侧增加显式的逐点控制采样适配器；禁止使用以下伪实现：

- 把所有 $\delta_i$ 平均后塞进一个全局 `distance`；
- 用移动 segment 坐标冒充移动 EPE 控制点；
- 只把旧代码中的 `tangent` 两行替换成 `normal`；
- 让 Golden evaluator 跟随 $q_i$ 一起移动。

仓库中的 `opc/simpleopc.py` 是旧 GDS/PatchSim 路径，只作为对照；README 官方 ICCAD 基线、项目 GLP/evaluation/lithography 导入和 v2 适配均应以 `pyilt/simpleopc.py` 为主参考。两条路径的输出轮次与内部最优选择必须单独版本化，不得顺手改变。

当前本地下载镜像位于 `third_party/OpenILT-main/OpenILT-main/`，但该目录没有独立 `.git`；从其中运行 Git 会向上解析到 `opc_agent`，因此此前读到的 `2b6beb...` 不是 OpenILT 镜像提交，不能使用。当前只读源码审查哈希为：`pyilt/simpleopc.py=CCBA9A814C6B0945A8A94363B7CFE227A63419D49C35E3E4049B1C86D2ED1322`、`utils/polygon.py=E39C5C6E3C7D327FAA96570F3646C08F9EC4A6C5CC780CF3FF45A33492B7184C`、`pyilt/evaluation.py=CC2C111993491F9F0123E0BBAD5E9815ACB849E0F9D36B3CCEFE0E12589F3D8C`。云端配置仍以 `third_party/OpenILT/` 和锁定提交 `dabb97c6ca3dfd159362e48273c436444c77353b` 为准；正式 preflight 必须重新核对提交、文件哈希和 tracked diff。

## 8. Golden EPE 防作弊边界

可学习控制点和正式评价点必须完全分离：

1. Recipe 控制点：$q_i=p_i+\delta_i n_i$，只用于指导 solver；
2. Golden EPE 点：固定在冻结的原始 target 边界，不随 $\delta_i$ 或 FRAG 参数移动；
3. Golden target、`nm_per_coordinate`、raster 坐标系统、阈值、边界采样点、evaluator 来源和测量规则在所有候选 Recipe 间相同；
4. 当前 Recipe-aware evaluator 先保存其实际提供的 Golden EPE violation count、L2、PVB 和原始加权损失；
5. 不允许用移动后的 Recipe 点直接计算 accepted 指标。

固定验收目标保持：

$$
J_{\mathrm{raw}}=L2+100\times EPE_{\mathrm{golden}}+PVB
$$

训练 reward 可以做正比例缩放，但验收只能读取未缩放的各项指标和 $J_{\mathrm{raw}}$。

当前 `SimpleOPCMetrics` 只有 `l2/epe/pvb`，其中 `epe` 是 `epe_in+epe_out` 的 violation count。EPE N、EPE D、MRC 和 mask complexity 必须等独立 evaluator contract、测试和产物字段实现后再加入；在此之前只能列为待扩展指标，禁止在运行报告中写成“已验收”。

还必须区分两个现有来源：上游内部控制窗口对照来自 `pyilt/simpleopc.py::checkEPE(distance=16)`，当前复检候选窗口为 24；项目固定 Golden EPE 来自 `pyilt/evaluation.py::boundaries/epecheck`。锁定云端工件已确认其约束常量为坐标值 `EPE_CONSTRAINT=15`、接口没有 `distance` 参数，source SHA256 为 `cc2c111993491f9f0123e0bbad5e9815acb849e0f9d36b3ccefe0e12589f3d8c`。正式 evaluator 必须把该值写入 `golden_evaluator.parameters.constraint_coordinate` 并由逐版图 `golden_evaluator.contract_sha256` 绑定；不能杜撰 `golden_epe_distance_nm=16` 或 `24`。

## 9. Episode 与 reward

### 9.1 两种 EPE-PPO 协议的公共规则

设默认零偏移完整 Recipe 的 Golden raw loss 为 $J_0$，设置前 $i$ 个动作后的完整 Recipe loss 为 $J_i$：

$$
J_i=L2_i+100\times EPE_i+PVB_i
$$

首轮实验采用固定正比例尺度 $s=10^{-5}$，并冻结在配置和产物中。当前阶段不启用 reward clipping、运行时 `VecNormalize` reward 标准化、每版图动态 scale 或分项 L2/EPE/PVB 归一化；这些都会改变优化条件，只能作为以后独立消融。原始指标和 $J_i$ 始终完整保存。

两种协议都必须满足：

- FRAG 在一个 EPE episode 内固定；
- reset 时从同一零偏移基准构造 Actor observation；
- 每个 `point_id` 恰好决策一次，无遗漏、无重复；
- 轨迹只从当前实际 Recipe 前进，不回退 episode best；
- `gamma=1`、`gae_lambda=1`，避免仅因位置靠前而折扣最终贡献；
- PPO 仍按 rollout/minibatch 更新网络，不把“每点求解”误写成“每点反向传播”；
- 正式结果始终来自全部点均已决定的 final Recipe。

### 9.2 协议 A：逐点即时 solver 的稠密 reward

一个 episode 按记录的 schedule 访问当前版图 EPE 点：

1. 当前点选择一个 $\delta_i$；
2. 将动作写入当前完整 Recipe；
3. 重跑一次完整 solver 和固定 Golden evaluator；
4. 返回相邻 Recipe 的改善量：

$$
r_i^{\mathrm{dense}}
=s\left(J_{i-1}^{\mathrm{raw}}-J_i^{\mathrm{raw}}\right)
$$

在 `gamma=1` 时，对最终相同的完整 Recipe 有：

$$
\sum_{i=1}^{N}r_i^{\mathrm{dense}}
=s\left(J_0^{\mathrm{raw}}-J_N^{\mathrm{raw}}\right)
$$

因此它优化的总目标与终局协议相同，但单步 credit 是给定当前 prefix 的条件边际，仍受访问顺序影响。训练时随机化 schedule 并保存顺序哈希；确定性 final replay 不读取该顺序。至少报告多种 schedule 下的最终 Recipe 一致性和 reward 轨迹方差。

每一步产物至少记录 `raw_weighted_loss_before/after`、三项 raw metrics、`scaled_training_reward`、solver 调用数、当前完整 Recipe、return、advantage 和 value prediction。逐点完整 solver 代价很高，但当前阶段按用户决策先实跑观察，不以教师、surrogate 或 GNN 作为前置条件。

### 9.3 协议 B：完整 Recipe 终局 reward

同一个逐点 `Discrete(9)` 环境可以只收集动作而不在中间求解：

1. 前 $N-1$ 个 step 只保存 $\delta_i$，`reward=0`；
2. 第 $N$ 个 step 组装完整 Recipe；
3. 只调用一次完整 solver 和固定 Golden evaluator；
4. 返回：

$$
r_i^{\mathrm{terminal}}=0\quad(i<N)
$$

$$
r_N^{\mathrm{terminal}}
=s\left(J_0^{\mathrm{raw}}-J_N^{\mathrm{raw}}\right)
$$

该模式与最终部署一致，同时保留可变数量点和标准逐点动作接口。`ppo_n_steps` 必须不小于最大 episode horizon，使 rollout 能覆盖完整终局反馈；若做不到，必须单独证明 bootstrap 边界正确，不能把截断 rollout 当完整 episode。

确定性回放时可以批量前向全部 patch。逐点收集和 batch 前向必须生成相同的 `point_id -> δ_i` 映射、完整 Recipe 哈希和 raw metrics。终局模式每个候选 episode 只允许一次 solver 调用，另加可缓存且明确计入成本的零偏移基准 solver。

### 9.4 best-prefix 与最终 Recipe

v2 可以记录 `diagnostic_best_prefix`，但禁止用它覆盖以下 canonical final 字段或流程：

- `epe_points[].normal_offset_nm` 组成的完整 `point_id -> delta_i` 映射；
- `raw_metrics` 及其 `raw_metrics_source`；
- validation/test accepted 指标；
- seed/model 选择；
- 后续部署或标签生成。

未访问点仍为 0 的 prefix 不是完整模型输出。当前 payload 的 canonical final Recipe 是按 `point_id` 排序的 `epe_points[].normal_offset_nm` 与两个全局 FRAG 参数，`final_recipe_sha256` 必须绑定这组映射；canonical final 指标是 `raw_metrics`，实际来源由 `raw_metrics_source` 标明。正式 accepted 时，`raw_metrics_source` 必须是完成 batched final replay 后的正式来源，同时 `full_recipe_replay_sha256` 非空并通过 sequential/batch gap 检查。这与 solver 内部在固定完整 Recipe 下选择哪一轮 mask 是两个不同层级，必须分别命名和版本化。

### 9.5 FRAG-PPO episode

FRAG-PPO 每张版图只有一个全局 action：

1. 输入整图 $O_x$ 和全局特征 $G_x$；
2. 输出一个合法 FRAG 参数对 $(L_{\mathrm{corner}},L_{\mathrm{uniform}})$；
3. 重新分段并生成该参数对对应的全部 EPE 点；
4. 调用冻结的 EPE-PPO 为这些点分别生成 $\delta_i$；
5. 完整运行 solver 和固定 Golden evaluator；
6. 用冻结 EPE-PPO 下默认 FRAG 与候选 FRAG 的 Golden loss 差定义唯一 reward：

$$
r_F=s\left[
J(F_{\mathrm{default}},\pi_{\mathrm{EPE}})
-J(F_{\mathrm{candidate}},\pi_{\mathrm{EPE}})
\right]
$$

FRAG-PPO 的一步 episode 不使用任何单点 patch。若实现为了复用代码而把某个 EPE patch 填给 FRAG observation，测试必须失败。

FRAG 候选动作来自冻结的合法参数对表；枚举该表用于合法性、去拓扑别名和建立默认/Oracle 诊断基线，不代表当前启用轻量 selector 或教师路线。

### 9.6 两套策略的训练隔离

- 训练 EPE-PPO 时，FRAG 参数来自冻结的默认值、冻结的 FRAG-PPO 或预先声明的合法采样分布；
- 训练 FRAG-PPO 时，EPE-PPO 权重完全冻结；
- 每个运行记录 `epe_model_sha256`、`frag_model_sha256`、当前更新对象和冻结对象；
- 禁止从同一个 reward 同时反向更新两套策略，否则无法归因是哪种粒度的动作带来改善；
- 最终验收对象是绑定了两个模型哈希的完整 pipeline，不允许为每张 test 版图重新组合 seed。

当前顺序是：先在默认 FRAG 上从相同初始化分别训练 dense/terminal EPE-PPO；比较它们的 final full-recipe replay 后冻结获胜协议；再训练 FRAG-PPO；最后可在冻结 FRAG 分布上对 EPE-PPO 微调一轮。任何 warm start 都必须作为第三个独立实验臂，不能覆盖两个从头训练的对照结果。

## 10. 产物协议

每个 v2 Recipe 至少保存：

```json
{
  "schema_version": "2.0",
  "environment": "simpleopc-recipe-local-epe-global-frag-v2",
  "label_version": "ppo-recipe-local-epe-global-frag-v2",
  "point_version": "target-epe-normal-point-v2",
  "action_semantics": "per_epe_point_normal_offset_and_two_global_fragment_lengths",
  "training_protocol": "ppo_dense_sequential",
  "accepted_metrics_source": null,
  "required_accepted_metrics_source": "batched_final_replay",
  "actor_observation_state": "frozen_zero_offset_baseline",
  "observation_version": "local-epe-frozen-baseline-5x128-geom12-v2-formal-candidate",
  "patch_shape": [5, 128, 128],
  "vector_shape": [12],
  "loss_version": "paper-weighted-sum-raw-v1",
  "gamma": 1.0,
  "gae_lambda": 1.0,
  "epe_probe_distance_nm": 24,
  "corner_length_nm": 16,
  "uniform_length_nm": 32,
  "epe_points": [
    {
      "point_id": "polygon-0-segment-0-epe",
      "base_xy": [64, 64],
      "normal_xy": [0, 1],
      "normal_offset_nm": 10,
      "realized_normal_offset_nm": 10,
      "quantization_error_nm": 0,
      "moved_xy": [64, 74],
      "segment_id": "polygon-0-edge-0-segment-0"
    }
  ],
  "raw_metrics": {
    "l2": 0,
    "epe": 0,
    "pvb": 0,
    "weighted_loss": 0
  },
  "raw_metrics_source": "ppo_dense_sequential_final",
  "optional_extended_metrics": {
    "epe_n": null,
    "epe_d": null,
    "mrc_violations": null
  },
  "golden_evaluator": {
    "version": "fixed-target-probe-golden-diagnostic-v1",
    "source_sha256": "<diagnostic-source-sha256>",
    "frozen_target_sha256": "<sha256>",
    "sampling_state_sha256": "<sha256>",
    "nm_per_coordinate": 1.0,
    "coordinate_system_sha256": "<sha256>",
    "contract_sha256": "<sha256>",
    "parameters": {
      "diagnostic_probe_distance_nm": 15,
      "nm_per_coordinate": 1.0,
      "raster_mapping_version": "db-coordinate-equals-raster-pixel-v1",
      "raster_scale": 1.0,
      "raster_offset_xy": [0, 0],
      "threshold": 0.5,
      "formal_openilt_epecheck": false,
      "reward_weights": {"l2": 1.0, "epe": 100.0, "pvb": 1.0},
      "golden_point_set_version": "frozen-target-boundary-points-v1",
      "golden_point_set_source_sha256": "<sha256>",
      "golden_point_set_sha256": "<sha256>"
    }
  },
  "training_reward_scale": 1e-5,
  "point_order_sha256": "<sha256>",
  "baseline_state_sha256": "<sha256>",
  "action_table_sha256": "<sha256>",
  "point_set_sha256": "<sha256>",
  "geometry_identity_sha256": "<sha256>",
  "final_recipe_complete": true,
  "final_recipe_sha256": "<sha256>",
  "full_recipe_replay_sha256": null,
  "sequential_final_replay_gap": null,
  "solver_call_counts": {
    "baseline_solver": 1,
    "candidate_solver": 1,
    "final_replay_solver": 0
  },
  "diagnostic_best_prefix": {
    "step": 0,
    "raw_metrics": {"l2": 0, "epe": 0, "pvb": 0},
    "raw_weighted_loss": 0
  },
  "epe_model_sha256": null,
  "frag_model_sha256": null,
  "layout_sha256": "<sha256>",
  "openilt_revision": "fake-openilt-not-formal",
  "source_hashes": {"recipe_v2.py": "<sha256>"},
  "status": "diagnostic_only"
}
```

上例对应 CPU/Fake solver 的 diagnostic payload 字段形状，不是实验结果，也不是正式 OpenILT Recipe。`accepted_metrics_source=null` 表示当前没有 accepted 指标；`required_accepted_metrics_source="batched_final_replay"` 只声明未来 accepted 必须采用的口径。实际证据必须联合读取 `status`、`raw_metrics_source`、`full_recipe_replay_sha256` 和 `sequential_final_replay_gap`。六张 train 版图的逐图 sampling/coordinate/evaluator contract 已冻结，但 validation/test 尚未完成正式 accepted 评价，因此仍不能生成伪装成正式 Golden 的 Recipe payload。

正式产物还要记录：

- EPE 动作候选表及 SHA256；
- FRAG 参数对候选表及 SHA256；
- 每个点的基准位置、法线、动作、移动后位置和合法性检查；
- 点 schedule、点顺序哈希、点 ID、segment 拓扑哈希和访问完整性；
- 冻结基准状态哈希、Actor observation 版本和 patch 规格；
- dense/terminal 协议身份、逐步或终局 reward、solver 调用次数和墙钟时间；
- sequential final 与 batched final 的 Recipe/metrics gap；
- 完整 final Recipe 与只作诊断的 best-prefix 分离字段；
- 版图、配置、项目源码和只读 OpenILT 来源哈希；
- `nm_per_coordinate`、raster mapping 版本/缩放/偏移、nm 换算和量化后坐标；
- 固定 Golden 评价协议版本；
- seed、模型哈希、split 和选模规则。

禁止再出现全局可学习字段 `epe_control_distance_nm`。固定控制窗口应明确命名为 `epe_probe_distance_nm`；正式 Golden 只能用独立 evaluator 身份记录，避免再次把控制检查半径误写成 Golden 或可学习 Recipe。

`optional_extended_metrics` 在 evaluator 尚未实现相应 contract 时必须为 `null`，不能用当前汇总 `epe` 推导或冒充 EPE D/MRC。

## 11. 验收规则

### 11.1 语义正确性

- EPE observation 明确绑定一个 `point_id`；
- 每个 EPE action 只更新该点的 $\delta_i$；
- dense 与 terminal 协议中的 Actor 都读取同一个冻结基准 observation；
- EPE-PPO 每次语义上输出一个点动作；batch 推理输出的每一行仍须绑定唯一 `point_id`；
- 逐点确定性收集与 batch 前向生成相同完整 Recipe，不受点存储顺序影响；
- FRAG-PPO observation 必须是整图/全局特征，且每张版图只输出一次参数对；
- 水平边只产生竖直法向候选，垂直边只产生水平法向候选；
- 不同 EPE 点允许输出不同动作；
- FRAG 不存在逐点 offset，只存在两个全局长度；
- PPO 不直接写 mask segment 坐标；
- Golden 点、Golden evaluator 来源/约束和验收 target 无法由 action 覆盖；
- 默认 Recipe `all δ_i=0, lenCorner=16, lenUniform=32` 在冻结环境下复现默认路径，差异在预先定义容差内。
- accepted 只能读取 final full-recipe replay；best-prefix 只能进入显式 `diagnostic_*` 字段。

### 11.2 数值稳定 smoke

- reward 缩放不改变 raw metrics 和候选排序；
- reward、return、advantage、value prediction、policy loss、value loss 全部有限；
- dense reward 满足望远镜一致性：$\left|\sum_i r_i-s(J_0-J_N)\right|\le10^{-6}$ 或预先冻结的浮点容差；
- terminal 模式只有最后一步产生 reward，且每个候选 episode 的 solver 调用数为一次；
- `p99(|return|) <= 10`；不再出现 $10^{12}$ 量级 value loss；
- 同时记录 explained variance 和归一化 value RMSE，不能只因绝对 value loss 下降就宣称 critic 已学会；
- 单个 terminal rollout 在 `gamma=1` 时可能产生方差为零或数值上可忽略的近常量 return；当 return 方差小于等于 `1e-12` 时，explained variance 数学上病态，工件必须写为 `null`，同时记录实际方差、标准差和原因。reward/return/value 张量仍必须全部有限，不能把结构性未定义误报为 CUDA 数值崩溃；
- 若连续三次 PPO 更新满足 `RMSE(V,G) > 10 * max(RMS(G), 1e-3)`，其中 `G` 为该批次 return/value target，则停止并检查 Critic/observation；
- 若 `approx_kl > 0.1` 或 `clip_fraction > 0.8` 连续三次更新，停止并检查步长/策略塌缩；
- 至少两个 EPE 点因局部 patch 不同而存在不同的有效最优动作；
- 至少两个合法法向动作产生不同的控制采样或 solver 响应；
- 候选 `d_probe=24nm` 与五动作表已通过六张 train 版图全点几何扫描，冻结 `both-sides-conflict-stay-v1`，且正式 solver、128 observation、dense/terminal 各两个完整 episode、两个单 rollout PPO CUDA smoke 及 terminal 三更新 pilot 均已通过；dense 三更新 pilot 已按 Critic RMSE 规则失败，原因诊断完成前长训练仍必须显式失败；
- 上游 tracked diff 保持 0；
- smoke 只能标记 `diagnostic_only`。

### 11.3 性能 accepted

- 只在 validation 上选模型、seed、动作表和停止点；
- test 在协议冻结后只运行一次，不参与调参；
- 一个共享模型服务全部 EPE 点，禁止逐点选择不同 seed；
- dense 与 terminal PPO 从相同初始化分别训练并分别报告，不允许在同一 rollout 混用两种 reward；
- 可选 warm-start 必须是第三个独立实验臂，不覆盖从头训练结果；
- EPE-PPO 与 FRAG-PPO 分别训练、分别冻结和分别记录模型哈希；
- 当前必需比较固定默认、dense PPO 的 final replay、terminal PPO 的 final replay，以及二者的 simulator calls/墙钟时间；Oracle、分类器、教师和 GNN 不作为当前验收前置；
- 当前 evaluator 分别报告 L2、汇总 EPE violation、PVB、加权损失、segment 数和运行时间；EPE N/D、MRC 只在扩展 evaluator 实现后追加；
- seed 数、改善 seed 比例、聚合改善阈值、逐图非退化规则和 L2/EPE/PVB 单项 guardrail 必须在 Phase 7 前根据 sensitivity/validation 证据写入版本化配置；当前 `configs/recipe_ppo_v2.yaml` 对应字段保持 `null`，任何字段未冻结时质量审核必须拒绝产生 `accepted`；
- 按全部 validation 版图的聚合结果选择一个全局 seed/model，禁止逐图挑 seed；
- “至少三个 seed、至少 `2/3` seed 改善、聚合改善 `1%`、L2/PVB 不恶化超过 `5%`”只保留为首轮候选门槛，不是当前生效的 v2 accepted contract，也不能从 point-v1 直接继承；
- 聚合 Golden EPE 是否要求严格不恶化，以及逐图与聚合 guardrail 如何组合，同样必须在正式运行前确认并冻结；
- 动作分布、点位覆盖、候选敏感性和多 seed 稳定性通过审核后，才允许生成新标签。

`dominant_action_fraction > 0.95` 当前也只作为候选诊断红旗，不单独构成拒绝；只有敏感性实验已证明不同局部点存在不同有效最优动作，而策略仍输出同一动作时，才判为塌缩。所有改善比例、动作分布阈值和 guardrail 必须在 Phase 7 前冻结，不能把 point-v1 的 accepted 直接继承为 v2 证据。

## 12. 测试清单

### 12.1 CPU 几何与协议测试

1. 水平边的 $q_i=p_i+\delta_i n_i$ 只改变 y；
2. 垂直边的候选只改变 x；
3. 正负 $\delta_i$ 分别沿内外方向，符号与 polygon winding 无关；
4. 凹角、凸角、线端附近候选不会绑定到错误 segment；
5. 越界 probe 在进入 NumPy 索引前显式失败；
6. 量化后重合的 EPE 动作被标记为动作别名或从动作表去除；
7. 每个 point action 只修改一个 `point_id`；
8. 两个 FRAG 参数传入每次相关 `dissect` 调用；
9. episode 内 FRAG 固定，重新分段会显式失败；
10. Golden evaluator 不读取移动后的 Recipe 点；
11. 产物包含逐点 `normal_offset_nm`，且不存在可学习的全局 EPE distance；
12. reward 缩放不改变 raw metrics 和 accepted/rejected；
13. Actor observation 在同一 episode 的不同 prefix 和不同 schedule 下保持相同；
14. Actor 输入不含绝对 `x/y`、progress、episode best 或接近常量的动作边界；
15. 正式 `128×128` 具有独立版本/哈希；底层 `64×64` 历史回归版本不得出现在 v2 训练工件中，多尺度若后续实现也必须使用新版本/哈希；
16. EPE N/D、MRC 未实现时产物对应字段必须为 `null`；
17. Golden 点集使用独立类型，Golden/solver 的 `nm_per_coordinate` 与 coordinate-system hash 不一致时构造失败；
18. solver raster/sign/hash 或返回类型非法时，在 observation/reward 前失败；
19. evaluator 参数 accessor 深拷贝且内部参数不可变，不能在 contract hash 不变时污染 payload；
20. `training_protocol`、点集/动作表/几何哈希等关键 episode 元数据不能在 final 后重绑定。

### 12.2 Fake solver 集成测试

- 同一共享策略可对两个不同局部 observation 输出不同动作；
- 一个 action 只更新当前 EPE 点；
- 批量传入 $N$ 个 EPE patch 时返回绑定 $N$ 个唯一 `point_id` 的逐点 logits，而不是固定长度 action head；
- point schedule 无遗漏、无重复；
- 候选 A/B 返回不同 raw loss 时，reward 排序一致；
- dense 模式每个动作后调用一次候选 solver，且 reward 总和与终局 raw loss 改善一致；
- terminal 模式前 $N-1$ 步不调用候选 solver且 reward 为 0，第 $N$ 步只调用一次；
- solver/evaluator 抛错或在回调中改变冻结身份时，Recipe/cursor/trajectory 原子回滚，只保留调用尝试计数；
- 同一确定性模型逐点收集与 batch 前向产生相同的完整 Recipe 哈希；
- sequential final 与 batch replay 的完整 Recipe hash、raw metrics 和 mask hash 一致；
- 打乱点存储顺序不改变 `point_id -> δ_i` 映射和 batch final metrics；
- best-prefix 进入 accepted/final 字段时测试必须失败；
- save/load 后同一 observation 输出同一确定性动作；
- validation 不允许逐点或逐图挑 seed；
- FRAG 参数变化后 EPE 点集合会整体重建，不复用旧 point ID。
- FRAG-PPO 接收局部 EPE patch 或在一张版图内输出多次时显式失败；
- 训练任一 PPO 时，另一 PPO 的参数哈希保持不变。

### 12.3 云端 CUDA/OpenILT 测试

- 默认 Recipe 复现；
- 水平边、垂直边、凹角和凸角的法向候选可视化人工通过；
- 至少两个点的不同 $\delta_i$ 产生不同真实 solver 响应；
- 单张 train 图逐点 sensitivity scan；
- 只对正式 `128×128` 输入运行接口与 PPO smoke；
- dense 与 terminal 模式分别完成至少两个完整 episode 的数值 smoke；
- 核对每种模式的真实 solver 调用次数、reward telescoping、return/value 有限性；
- 生成并比较 sequential final 与 batched final Recipe/metrics gap；
- 单 seed、train/validation 小诊断；
- 最后才进行多 seed 正式 validation；
- 每一步前后核验上游源码哈希和 tracked diff。

## 13. 实施顺序

### Phase 0：冻结证据

- 保留 `simpleopc-recipe-point-v1` 运行、模型和质量报告，不改写历史状态；
- 将其明确标记为“EPE 切向 + FRAG 逐点切向”的旧自定义协议；
- 记录当前项目相关源码、配置和上游文件哈希；
- 不提交 Git，除非用户另行授权。

### Phase 1：逐点 EPE 法向 contract 与几何测试

- 新建版本化 v2 contract；
- EPE Recipe 保存 `base_xy/normal_xy/normal_offset_nm/moved_xy`；
- 固定 FRAG 为 $(16,32)$；
- 实现水平、垂直、凹角、凸角和越界测试；
- 生成默认点与 `±40nm` 法向候选可视化。

### Phase 2：项目侧控制点 adapter 与 Golden evaluator

- 在项目侧实现移动 crossing 的固定 probe window；
- 保留原始 solver mask 步长和最大位移逻辑；
- 建立完全独立的固定 Golden 评价入口；
- 禁止直接修改只读 OpenILT 源码；
- 先用 fake solver 证明动作、方向和 reward 链路。

### Phase 3：冻结 Actor observation 与数值协议

- 把旧 EPE `tangent` 动作替换为经过测试的 point-local `normal` contract；
- 从 point schedule 删除所有 FRAG 逐点动作；
- 建立 `all δ_i=0` 的冻结基准 observation，禁止 prefix/best 状态回灌 Actor；
- 删除 Actor 的绝对坐标、progress、当前 offset、常量边界和全局 loss；
- 冻结 `128×128` 单尺度输入并预留多尺度扩展；
- 固定 `s=1e-5`、`gamma=1`、`gae_lambda=1`、raw metric 记录和有限性诊断；
- 完成 fake solver、单图 preflight 和 CUDA 接口 smoke。

### Phase 4：EPE-PPO 逐点即时 solver 实验

- 每个点动作后重跑完整 solver；
- 使用 `s(J_{i-1}-J_i)` 的 dense reward；
- 记录 point order、条件边际、solver calls 和 best-prefix 诊断；
- 只用 final complete Recipe 做确定性回放；
- 完成单 seed 小训练，再决定是否进入多 seed，不因 value loss 数字变小直接宣称收敛。

### Phase 5：EPE-PPO 完整 Recipe 终局实验

- 从相同初始化独立训练，不复用 Phase 4 权重作为默认；
- 前 $N-1$ 步只收集动作，终点只调用一次 solver；
- 使用 `s(J_0-J_N)` 终局 reward；
- 保证 rollout 覆盖完整 episode；
- 实现确定性 batch final replay，并与逐点收集结果核对 Recipe/metrics hash；
- 比较 Phase 4/5 的 final quality、顺序敏感性、value 诊断、仿真次数和墙钟时间；
- 两个独立臂均稳定后，才允许另做“Phase 4 Actor warm start Phase 5”的 PPO-only 消融。

### Phase 6：两个全局 FRAG 参数的 PPO

- 建立合法 FRAG 参数对表，完成栅格、拓扑、最小 segment 和动作别名检查；MRC 只在 checker 实现并冻结后作为硬门槛；
- 选择 Phase 4/5 中通过 final replay 的 EPE 协议并冻结其模型；
- 训练 FRAG-PPO，每张版图只输出一次合法参数对；
- 每个候选重新 dissect、重建 segment/EPE 点/基准 observation，再由冻结 EPE-PPO 产生完整 Recipe；
- 与默认 FRAG 和共同最佳诊断基线比较，但不把轻量 selector 或教师作为当前前置；
- 冻结 FRAG 后最多对 EPE-PPO 微调一轮，必须作为单独运行记录。

### Phase 7：正式 validation

- 冻结动作表、Golden 协议、reward、split 和选模规则；
- 多 seed 训练只使用 train；
- validation 决定 accepted/rejected；
- test 只在冻结后运行一次；
- accepted 只读取 batched final replay，不读取 best-prefix；
- 当前阶段不生成教师/GNN/决策树标签。

## 14. 延期研究路线（当前不实施）

以下流程保留为 PPO full replay 未通过质量门槛、顺序敏感性过高或仿真预算不可接受时的后续路线，不属于当前 Phase 0–7 的实现或验收前置条件：

```text
FRAG 合法参数对穷举/轻量选择器
        ↓
反事实 + 坐标下降教师
        ↓
EPE 图数据和软标签
        ↓
多尺度 CNN + GNN/Transformer
        ↓
一次输出完整 Recipe
        ↓
SimpleOPC
        ↓
固定 Golden signoff
        ↓
低置信度回退默认 Recipe
```

当前不实现或不启用：教师 imitation、坐标下降标签、graph policy、surrogate simulator、低置信度/OOD 回退、MLLM/Qwen、决策树，以及 EPE-PPO/FRAG-PPO 联合反向更新。延期路线若被触发，必须另建版本化设计和实验记录，不能回填成当前 PPO 已具备的能力。

## 15. 尚待证据决定的问题

1. `24nm probe + [-20,-10,0,10,20]nm` 候选是否能在六张 train 版图全部通过，还是仍需调整动作范围/探针；
2. `d_probe` 的固定值、`nm_per_coordinate` 与 raster mapping 换算；
3. 移动 crossing 的 probe 规则是否能稳定产生正确 `hmoves/vmoves`；
4. 凹角、凸角和邻近边情况下法线候选的有效区域；
5. `128×128` 是否足以覆盖关键局部几何与光学邻近范围，后续是否需要多尺度；
6. 每点重跑完整 solver 的成本以及安全缓存边界；
7. 哪些 FRAG 参数对在十图上产生真实不同而非拓扑别名；
8. validation 改善阈值与 L2/EPE/PVB 单项 guardrail；
9. segment 数、运行时间和 MRC 应作为硬约束还是报告项。
10. frozen-baseline Actor observation 是否足以支持逐点 dense reward；
11. 不同 point schedule 对 dense reward 轨迹、最终动作和 final metrics 的影响；
12. sequential final 与 batched final 的 Recipe/metrics gap；
13. terminal PPO 的 centralized critic 是否足以解决终局 credit assignment；
14. dense、terminal 和可选 warm-start 三种 PPO 的质量/仿真成本权衡。

这些问题必须由几何测试、敏感性扫描和 validation 回答，不能从论文未公开信息、旧 v1 accepted 或原始 `distance=16` 猜测。

## 16. 下次调用提示

下次开发可直接使用：

> 请严格按照 `docs/recipe_ppo_global_parameters_v2_design.md` 继续。六图 preflight、M1_test1 conflict-stay、128 episode smoke、PPO 输入样例、六图 Golden contract、dense/terminal 单 rollout CUDA PPO smoke 及两个三更新 pilot 均已完成；dense pilot 的 value RMSE 连续三次超限。`v2-ppo-small-train` 只允许 M1_test5/6 两个环境共同训练一个 terminal PPO 模型。首次云端尝试已完成首个 984-step rollout，但旧诊断误把 SB3 更新后 env-major 展平数组当作 step×env，因实现缺陷停止；当前代码已按 `buffer_size/n_envs` 恢复版图列并补充两种 buffer 形状测试。下一步同步修复代码，在 CUDA/OpenILT 环境重新跑全量 pytest 与 OpenILT clean check，再从一个新 run 执行相同小训练命令并下载完整工件；不得把上次异常解释为 PPO 失败，也不得把后续 `execution_pass` 写成质量改善、收敛或 accepted。保持 `training.enabled=false`、FRAG=`16/32`，不得启用 dense 小训练或通用长训练，不得调高阈值。多尺度、教师、坐标下降、GNN/Transformer、低置信度回退和决策树继续延期；不要修改上游 OpenILT，不要提交 Git。

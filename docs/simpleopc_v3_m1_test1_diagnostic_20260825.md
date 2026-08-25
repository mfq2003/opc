# SimpleOPC 多步 PPO v3：M1_test1 云端诊断报告

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-25
- Verification Status: ANALYZED
- Version Label: validation_v1
- Source: 用户提供的 AutoDL 云端终端输出与可审计 JSON 产物

## 一句话结论

`simpleopc-multistep-v3` 已在固定 OpenILT 提交和真实 CUDA 环境中跑通。单版图、单 seed、10000 timestep 的 PPO 将论文形式加权损失相对初始掩模降低 `14.1968%`，动作没有坍缩，但仍比同条件 SimpleOPC 启发式差 `0.3550%`；因此已经证明策略能学到有效位移，尚不能声称通过正式 validation 质量验收。

## 实验身份

| 项目 | 已验证值 |
|---|---|
| OpenILT commit | `dabb97c6ca3dfd159362e48273c436444c77353b` |
| OpenILT tracked diff | `0`，无已跟踪改动 |
| 环境版本 | `simpleopc-multistep-v3` |
| 损失版本 | `paper-weighted-sum-initial-normalized-v1` |
| 原始损失 | `L2 + 100×EPE + PVB` |
| 位移代表值 | `[-40,-30,-20,-10,0,10,20,30,40] nm` |
| 每轮步长 | `[10,10,10,10] nm` |
| 版图 / seed | `M1_test1 / 0` |
| 边段数 | `242` |
| 云端设备 | AutoDL `4080(S)-32G` 规格；实际设备身份仍以 `nvidia-smi` 为准 |

## 两次 v3 运行

### 256 timestep 冒烟

- Run ID：`20260825T030855Z-train-oracle-7363bc90`
- 模型 SHA-256：`1059f61fd7d4ad5c4364eddd30e52f38a23aed57df3428421abc56728be043fb`
- PPO normalized losses：`[1.0, 1.06183126, 1.14999004, 1.25647787, 1.41971077]`
- 最佳点：step 0，导出 Recipe 为 `0 nm × 242`。
- 启发式最佳：step 2，normalized loss `0.85499654`。
- 解释：冒烟验证了 v3 版本、损失、CUDA、OpenILT 和产物链路；256 timestep 不足以判断最终学习能力。

### 10000 timestep 诊断

- Run ID：`20260825T031554Z-train-oracle-ff3b8ee9`
- 实际 timestep：`10000`
- 耗时：`1787.6209 s`，约 `29 min 47.6 s`
- 模型 ZIP：`5,214,273 bytes`，ZIP 结构有效，产物中的模型哈希复核一致。
- Recipe、metadata、heuristic 和 `stage-result.json` 均存在且是有效 JSON。

## 物理质量比较

| 方法 | best step | L2 | EPE | PVB | 原始加权损失 | 归一化损失 | 相对初始改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 初始掩模 | 0 | 116184 | 86 | 45874 | 170658 | 1.000000 | 0% |
| PPO | 1 | 96275 | 58 | 44355 | 146430 | 0.858032 | 14.1968% |
| SimpleOPC 启发式 | 2 | 64890 | 25 | 78522 | 145912 | 0.854997 | 14.5003% |

PPO 与启发式的损失差为：

```text
146430 - 145912 = 518
518 / 145912 = 0.3550%
```

分项上，PPO 比启发式多出 `31385` 的 L2 和 `33×100=3300` 的 EPE 惩罚，但减少 `34167` 的 PVB，最终净差为 `518`。这说明 PPO 学到的是与启发式不同的权衡，不是简单复刻 EPE 方向规则。

## PPO 轨迹与动作分布

```text
PPO raw losses:
[170658, 146430, 169713, 200568, 229565]

PPO normalized losses:
[1.000000, 0.858032, 0.994463, 1.175263, 1.345176]

最佳 Recipe：
-10 nm: 73
  0 nm: 79
+10 nm: 90
```

最大类别占比为 `90/242=0.3719`，低于配置门槛 `0.95`，不存在单一动作类别坍缩。本次只有 `-10/0/+10 nm` 并不表示九分类失效：历史最佳出现在第一步，而第一步从零位移出发只能精确到达这三类；四步累计后九个代表值均可到达。

训练日志中，`ep_rew_mean` 从约 `-4.33` 改善到 `-4.15`，最终 `explained_variance=0.992`；但 `approx_kl` 约为 `0.18–0.30`、`clip_fraction` 约为 `0.66–0.71`，且策略熵接近 242 个三分类动作的理论最大值 `242×ln(3)≈265.86`。这些是需要在多 seed 运行中继续观察的稳定性信号，不能单独当作失败证据。

## 当前质量门判断

| 门槛 | 配置要求 | 当前结果 | 状态 |
|---|---:|---:|---|
| 相对初始改善 | `≥1%` | `14.1968%` | 通过 |
| 相对启发式损失 | `≤0%` | `+0.3550%` | 未通过 |
| 单一类别占比 | `≤0.95` | `0.3719` | 通过 |
| 三 seed 损失 CV | `≤0.20` | 只有 seed 0 | 未验证 |
| validation 决策 | 只用 M1_test7/8 | 当前是 train 的 M1_test1 | 不适用 |

因此，当前最准确的表述是“单图诊断接近启发式并显著优于初始掩模”，不能表述为“PPO 已正式 accepted”。

## 可直接用于汇报的表述

> 我们完成了基于只读 OpenILT 的 SimpleOPC 多步 PPO v3 适配，并把训练与验收统一为论文形式的 `L2+100×EPE+PVB` 目标。在 M1_test1 的 10000 timestep 单种子诊断中，PPO 将总损失相对初始掩模降低了 14.20%，最佳 Recipe 在 242 个边段上产生了 -10、0 和 +10 nm 三类非坍缩动作。PPO 与启发式基线仅相差 0.355%，说明策略已经学到有效的物理优化信号，但目前仍是单版图、单种子结果，尚未通过正式 validation 和多种子稳定性验收。

## GPU 迁移判断

当前一次 10000 timestep 模型耗时约 `1787.6 s`。按相同耗时线性估算，十图、三 seed 共 30 个模型约需 `14.90 h`，尚未计入不同版图复杂度和外围开销。

AutoDL 将当前设备列为 `4080(S)-32G` 性能型规格；这不是 NVIDIA 标准零售 RTX 4080 的 16GB 配置，因此真实设备、功率和虚拟化状态必须以 `nvidia-smi` 为准。AutoDL 当前页面列出的理论单精度性能为：A800-80GB `19.5 TFLOPS`，RTX PRO 6000 Blackwell 96GB `126.0 TFLOPS`。NVIDIA 给出的 RTX PRO 6000 Blackwell Workstation Edition规格为 96GB GDDR7 ECC、`125 TFLOPS` FP32、`1792 GB/s` 带宽。

对本项目的建议分两层：

1. **只比较潜在速度：选 RTX PRO 6000 Blackwell。** 当前任务是单 GPU、FP32 为主的小 PPO 网络加 OpenILT 光刻计算，32GB 已能运行，A800 的 80GB 容量不会自动转化为速度；PRO 6000 的单精度吞吐明显更高。
2. **只比较迁移风险：选 A800 80GB。** 项目当前锁定 Python 3.8、PyTorch `2.0.1+cu118`，Ampere A800 可沿用；Blackwell 首次由 CUDA 12.8 支持，PyTorch 从 2.7 的 CUDA 12.8 wheel 开始正式支持，因此 PRO 6000 需要新建 Python 3.10+ / PyTorch 2.7+cu128 环境，并重新运行全部测试和 256/2000 timestep 基准。

不能根据峰值 TFLOPS宣称 PRO 6000 会获得固定倍数加速：OpenILT 的实际瓶颈可能是内存访问、FFT/卷积、CPU 预处理或 Python 调度。正确做法是在新卡上先跑相同配置的短基准，比较 `fps` 和 `elapsed_seconds`，再外推完整时间。

参考资料：

- [NVIDIA RTX PRO 6000 Blackwell 官方规格](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [NVIDIA CUDA 架构兼容矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
- [PyTorch 2.7：Blackwell 与 CUDA 12.8 支持](https://pytorch.org/blog/pytorch-2-7/)
- [AutoDL 当前 GPU 规格与价格页](https://www.autodl.com/home)
- [AutoDL 4080(S)-32G 与 PRO6000-96G 规格标识](https://www.autodl.com/docs/instance_pro_api/)

## 方法学风险扫描

- Coverage: `11/11` fallacy types checked

| 风险类型 | 结论 | 说明 |
|---|---|---|
| Simpson's paradox | 不适用 | 当前只有一个 clip 和一个 seed，没有可聚合分组。 |
| Ecological fallacy | 未发现 | 没有从版图级指标推断个体人群属性。 |
| Berkson's paradox | 未发现 | 当前不是相关性抽样研究。 |
| Collider bias | 未发现 | 未进行含控制变量的因果回归。 |
| Base-rate neglect | 不适用 | 当前不是诊断分类率报告。 |
| Regression to the mean | 未发现 | 版图不是因极端表现而从更大样本中筛选。 |
| Survivorship bias | 低风险 | v2 失败、v3 冒烟和完整诊断均被保留，没有只报告成功值。 |
| Look-elsewhere effect | 注意 | 只观察了一个 seed；不得选择该结果代表总体稳定性。 |
| Garden of forking paths | 注意 | 10nm×4、10000 timestep 等是公开细节缺失后的实现选择，应持续披露。 |
| Correlation ≠ causation | 不适用 | 当前报告的是同版图受控算法结果，没有作人群因果推断。 |
| Reverse causality | 不适用 | 当前不是观察性相关研究。 |

## 可复现性结论

- Method: 云端产物结构、模型 ZIP、JSON、模型哈希、OpenILT commit 和 tracked diff 已由用户执行脚本核验；本地未重新运行 GPU 实验。
- Verdict: `PARTIALLY_REPRODUCIBLE`。
- 已验证：环境/损失版本、10000 timestep、轨迹公式、模型哈希一致、OpenILT 无改动。
- 未验证：相同 seed 的独立重复运行、多 seed 稳定性、validation/test 质量和跨 GPU 数值一致性。

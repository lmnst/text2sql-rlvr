# 最终实验报告

## 研究问题

本项目验证两件事：

1. Qwen3-1.7B 的 Text-to-SQL 主要受模型能力、schema 干扰还是上下文截断限制；
2. 在 prompt 与 reward 口径修正后，普通 GRPO 能否在强 SFT 起点上提供额外收益。

所有训练 reward 只使用 BIRD train 的 gold SQL 和对应数据库。train 中固定切出 val 用于诊断与
checkpoint 选择；BIRD dev 没有被用于调参或最终表格。

## 系统实现

完整链路包括：

- BIRD SQLite 数据发现、校验和固定 train/val 切分；
- 与训练共用的 prompt 构造和 Qwen3 chat template，默认关闭 thinking；
- full、linked、oracle 三种 schema 模式，其中 oracle 只用于诊断；
- Qwen3-1.7B LoRA SFT、vLLM 推理和 verl GRPO；
- 只读 SQLite 执行：immutable/read-only URI、authorizer、单语句验证、硬超时、结果上限、
  连接复用和执行缓存；
- BIRD official EX 与保留重复行/列数的 strict EX；
- 逐题 outcomes、自动实验台账和 verl `.pt` LoRA checkpoint 到 Hugging Face 模型的导出。

## 可报告结果

| 阶段 | 划分 | n | official EX | strict EX | run_id |
|---|---|---:|---:|---:|---|
| Qwen3-1.7B baseline | Mini-Dev | 500 | 19.80% | 17.40% | `43bb66bbe1e8` |
| LoRA SFT | Mini-Dev | 500 | 34.60% | 30.00% | `f1ef9b247255` |
| 强 SFT + full schema | 固定 train-val | 788 | 32.87% | 28.68% | `74d4d83c087c` |
| 强 SFT + linked schema | 固定 train-val | 788 | 37.94% | 33.63% | `3b9e91f891d8` |
| official-reward GRPO + linked schema | 固定 train-val | 788 | 38.20% | 33.88% | `53d1586c7d33` |

Mini-Dev 表示独立的 500 题公开小型开发集；“固定 train-val”表示从 BIRD train 固定切出的
验证集。不同划分之间不做直接增量归因。

## 实验一：schema 选择

### 设计

full 与 linked 使用同一个强 SFT checkpoint、同一固定 val 788、temperature=0、top_p=1、seed=0、
max_new_tokens=512，并关闭 thinking。唯一变量是 prompt 中保留的 schema。

tokenizer 诊断显示 full prompt 没有超过 8192 上限，因此“关键问题被截断”不是主要解释。
linked 只用问题和 evidence 做词面匹配，并补充外键相邻表，不读取 gold SQL。

### 结果

official EX 从 32.87%（259/788）提高到 37.94%（299/788），增加 5.07 个百分点；strict EX
从 28.68%（226/788）提高到 33.63%（265/788），增加 4.95 个百分点。

逐题配对：

| 指标 | 两者都对 | linked 新增答对 | linked 丢失 | 两者都错 | 净增 |
|---|---:|---:|---:|---:|---:|
| official | 223 | 76 | 36 | 453 | +40 |
| strict | 194 | 71 | 32 | 491 | +39 |

两种指标同向提升且净增接近，说明收益不是单纯利用 official 忽略重复行。结论是：对 1.7B 模型，
大量无关 schema 的干扰比上下文截断更值得优先处理。

### 限制

linked 是轻量词面方法，仍会漏必要表、漏多跳外键桥接表，也会保留大量干扰表。它证明 schema
筛选方向有效，但不代表 schema linking 已解决。oracle 结果只说明诊断上界，不可用于训练或对外成绩。

## 实验二：official-reward 普通 GRPO

### 设计

- 起点：与 linked SFT 对照相同的强 SFT 模型；
- 训练数据：BIRD train 固定 500 prompt，seed=0，与 val 200 不重叠；
- prompt：linked schema，thinking 关闭；
- reward：BIRD official EX 为主奖励，strict 只记录；
- 普通 GRPO：LoRA rank 32 / alpha 64，rollout n=4，30 step；
- checkpoint：只根据固定 val 200 选择，不读取完整 val 788 的结果挑点。

val 200 上的 SFT 起点为 39.5%，step 10、20、30 均为 40.0%。按预设的“最高分并列取最早”
规则选择 step 10。训练过程没有 prompt 截断、响应中止或 entropy collapse。

### 结果

完整固定 val 788 上，linked SFT 为 37.94% official / 33.63% strict，GRPO 为 38.20% / 33.88%。

逐题配对：

| 指标 | 两者都对 | GRPO 新增答对 | GRPO 丢失 | 两者都错 | 净增 |
|---|---:|---:|---:|---:|---:|
| official | 289 | 12 | 10 | 477 | +2 |
| strict | 256 | 11 | 9 | 512 | +2 |

净增 2/788，即 +0.26 个百分点。这个幅度和 gain/loss 交换都不足以证明稳定提升，因此正式结论是
“没有明确额外收益”。不把 38.20% 相对 37.94% 包装成 GRPO 成功。

## 为什么 GRPO 没有明显提升

以下是与证据一致、但尚未全部验证的解释：

1. **二值执行奖励稀疏。** SQL 只有完整执行结果正确才得 1 分。n=4 时，如果一个 group 全错或
   全对，组内相对 advantage 为零；有效 group 比例尚未离线统计。
2. **训练规模有限。** 本次只使用一个 seed、500 prompt、30 step 和 LoRA，适合验证方向，不能
   覆盖更大规模训练的可能性。
3. **模型能力上限。** linked schema 解决的是输入干扰，复杂 JOIN、聚合、子查询和组合条件仍要求
   1.7B 模型本身偶尔探索到正确答案，GRPO 不能从全错采样中凭空学习完整解法。
4. **official reward 粗粒度。** 它与最终指标对齐，但错误 SQL 与接近正确的 SQL 都是 0，且集合
   比较可能接受某些语义不完整但当前数据库结果相同的查询。
5. **linker 仍不完美。** 必要表缺失时 RL 无法补回 prompt 中不存在的信息，干扰表过多时探索空间
   仍然较大。

当前证据不支持“代码坏了”：训练跑满、reward 接入已校验、checkpoint 精确重建、确定性评测完成，
且没有截断、超时或输出损坏。更合理的说法是，本次配置下可用学习信号不足以产生可确认的泛化收益。

## 作废或仅供诊断的历史结果

旧 strict-reward GRPO、thinking 模板不一致的训练、dirty 工作树上的 val 200 schema 对照都保留在
`docs/PROGRESS.md` 作为排障历史，但不进入本报告主表。它们不能与最终强 SFT 起点做公平归因。

## 复现与证据

- 正式台账：`results/runs.jsonl`；
- 最终训练配置：`configs/grpo/run_grpo.sh`；
- checkpoint 导出：`scripts/convert_checkpoint.py`；
- 逐题生成与评测：`scripts/generate.py`、`scripts/evaluate.py`；
- linker/token 诊断：`scripts/diagnose_prompts.py`；
- 全部里程碑与订正：`docs/PROGRESS.md`；
- 云端产物路径与磁盘状态：`HANDOFF.md`。

最终实验记录提交为 `e945064`。本地测试在该提交后运行通过。README、简历或面试中出现的主表
数字必须按 run_id 回到 `results/runs.jsonl`，不能从聊天记录或记忆补录。

## 项目最终结论

项目完成了可复现的 Text-to-SQL RLVR 工程闭环，并通过公平实验识别出真正有效的改动：减少无关
schema 干扰。普通 GRPO 在本次小模型、小数据、稀疏二值奖励设置下没有明确额外收益。这个负结果
限定了当前方法的边界，也为后续是否投入 Dynamic Sampling、更强模型或更密集反馈提供了依据。

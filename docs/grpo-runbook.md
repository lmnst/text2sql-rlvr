# GRPO 最终运行手册

这份手册记录已经跑通的普通 GRPO 路径，不再是早期的换卡计划。历史 OOM、thinking 模板错误、
旧 strict-reward 训练和每次订正保留在 `docs/PROGRESS.md`，不作为当前默认配置。

## 最终结论

最终实验使用 linked schema 和 BIRD official EX 主奖励，在固定 train-val 788 上从 SFT 的
37.94% official EX 变为 GRPO 的 38.20%。这个变化按“没有明确额外收益”处理。

GRPO 链路本身已经验证完成：训练、定期 val、LoRA checkpoint、verl 0.9 `.pt` 权重重建、
Hugging Face 合并模型、vLLM 服务、确定性生成和双指标评测均已跑通。

## GRPO 在这里做什么

同一道题生成多个 SQL，分别在只读数据库上执行。BIRD official EX 正确记 1，错误记 0；GRPO
根据同组样本的相对奖励更新策略。因此一组样本全 0 或全 1 时没有组内排序信号，只有同时包含
正确和错误答案的 mixed group 能提供直接相对优势。

SFT 是模仿 gold SQL；GRPO 允许模型使用与 gold 写法不同、但执行结果等价的 SQL。代价是
execution reward 只有整题 0/1，不能告诉模型错在表、JOIN 还是过滤条件。

## 当前奖励口径

- 主奖励：BIRD official Execution Accuracy；
- 监控指标：strict execution equivalence；
- `format_bonus=0`；
- `execution_bonus=0`；
- 训练期 SQL timeout 默认 10 秒；
- rollout 逐条写入 `TEXT2SQL_ROLLOUT_LOG`。

为什么不是 strict 主奖励：最终 checkpoint 和对外成绩按 BIRD official 选择，训练目标必须与考试
口径一致。为什么还保留 strict：official 集合比较会忽略重复行，strict 能发现某些官方判对但结果
语义不完整的查询。两者职责不同。

## 已验证环境

依赖和 Blackwell workaround 以 `requirements-train.txt` 为准：

- Linux + 单张 Blackwell GPU；
- torch 2.11.0+cu130；
- vLLM 0.24.0；
- transformers 5.5.3；
- verl commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`。

不要在已工作的环境里随意升级其中一个包。verl、vLLM、transformers 和 torch 的接口强耦合，
单独升级会把已确认的配置问题重新引入。

## 训练前检查

### 1. 本地测试

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

### 2. 数据纪律

- reward 只能读取 BIRD train gold SQL 和 train 数据库；
- train/val 固定切分和 seed 必须写入 manifest；
- val 用于 checkpoint 选择；
- BIRD dev 不参与训练、调参或 early stopping。

linked-schema RL 数据由同一份 prompt 代码生成：

```bash
python scripts/build_rl_data.py \
  --root /root/autodl-tmp \
  --train /path/to/fixed_train.json \
  --val /path/to/fixed_val.json \
  --out-dir /root/autodl-tmp/rl-linked \
  --schema-mode linked \
  --val-subset 200 \
  --val-seed 0 \
  --val-subset-json /root/val200.json
```

不要重新随机切分后与旧结果比较。

### 3. 奖励自检

在 GPU 训练前单独调用 `scripts/verl_reward.py`，确认：

- 正确 SQL：official reward 1；
- `SELECT 1`：0；
- 非 SQL：0；
- 写库语句：拒绝；
- strict 与 official 都写入 rollout 日志。

### 4. 显存与残留进程

每次失败后先检查 `nvidia-smi`。Ray/vLLM worker 可能不会随主进程退出；残留显存会让下一次运行
报出误导性的 NCCL 或微小分配 OOM。确认没有需要保留的任务后再清理：

```bash
ray stop --force
pkill -9 -f 'ray::'
pkill -9 -f vllm
nvidia-smi
```

## 三步 smoke

先用 `configs/grpo/run_smoke.sh`。目标只包括：

1. verl 能读取 parquet；
2. custom reward 确实被 `reward.custom_reward_function.path/name` 加载；
3. actor、vLLM rollout、SQL reward、反向传播和 checkpoint 能走完；
4. rollout 日志中确实有 official/strict 字段。

smoke 跑通不代表超参数有效，只代表链路可用。

## 正式训练

当前入口是 `configs/grpo/run_grpo.sh`。最终公平实验覆盖的关键值：

| 参数 | 值 |
|---|---|
| 起点 | 强 SFT 合并模型 |
| schema | linked |
| train prompt | 固定 500，seed=0 |
| val | 固定 200，与 train 不重叠 |
| rollout n | 4 |
| train batch | 32 |
| LoRA | rank 32 / alpha 64 / all-linear |
| actor lr | 1e-6 |
| steps | 30 |
| save / test | 每 10 step |
| thinking | false |
| max prompt / response | 8192 / 512 |
| Dynamic Sampling | 未使用 |

示例：

```bash
MODEL=/root/autodl-tmp/Qwen3-1.7B-sft-strong-merged-f78ab16a \
DATA=/root/autodl-tmp/rl-linked \
OUT=/root/autodl-tmp/out/grpo-linked-official \
EXP=grpo-linked-official \
TOTAL_STEPS=30 \
bash configs/grpo/run_grpo.sh
```

运行开始后，从 resolved config 核对以下项目，不要只相信 shell 参数：

- `enable_thinking=False`；
- `tensor_model_parallel_size=1`；
- `rollout.n=4`；
- custom reward 路径和函数名；
- train/val parquet 的实际路径；
- `TEXT2SQL_REWARD_OFFICIAL=1`；
- 输出目录没有复用旧实验。

## 训练时看什么

| 信号 | 正常含义 | 异常解释 |
|---|---|---|
| `critic/rewards/mean` | 当前 batch 的 rollout 正确率 | 不能单独证明泛化 |
| `actor/entropy` | 输出分布仍有探索 | 持续接近 0 才怀疑 collapse |
| prompt/response clip ratio | 应为 0 | 非 0 说明长度配置破坏实验 |
| aborted ratio | 应接近 0 | 生成或服务不稳定 |
| val official | checkpoint 选择依据 | 只能在固定 val 上比较 |
| group reward variance | 有效 GRPO 信号 | 全 0/全 1 group 不提供相对排序 |

最终 run 没有发生 prompt 截断、响应中止或 entropy collapse；但 val 在 step 10 后没有继续改善。

## checkpoint 选择与导出

最终 run 的 step 10/20/30 在固定 val 200 上并列，按预先规则选最早的 step 10。不能在完整
val 788 上重新挑 checkpoint。

verl 0.9 保存的是带 PEFT LoRA 参数的 `.pt` state dict，不是可直接给 vLLM 的 Hugging Face 目录。
必须用真实 GRPO 起点模型重建 LoRA，然后精确检查 missing/unexpected keys，再 merge：

```bash
python scripts/convert_checkpoint.py \
  --ckpt /root/autodl-tmp/out/grpo-linked-official/global_step_10/actor \
  --base /root/autodl-tmp/Qwen3-1.7B-sft-strong-merged-f78ab16a \
  --out /root/autodl-tmp/Qwen3-1.7B-grpo-linked-step10
```

如果报 `No space left on device`，先删失败留下的半截输出和明确可重建的旧合并模型；不要删除原始
verl checkpoint 或唯一的 SFT 起点。最终实验第一次后处理失败就是数据盘写满，不是权重损坏。

## 确定性评测

Blackwell 上启动 vLLM：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
vllm serve /root/autodl-tmp/Qwen3-1.7B-grpo-linked-step10 \
  --served-model-name text2sql-grpo \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8704 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

生成与评测：

```bash
python scripts/generate.py \
  --root /root/autodl-tmp \
  --questions /path/to/fixed_val788.json \
  --split train \
  --schema-mode linked \
  --model text2sql-grpo \
  --temperature 0 \
  --top-p 1 \
  --seed 0 \
  --max-tokens 512 \
  --out /path/to/grpo-linked.jsonl

python scripts/evaluate.py \
  --root /root/autodl-tmp \
  --questions /path/to/fixed_val788.json \
  --split train \
  --predictions /path/to/grpo-linked.jsonl \
  --stage grpo \
  --checkpoint /root/autodl-tmp/Qwen3-1.7B-grpo-linked-step10 \
  --config-path configs/grpo/run_grpo.sh \
  --outcomes /path/to/grpo-linked-outcomes.jsonl
```

评测前要求仓库干净。`evaluate.py` 会自动记录 Git SHA、dirty state、配置哈希、解码参数和结果。

## 当前不建议继续做什么

- 不直接增加训练步数；
- 不先上 Dynamic Sampling；
- 不用完整 val 788 反复挑 checkpoint；
- 不读取 BIRD dev 来美化结果；
- 不运行未设计清楚的 execution bonus 对照；
- 不为了修一个配置问题升级整套依赖。

如果以后继续，第一步应离线统计 group reward variance。只有确认大量 group 为全 0/全 1，才有
证据支持 Dynamic Sampling 或增加 rollout n；否则先查更新量与 train/val 泛化差异。

## 云端最终产物

完整路径、磁盘状态和保留/清理建议见仓库根目录 `HANDOFF.md`。实验结束后 vLLM 和训练进程都已
退出；AutoDL 按开机时间计费，确认产物写完后应立即关机。

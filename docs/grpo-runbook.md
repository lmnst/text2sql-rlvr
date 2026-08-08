# GRPO：让模型自己试错

到这一步为止的成绩：

| | 未训练 | SFT |
|---|---|---|
| val（788 题） | 21.83% | 33.38% |
| mini-dev（500 题） | 19.80% | 34.60% |

## 这一步和 SFT 有什么不同

SFT 是**照着标准答案抄**。GRPO 是：同一道题让模型写 8 个不同的答案，
把 8 个答案分别拿到数据库上执行，对的给 1 分错的给 0 分，
然后**鼓励得分高于这组平均分的写法，抑制低于平均的**。

关键区别在于，模型可以写出和标准答案完全不同、但执行结果一样正确的 SQL，照样得满分。
SFT 做不到这一点——写法不同就算错。

奖励不来自人工标注，也不来自另一个模型打分，而是来自**真实数据库的执行结果**。
这就是这个项目名字里 RLVR 的含义。

## 奖励怎么设计的，以及为什么

**用严格验证器，不用 BIRD 官方口径。** 这不是偏好问题，是里程碑 9 量出来的：

SFT 之后有 25 道题，官方口径给分而严格口径不给。全部是同一个形状——
标准答案写了 `SELECT DISTINCT` 返回 21 行，模型漏写返回 933 行，
官方口径两边各自去重后判为相同，照给满分。

**如果拿官方口径当奖励，模型在"要不要去重"上完全不受约束**：
返回 933 行和返回正确的 21 行得分一模一样。强化学习会放大一切没有压力的方向。

**默认不给部分分。** 奖励函数里有两个开关（`format_bonus`、`execution_bonus`），
默认都是 0。`execution_bonus` 是"SQL 能跑就给分"，而 `SELECT 1` 就能跑——
这是个明摆着的漏洞。

**留着这个开关不是疏忽，它就是实验本身**：打开它，量模型多快找到这个漏洞、
输出退化成什么样；关掉它，再量一次。两种设置都是一等公民，都会被记录。

---

## 第 0 步：本地已经做完

```
data/processed/rl/train.parquet    8191 条
data/processed/rl/val.parquet       200 条（训练中定期评测用）
configs/grpo/dataset.json          这份数据怎么来的
scripts/verl_reward.py             奖励函数，verl 调用它
configs/grpo/run_grpo.sh           正式训练
configs/grpo/run_smoke.sh          三步冒烟
```

奖励函数有 30 多条单元测试，逐一钉死了每种输出形状的得分：
正确答案 1 分；写法不同但结果对，也是 1 分；`SELECT 1` 零分；
漏写 DISTINCT 零分（并记录"官方口径本会给分"）；企图写库的直接拒绝。

## 第 1 步：把 SFT 的适配器合并成完整模型

GRPO 要在 SFT 的基础上继续训。适配器套适配器容易出问题，先合并成一个完整模型：

```bash
llamafactory-cli export --model_name_or_path /root/autodl-tmp/Qwen3-1.7B --adapter_name_or_path /root/autodl-tmp/out/qwen3-1.7b-sft-lora --template qwen3 --finetuning_type lora --export_dir /root/autodl-tmp/Qwen3-1.7B-sft-merged --export_size 5
```

合并完确认目录里有 `config.json` 和 `.safetensors`。

## 第 2 步：装 verl，先跑通它自带的例子

**这一步不要跳。** 它把"我的奖励函数写错了"和"我的环境装坏了"这两种情况分开——
这两者的报错长得一模一样。

```bash
git clone https://github.com/volcengine/verl /root/verl
cd /root/verl && git checkout 4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71 && pip install -e .
```

这个 commit 上有个新加的数据组件 `transfer_queue`，任务运行器会 import 它，
但它还不是依赖。**用命令行关掉即可**，见第 5 步。

### 这一段是踩过两次坑之后写的

第一次：手册写的是 `git clone`，拿到 HEAD，跑到一半死于
`ModuleNotFoundError: No module named 'transfer_queue'`。
AGENTS.md 从第一天就要求钉版本，规则写了没照做。

第二次（更糟）：为了"修好"它去 checkout 了 v0.8.0 标签，
`pip install -e .` 把 numpy 从 2.3.2 降到 1.26.4，
连带 scipy、opencv、transformers 全部报错——
**用一个只有一处导入问题的可用环境，换来了一个彻底坏掉的环境。**

真正的教训不是"要钉版本"，而是：**在一个已经能跑的环境里改版本，
本身就是一次依赖变更。** 恢复方式是回到原 commit 并显式保护依赖：

```bash
cd /root/verl && git checkout 4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71
pip install --no-deps -e .
pip install "numpy==2.3.2"
```

`--no-deps` 是重点：只刷新 verl 本身，不让它重新解析并改动别的包。

按 verl 官方 README 跑通它自带的 gsm8k GRPO 例子，再往下走。

## 第 3 步：传数据和奖励函数

本地：

```bash
scp -P 端口号 -r data/processed/rl root@你的地址:/root/autodl-tmp/
```

```bash
scp -P 端口号 scripts/verl_reward.py scripts/analyze_rollouts.py configs/grpo/run_grpo.sh configs/grpo/run_smoke.sh root@你的地址:/root/autodl-tmp/
```

**还要把 `src/` 传上去。** 奖励函数不是一个独立文件，它 import 了整个包
（执行沙箱、结果比较、SQL 解析）：

```bash
scp -P 端口号 -r src root@你的地址:/root/autodl-tmp/
```

传完在云上确认：

```bash
ls /root/autodl-tmp/src/text2sql_rlvr/
```

脚本会自己在几个位置找这个包（同级 `src/`、上一层 `src/`、以及同目录），
都找不到时会打印它找过哪些路径。也可以用 `TEXT2SQL_SRC` 指定。

奖励函数要执行 SQL，所以**这次数据库必须传**（train_databases，约 30 GB）。
如果嫌大，也可以在云上直接下载 BIRD 训练集，比从德国上传快得多。

## 第 4 步：单独测奖励函数（不启动训练）

奖励函数自带一个自检入口：

```bash
cd /root/autodl-tmp && python verl_reward.py /root/autodl-tmp/bird/train/train_databases california_schools "SELECT COUNT(*) FROM schools"
```

（数据库名和 SQL 换成 train 里真实存在的）

期望输出：

```
perfect                reward=1.0
degenerate SELECT 1    reward=0.0
not sql                reward=0.0
write attempt          reward=0.0
```

**四行不是这个结果就停下。** 奖励错了，后面训练越久错得越离谱，而且不会报错。

## 第 5 步：三步冒烟

```bash
chmod +x /root/autodl-tmp/run_grpo.sh /root/autodl-tmp/run_smoke.sh
```

```bash
/root/autodl-tmp/run_smoke.sh ++transfer_queue.enable=False
```

**两个加号。** Hydra 里 `+key=value` 表示"新增一个原本不存在的键"，
而这个键已经存在，覆盖已有的键要用 `++`。写成一个加号会报
`Could not append to config. An item is already at ...`，
报错里其实已经给出了答案。

这一步专门用来发现配置问题。verl 的配置在版本间会变，前几次报错是正常的，
按下表自己处理，处理不了再贴出来。

| 报错 | 含义 | 处理 |
|---|---|---|
| `Please set at least one of 'X' or 'X_per_gpu'` | 这个键存在但默认为空，必须显式设 | 在脚本里加 `X_per_gpu=1` |
| `Could not override 'xxx'` | 这个键在当前版本**不存在** | 去 verl 的 `trainer/config/` 里找对应新名字 |
| `CUDA out of memory` | 显存不够 | 依次降 `rollout.gpu_memory_utilization`、`rollout.n`、`data.train_batch_size` |
| 卡在加载模型不动 | 通常在编译或初始化 vLLM | 等几分钟；超过十分钟再看日志 |

**注意区分前两种。** 第一种说明键名是对的、只是没赋值，加上即可；
第二种说明键名在这个版本里改掉了，得去源码里查。二者的处理方式完全不同。

`log_prob_micro_batch_size_per_gpu` 在 `actor_rollout_ref` 下有三处
（`actor`、`rollout`、`ref`），三处都要设，缺一处就报第一种错。

跑完检查奖励日志确实产生了：

```bash
wc -l /root/autodl-tmp/out/grpo_smoke/rollouts.jsonl
```

应该有几十行，每行是一次 rollout 的完整打分记录。

## 第 6 步：正式训练

```bash
/root/autodl-tmp/run_grpo.sh
```

盯三个数：

| 指标 | 期望 | 异常信号 |
|---|---|---|
| `critic/rewards/mean` | 缓慢上升 | 暴涨到接近 1，多半是奖励算错了 |
| `actor/entropy` | 缓慢下降 | 掉到 0.1 以下并持续，是熵坍缩，输出多样性没了 |
| `val` 分数 | 高于 33.38% | 长期不动说明没学到东西 |

**熵坍缩是这一步最典型的失败**，也正是后面消融实验要研究的现象，
所以看到它不算灾难，记下发生在第几步。

## 第 7 步：评测

和 SFT 一样：先 val，再 mini-dev。检查点导出后用 vLLM 起服务，
然后本地跑 `generate.py` 和 `evaluate.py`，`--stage grpo`。

## 第 8 步：反作弊实验（这是这个项目最值钱的部分）

在同样的配置下，**只改一个环境变量**再训一次：

```bash
TEXT2SQL_REWARD_EXEC=0.3 EXP=grpo_naive OUT=/root/autodl-tmp/out/grpo_naive /root/autodl-tmp/run_grpo.sh
```

这是给"SQL 能跑就给 0.3 分"。然后对比两次的 `rollouts.jsonl`：

```bash
python scripts/analyze_rollouts.py --rollouts /path/to/rollouts.jsonl
```

要看的是 `no_from_rate`（输出 `SELECT 1` 这种不读库的查询的比例）
和 `hack_rate`（拿到奖励但严格口径判错的比例）随训练步数怎么变。

**如果朴素奖励下这两个数明显上升、严格奖励下不上升，你就有了一个完整的、
自己造出来又自己测出来的 reward hacking 案例。** 这比引用别人论文里的现象值钱得多。

---

## 成本预估

| | 估算 |
|---|---|
| 一步 GRPO（32 题 × 8 个答案 = 256 次生成 + 256 次 SQL 执行） | 30–90 秒 |
| 300 步 | 3–7 小时 |
| 两次训练（严格 + 朴素） | 6–14 小时，一百多块 |

**这些是估算，没有实测。** 前面 SFT 我估 2–4 小时实际 30 分钟，估算就是估算。
冒烟跑完看一步实际多久，再决定跑多少步。

## 可信度

| 步骤 | 依据 |
|---|---|
| 奖励函数本身 | **30+ 条单元测试，本地全绿** |
| 训练数据格式 | 本地生成并读回验证 |
| `max_prompt_length=8192` | **实测**：1288 次真实生成中最长 6731 token |
| verl 配置项名称 | **从未验证**，最可能卡在这里 |
| 显存能否放下 | 未验证，放不下就调低 `gpu_memory_utilization` 和 `rollout.n` |
| 超参数 | 有依据的起点，非实测最优 |

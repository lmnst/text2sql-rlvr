# SFT：第一次真正训练模型

目标：用 8191 道训练题教会 Qwen3-1.7B 按 BIRD 的风格写 SQL，然后看分数比基线高多少。

基线（未训练，mini-dev 500 题，提示词 v1）：**官方口径 19.80%**。

## 每一步的可信度

这份手册不是实测记录，是计划。下表说明每一步的依据强度，
碰到与描述不符的情况时先看这里，再决定是照着改还是停下来问。

| 步骤 | 依据 | 出问题的概率 |
|---|---|---|
| 1 生成训练数据 | **已在本地实际跑过**，输出与统计都是真的 | 无 |
| 2 传文件 | 普通 scp | 低（易错点是跑错机器，见步骤内说明） |
| 3 装框架、注册数据集 | 安装已实测可用；**数据集注册格式未验证** | 中 |
| 4 冒烟训练 | 配置从未被框架加载过 | **中高** |
| 5 验证对话模板 | transformers 标准接口 | 低 |
| 6 正式训练 | 超参是有依据的起点，**非实测**；耗时为估算 | 中 |
| 7 起服务加载适配器 | vLLM 的 LoRA 参数**未实测** | **中高** |
| 8 生成与评测 | **这两个脚本已跑通过完整基线**，命令是可靠的 | 低 |

第 4、7 步是最可能卡住的地方，都是"框架参数对不对"的问题，报错原文贴出来就能定位。
第 1、8 步是实测过的，可以放心。

## 先理解这一步在干什么

前面的基线是"模型凭自己的常识做题"。SFT 是给它看 8191 道题的标准答案，让它模仿。
这不是强化学习——没有奖励、没有试错，就是照着抄，学的是**格式和套路**。

真正的强化学习在下一步。

## 唯一需要小心的地方

**训练时喂给模型的输入，必须和评测时一模一样。**

如果训练时用的提示词格式和评测时差一点，模型学会的是一种它以后再也见不到的输入格式，
分数就毫无意义。数据生成脚本已经用了和评测完全相同的代码路径来构造提示词，
但**对话模板**（把 system / user / assistant 拼成一个字符串的规则）是训练框架加的，
vLLM 在推理时也会加一次——这两次必须一致。

这是这一步唯一真正的风险，下面第 5 步专门验证它。

---

## 第 1 步：本地已经做完了

训练数据已生成：

```
data/processed/sft_train.jsonl     8191 条，66 个数据库
configs/sft/dataset.json           这份数据是怎么来的（提示词配置、长度统计、git 版本）
```

长度实测：一半的样本在 2954 字符以内，最长 14599 字符（约 4400 token）。
`cutoff_len` 设 6144 就零截断，配置里用了 8192 留余量。

**截断为什么危险**：被截掉的是序列末尾，而末尾正是答案。那些样本会教模型"什么都不要输出"。

## 第 2 步：把数据和配置传上去

**注意：scp 要在你自己电脑上跑，不是在 ssh 进去的云端终端里跑。**
提示符如果是 `root@autodl-container-...` 就说明跑错地方了。

先在**云上**建目录：

```bash
mkdir -p /root/autodl-tmp/sft_data
```

然后回到**本地**，另开一个 PowerShell 窗口（不要用打隧道那个）：

```bash
cd D:\Code\Demo\text2sql-rlvr
```

```bash
scp -P 端口号 data/processed/sft_train.jsonl configs/sft/dataset_info_entry.json root@你的地址:/root/autodl-tmp/sft_data/
```

```bash
scp -P 端口号 configs/sft/qwen3_1.7b_lora.yaml configs/sft/qwen3_1.7b_lora_smoke.yaml root@你的地址:/root/autodl-tmp/
```

一共约 32 MB。数据库不用传——SFT 阶段不执行 SQL。

传完在**云上**确认三个文件都到位，再往下走：

```bash
ls -la /root/autodl-tmp/sft_data/ /root/autodl-tmp/*.yaml
```

## 第 3 步：装 LLaMA-Factory 并注册数据集

```bash
pip install llamafactory[torch,metrics]
```

`dataset_info_entry.json` 本身就是一份完整可用的 `dataset_info.json`，在**云上**改个名即可：

```bash
mv /root/autodl-tmp/sft_data/dataset_info_entry.json /root/autodl-tmp/sft_data/dataset_info.json
```

## 第 4 步：先用 32 条样本跑两分钟

**不要直接开全量训练。**

```bash
llamafactory-cli train /root/autodl-tmp/qwen3_1.7b_lora_smoke.yaml
```

看到 loss 在下降、没有报错，就说明环境和数据格式都对。

**注意不要给这条命令加命令行参数。** llamafactory-cli 传了 yaml 之后，
会把 yaml 当作完整的参数集合，额外的 `--max_samples` 之类会被判为多余参数并报错
（`Some keys are not used by the HfArgumentParser`）。所以冒烟用的是一个单独的配置文件，
除了四行标了 `smoke only` 的之外，其余与正式配置逐行相同——
一个跑在不同配置上的冒烟测试，证明不了正式配置能跑。

配置里有两处标了 `VERIFY`，就是在这一步确认：

- `template: qwen3` —— 如果这个版本没有 qwen3 模板，会报错，改成 `qwen`
- `enable_thinking: false` —— 如果这个参数不被识别会报错，那就说明该版本用别的方式控制，
  贴报错给我

## 第 5 步：验证对话模板是否一致（最关键的一步）

这一步是防上面说的那个唯一风险。把脚本传上去（本地）：

```bash
scp -P 端口号 scripts/check_chat_template.py root@你的地址:/root/autodl-tmp/
```

在云上跑：

```bash
python /root/autodl-tmp/check_chat_template.py --model /root/autodl-tmp/Qwen3-1.7B --data /root/autodl-tmp/sft_data/sft_train.jsonl
```

**要看的不是"有没有 `<think>`"。** Qwen3 的非思考格式本来就包含一对空的
`<think></think>`，看到它是正常的。

要看的是最后一行是 `PASS` 还是 `FAIL`。判据是：
**推理时交给模型的那串字符，是否恰好是训练序列的前缀。**

- 是前缀 → 模型被训练成"从生成起点继续往下写"，两边严丝合缝，没问题。
- 不是前缀 → 推理时给模型的开头是它训练时从没见过的，学到的东西用不上。

`FAIL` 就停下，别开正式训练，把输出贴出来。

这个脚本只验证了分词器自带的模板（也就是 vLLM 推理时用的那套）。
**训练框架实际用的是不是同一套，还要看第 4 步冒烟训练打印出来的那个样例**——
LLaMA-Factory 会在开始训练前打印第一条样本的原文，对照着看 assistant 段长什么样。

## 第 6 步：正式训练

```bash
llamafactory-cli train /root/autodl-tmp/qwen3_1.7b_lora.yaml
```

先跑 1 个 epoch。8191 条样本，在一张 4090 上**实测约 30 分钟**（507 步，约 3.5 秒一步），
成本一块多。

（此处原先估的是 2–4 小时，差了四到八倍。估算就是估算，实测覆盖之。）

**先只跑 1 个 epoch，评测完再决定要不要第二个。** 既然一轮只要半小时，
加第二轮的成本可以忽略，更没有理由预先猜。

### 怎么看 loss

实测的下降形态：`1.74 → 0.46 → 0.28 → 0.24 → 0.22 → 0.22`（每点相隔 10 步）。

**前几步暴跌是正常的，而且不代表学到了东西。** loss 只计算答案部分，
而答案就是一句几十个 token 的 SQL，其中固定格式（代码块标记、结束符）占了相当比例，
几十步就学完了；表名列名基本是从提示词里抄过来的，模型做这件事很轻松。

所以这段暴跌反映的是"学会了长什么样"，不是"学会了怎么查数据"。

**关键提醒：loss 低不等于 SQL 对。** loss 衡量的是与标准答案逐字的相似度，
但同一个问题有无数种正确写法——写法不同则 loss 高而结果全对；
抄得很像但漏一个连接条件则 loss 很低而结果全错。

**唯一算数的指标是执行准确率。** loss 只用来判断训练有没有崩，不能用来判断好坏。

需要警惕的异常形态：一开始就接近 0（多半是数据或掩码有问题）、
长时间纹丝不动、或者中途突然飙升。出现这些停下来贴出来。

## 第 7 步：用训练好的模型起服务

LoRA 训练出来的是一个小的适配器，不是完整模型。vLLM 可以直接加载：

```bash
vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --enable-lora --max-lora-rank 32 --lora-modules sft=/root/autodl-tmp/out/qwen3-1.7b-sft-lora
```

**`--max-lora-rank 32` 不能省。** vLLM 这个参数默认是 16，而训练配置里用的秩是 32，
不显式抬高会在加载适配器时直接失败。两个数字必须一致：改了 `lora_rank` 就要同步改这里。

然后本地打隧道（和之前一样），调用时**模型名用 `sft`**（就是 `--lora-modules` 里等号左边那个）。

起来之后先确认适配器真的挂上了：

```bash
curl http://localhost:8000/v1/models
```

返回的列表里应该**同时**有 `Qwen3-1.7B` 和 `sft` 两个条目。只有前者说明适配器没加载成功，
这时候继续跑评测，测的是没训练过的原模型，分数会和基线一模一样——
这种失败不会报错，只会让你以为训练没效果。

## 第 8 步：先评验证集，再评 mini-dev

**顺序不能反。** 验证集用来判断训练有没有效、要不要再来一个 epoch；
mini-dev 只在最后确定下来之后测一次。

```bash
python scripts/generate.py --questions data/processed/val.json --split train --model sft --instruction-version v1 --out results/preds/val_sft.jsonl
```

```bash
python scripts/evaluate.py --questions data/processed/val.json --split train --predictions results/preds/val_sft.jsonl --stage sft --notes "SFT 1 epoch, val"
```

和基线比：**val 上未训练时是 21.83%**。

如果验证集有明显提升，再跑 mini-dev：

```bash
python scripts/generate.py --root data/bird --split mini_dev --model sft --instruction-version v1 --out results/preds/sft_minidev.jsonl
```

```bash
python scripts/evaluate.py --root data/bird --split mini_dev --predictions results/preds/sft_minidev.jsonl --stage sft --notes "SFT 1 epoch, mini-dev"
```

和基线比：**mini-dev 上未训练时是 19.80%**。

## 第 9 步：关机

---

## 预期与判断标准

| 情况 | 说明 |
|---|---|
| val 提升到 30% 以上 | 正常，SFT 主要修掉格式和方言问题，这部分收益很大 |
| val 提升不到 3 个点 | 训练没吃进去，检查第 5 步的模板一致性 |
| val 反而下降 | 大概率是模板不一致，模型学的格式和评测要的对不上 |
| 生成结果里 `sql` 字段大量为空 | 同上，模型不再输出 ```sql 代码块 |

**注意一件事**：SFT 的提升里，有一部分只是"学会了这是 SQLite、学会了输出格式"，
不是"更懂怎么查数据了"。里程碑 6 量过：基线里有 9.2% 的题纯粹败在方言和格式上。
所以看到提升别急着高兴，**评测报告里的 `prediction status` 那行——执行失败的数量掉了多少
——才说明修掉了多少格式问题**。这个区分在写简历时要讲清楚。

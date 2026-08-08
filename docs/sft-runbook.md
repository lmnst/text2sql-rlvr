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

这一步是防上面说的那个唯一风险。在云上：

```bash
python -c "
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-1.7B')
r = json.loads(open('/root/autodl-tmp/sft_data/sft_train.jsonl').readline())
s = tok.apply_chat_template(r['messages'][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(s[-300:]))
"
```

看最后 300 个字符。**关键是结尾不能有 `<think>`**。如果有，说明推理时模型会被要求先思考，
而训练数据里没有思考过程，两边对不上。出现这种情况把输出贴给我。

## 第 6 步：正式训练

```bash
llamafactory-cli train /root/autodl-tmp/qwen3_1.7b_lora.yaml
```

先跑 1 个 epoch。8191 条样本，4090 上估计 2–4 小时，十几块钱。

**先只跑 1 个 epoch，评测完再决定要不要第二个**，而不是先花两倍的钱、再发现一个就够了。

训练中盯 loss：应该从 1.5 左右稳步下降到 0.5 以下。如果一开始就接近 0，或者一直不动，
停下来贴给我看。

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

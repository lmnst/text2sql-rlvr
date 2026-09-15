# Schema selector：上 AutoDL 训练和评测的操作手册

目标：用 666 道题、2 个 epoch，给 Qwen3-1.7B 训一个 LoRA，让它看到问题和完整 schema 后
输出需要哪些表；然后在 selector 自己的 10 个 held-out 库（1598 题）和生成模型的固定
val 788 上，和现在的词面 linker 比"每题 gold 全保留率、准确率、平均选中表数"。

## 整体思路：云上只训练和起服务，评测在本地

和 `docs/gpu-runbook.md` 一样：租来的机器上只做两件事，训练 LoRA、跑 `vllm serve`。
评测脚本在你的 Windows 上跑，通过一条 SSH 隧道把请求发到云上。数据库、评测文件、结果
都留在本地，云上机器随时可以关。

整个过程会同时开三个窗口，先记住它们的分工：

| 窗口 | 在哪 | 干什么 |
|---|---|---|
| A：云端终端 | 本地 PowerShell 里 `ssh` 登进去，或 AutoDL 网页的 JupyterLab 终端 | 装环境、训练、起服务 |
| B：隧道 | 本地 PowerShell | 只跑一条 `ssh -L`，然后放着不动 |
| C：本地评测 | 本地 PowerShell，`cd D:\Code\Demo\text2sql-rlvr` | 跑 `scripts/evaluate_selector.py` 等 |

怎么分辨自己在哪：提示符是 `root@autodl-container-xxx:~#` 就是云端；是 `PS D:\...>` 就是本地。
**scp 和评测脚本都在本地跑，训练和 vllm 在云端跑。** 跑错地方是最常见的错误。

## 第 0 步：本地已经准备好的东西

这些已经生成，不用再做：

```text
data/processed/selector/v1/train.jsonl     666 条训练样本，2.1 MB   <- 唯一要传上去的数据
data/processed/selector/v1/heldout.jsonl   1598 题评测文件，留本地
data/processed/selector/v1/val788.jsonl    788 题评测文件，留本地
configs/selector/dataset_v1.json           数据怎么来的，以及词面 linker 的基线
scripts/train_sft.py                       训练脚本，不依赖本项目其他代码，单文件可传
scripts/check_chat_template.py             训练前验证对话模板，单文件可传
```

## 第 1 步：租机器

- 卡：一张 **RTX 4090（24 GB）** 足够。1.7B 模型加 LoRA，训练样本最长 3010 token，显存绰绰有余。
- 镜像：**PyTorch 2.x + CUDA 12.x** 基础镜像。如果镜像市场有带 vLLM 的，直接用更省事。
- **先用无卡模式开机**装环境、下模型，都弄好再关机换带卡开机。无卡模式每小时几毛钱。
- 系统盘只有 30 GB，所有东西放 `/root/autodl-tmp`。

开机后，AutoDL 控制台的实例列表里有一栏"SSH 登录指令"和"密码"，形如：

```text
ssh -p 12345 root@region-x.autodl.com
```

下面所有命令里的 `端口` 和 `地址` 就是这两个值，照抄替换。

如果是之前那台已经装好 vLLM 和 verl 的实例（`requirements-train.txt` 记录的那套），
第 4 步的 `pip install vllm` 跳过，不要动那套环境。

## 第 2 步：登录云端（窗口 A）

本地打开 PowerShell：

```bash
ssh -p 端口 root@地址
```

- 第一次连会问 `Are you sure you want to continue connecting (yes/no)?`，输 `yes` 回车。
- 然后要密码。**输密码时屏幕上什么都不显示**，这是正常的，输完直接回车。
- 提示符变成 `root@autodl-container-xxx:~#` 就是登进去了。

在云端建目录：

```bash
mkdir -p /root/autodl-tmp/selector /root/autodl-tmp/out
```

## 第 3 步：传文件（本地，另开一个 PowerShell）

**这一步在本地跑。** 新开一个 PowerShell 窗口（先别关窗口 A）：

```bash
cd D:\Code\Demo\text2sql-rlvr
```

```bash
scp -P 端口 data/processed/selector/v1/train.jsonl scripts/train_sft.py scripts/check_chat_template.py root@地址:/root/autodl-tmp/selector/
```

- `scp` 的端口参数是**大写** `-P`，`ssh` 的是小写 `-p`。
- 会再要一次密码。
- 一共约 2.2 MB，几秒钟。

传完回到窗口 A 确认三个文件都在：

```bash
ls -la /root/autodl-tmp/selector/
```

应该看到 `train.jsonl`（约 2.1 MB）、`train_sft.py`、`check_chat_template.py`。

不想用 scp 的话，AutoDL 网页里的 JupyterLab 文件浏览器可以直接拖拽上传，效果一样。

## 第 4 步：装环境、下模型（窗口 A，无卡模式下做）

```bash
pip install vllm
```

十几分钟正常。装完再装训练要的两个包：

```bash
pip install peft datasets accelerate
```

下模型，用魔搭，国内快：

```bash
pip install modelscope
```

```bash
modelscope download --model Qwen/Qwen3-1.7B --local_dir /root/autodl-tmp/Qwen3-1.7B
```

约 3.5 GB。下完确认：

```bash
ls -la /root/autodl-tmp/Qwen3-1.7B
```

要能看到 `config.json`、`tokenizer.json` 和 `.safetensors` 文件。

**做完这步：AutoDL 控制台关机，改成带卡开机，然后重新 ssh 登录。**

## 第 5 步：验证对话模板（窗口 A）

训练序列必须以推理时给模型的那串字符开头，否则训了白训。这个脚本只用 CPU：

```bash
python /root/autodl-tmp/selector/check_chat_template.py --model /root/autodl-tmp/Qwen3-1.7B --data /root/autodl-tmp/selector/train.jsonl
```

看最后一行是 `PASS` 还是 `FAIL`。`PASS` 才往下走；`FAIL` 把整段输出贴出来。
看到 `<think></think>` 是正常的，Qwen3 关闭思考模式时就长这样。

## 第 6 步：训练（窗口 A，带卡）

```bash
cd /root/autodl-tmp/selector
```

```bash
python -u train_sft.py --model /root/autodl-tmp/Qwen3-1.7B --data train.jsonl --out /root/autodl-tmp/out/selector-v1-lora --epochs 2 --max-length 4096 --grad-accum 8 2>&1 | tee train_selector_v1.log
```

`-u` 让 Python 不缓冲输出；不加的话经过 `tee` 时打印会攒着，看起来像没反应。头一两分钟只有
加载模型的进度条，之后才出现 `{'loss': ...}` 行。怀疑卡住时在另一个终端看
`nvidia-smi` 有没有 python 进程占显存，以及 `tail -20 train_selector_v1.log`。

参数含义：

- `--epochs 2`：按我们商量的，小数据两轮。
- `--max-length 4096`：训练样本最长 3010 token，4096 不会截断。截断会切掉答案，绝对不能发生。
- `--grad-accum 8`：每 8 条算一步，666 条一轮约 84 步，两轮约 168 步。
- LoRA 秩默认 32，和后面起服务的 `--max-lora-rank 32` 对应。
- `tee` 把日志同时存成文件，出问题可以回看。

预计几分钟到十几分钟。loss 怎么看：

- 前十几步从 2 点几暴跌到 0.5 以下是正常的，那是在学 JSON 列表的格式。
- 之后缓慢下降到 0.1 到 0.3 之间都合理。
- 一开始就接近 0、长时间纹丝不动、或中途突然飙升，停下来把日志贴出来。
- **loss 低不等于选表对**，唯一算数的是第 9 步的指标。

结束时看到 `LoRA adapter saved to /root/autodl-tmp/out/selector-v1-lora`。

## 第 7 步：起服务（窗口 A）

```bash
vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --enable-lora --max-lora-rank 32 --lora-modules selector=/root/autodl-tmp/out/selector-v1-lora
```

- `--max-model-len 8192` 不能小：val 788 的 prompt 最长 6763 token。
- `--max-lora-rank 32` 不能省，默认 16，和训练的秩不一致会加载失败。
- `selector=...` 等号左边就是评测时用的模型名。

**先等训练结束、`/root/autodl-tmp/out/selector-v1-lora` 里有 `adapter_model.safetensors` 再起服务**，
训练和 vLLM 不能同时占显卡。

如果租到的是 Blackwell 卡（5090、RTX 6000D、PRO 6000，`nvidia-smi -L` 能看到型号），vllm 一启动
就会报 `AssertionError: duplicate template name`。这是 `requirements-train.txt` 第 1 条记录的
torch 2.11 import bug，修法是注释掉 torch 里两行断言（不用 torch.compile 时安全），一条命令完成：

```bash
F=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'_inductor','select_algorithm.py'))"); sed -i 's/^\(\s*\)assert name not in self.all_templates, "duplicate template name"/\1pass  # patched/' "$F"; sed -i 's/^\(\s*\)assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"/\1pass  # patched/' "$F"; grep -n "patched" "$F"
```

最后要打印出两行带 `patched` 的。然后起服务时命令前面加环境变量并加 `--enforce-eager`：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --enforce-eager --enable-lora --max-lora-rank 32 --lora-modules selector=/root/autodl-tmp/out/selector-v1-lora
```

看到 `Uvicorn running on http://0.0.0.0:8000` 就是起来了。**这个窗口不要关。**

## 第 8 步：打隧道（窗口 B，本地）

本地新开一个 PowerShell：

```bash
ssh -p 端口 -L 8000:127.0.0.1:8000 root@地址 -N
```

输密码后**没有任何输出，光标停着**，这就是通了。这个窗口也不要关。

验证（窗口 C，本地）。PowerShell 里 `curl` 是别名，要写 `curl.exe`：

```bash
curl.exe http://localhost:8000/v1/models
```

返回的 JSON 里要**同时**有 `Qwen3-1.7B` 和 `selector` 两个条目。只有前者说明 LoRA 没挂上，
这时评的是没训练的原模型，不会报错，只会让你以为训练没用。

## 第 9 步：本地评测（窗口 C）

```bash
cd D:\Code\Demo\text2sql-rlvr
```

先 20 题冒烟，不写台账：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model selector --out results/selector/smoke.jsonl --limit 20 --no-ledger
```

看两件事：`empty predictions` 是不是 0；随便看几条 `completion` 是不是一个 JSON 列表：

```bash
python -c "import json;[print(json.loads(l)['completion']) for l in open('results/selector/smoke.jsonl',encoding='utf-8')]"
```

没问题就跑正式的两个评测集：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model selector --out results/selector/heldout_v1.jsonl --notes "selector v1: 666 examples, 2 epochs, lora r32"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/val788.jsonl --model selector --out results/selector/val788_v1.jsonl --notes "selector v1: 666 examples, 2 epochs, lora r32"
```

顺手把没训练的原模型也评一遍当零样本对照，只要换模型名：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model Qwen3-1.7B --out results/selector/heldout_base.jsonl --notes "zero-shot base, no selector training"
```

每次都会往 `results/runs.jsonl` 追加一行，`stage` 是 `selector`，并写一个 `.summary.json`。

### 怎么读结果

每组三行：`model` 是模型原始输出，`expanded` 是原始输出并上外键一跳邻居和词面前 2 的表
（默认开启，`--expand none` 关闭），`linker` 是词面 linker。每行 `all-gold / precision / kept`
三个数。真正交给 SQL 生成模型的是 `expanded` 那一行，它写在预测文件的 `expanded_tables` 字段。

已经跑过的预测文件可以离线换扩展参数重打分，不用 GPU，加 `--resume` 就不会重新请求模型：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model selector --out results/selector/heldout_v1.jsonl --resume --expand fk+lex --fk-hops 1 --lex-top-k 2 --no-ledger
```

v1 的实测（held-out）：model 全保留 0.786、expanded 0.948、linker 0.987；选中表数分别是
2.3、5.1、8.7。判断标准看 expanded 那一行：

| 数 | 含义 | 词面 linker 现在的值 |
|---|---|---|
| all-gold | 每题 gold 表全部保留的比例，漏一张就算 0 | held-out 0.987，val 788 0.980 |
| precision | 选出的表里 gold 真用到的比例 | held-out 0.310，val 788 0.142 |
| kept | 平均选了几张表（gold 平均 2 张） | held-out 8.7，val 788 25.5 |

判断：

- expanded 的 all-gold 不低于 0.95、kept 明显少于 linker：有效，进第 10 步。
- model 的 all-gold 比零样本 base 没有提高：训练只学到了输出长度，需要更多多表样本，
  用 v2 全量数据重训（见第 12 步）。
- `hard` 那一行的 all-gold 远低于 `easy`：漏的是多表 JOIN 的桥接表，是 v1 的主要失败模式。
- completion 里大量不是 JSON：把原文贴出来，可能是格式没学会，也可能是思考模式没关。

## 第 10 步：接进两阶段，在 val 788 上比 SQL 准确率

这一步需要 SQL 生成模型。selector 的 LoRA 是在 base Qwen3-1.7B 上训的，只能挂在 base 上，
所以顺序是：**先用第 7 步的服务把 val 788 的选表结果跑出来**（第 9 步已经做了，就是
`results/selector/val788_v1.jsonl`），**再换生成模型起服务**。

生成模型两种情况：

- 强 SFT 合并模型还在数据盘（`/root/autodl-tmp/Qwen3-1.7B-sft-strong-merged-f78ab16a`）：
  Ctrl+C 停掉第 7 步的服务，用它作为 `vllm serve` 的模型路径重新起，`--served-model-name sft`，
  不需要 `--enable-lora`。
- 找不到了：直接用 base Qwen3-1.7B 做生成模型，第 7 步的服务不用动。这样得到的是同一个
  生成模型下 full / linked / predicted / oracle 四组对照，结论一样成立，只是绝对分数低。

然后本地（窗口 C）跑生成和评测，`<模型名>` 填 `sft` 或 `Qwen3-1.7B`。`--selected-field
expanded_tables` 表示用扩展后的选集，这是 v1 实测后定下的工作点；不加则用模型原始输出：

```bash
python scripts/generate.py --questions data/processed/val.json --split train --model <模型名> --selected-tables results/selector/val788_v1.jsonl --selected-field expanded_tables --out results/preds/val788_selector_v1.jsonl
```

```bash
python scripts/evaluate.py --questions data/processed/val.json --split train --predictions results/preds/val788_selector_v1.jsonl --stage ablation --notes "val788, schema = selector v1 expanded, generator <模型名>"
```

对照组各跑一次，把 `--selected-tables ... --selected-field ...` 换成 `--schema-mode full`、
`--schema-mode linked`、`--schema-mode oracle`，输出文件名对应改（`val788_full.jsonl` 等），
evaluate 的 `--notes` 写清 schema 和生成模型。oracle 读了标准答案，只作上界，不进成绩表。
四组必须用同一个生成模型、同一批题、同样解码参数，只有 schema 不同，才能互相比较。

生成 788 题 × 4 组，每组十几分钟。生成模型是 base 时，四组预期的顺序是
full < linked ≈ 扩展选集 < oracle；扩展选集如果明显低于 linked，说明漏掉的 6% 题损失
超过了少给 17 张表的收益。

生成脚本的 `.meta.json` 里会记 `n_fallback`：selector 一张表都没选出来的题，会退回到完整
schema，这个数应该接近 0。

## 第 11 步：关机

AutoDL 按开机时间计费，跑完立刻关机。想把 LoRA 存到本地（约 100 MB），关机前在**本地**跑：

```bash
scp -P 端口 -r root@地址:/root/autodl-tmp/out/selector-v1-lora results/selector/
```

## 第 12 步：v2 全量数据重训

v1 的 666 题只教会了输出长度，漏多表题。v2 用 56 个训练库全部 6593 题，本地已生成在
`data/processed/selector/v2/train.jsonl`（27 MB）。传上去：

```bash
scp -P 端口 data/processed/selector/v2/train.jsonl root@地址:/root/autodl-tmp/selector/train_v2.jsonl
```

云端训练，输出到新目录，不覆盖 v1：

```bash
cd /root/autodl-tmp/selector && python -u train_sft.py --model /root/autodl-tmp/Qwen3-1.7B --data train_v2.jsonl --out /root/autodl-tmp/out/selector-v2-lora --epochs 2 --max-length 4096 --grad-accum 8 2>&1 | tee train_selector_v2.log
```

6593 条两轮约 1650 步，在 RTX PRO 6000 上预计半小时上下。训练前先 Ctrl+C 停掉 vllm。
训完起服务时把两个版本一起挂上，评测时用 `--model selector_v2`，和 v1 直接对比：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --enforce-eager --enable-lora --max-lora-rank 32 --lora-modules selector=/root/autodl-tmp/out/selector-v1-lora selector_v2=/root/autodl-tmp/out/selector-v2-lora
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model selector_v2 --out results/selector/heldout_v2.jsonl --notes "selector v2: 6593 examples, 2 epochs, lora r32"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/val788.jsonl --model selector_v2 --out results/selector/val788_v2.jsonl --notes "selector v2: 6593 examples, 2 epochs, lora r32"
```

要看的是 `model` 那一行的 all-gold 有没有从 0.79 往上走，尤其是 `hard` 那一行的 0.51。
评测文件 v1 和 v2 目录下是同一份，用哪个都一样；v3 起 prompt 不同，必须用对应目录的。

## 第 13 步：先试多样本并集（不训练，用现有服务）

v2 的诊断结论是模型"提前停"：贪心解码在第二张表之后，"结束"这个 token 比任何一张
不确定的第三张表都更可能，于是就不再列了。让模型在温度 0.8 下采样 8 次、把 8 次的表
取并集，能把它拿不准的第三张表捞回来，而且不依赖外键声明，对没有外键的库也有效：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/heldout.jsonl --model selector_v2 --out results/selector/heldout_v2_n8.jsonl --n-samples 8 --temperature 0.8 --notes "selector v2, union of 8 samples at T=0.8, + FK1/lex2"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v1/val788.jsonl --model selector_v2 --out results/selector/val788_v2_n8.jsonl --n-samples 8 --temperature 0.8 --notes "selector v2, union of 8 samples at T=0.8, + FK1/lex2"
```

看 `model` 那一行：all-gold 应该明显高于贪心的 0.787 / 0.728，kept 会涨到 3 张左右。
`expanded` 那一行是并集再加外键扩展，是交给生成模型的最终选集。

## 第 14 步：v3，让模型读懂 schema

v1 到 v2 数据翻了十倍，全保留率没动，两版在 40% 的题上答案不同但错误只是换了位置。
缺的不是题量，是"这个没见过的库里哪张表放着这个属性"：例如 movies_4 的性别在单独的
gender 表里，模型猜在 person 里。v3 给它两样帮助，各自单独一版，方便看哪个有用：

- v3a：prompt 里每列附上 BIRD 的列描述（截到 80 字符，去掉冗长的取值说明），目标不变。
- v3b：同样加描述，目标改为先列出 gold SQL 用到的每个 `table.column`，再列表。逼模型先
  定位属性，再决定表。标签同样从 gold SQL 自动解析，96.5% 的题能干净解析，其余保留表标签、
  列表可能不全。

prompt 变长了：训练题最长 5600 token，val 788 的 works_cycles 到 12.4k token。所以训练
用 `--max-length 8192`，起服务用 `--max-model-len 16384`。评测文件必须用对应版本目录下的
（prompt 带描述），不能再用 v1 的。

传数据（本地，两个文件各约 50 MB）：

```bash
scp -P 端口 data/processed/selector/v3a/train.jsonl root@地址:/root/autodl-tmp/selector/train_v3a.jsonl
```

```bash
scp -P 端口 data/processed/selector/v3b/train.jsonl root@地址:/root/autodl-tmp/selector/train_v3b.jsonl
```

云端训练，先停掉 vllm，两版依次跑，每版约 1650 步：

```bash
cd /root/autodl-tmp/selector && python -u train_sft.py --model /root/autodl-tmp/Qwen3-1.7B --data train_v3a.jsonl --out /root/autodl-tmp/out/selector-v3a-lora --epochs 2 --max-length 8192 --grad-accum 8 2>&1 | tee train_selector_v3a.log
```

```bash
cd /root/autodl-tmp/selector && python -u train_sft.py --model /root/autodl-tmp/Qwen3-1.7B --data train_v3b.jsonl --out /root/autodl-tmp/out/selector-v3b-lora --epochs 2 --max-length 8192 --grad-accum 8 2>&1 | tee train_selector_v3b.log
```

起服务，把 v2、v3a、v3b 一起挂上，注意 `--max-model-len 16384`：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.9 --enforce-eager --enable-lora --max-lora-rank 32 --lora-modules selector_v2=/root/autodl-tmp/out/selector-v2-lora selector_v3a=/root/autodl-tmp/out/selector-v3a-lora selector_v3b=/root/autodl-tmp/out/selector-v3b-lora
```

本地评测，每版跑 held-out 和 val 788，用各自目录的评测文件。v3b 的回答更长，`--max-tokens 256`：

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v3a/heldout.jsonl --model selector_v3a --out results/selector/heldout_v3a.jsonl --notes "selector v3a: descriptions, tables target, 6593 ex, 2 epochs"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v3a/val788.jsonl --model selector_v3a --out results/selector/val788_v3a.jsonl --notes "selector v3a: descriptions, tables target, 6593 ex, 2 epochs"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v3b/heldout.jsonl --model selector_v3b --out results/selector/heldout_v3b.jsonl --max-tokens 256 --notes "selector v3b: descriptions, columns target, 6593 ex, 2 epochs"
```

```bash
python scripts/evaluate_selector.py --eval data/processed/selector/v3b/val788.jsonl --model selector_v3b --out results/selector/val788_v3b.jsonl --max-tokens 256 --notes "selector v3b: descriptions, columns target, 6593 ex, 2 epochs"
```

v3b 的输出是 JSON 对象，解析时列名前缀的表也算选中，所以只要模型列出了 `Person.FirstName`，
Person 就不会漏。对比三行：v2 贪心的 model all-gold 是 0.787 / 0.728（held-out / val 788），
`hard` 行是 0.526 / 0.451。v3 要看的就是这两个数有没有实质上升。

## 第 15 步：agent 循环，单次生成对比带自纠错

`scripts/run_agent.py` 和 generate.py 一样在本地跑，模型每轮一个动作：`DESCRIBE 表名`、
一个 ```sql 块（执行并把结果或报错回给它）、或 `FINAL` 加 ```sql 块（提交）。默认每题
最多 4 轮。输出文件 evaluate.py 直接能打分。

先 20 题冒烟，看模型守不守协议：

```bash
python scripts/run_agent.py --questions data/processed/val.json --split train --model Qwen3-1.7B --schema-mode linked --out results/preds/agent_smoke.jsonl --limit 20 --concurrency 4
```

看屏幕上的 `agent summary`：`stop_reasons` 里 `final` 应该占多数，`no_sql` 接近 0；
`n_recovered_from_first_error` 是第一次执行报错、最终 SQL 能执行的题数。再看两条轨迹：

```bash
python -c "import json;r=json.loads(open('results/preds/agent_smoke.jsonl',encoding='utf-8').readline());[print(m['role'].upper(),':',m['content'][-300:],'\n') for m in r['messages'][2:]]"
```

正式跑 val 788，schema 用 selector 扩展后的选集，没展示的表模型可以 DESCRIBE：

```bash
python scripts/run_agent.py --questions data/processed/val.json --split train --model Qwen3-1.7B --selected-tables results/selector/val788_v2.jsonl --selected-field expanded_tables --out results/preds/val788_agent_selector_v2.jsonl
```

```bash
python scripts/evaluate.py --questions data/processed/val.json --split train --predictions results/preds/val788_agent_selector_v2.jsonl --stage ablation --notes "val788, agent loop max 4 turns, schema = selector v2 expanded, generator Qwen3-1.7B base"
```

和第 10 步的四组放在一起就是完整对照：full、linked、selector 扩展、oracle 是单次生成，
这一行是 selector 扩展加循环。差值就是"执行反馈自纠错"带来的提升，也是这个项目要讲的故事。

每题最多 4 次模型调用，788 题大约是单次生成的 2 到 3 倍时间。

## 出问题时对照这里

| 现象 | 原因 | 处理 |
|---|---|---|
| `ssh: connect to host ... refused` 或超时 | 端口/地址抄错，或实例没开机 | 回控制台重新复制登录指令 |
| `Permission denied (publickey,password)` | 密码抄错，常见是多了空格 | 重新复制密码 |
| scp 报 `No such file or directory` | 在云端跑了 scp，或没 cd 到项目根 | scp 在本地跑，先 `cd D:\Code\Demo\text2sql-rlvr` |
| `check_chat_template.py` 输出 FAIL | tokenizer 模板和预期不一致 | 停下，贴全部输出 |
| 训练命令没有任何输出 | 还在加载模型，或 `tee` 缓冲了打印 | 另开终端看 `nvidia-smi` 和日志；用 `python -u` |
| 训练报 CUDA out of memory | 显存被别的进程占着 | `nvidia-smi` 看，杀掉残留进程再跑 |
| vllm 一启动就 `duplicate template name` | Blackwell 卡上 torch 2.11 的 import bug | 第 7 步的补丁命令，再加环境变量和 `--enforce-eager` |
| `curl.exe` 连不上 8000 | 隧道断了，或 vllm 还没起来 | 看窗口 B 还在不在，看窗口 A 有没有 Uvicorn 那行 |
| `/v1/models` 里没有 `selector` | LoRA 没挂上 | 检查 `--lora-modules` 路径、`--max-lora-rank 32` |
| 评测 `completion` 全空或 `error` 非空 | 服务挂了或隧道断了 | 同上两行 |
| 评测 `empty predictions` 很多 | 模型没按 JSON 列表输出 | 打印几条 completion 原文看 |
| 报 `maximum context length` | prompt 超长 | 确认 `--max-model-len 8192` |

任何报错把**原文**整段贴出来，不要只说"报错了"。

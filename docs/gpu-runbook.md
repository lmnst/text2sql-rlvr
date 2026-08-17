# 第一次上 GPU：跑出 baseline 分数

> 最终状态（2026-08-17）：baseline 路径已经跑通。本文件是第一次 baseline 的历史操作手册，
> 不是当前 GRPO 环境说明；最终训练环境见 `requirements-train.txt` 和 `docs/grpo-runbook.md`。

目标：让 Qwen3-1.7B 在完全没训练的情况下做完 mini-dev 的 500 道题，得到项目的第一个真实数字。

## 整体思路：GPU 上只跑模型，别的都留在本地

租来的机器上**只做一件事**：把模型跑起来，开一个 HTTP 服务。

数据、判分、评测全部留在你自己的 Windows 上，通过一条 SSH 隧道连过去。

这样做的好处：

- 不用往云上传 3.3 GB 的数据库
- 不用在云上装这个项目的环境
- 云上机器出任何问题，关掉重开就行，本地什么都不会丢
- 计费只覆盖"模型真的在跑"的时间

代价只是提示词要走一趟网络，文本很小，可以忽略。

---

## 第 1 步：租机器

AutoDL 上选卡：

- **一张 RTX 4090（24 GB）就够**，不用 A100。模型只有 1.7B，24 GB 显存绰绰有余，价格差好几倍。
- 镜像选 **PyTorch 2.x + CUDA 12.x** 的基础镜像。

**省钱技巧：先用「无卡模式」开机。** AutoDL 的无卡模式每小时几毛钱，可以用来装环境、下模型
这些不需要 GPU 的活。等一切就绪再关机、换成带卡开机。装环境加下模型可能要半小时以上，
这段时间用无卡模式能省下大部分钱。

**注意系统盘只有 30 GB 左右，模型一定要放 `/root/autodl-tmp`**，那是数据盘。

## 第 2 步：装 vLLM（无卡模式下做）

```bash
pip install vllm
```

装得比较久，十几分钟正常。如果 AutoDL 的镜像市场里有现成带 vLLM 的镜像，直接用那个更快。

## 第 3 步：下模型（无卡模式下做）

用魔搭（ModelScope）下，它是国内的，比 HuggingFace 快很多：

```bash
pip install modelscope
```

```bash
modelscope download --model Qwen/Qwen3-1.7B --local_dir /root/autodl-tmp/Qwen3-1.7B
```

大约 3.5 GB。下完确认一下：

```bash
ls -la /root/autodl-tmp/Qwen3-1.7B
```

应该能看到 `config.json`、`tokenizer.json` 和几个 `.safetensors` 文件。

**做完这一步就关机，改成带卡开机。**

## 第 4 步：启动模型服务（带卡开机后）

```bash
vllm serve /root/autodl-tmp/Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9
```

参数说明：

- `--served-model-name Qwen3-1.7B`：给服务起个短名字，本地脚本用这个名字调用
- `--max-model-len 8192`：单次能处理的最大长度。实测最长的提示词约 2600 个 token，
  加上输出也远远够用，8192 是留足余量的选择
- `--gpu-memory-utilization 0.9`：允许 vLLM 用掉 90% 显存

看到类似 `Uvicorn running on http://0.0.0.0:8000` 就是起来了。**这个终端不要关**，让它一直开着。

## 第 5 步：打隧道（在你自己的 Windows 上，另开一个终端）

AutoDL 会给你一条登录命令，形如 `ssh -p 12345 root@region-x.autodl.com`。
把端口号和地址换成你自己的：

```bash
ssh -p 12345 -L 8000:127.0.0.1:8000 root@region-x.autodl.com -N
```

- `-L 8000:127.0.0.1:8000` 的意思是：把云上的 8000 端口，映射成你本机的 8000 端口
- `-N` 表示只做转发、不开远程命令行

输密码后**没有任何输出，光标就停在那里，这是正常的**，说明隧道通了。
**这个窗口也不要关。**

验证一下（再开第三个终端，在本地）：

```bash
curl http://localhost:8000/v1/models
```

能返回一段带 `Qwen3-1.7B` 的 JSON，就说明本地已经能调用云上的模型了。

## 第 6 步：先跑 20 题试水

**不要一上来就跑 500 题。** 先确认整条链路是通的：

```bash
python scripts/generate.py --root data/bird --split mini_dev --model Qwen3-1.7B --out results/preds/smoke.jsonl --limit 20
```

跑完看一眼生成的内容：

```bash
python -c "import json;[print(json.loads(l)['sql'][:120]) for l in open('results/preds/smoke.jsonl',encoding='utf-8')]"
```

应该看到 20 条像模像样的 SQL。如果全是空的，说明模型没按格式输出，先解决这个再往下走。

然后评一下这 20 题：

```bash
python scripts/evaluate.py --root data/bird --split mini_dev --predictions results/preds/smoke.jsonl --stage smoke --only-predicted
```

`--only-predicted` 表示只算这 20 道题。**不加这个参数，另外 480 道没生成的题会被算成答错，
分数就没有意义了。**

## 第 7 步：跑完整的 500 题

试水没问题就跑全量：

```bash
python scripts/generate.py --root data/bird --split mini_dev --model Qwen3-1.7B --out results/preds/base_minidev.jsonl
```

中途断了直接重跑同一条命令，加 `--resume` 会接着上次的继续：

```bash
python scripts/generate.py --root data/bird --split mini_dev --model Qwen3-1.7B --out results/preds/base_minidev.jsonl --resume
```

## 第 8 步：出分数

```bash
python scripts/evaluate.py --root data/bird --split mini_dev --predictions results/preds/base_minidev.jsonl --stage baseline
```

这次**不要**加 `--only-predicted`，因为 500 题全跑了。

输出里最重要的是这两行：

```
official EX          xx.xx%   <- 官方口径，对外汇报只能用这个
strict EX            xx.xx%   <- 自己的严格口径
```

**这是这个项目的第一个真实数字。** 同时会往 `results/runs.jsonl` 追加一条记录，
把模型、随机种子、提示词配置、硬件全部记下来。

## 第 9 步：关机

**AutoDL 是按开机时间计费的，跑完立刻关机。**

生成结果都在本地 `results/` 下，关机不影响。想再跑别的，重新开机、重复第 4、5 步即可。

---

## 出问题时对照这里

| 现象 | 原因 | 处理 |
|---|---|---|
| `curl` 连不上 8000 | 隧道断了 | 回第 5 步重新连，检查那个窗口还在不在 |
| 生成结果里 `sql` 全是空 | 模型没按 ```sql 格式输出 | 把某条 `completion` 原文贴出来看 |
| `error` 字段里有 `ConnectError` | 隧道断了或服务挂了 | 检查云上那个终端是否还在跑 |
| 报 `maximum context length` | 提示词超长 | 提高 `--max-model-len` |
| 显存不足 | `--gpu-memory-utilization` 太高 | 降到 0.8 |

任何一步的报错，把**原文**贴出来，不要只说"报错了"。

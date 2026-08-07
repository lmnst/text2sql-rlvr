# 项目协作约定

## 目标

尽快做出一个可复现、可投递的 Text-to-SQL RLVR 项目：Qwen3-1.7B、BIRD、LoRA SFT、verl GRPO、vLLM rollout、数据库执行奖励。

## 当前状态

仓库只有骨架，尚未运行任何实验。不得把计划值写成已完成结果。

## 工作优先级

1. 先打通 BIRD Mini-Dev 的数据、推理和官方 Execution Accuracy 评测。
2. 再实现 SQL 只读执行、超时、结果集规范化和单元测试。
3. 跑 Qwen3-1.7B baseline。
4. 跑 LoRA SFT 并评测。
5. 接入 verl 自定义 reward，先做小规模 GRPO smoke test，再做正式训练。
6. 最后做 dynamic sampling / clip-higher 消融和错误分析。

第 1、2 步不需要 GPU，只依赖 sqlite 和纯 Python，必须先在本地 Windows 上完成并通过测试。
全链路先在 Mini-Dev 上跑通一遍，再上全量 dev；不要一开始就用全量数据迭代。

## 数据划分纪律

- RL 训练期间的 reward 只能用 BIRD **train** 划分的 gold SQL，在 train 对应的数据库上执行。
- 从 train 中切出固定的 val 子集，用于超参选择和 checkpoint 选择；切分方式和种子写进配置，不得每次重切。
- **dev 划分只在最终评测时使用一次性读取**，不得用于调参、early stopping 或 checkpoint 挑选。
- 任何指标必须注明它来自 train / val / dev 中的哪一个，以及样本数。
- 若某次实验违反了上述任一条，该实验的数字作废，不得进入 README 或简历。

## 实验记录

每次产生指标的运行都要向 `results/runs.jsonl` 追加一行 JSON，字段至少包含：

```json
{
  "run_id": "",
  "timestamp": "",
  "git_sha": "",
  "git_dirty": false,
  "stage": "baseline | sft | grpo | ablation",
  "config_path": "",
  "config_sha256": "",
  "command": "",
  "model": "",
  "checkpoint": "",
  "split": "train | val | dev | mini-dev",
  "n_samples": 0,
  "seed": 0,
  "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 0},
  "hardware": "",
  "metrics": {},
  "log_path": "",
  "notes": ""
}
```

- 追加操作由脚本自动完成，不允许手写补录。
- `git_dirty` 为 true 的运行结果不得用于对外汇报。
- README 和简历里出现的每个数字，都必须能在 `results/runs.jsonl` 里定位到唯一一行。

## 进度记录

每完成一个里程碑，在 `docs/PROGRESS.md` **追加**（不是覆盖）一段中文说明，包含三部分：

1. **做了什么** —— 这次落地了哪些能力。
2. **为什么这么设计** —— 关键取舍及其理由，尤其是"看起来更简单的做法为什么不行"。
3. **还没验证的** —— 哪些是已经跑通并有证据的，哪些还只是设想、假设或待测。

写作要求：

- 面向**不熟悉本项目的读者**。第一次出现的概念用一句白话解释，不要堆术语。
- 不要罗列文件名和函数名当作内容，要讲清楚这一步解决了什么问题。
- 结论和猜测必须分开。没跑过的就写"还没验证"，不要用"已实现 X 提升"这类措辞。
- 追加在文件末尾，保留全部历史条目。改写历史条目只允许用于订正错误，且要注明。

## 真实性要求

- 简历、README 和实验表只写日志或结果文件能够证明的数字。
- 每个指标注明数据划分、样本数、随机种子、checkpoint、评测脚本和硬件。
- reward hacking 必须给出明确定义、分母和可复现样例。
- 不预设 GRPO 一定提升，也不预设一定发生 entropy collapse。

## 环境与依赖

开发环境和训练环境是两套，不要合并成一个依赖列表。

- `requirements-dev.txt`：数据处理、SQL 沙箱、结果规范化、官方评测 wrapper、pytest、ruff。
  只依赖 sqlite 和纯 Python，必须能在 Windows 本地安装和跑通全部单元测试，不引入 torch。
- `requirements-train.txt`：torch、vllm、verl、transformers 等训练侧依赖，只在租用的 Linux GPU 机器上安装。

约束：

- 训练侧依赖**全部钉死版本**，verl 钉到具体 commit sha 并在文件里注释锁定日期。verl 接口变动频繁，不钉版本会导致同一份代码在不同时间跑出不同行为。
- `pyproject.toml` 的 `dependencies` 保持与 `requirements-dev.txt` 一致，训练侧依赖只走 optional-dependencies 或独立 requirements 文件。
- 首次在 GPU 机器上配环境时，**先跑通 verl 官方 gsm8k GRPO 示例**确认环境可用，再接入本项目的自定义 reward。不要一上来就调自定义逻辑。

## 工程约束

- 第一版只支持 SQLite，不提前加入 MySQL/PostgreSQL 兼容代码。
- 模型生成 SQL 只能在只读数据库上执行；拒绝多语句和 DDL/DML，并设置超时。
- 结果匹配必须保留重复行，显式处理 NULL、浮点值、列数和无 ORDER BY 时的行序。
- 优先复用 BIRD 官方评测脚本，项目内 wrapper 不得悄悄改变其指标口径。
- 配置、命令和实验元数据必须可复现；大数据、数据库和 checkpoint 不提交 Git。
- Qwen3 的 chat template 默认关闭 thinking 模式（`enable_thinking=False`）。BIRD 的 schema prompt 本身很长，开启 thinking 会显著推高 rollout 长度和显存占用，单卡难以承受。推理行为应当作为 GRPO 训练后被观察到的现象，而不是预先灌入的格式。若后续要开启，必须作为独立消融并记录长度和吞吐代价。
- reward 计算是训练吞吐的主要瓶颈，不是 GPU。SQL 执行侧从一开始就要做：`file:...?immutable=1` 只读打开、进程池并发、硬超时（训练期 5–10 秒，不沿用官方评测的 30 秒）、按 `(db_id, sql_hash)` 缓存执行结果。

## 沟通

- 默认中文，简洁直接。
- 发现设计问题直接提出修改方案。
- 未验证完成前使用“计划”“待验证”，不要写成“已实现”“已提升”。


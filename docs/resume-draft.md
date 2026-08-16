# 简历项目经历草稿（Text2SQL-RLVR）

> 状态：草稿。带 `___` 的数字是还没跑出来的（GRPO 训练中），**填之前必须能从
> `results/runs.jsonl` 定位到唯一一行**，否则按 AGENTS.md 一律不写。
> 已填的数字全部有台账依据：baseline/SFT 来自 runs.jsonl 第 1-6 行。

---

## 项目经历（简历 bullet 版）

**Text2SQL-RLVR：基于执行反馈的 Text-to-SQL 强化学习**（Qwen3-1.7B · verl/GRPO · vLLM · LoRA）

- 构建「生成 SQL → 只读数据库执行 → 执行结果正确性作为奖励」的完整 RLVR 管线，在 BIRD 基准上以 GRPO 训练 Qwen3-1.7B；BIRD Mini-Dev（500 题）执行准确率从基线 19.80% 提升至 SFT 后 34.60%、GRPO 后 ___%。
- 设计并实现双验证器：严格复刻 BIRD 官方 `set(pred)==set(gold)` 口径用于对外汇报，自研保留行序与重复行的严格验证器用于训练奖励；两者之差即 reward hacking 的直接度量，实测训练集 8.7%（770/9428）的题目可利用「空结果 / 去重」在未答对时白拿官方分数。
- 实现安全 SQL 执行沙箱（三层防护：只读打开 + SQLite authorizer + 文本预检；硬超时 + 结果缓存 + 连接池），支撑 GRPO 训练中每步数百次 SQL 执行；训练吞吐瓶颈从 GPU 转移到 reward 计算，已针对性优化。
- 主导 GRPO 环境攻坚：在 Blackwell 新架构（RTX 6000D）+ verl 0.9 配置大改版 + vLLM LoRA API 重构的三重版本冲突下，定位并解决 PyTorch #186220（torch 2.11 Blackwell 模板重复注册）、verl 0.9 奖励配置键迁移失效等十余个兼容性问题，锁定一套可复现版本组合。
- 设计并运行 reward hacking 对照实验：开关「SQL 可执行即给分」的奖励项，对比训练过程中 `SELECT 1` 类退化输出占比（hack rate）与正确率曲线，量化复合奖励下的投机行为（结果：___）。
- 工程规范：实验台账（runs.jsonl）自动记录 git 版本/硬件/解码参数/划分/样本数；train/val/dev 数据划分纪律（dev 只在最终评测使用一次）；训练与评测 prompt 逐字节一致性校验。

## 量化结果（表格版，用于面试展开）

| 阶段 | 数据划分 | 官方 EX | 严格 EX | 依据 |
|---|---|---|---|---|
| Qwen3-1.7B 基线 | Mini-Dev 500 题 | 19.80% | 17.40% | runs.jsonl |
| + LoRA SFT 1 epoch | Mini-Dev 500 题 | 34.60% | 30.00% | runs.jsonl |
| + GRPO（严格奖励） | Mini-Dev 500 题 | ___ | ___ | 训练中 |
| GRPO 对照（exec bonus=0.3） | Mini-Dev 500 题 | ___ | ___ | 未跑 |

关键中间数据（可支撑面试问答）：
- 训练集 gold 自身不可执行 370/9428（3.9%），已在切分时过滤；gold 空结果 283 题（3.0%）、含重复行 487 题（5.2%），合计 8.2% 官方口径可利用面。
- SFT 提升拆解：mini-dev 净增 74 题中，53 题来自「修掉格式/方言错误」（执行失败 207→107，方言类 63→1），51 题来自真实查询能力提升——报告中主动区分，不把格式修复算作能力提升。
- 官方口径与严格口径分歧：SFT 后 mini-dev 上 25 题官方给分、严格不给分，全部为「漏写 DISTINCT」形状（如模型 933 行 vs gold 21 行），直接支撑「用严格验证器当奖励」的决策。

## 技术栈

PyTorch · verl(GRPO) · vLLM · LoRA/PEFT · SQLite · BIRD · Ray · Python

---

## 待补（跑完后回来填）

- [ ] GRPO 严格奖励最终 EX（val + mini-dev，两个口径）
- [ ] 对照实验的 hack rate / no_from rate 曲线结论
- [ ] 错误分析结论（官方-严格分歧随训练的变化）
- [ ] 硬件与训练耗时（如实写：单卡 84GB，300→150 步，每步耗时）
- [ ] 若 GRPO 无明显提升，简历改为「验证了 RLVR 在此设定下的边界 + reward hacking 案例」，同样是诚实且有价值的叙事

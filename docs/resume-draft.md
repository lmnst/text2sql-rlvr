# Text2SQL-RLVR 简历素材

本文件只使用 `results/runs.jsonl` 中可唯一定位、`git_dirty=false` 的正式实验数字。
推荐简历正文使用前三条；GRPO 的负结果留到面试追问时展开。

## 中文版：推荐三条

**Text2SQL-RLVR：基于执行反馈的 Text-to-SQL 训练与评测系统**

Qwen3-1.7B · verl/GRPO · vLLM · LoRA/PEFT · SQLite · BIRD

- 构建从 BIRD 数据处理、LoRA SFT、vLLM 并发生成到数据库执行奖励和 verl GRPO 的完整链路；
  Qwen3-1.7B 在 BIRD Mini-Dev 500 题上的官方执行准确率由 19.80% 提升至 SFT 后 34.60%。
- 实现基于问题与 evidence 的 schema linking，并通过同 checkpoint、同 788 题、确定性解码的
  受控实验，将官方 EX 从 full schema 的 32.87% 提升至 linked schema 的 37.94%。
- 实现 SQLite 只读执行沙箱与官方/严格双验证器：使用 immutable 只读连接、authorizer、单语句
  校验、硬超时和结果缓存隔离模型 SQL；主指标复刻 BIRD official EX，严格指标保留重复行并监控
  指标利用，实验台账自动记录 Git SHA、配置哈希、划分、解码参数和逐题结果。

## 中文版：如果版面允许加第四条

- 对齐训练与评测口径，用 BIRD official EX 重跑 linked-schema 普通 GRPO；固定 train-val 788 上
  由 37.94% 变为 38.20%，判定为无明确增益，并将稀疏二值奖励和有效 group
  不足作为下一步诊断方向，而非选择性汇报正向数字。

第四条适合研究型岗位或面试材料；一页中文简历通常不放，以免弱结果抢占项目主贡献。

## English version

**Text2SQL-RLVR — Execution-grounded training and evaluation for Text-to-SQL**

Qwen3-1.7B · verl/GRPO · vLLM · LoRA/PEFT · SQLite · BIRD

- Built an end-to-end pipeline covering BIRD preprocessing, LoRA SFT, concurrent vLLM
  generation, database-execution rewards, and verl GRPO; improved BIRD Mini-Dev official
  execution accuracy from 19.80% to 34.60% after SFT on Qwen3-1.7B.
- Implemented question/evidence-based schema linking and evaluated it with a controlled,
  deterministic 788-example comparison on the same checkpoint, improving official EX from
  32.87% with the full schema to 37.94% with the linked schema.
- Implemented a read-only SQLite execution sandbox and dual verifiers using immutable
  connections, an authorizer, single-statement validation, hard timeouts, and result caching;
  recorded Git revision, config hash, split, decoding settings, and per-example outcomes in an
  append-only experiment ledger.

Optional research-oriented bullet:

- Re-ran ordinary GRPO with BIRD official EX as the reward and linked-schema prompts; the paired
  788-example evaluation changed official EX from 37.94% to 38.20%, which was reported
  as no clear gain and motivated an analysis of sparse binary rewards and zero-variance groups.

## 正式结果与依据

| 阶段 | 划分 | n | official EX | strict EX | run_id |
|---|---|---:|---:|---:|---|
| Qwen3-1.7B baseline | Mini-Dev | 500 | 19.80% | 17.40% | `43bb66bbe1e8` |
| LoRA SFT | Mini-Dev | 500 | 34.60% | 30.00% | `f1ef9b247255` |
| 强 SFT + full schema | 固定 train-val | 788 | 32.87% | 28.68% | `74d4d83c087c` |
| 强 SFT + linked schema | 固定 train-val | 788 | 37.94% | 33.63% | `3b9e91f891d8` |
| official-reward GRPO + linked schema | 固定 train-val | 788 | 38.20% | 33.88% | `53d1586c7d33` |

Mini-Dev 与固定 train-val 是不同划分，不跨表直接比较。38.20% 不是 BIRD dev 成绩。

## 面试时怎么讲

### 30 秒版本

项目目标是验证执行反馈能否改进小模型 Text-to-SQL。我先做了只读 SQL 沙箱和官方评测复刻，
再完成 Qwen3-1.7B 的 SFT 与 GRPO。排查发现完整 schema 并未超过上下文上限，但大量无关表会
干扰 1.7B 模型；加入轻量 schema linking 后，固定 val 788 上官方 EX 提升 5.07 个百分点。
之后把 GRPO reward 与 BIRD official 对齐并公平重跑，结果只净增 2 题，因此最终把 schema
筛选作为主要贡献，把 GRPO 作为边界验证而不是硬说成功。

### 如果追问“GRPO 为什么没涨”

先说证据，再说假设：训练正常跑满，没有截断、显存错误或 entropy collapse；多个 checkpoint
的固定 val 没有继续改善，完整 val 的变化也很小。最可能的机制是 SQL 的 0/1 execution reward
过于稀疏，n=4 时很多组可能全错或全对，组内 advantage 为零；其次是 500 prompt、30 step 和
1.7B 模型能力限制。有效 group 比例尚未统计，所以必须说“待验证假设”，不能说已经证明。

### 如果追问“为什么 official 和 strict 都要有”

BIRD official 是最终考试口径，所以主奖励必须与它对齐；但 official 使用集合比较，会忽略重复
行。strict 保留重复行并检查列数，用来监控模型是否只学会利用评分规则。两者职责不同，不应该
用 strict 替换最终指标，也不应该因为 official 是官方指标就停止监控。

## 不要写的内容

- “GRPO 显著提升了准确率”——正式对照只从 37.94% 变为 38.20%。
- “BIRD dev 达到 38.20%”——该数字来自 BIRD train 中固定划出的 val，dev 未读取。
- “解决了 schema linking”——轻量 linker 仍会漏必要表并保留干扰表。
- “完成 reward-hacking 对照实验”——execution bonus 对照没有运行。
- 训练集攻击面、吞吐优化幅度、错误类型数量等没有唯一 ledger 行的数字。

## 写简历时的取舍

一页简历优先保留 schema linking、SFT 提升和安全评测基础设施。不要为了让标题中的 RL 看起来
更成功而虚构增益；项目真正有价值的地方是完成了公平实验，并能解释哪个改动有效、哪个无效。

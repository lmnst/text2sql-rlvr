# 项目交接（2026-08-16 晚更新）

## 先读什么

1. `AGENTS.md`：数据划分、实验记录和真实性规则（不可违反）。
2. `docs/PROGRESS.md`：全部里程碑历史，最新到「里程碑 15：正式 GRPO 完成，未超过 SFT」。
3. `docs/resume-draft.md`：简历草稿，GRPO 真实数字已填。
4. `requirements-train.txt`：训练环境钉死的版本组合 + 6 条 Blackwell workaround。
5. `results/runs.jsonl`：可对外引用的实验台账（含 GRPO val 一行，run_id=a8f7fa3bdecc）。

## 当前状态（最重要的一段）

**主线：正式 GRPO 训练已完成，结果为负（诚实记录，不是失败）。**

| 阶段 | val 788 题 official | val 788 题 strict | mini-dev 500 题 official |
|---|---|---|---|
| 基线 | 21.83% | 19.54% | 19.80% |
| SFT 后 | 33.38% | 29.06% | 34.60% |
| GRPO 后（150 步、n=4、严格奖励） | **29.70%** | **25.76%** | 未评测 |

- 训练集 reward 从 0.20 涨到 0.375（持续上升），训练中 val 子集横盘 0.235–0.245。
- 结论：**GRPO 未超过 SFT**。训练 reward 上升、验证准确率不涨，是 RLVR 教科书级现象
  （模型学到刷分技巧而非可迁移的查询能力）。这个负结果 + 接下来的对照实验构成完整诚实叙事。
- 台账依据：GRPO val 在 `results/runs.jsonl`（run_id=a8f7fa3bdecc，官方 29.70/严格 25.76）。

**已完成链路**（全部有证据）：

| 环节 | 状态 |
|---|---|
| BIRD 数据/评测/只读 SQL 沙箱/双验证器 | ✅ 长期完成 |
| Qwen3-1.7B 基线 | ✅ mini-dev 官方 EX 19.80% |
| LoRA SFT 1 epoch | ✅ mini-dev 官方 EX 34.60%；`scripts/train_sft.py` 可重训复现 |
| GRPO 三步 smoke | ✅ 3 step 跑通，rollouts.jsonl 212 行 |
| 正式 GRPO 训练（150 步） | ✅ 完成，val 29.70%（低于 SFT） |
| checkpoint 导出 | ✅ 实测通过（missing=0/unexpected=0），`scripts/convert_checkpoint.py` 或手工 peft 合并 |
| vLLM 起服务 | ✅ 需 `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN --enforce-eager` |
| 对照实验（exec bonus） | ⬜ **未跑（建议下一步第一优先）** |
| mini-dev 最终评测 | ⬜ 未跑（val 已降，mini-dev 大概率同样低于 SFT，但按纪律未验证） |
| 错误分析 | ⬜ 未做（正式训练 rollouts.jsonl 22024 条可支撑） |

## 云上环境（AutoDL，勿轻易动）

- 机器：单卡 RTX 6000D 84GB（Blackwell sm_120）。**关机安全（数据保留），释放/保存镜像会丢数据盘。**
- 版本组合（钉死，见 `requirements-train.txt`）：torch 2.11.0+cu130、vLLM 0.24.0、
  verl commit `4a2cba76`（0.9.0.dev0，源码在 `/root/verl`）、transformers 5.5.3。
- 数据盘 `/root/autodl-tmp/` 关键内容：
  - `Qwen3-1.7B/`（基座）、`Qwen3-1.7B-sft-merged/`（SFT 合并）、`Qwen3-1.7B-grpo/`（GRPO 导出）
  - `train/`（BIRD train：train.json + train_databases/ 69 库）
  - `rl/`（parquet）、`src/`、`verl_reward.py`、`run_grpo.sh` / `run_smoke.sh`
  - `out/grpo/`（正式训练：checkpoints global_step_50/100/150 + rollouts.jsonl 22024 条）
  - `grpo_train.log`（第一次白跑）、`grpo_train2.log`（有效训练日志）

**云上对第三方源码的手改（换机器必须重做，方法见 requirements-train.txt）**：
1. `torch/_inductor/select_algorithm.py`：注释两行 assert（PyTorch #186220）。
2. `verl/utils/attention_utils.py`：flash_attn.bert_padding → transformers + einops 回退。
3. `run_grpo.sh` 的奖励键名 `reward.custom_reward_function.*` 等配置修正。

## 本地仓库状态

- `main` 领先 `origin/main` 9+ 个 commit。**待 push**（需要 GitHub 凭证）。
- 台账 `results/runs.jsonl` 已清理无效记录（0.00% 那条空评测已删），现为 7 行有效记录。

## 下一步（建议顺序）

1. **对照实验（第一优先，最值钱的素材）**：
   `TEXT2SQL_REWARD_EXEC=0.3 OUT=/root/autodl-tmp/out/grpo_naive /root/autodl-tmp/run_grpo.sh`
   再训一次（约 4 小时），然后
   `python scripts/analyze_rollouts.py --rollouts /root/autodl-tmp/out/grpo_naive/rollouts.jsonl`
   对比严格版与 naive 版的 no_from / hack 率。**这个实验不依赖 GRPO 提升，大概率有戏剧性结果。**
2. **mini-dev 评测**（可选但建议做，让表格完整）：vLLM serve GRPO 模型 → 本地
   generate/evaluate（`--split mini_dev --stage grpo`）。预期低于 34.60%，如实记录。
3. **错误分析**：对比 SFT 与 GRPO 的 outcomes（`results/outcomes/`），写进 PROGRESS。
4. **收尾**：README Status 更新 → PROGRESS 里程碑 16 → push 全部。

## 换电脑恢复

```bash
git clone https://github.com/lmnst/text2sql-rlvr.git
cd text2sql-rlvr
pip install -r requirements-dev.txt
python -m pytest          # 本地测试应全绿（数据不依赖 BIRD）
```

BIRD 数据按 `docs/data.md` 重下；GPU 环境按 `requirements-train.txt` + 本文件「云上环境」一节重建。

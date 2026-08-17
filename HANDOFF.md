# 项目交接（2026-08-16 晚更新）

## 先读什么

1. `AGENTS.md`：数据划分、实验记录和真实性规则（不可违反）。
2. `docs/PROGRESS.md`：全部里程碑历史；里程碑 15 的结论已在文末追加订正。
3. `docs/resume-draft.md`：简历草稿，GRPO 真实数字已填。
4. `requirements-train.txt`：训练环境钉死的版本组合 + 6 条 Blackwell workaround。
5. `results/runs.jsonl`：可对外引用的实验台账（含 GRPO val 一行，run_id=a8f7fa3bdecc）。

## 当前状态（最重要的一段）

**主线：先验证 schema linking，再公平重跑 official-reward GRPO。**

| 阶段 | val 788 题 official | val 788 题 strict | mini-dev 500 题 official |
|---|---|---|---|
| 基线 | 21.83% | 19.54% | 19.80% |
| 强 SFT（checkpoint 已恢复） | 33.38% | 29.06% | 34.60% |
| 新 SFT（GRPO 的真实起点，诊断评测） | 29.19% | 25.76% | 未评测 |
| 旧 GRPO（150 步，诊断评测） | 29.70% | 25.76% | 未评测 |
| corrected GRPO step 10（诊断评测） | 29.31% | 25.63% | 未评测 |

- 旧 GRPO 实际从 2026-08-16 重训的新 SFT 出发；33.38% 的 SFT 数字来自已丢失的旧 checkpoint，
  不能作为直接对照。诊断结果显示旧 GRPO 相对真实起点为 official +0.51、strict +0.00 个点。
- `grpo_train2.log` 的 `data.apply_chat_template_kwargs={}`，训练误用了 Qwen3 thinking；最终评测却关闭 thinking。
- corrected run 显式关闭 thinking，val 200 条用 seed=0 随机抽样并覆盖 3 个数据库。step 0→10
  的 strict reward 为 0.265→0.260、official 为 0.320→0.310，因此在第一个 checkpoint 止损。
- step 10 完整 val 788 题相对直接 SFT 是 official +0.12、strict -0.13 个百分点，本质无变化。
- 上述新 SFT 诊断数字尚未写入正式台账，因此不能用于简历或 README。
- 原强 SFT adapter 已恢复并按 SHA-256 校验，合并模型位于 F37 数据盘
  `/root/autodl-tmp/Qwen3-1.7B-sft-strong-merged-f78ab16a`。
- 同一强 SFT、固定 val 200 的 schema 诊断中，linked 相对 full 的 official 为
  39.5% 对 33.0%，strict 为 34.0% 对 27.0%；这些是 dirty 诊断结果，尚不能对外报告。

**已完成链路**（全部有证据）：

| 环节 | 状态 |
|---|---|
| BIRD 数据/评测/只读 SQL 沙箱/双验证器 | ✅ 长期完成 |
| Qwen3-1.7B 基线 | ✅ mini-dev 官方 EX 19.80% |
| LoRA SFT 1 epoch | ⚠️ 旧 checkpoint 为 34.60%；新脚本重训结果未复现，正在排查 |
| GRPO 三步 smoke | ✅ 3 step 跑通，rollouts.jsonl 212 行 |
| 正式 GRPO 训练（150 步） | ⚠️ 完成，但模板不一致且直接 SFT 对照缺失，结论作废 |
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

1. 在干净 commit 上完成强 SFT 的 full / linked 完整 val 788 配对评测，确认 200 题提升能否复现。
2. linked 静态诊断同时报告逐题 gold 表全保留率、外键连通保留率和干扰表数量；不要只报平均表召回。
3. 若 linked 提升稳定，用 linked schema 和 BIRD official 主奖励重跑普通 GRPO；strict 仅作监控指标。
4. 固定相同 val、解码和 checkpoint 对照，先不做 Dynamic Sampling 或大规模数据清洗。

## 换电脑恢复

```bash
git clone https://github.com/lmnst/text2sql-rlvr.git
cd text2sql-rlvr
pip install -r requirements-dev.txt
python -m pytest          # 本地测试应全绿（数据不依赖 BIRD）
```

BIRD 数据按 `docs/data.md` 重下；GPU 环境按 `requirements-train.txt` + 本文件「云上环境」一节重建。

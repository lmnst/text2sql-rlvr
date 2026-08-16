# 项目交接（2026-08-16 更新）

## 先读什么

1. `AGENTS.md`：数据划分、实验记录和真实性规则（不可违反）。
2. `docs/PROGRESS.md`：全部里程碑历史，最新到「里程碑 14：正式 GRPO 训练启动」。
3. `docs/resume-draft.md`：简历草稿，GRPO 数字留空待填。
4. `requirements-train.txt`：训练环境钉死的版本组合 + 5 条 Blackwell workaround。
5. `results/runs.jsonl`：可对外引用的实验台账。

## 当前状态（最重要的一段）

**主线：正式 GRPO 训练正在 AutoDL 上跑**（2026-08-16 14:40 左右启动，预计 3–4 小时）。

- 训练配置：严格奖励、150 步、rollout n=4、关 gradient checkpointing、batch 32、
  `test_freq=25`（每 25 步自动测验证集）、SFT 合并模型为起点。
- 云上日志：`/root/autodl-tmp/grpo_train.log`（nohup 后台，SSH 断连不中断）。
- 查进度：`grep "Training Progress" /root/autodl-tmp/grpo_train.log | tail -1`
- 查 reward/val：`grep -oE "(reward/mean|val-core)[^ ]*" /root/autodl-tmp/grpo_train.log | tail`

**已验证完成的链路**（全部有证据）：

| 环节 | 状态 |
|---|---|
| BIRD 数据/评测/只读 SQL 沙箱/双验证器 | ✅ 长期完成，本地测试全绿 |
| Qwen3-1.7B 基线（mini-dev 500 题） | ✅ 官方 EX 19.80%（runs.jsonl） |
| LoRA SFT 1 epoch | ✅ 官方 EX 34.60%（runs.jsonl）；本次用 `scripts/train_sft.py` 重训复现，loss 0.2249 与历史一致 |
| GRPO 三步 smoke | ✅ 3 step 跑通，rollouts.jsonl 212 行，第一条即复现「official 给分 / strict 不给分」案例 |
| 正式 GRPO 训练 | ⏳ **进行中**（本交接时的主线） |
| 对照实验（exec bonus） | ⬜ 未跑 |
| val/mini-dev 最终评测 | ⬜ 待 GRPO 完成后 |

## 云上环境（AutoDL，勿轻易动）

- 机器：单卡 RTX 6000D 84GB（Blackwell sm_120），已开机，**关机安全（数据保留），
  释放/保存镜像会丢数据盘**。
- 版本组合（已钉死，见 `requirements-train.txt`）：torch 2.11.0+cu130、vLLM 0.24.0、
  verl commit `4a2cba76`（0.9.0.dev0，源码在 `/root/verl`）、transformers 5.5.3。
- 数据盘 `/root/autodl-tmp/` 关键内容：
  - `Qwen3-1.7B/`（基座）、`Qwen3-1.7B-sft-merged/`（SFT 合并模型，3.4GB）
  - `train/`（BIRD train 数据：`train.json` + `train_databases/` 69 库）
  - `rl/`（train.parquet 8191 行 + val.parquet 200 行）、`src/`（项目包）、`verl_reward.py`
  - `run_grpo.sh` / `run_smoke.sh`（训练脚本，已含全部配置修正）
  - `out/qwen3-1.7b-sft-lora/`（SFT adapter）、`out/grpo_smoke/rollouts.jsonl`（smoke 日志）

**云上对第三方源码做过 3 处手改（换机器必须重做，方法见 requirements-train.txt）**：
1. `torch/_inductor/select_algorithm.py`：注释两行 assert（PyTorch #186220 Blackwell 重复注册 bug）。
2. `verl/utils/attention_utils.py`：flash_attn.bert_padding 改为 transformers + einops 回退。
3. `run_grpo.sh`：奖励键名 `reward.custom_reward_function.*`（verl 0.9 顶层键迁移只在
   fully-async 路径执行，v0 路径必须直接写新键名）、`tensor_model_parallel_size=1`、
   `agent.num_workers=4`。

## 本地仓库状态

- `main` 领先 `origin/main` 8+ 个 commit（含 3 处可投递性隐患修复、question_id 修复、
  训练版本钉死、PROGRESS 里程碑 12-14）。**待 push**（需要 GitHub 凭证）。
- 未追踪新文件：`scripts/train_sft.py`、`docs/resume-draft.md`（本交接后应 commit）。

## 下一步（GRPO 跑完后，按顺序）

1. **看 val 曲线决定是否续跑**：`grep "val-core" grpo_train.log`。若 150 步内 val 已见顶，
   不必续；若仍在涨，把 `TOTAL_STEPS=150` 提到 200 续跑。
2. **导出 checkpoint → vLLM serve → 本地 generate/evaluate**（先 val 再 mini-dev，
   `--instruction-version v1`，两套口径都报，`--stage grpo` 写台账）。
3. **对照实验**：`TEXT2SQL_REWARD_EXEC=0.3 OUT=/root/autodl-tmp/out/grpo_naive /root/autodl-tmp/run_grpo.sh`
   再训一次，用 `scripts/analyze_rollouts.py --rollouts out/grpo_naive/rollouts.jsonl`
   对比 no_from / hack 率。
4. **收尾**：错误分析 → 填 `docs/resume-draft.md` 的数字 → 更新 README 的 Status →
   追加 PROGRESS 里程碑 15 → push 全部。

## 换电脑恢复

```bash
git clone https://github.com/lmnst/text2sql-rlvr.git
cd text2sql-rlvr
pip install -r requirements-dev.txt
python -m pytest          # 本地测试应全绿（数据不依赖 BIRD）
```

BIRD 数据按 `docs/data.md` 重下；GPU 环境按 `requirements-train.txt` + 本文件「云上环境」一节重建。

# 项目交接

## 先读什么

按以下顺序了解项目，不要依赖旧聊天记录：

1. `AGENTS.md`：数据划分、实验记录和真实性规则。
2. `docs/PROGRESS.md`：按时间记录已完成工作、设计理由和未验证项。
3. `results/runs.jsonl`：能够对外引用的实验台账。
4. `docs/grpo-runbook.md`：当前 GRPO 环境与操作步骤。

## 当前状态

当前主分支为 `main`。截至 2026-08-09：

- Qwen3-1.7B 未训练基线在 BIRD mini-dev 500 题上为
  `official_ex=19.80%`、`strict_ex=17.40%`。
- 一轮 LoRA SFT 后为 `official_ex=34.60%`、`strict_ex=30.00%`。
- GRPO 的数据、严格执行奖励、运行脚本和 rollout 分析工具已经实现。
- GRPO smoke test 尚未跑通，因此没有 GRPO 结果，也没有 reward-hacking
  对照实验结果。不要把配置或计划写成已经验证。

当前唯一主线是先跑通 AutoDL 上的 GRPO smoke，再决定正式训练。

## 当前卡点

最近一次 AutoDL 调试停在 vLLM 报错：

```text
cumem allocator is not supported on current platform
```

容器不支持 vLLM 睡眠模式需要的 CUDA 虚拟内存接口。下一次 smoke 应在现有命令上增加：

```bash
actor_rollout_ref.rollout.free_cache_engine=False
```

如果显存不足，再把 `actor_rollout_ref.rollout.gpu_memory_utilization` 从 `0.4`
降到 `0.3`。不要先改依赖或安装 flash-attn。

已确认的另外两项环境处理：

- `trainer.use_v1=False`：绕过会导入缺失 `transfer_queue` 的新版任务运行器；
  该项已经写入 `configs/grpo/run_grpo.sh`。
- FlashAttention2 缺失时改用 `sdpa`；具体覆盖方式见
  `docs/grpo-runbook.md`。

每次训练进程崩溃后，先清掉残留 Ray/vLLM 进程并用 `nvidia-smi` 确认显存归零，
再重试。命令在 `docs/grpo-runbook.md`。

## 下一步

1. 在 AutoDL 恢复此前可运行的 verl HEAD 环境，不升级或降级依赖。
2. 清理残留进程。
3. 按 `docs/grpo-runbook.md` 重新运行三步 smoke，并增加
   `actor_rollout_ref.rollout.free_cache_engine=False`。
4. 确认 trainer 启动、至少完成一个训练 step，并产生
   `rollouts.jsonl`；失败时保存完整报错和实际 verl commit。
5. smoke 通过后先提交环境配置与证据，再运行严格奖励版本。
6. 严格版本稳定后，才运行 `TEXT2SQL_REWARD_EXEC=0.3` 的朴素奖励对照。

## 换电脑后恢复

```bash
git clone https://github.com/lmnst/text2sql-rlvr.git
cd text2sql-rlvr
pip install -r requirements-dev.txt
python -m pytest
```

Git 仓库不包含 BIRD 数据、模型、checkpoint、预测文件或训练输出。
本地数据按 `docs/data.md` 重新获取；GPU 环境按 `docs/gpu-runbook.md` 和
`docs/grpo-runbook.md` 重建。密钥和云服务器登录信息不应写入仓库。

继续工作前先运行 `git status`，再阅读最新 commit 和本文件。

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

当前唯一主线是迁移到单卡 48 GB 实例，先跑通 GRPO smoke，再决定正式训练。

## 当前卡点

最近一次在单卡 RTX 4090 24 GB 上的调试已经依次越过配置解析、Ray 初始化、
vLLM 初始化，并走到第一次把 actor 权重同步给 vLLM。已确认三处兼容问题：

- `trainer.use_v1=False`：绕开会导入缺失 `transfer_queue` 的新版任务运行器。
- 模型要求 FlashAttention2、环境未安装：覆盖为 PyTorch `sdpa`。
- AutoDL 容器不支持 vLLM 睡眠模式需要的 cuMem 接口：必须同时设置
  `rollout.enable_sleep_mode=False` 和 `rollout.free_cache_engine=False`。

只关 `free_cache_engine` 不够：它只控制后续是否调用 sleep；
`enable_sleep_mode` 会直接传给 vLLM，并在初始化时触发 cuMem 检查。

关闭睡眠模式后，当前 vLLM/verl 组合又在 LoRA IPC 注入时触发
`IndexError: tuple index out of range`。设置 `model.lora.merge=True` 后确认切换到了
“先把 LoRA 合入权重、再走普通权重同步”的路径，绕过了 `add_lora()`。

4090 上最后的真实失败是第一次完整权重同步时 OOM，而非显存碎片：actor/FSDP
约占 13.21 GiB，vLLM 约占 10.20 GiB，整卡只剩 93 MiB，随后复制
`lm_head.weight` 还需约 1.16 GiB。**尚未产生 rollout、reward、训练 step 或 checkpoint。**

因此当前决定是换一张 48 GB 卡，不先做双卡。原因是双卡会同时引入 FSDP、张量并行和
NCCL 的新变量；48 GB 单卡更适合先证明链路成立。A40 更省钱，L40/L40S/RTX 6000 Ada
更快；这只是选型建议，不是已经租用或跑通的结果。

当前 AutoDL 旧实例已关机、未释放。关机前已运行 `sync`，并生成：

```text
/root/autodl-tmp/handoff_20260809.tar.gz
```

截图确认合并后的 SFT 模型约 3.3 GB、RL 数据目录约 3.3 MB；
`/root/autodl-tmp/bird` 当时不存在，所以迁移后必须重新定位或下载 BIRD train 数据库。
没有有效的 GRPO checkpoint 需要抢救。

## 下一步

1. 在 AutoDL 同一区域用“克隆实例”迁移旧实例到单卡 48 GB，并勾选复制数据盘；
   旧实例先保留，等新实例验收后再释放。
2. 在新实例确认模型、RL 数据、交接压缩包和 verl commit
   `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`，不要升级或降级依赖。
3. 找回或重新下载 BIRD train 数据库，确认
   `$TEXT2SQL_DB_ROOT` 指向真实的 `train_databases`。
4. 从最新 Git 仓库复制 `configs/grpo/run_grpo.sh`、`run_smoke.sh`、奖励脚本和 `src/`
   到 `/root/autodl-tmp/`，再按 `docs/grpo-runbook.md` 运行奖励自检和三步 smoke。
5. 确认至少完成一个训练 step，并产生
   `rollouts.jsonl`；失败时保存完整报错和实际 verl commit。
6. smoke 通过后先提交环境配置与证据，再运行严格奖励版本；严格版本稳定后，
   才运行 `TEXT2SQL_REWARD_EXEC=0.3` 的朴素奖励对照。

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

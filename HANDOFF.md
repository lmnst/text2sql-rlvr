# 项目最终交接（2026-08-17）

## 一句话状态

项目已经收尾：BIRD 数据、只读 SQL 执行、官方/严格双评测、Qwen3-1.7B baseline、LoRA SFT、
linked schema、verl GRPO、checkpoint 导出和完整 val 评测全部跑通。确定性成果是 schema linking；
本次普通 GRPO 基本持平，不能写成有效提升。

## 可对外引用的结果

| 阶段 | 划分 | n | official EX | strict EX | run_id |
|---|---|---:|---:|---:|---|
| Qwen3-1.7B baseline | Mini-Dev | 500 | 19.80% | 17.40% | `43bb66bbe1e8` |
| LoRA SFT | Mini-Dev | 500 | 34.60% | 30.00% | `f1ef9b247255` |
| 强 SFT + full schema | 固定 train-val | 788 | 32.87% | 28.68% | `74d4d83c087c` |
| 强 SFT + linked schema | 固定 train-val | 788 | 37.94% | 33.63% | `3b9e91f891d8` |
| official-reward GRPO + linked schema | 固定 train-val | 788 | 38.20% | 33.88% | `53d1586c7d33` |

公平配对结论：

- linked 相对 full：official gain 76、loss 36，净增 40/788，即 +5.07 个百分点；
- GRPO 相对 linked SFT：official gain 12、loss 10，净增 2/788，即 +0.26 个百分点；
- 后者按“没有明确额外收益”处理，不声称 GRPO 有效涨分；
- BIRD dev 没有用于训练、调参、early stopping 或 checkpoint 选择，尚无 dev 成绩。

每个表格数字都能在 `results/runs.jsonl` 里按 run_id 唯一定位。详细解释见
`docs/FINAL_REPORT.md`，完整历史见 `docs/PROGRESS.md`。

## 本地仓库

- 当前分支：`main`。
- 最终实验记录提交：`e945064`；本次文档收尾会再产生一个提交。
- 本地完整测试：222 passed；pytest 因工作区权限无法写 `.pytest_cache`，不影响测试结果。
- 本地分支领先 `origin/main`，尚未 push；push 需要 GitHub 凭证或用户授权。
- `.pt2/` 和 `.pytest_tmp/` 是无法读取的忽略目录，Git 会打印 permission warning，tracked
  worktree 不受影响。

## 云端 F37 状态

评测结束时后处理状态为 `COMPLETE`，`nvidia-smi` 没有计算进程。实例可以直接关机止费；
关机保留数据，释放实例或制作不含数据盘的镜像可能丢失 `/root/autodl-tmp`。

关键产物：

```text
/root/autodl-tmp/Qwen3-1.7B-sft-strong-merged-f78ab16a
    最终公平对照使用的强 SFT 起点

/root/autodl-tmp/schema-val788-c9cf559/
    full / linked 的 788 题预测、逐题 outcomes 和评测日志

/root/autodl-tmp/out/grpo-linked-official-b1d7bdb-n500/
    最终普通 GRPO 的原始 verl checkpoints 和训练产物

/root/autodl-tmp/Qwen3-1.7B-grpo-b1-best-c1f62a1/
    选中的 step 10 LoRA checkpoint 合并后的 Hugging Face 模型

/root/autodl-tmp/grpo-val788-c1f62a1/
    GRPO 的 788 题预测、outcomes、评测日志和 paired 对比

/root/grpo_linked_official_b1d7bdb_train.log
    最终训练日志

/root/paired_sft_vs_grpo_c1f62a1.json
    SFT 与 GRPO 的逐题配对汇总
```

数据盘约 150 GB，收尾时只剩约 1.4 GB。不要在未清理前再导出模型。优先删除可以重建的旧
合并模型和作废实验 checkpoint；必须保留上面列出的强 SFT、最终 GRPO 原始 checkpoint、最终
合并模型、三组 outcomes 和日志。曾删除的只有一个可重建的 step 10 转换器测试模型，没有删除
训练 checkpoint。

## 最终 GRPO 配置

- 起点：强 SFT 合并模型；
- 数据：BIRD train 固定 500 prompt，seed=0，与固定 val 200 不重叠；
- schema：linked；
- 主奖励：BIRD official EX；strict 只作监控；
- 普通 GRPO，LoRA rank 32 / alpha 64，rollout n=4，30 step；
- thinking 关闭，max prompt 8192，max response 512；
- step 10/20/30 在 val 200 上均为 40.0%，按预设规则取并列中最早的 step 10；
- 不使用 Dynamic Sampling，也没有做大规模数据清洗。

训练代码提交为 `b1d7bdb`，checkpoint 转换与最终评测代码提交为 `c1f62a1`。verl 0.9 的
`.pt` LoRA checkpoint 已实测可以精确重建 PEFT 权重并合并成标准 Hugging Face 模型。

## 如果以后继续实验

现在不建议继续烧 GPU。先离线统计最终 rollout 中每个 prompt group 的 reward 组成：全 0、全 1、
有 0 有 1分别占多少。只有 mixed group 才会给普通 GRPO 提供组内相对优势。

判断顺序：

1. mixed group 很少：优先改善探索、模型能力或采样策略，再考虑 Dynamic Sampling；
2. mixed group 足够但 train/val 都不涨：检查学习率、LoRA 更新量和 reward 接入；
3. train 涨而 val 不涨：按过拟合或 official reward 偶然命中分析；
4. 没有新的诊断证据前，不重跑更长 GRPO，也不读取 BIRD dev。

## 写简历时

直接使用 `docs/resume-draft.md` 的推荐三条。核心叙事是：

1. 做成了可复现的执行反馈 RLVR 全链路；
2. 通过受控实验发现主要瓶颈是无关 schema 干扰，official EX 从 32.87% 提升至 37.94%；
3. 修正 reward 和 prompt 后，普通 GRPO 为 38.20%，因此诚实报告为无明确增益。

不要写“GRPO 显著提升”“BIRD dev 38.20%”“解决了 schema linking”或未运行的 reward-hacking
对照实验。

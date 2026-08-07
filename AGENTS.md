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

## 真实性要求

- 简历、README 和实验表只写日志或结果文件能够证明的数字。
- 每个指标注明数据划分、样本数、随机种子、checkpoint、评测脚本和硬件。
- reward hacking 必须给出明确定义、分母和可复现样例。
- 不预设 GRPO 一定提升，也不预设一定发生 entropy collapse。

## 工程约束

- 第一版只支持 SQLite，不提前加入 MySQL/PostgreSQL 兼容代码。
- 模型生成 SQL 只能在只读数据库上执行；拒绝多语句和 DDL/DML，并设置超时。
- 结果匹配必须保留重复行，显式处理 NULL、浮点值、列数和无 ORDER BY 时的行序。
- 优先复用 BIRD 官方评测脚本，项目内 wrapper 不得悄悄改变其指标口径。
- 配置、命令和实验元数据必须可复现；大数据、数据库和 checkpoint 不提交 Git。

## 沟通

- 默认中文，简洁直接。
- 发现设计问题直接提出修改方案。
- 未验证完成前使用“计划”“待验证”，不要写成“已实现”“已提升”。


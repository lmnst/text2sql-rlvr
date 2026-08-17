#!/usr/bin/env bash
# GRPO on one GPU, starting from the merged SFT model.
#
# Run run_smoke.sh before a full experiment. The AutoDL compatibility
# overrides below are observed fixes; hyperparameters still require validation.
set -euo pipefail

MODEL=${MODEL:-/root/autodl-tmp/Qwen3-1.7B-sft-merged}
DATA=${DATA:-/root/autodl-tmp/rl}
OUT=${OUT:-/root/autodl-tmp/out/grpo}
EXP=${EXP:-grpo_official}

# --- reward configuration, read by verl_reward.py -------------------------
# The main reward matches BIRD official EX. Strict equivalence is still logged
# for every rollout so duplicate-row metric exploitation remains visible.
export TEXT2SQL_DB_ROOT=${TEXT2SQL_DB_ROOT:-/root/autodl-tmp/train/train_databases}
export TEXT2SQL_REWARD_FORMAT=${TEXT2SQL_REWARD_FORMAT:-0}
export TEXT2SQL_REWARD_EXEC=${TEXT2SQL_REWARD_EXEC:-0}
export TEXT2SQL_REWARD_OFFICIAL=${TEXT2SQL_REWARD_OFFICIAL:-1}
export TEXT2SQL_REWARD_TIMEOUT=${TEXT2SQL_REWARD_TIMEOUT:-10}
export TEXT2SQL_ROLLOUT_LOG=${TEXT2SQL_ROLLOUT_LOG:-$OUT/rollouts.jsonl}

TOTAL_STEPS=${TOTAL_STEPS:-50}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/val.parquet" \
  data.train_batch_size=32 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=8192 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  ++actor_rollout_ref.model.lora.merge=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_model_len=8704 \
  actor_rollout_ref.rollout.load_format=safetensors \
  ++actor_rollout_ref.rollout.enable_sleep_mode=False \
  ++actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  custom_reward_function.path=/root/autodl-tmp/verl_reward.py \
  custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.logger=[console] \
  trainer.project_name=text2sql-rlvr \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$OUT" \
  "$@"

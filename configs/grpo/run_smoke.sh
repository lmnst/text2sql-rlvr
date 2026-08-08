#!/usr/bin/env bash
# Three conservative GRPO steps. These settings are the next planned attempt
# on one 48 GB GPU; they have not yet completed a training step.
set -euo pipefail

export TOTAL_STEPS=3
export EXP=grpo_smoke
export OUT=${OUT:-/root/autodl-tmp/out/grpo_smoke}

exec "$(dirname "$0")/run_grpo.sh" \
  data.train_batch_size=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"

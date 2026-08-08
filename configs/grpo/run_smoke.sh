#!/usr/bin/env bash
# Three steps of GRPO on a handful of prompts. The point is not to learn
# anything -- it is to find out which config keys this verl build accepts, and
# whether the reward function is reachable from inside the trainer, before
# paying for a real run.
set -euo pipefail

export TOTAL_STEPS=3
export EXP=grpo_smoke
export OUT=${OUT:-/root/autodl-tmp/out/grpo_smoke}

exec "$(dirname "$0")/run_grpo.sh" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.n=4 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"

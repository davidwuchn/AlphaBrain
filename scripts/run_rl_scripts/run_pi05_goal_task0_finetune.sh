#!/bin/bash
# Sanity-check pi05 finetune: LIBERO-goal task 0 ONLY, all ~43 demos.
# task 0 here = LIBERO benchmark's task_id 0 = "open the middle drawer of the
# cabinet" (the same target as RL's --task_id 0; NOT the lerobot dataset's
# task_index=0 which is "put the bowl on the plate").
#
# Recipe (post openpi-key-remap fix in trainer_tools.py):
#   per_device_batch_size 8 × 4 GPU × 1 accum = total batch 32  (small + fast)
#   max_train_steps 30000, save_interval 1000  (dense early ckpts for fast eval)
#   LR base/action/vl = 5e-5, warmup 10000, constant LR (min_lr=base)
#   pretrained_checkpoint: /datasets/pi05/model.safetensors  (raw openpi pi05)
# Run_id changed to Pi05-goal-task0-openpi so it doesn't collide with the
# previous random-init Pi05-goal-task0 dir (which was trained with the bug).
#
# GPUs: 1,2,3,4 (4 GPUs; skips 0/5/6 owned by other users / processes).
# Produces: results/training/Pi05-goal-task0-openpi/checkpoints/steps_{1000,...,30000}
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"

# Pin to user's available GPUs (1,2,3,4,6). With CUDA_VISIBLE_DEVICES set, the
# 5 visible devices are renumbered 0..4 inside the process — this matches
# `num_gpus: 5` in the mode config.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

exec bash scripts/run_finetune.sh pi05_goal_task0 "$@"

#!/bin/bash
# Pi05 finetune: LIBERO-goal multi-task, 5 traj per task (50 demos total).
# Mirrors run_pi05_goal_task0_finetune.sh — same recipe (raw openpi pi05 init,
# batch 32, constant LR after 10k warmup) — only difference: dataset_mix uses
# all 10 tasks with num_traj_per_task=5, no task_whitelist.
#
# Requires fixes already in place:
#   trainer_tools.py:load_pretrained_backbones — openpi key remap (97% load)
#   model2libero_interface.py:M1Inference — 'PaliGemmaPi05' in fw whitelist (no q99 double-unnorm at eval)
#
# GPUs: 0,1,2,3,4 (5 GPUs, total batch = 5×8×1 = 40).
# Produces: results/training/Pi05-goal-5traj-openpi/checkpoints/steps_{2500,5000,...,30000}
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"

exec bash scripts/run_finetune.sh pi05_goal_5traj_openpi "$@"

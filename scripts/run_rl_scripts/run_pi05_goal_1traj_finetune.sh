#!/bin/bash
# Pi05 finetune: LIBERO-goal multi-task, 1 traj per task (10 demos total).
# Mirrors run_pi05_goal_5traj_finetune.sh — only num_traj_per_task changed
# (5 → 1) and save_interval (2500 → 10000, since smaller run).
#
# Requires fixes already in place:
#   trainer_tools.py:load_pretrained_backbones — openpi key remap (97% load)
#   model2libero_interface.py:M1Inference — 'PaliGemmaPi05' in fw whitelist
#
# GPUs: 0,1,2,3,4 (5 GPUs, total batch = 5×8×1 = 40).
# Produces: results/training/Pi05-goal-1traj-openpi/checkpoints/steps_{10000,20000,30000}
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"

exec bash scripts/run_finetune.sh pi05_goal_1traj_openpi "$@"

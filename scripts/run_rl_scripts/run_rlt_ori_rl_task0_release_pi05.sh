#!/bin/bash
# Pi05 (PaliGemmaPi05) variant of run_rlt_ori_rl_task0_release.sh.
# Same hyperparams + steplock; only the VLA backbone and the Phase-1
# encoder are swapped to the Pi05 5traj pair.
#
# Phase-2 RL is wired through the new Pi05 inference adapter
# (AlphaBrain/training/reinforcement_learning/algos/RLT_ori/pi05_inference_zhanghe.py)
# which runs PaliGemma's prefix Gemma forward + flow-matching diffusion
# in a single fused call per rollout step. Trainer/rollout/eval all
# dispatch on framework type (is_pi05) — no behavior change for Qwen runs.
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_ori_rl_task0_release_pi05.sh [GPU_ID]
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_ID=${1:-0}
TASK_ID=${TASK_ID:-0}

CKPT_PATH="results/training/Pi05-5traj-libero_goal/checkpoints/steps_50000"
ENCODER_PATH="results/rlt_ori_training/pi05_5traj_libero_goal_strict_0426_0921/pretrain/checkpoints/pretrain_best/encoder.pt"

RUN_NAME="rlt_ori_rl_t${TASK_ID}_release_pi05"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_ori_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT_ori encoder not found: ${ENCODER_PATH}"
    echo "       Run scripts/run_rl_scripts/run_rlt_ori_pretrain.sh first."
    exit 1
fi

echo "============================================================"
echo " RLT_ori Phase-2 TD3 (release-pi05, libero_goal task ${TASK_ID})"
echo "   GPU:        ${GPU_ID}"
echo "   ckpt:       ${CKPT_PATH}"
echo "   encoder:    ${ENCODER_PATH}"
echo "   output:     ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --encoder_mode rlt_ori \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --task_id ${TASK_ID} \
    --rollout_gpus 0 \
    --train_gpu 0 \
    --bottleneck_dim 2048 \
    --encoder_layers 2 \
    --encoder_heads 8 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.1 \
    --G 64 \
    --group_size 8 \
    --num_envs 64 \
    --reward_coef 5.0 \
    --lr_actor 1e-3 \
    --lr_critic 1e-3 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 256 \
    --warmup_iters 5 \
    --td_updates_per_iter 10000 \
    --utd_ratio 10.0 \
    --td_batch_size 1024 \
    --tau 0.005 \
    --beta 1.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 300 \
    --eval_interval 10 \
    --eval_n_episodes 20 \
    --save_interval 25 \
    --save_video_interval 999 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1 \
    --use_steplock \
    2>&1 | tee "${TRAIN_LOG}"

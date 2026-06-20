#!/usr/bin/env bash
#SBATCH --job-name=lewm_train
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/%j.out
#SBATCH --error=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/%j.err

set -euo pipefail

IB=/lustre/work/vivatech-yentlteam/ibacry
G=/lustre/work/vivatech-yentlteam/gklajer
REPO="$G/eb_jepa"                 # gklajer's lewm-reproduction branch
cd "$REPO"

PY="$IB/venvs/eb_jepa_aarch64/bin/python"
RUN_DIR="${RUN_DIR:-$G/lewm_train/run1}"
mkdir -p "$RUN_DIR"
# Put the branch on the path so `eb_jepa.lewm` + `examples.lewm` resolve from here.
export PYTHONPATH="$REPO"
# WANDB_MODE (online/offline) and WANDB_API_KEY come from the submit environment
# (propagated via sbatch --export=ALL); never written to this tracked file.
export WANDB_MODE="${WANDB_MODE:-online}" \
       WANDB_DIR="$RUN_DIR" WANDB_CACHE_DIR="$G/.cache/wandb" \
       HF_HOME="$G/.cache/huggingface" TORCH_HOME="$G/.cache/torch" \
       TRITON_CACHE_DIR="$G/.cache/triton" XDG_CACHE_HOME="$G/.cache" \
       EBJEPA_CKPTS="$G/checkpoints"

echo "Host: $(hostname)"; echo "GPUs: $(nvidia-smi -L)"

srun "$PY" -m examples.lewm.main_lewm \
    --fname "${CFG:-examples/lewm/cfgs/lewm_two_rooms.yaml}" \
    --folder "$RUN_DIR" \
    ${EXTRA:-}

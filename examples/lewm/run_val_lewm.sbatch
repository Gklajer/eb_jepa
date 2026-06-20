#!/usr/bin/env bash
#SBATCH --job-name=lewm_val
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/val_%j.out
#SBATCH --error=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/val_%j.err
set -euo pipefail
IB=/lustre/work/vivatech-yentlteam/ibacry; G=/lustre/work/vivatech-yentlteam/gklajer
cd "$G/eb_jepa"
PY="$IB/venvs/eb_jepa_aarch64/bin/python"
export PYTHONPATH="$G/eb_jepa" XDG_CACHE_HOME="$G/.cache" TRITON_CACHE_DIR="$G/.cache/triton" \
       HF_HOME="$G/.cache/huggingface" TORCH_HOME="$G/.cache/torch" EBJEPA_CKPTS="$G/checkpoints"
srun "$PY" -m examples.lewm.val_lewm --ckpt "${CKPT:?}" ${EXTRA:-}

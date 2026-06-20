#!/usr/bin/env bash
#SBATCH --job-name=lewm_viz
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/viz_%j.out
#SBATCH --error=/lustre/work/vivatech-yentlteam/gklajer/lewm_train/logs/viz_%j.err

set -euo pipefail
IB=/lustre/work/vivatech-yentlteam/ibacry
G=/lustre/work/vivatech-yentlteam/gklajer
REPO="$G/eb_jepa"
cd "$REPO"
PY="$IB/venvs/eb_jepa_aarch64/bin/python"
export PYTHONPATH="$REPO" XDG_CACHE_HOME="$G/.cache" TRITON_CACHE_DIR="$G/.cache/triton" \
       HF_HOME="$G/.cache/huggingface" TORCH_HOME="$G/.cache/torch" EBJEPA_CKPTS="$G/checkpoints"

CKPT="${CKPT:?set CKPT}"
OUT="${OUT:-$(dirname "$CKPT")/viz}"
echo "Host: $(hostname)"; nvidia-smi -L

srun "$PY" -m examples.lewm.viz_lewm --ckpt "$CKPT" --out_dir "$OUT" ${EXTRA:-}

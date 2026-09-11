#!/bin/sh
#SBATCH --partition=<gpu-partition>
#SBATCH -N 1-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=100G

# Evaluate the unconstrained arm at its pre-divergence 200k-step archive.
#
#     sbatch -J eval_norenorm200k_<set> eval_runs/eval_spin_norenorm200k_3090.sh <set>
#
# set: mp | gnome. --constraint none is forward-identical to spin_renorm: False; never run
# the csv/oracle modes against this checkpoint.

set -e

SET="${1:-mp}"
case "$SET" in
    mp)    OOD_ARGS="" ;;
    gnome) OOD_ARGS="--ood_path /path/to/gnome_ecd --ood_name gnome_ecd" ;;
    *) echo "unknown set: $SET (expected mp or gnome)" >&2; exit 2 ;;
esac
CKPT="trained_models/spin_density/unconstrained_spin_electrafi.ckpt"
GRID_DIR="spin_eval_grids/norenorm200k_${SET}"

# Load your cluster's Python 3.11 / CUDA 12.1 / cuDNN / cuTENSOR modules here, e.g.
# module load <python> <cuda> <cudnn> <cutensor>
source .venv/bin/activate

if [ -z "$CUDA_HOME" ]; then
    echo "FATAL: CUDA module did not load; KeOps would run CPU-only and segfault." >&2
    exit 1
fi
python -c "
from keopscore.config.cuda import CUDAConfig
assert CUDAConfig()._use_cuda, 'KeOps is CPU-only: check the module line'
print('[preflight] KeOps CUDA OK')" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

mkdir -p eval_runs/configs logs spin_eval_csv
CFG="eval_runs/configs/eval_spin_norenorm200k_${SET}.yaml"

# Training config recipe minus training-only settings.
sed -e 's/^data_split:.*/data_split: mpfull2025/' \
    -e 's/^wandb:.*/wandb: False/' \
    -e 's/^ood_eval:.*/ood_eval: False/' \
    -e 's/^construct_test_cd:.*/construct_test_cd: False/' \
    -e 's/^write_test_chgcars:.*/write_test_chgcars: False/' \
    -e 's/^save_model:.*/save_model: False/' \
    -e 's/^save_memory:.*/save_memory: True/' \
    hpc_conf.yaml > "$CFG"
cat >> "$CFG" <<'EOF'

# --- spin channel, matching the full_spin_norenorm training config ---
spin_type: total_diff
spin_loss_weight: 0.2
spin_mag_min: 0.1
spin_warmup_steps: 500
sigma_trace_max: 0.0
wsum_rel_floor: 0.001
model_name: mpfull2025_spin_norenorm
EOF

srun python eval_spin_constrained.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --constraint none \
    $OOD_ARGS \
    --out_csv "spin_eval_csv/spin_eval_norenorm200k_${SET}.csv" \
    --save_grids "$GRID_DIR"

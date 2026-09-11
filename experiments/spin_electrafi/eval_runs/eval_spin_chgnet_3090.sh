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

# Evaluate the constrained spin model with an external net-moment constraint.
#
#     sbatch -J eval_spin_<set>_<mode> eval_runs/eval_spin_chgnet_3090.sh <set> <mode> [ckpt]
#
# set:  mp (mpfull2025 test split) | gnome (OOD .lz4 CHGCAR dir; dedupe the CSV by id)
# mode: csv (CHGNet unsigned site-moment sum, a ferromagnetic assumption) | oracle (target grid) | none
# Output: spin_eval_csv/spin_eval_<set>_<mode>.csv and per-structure spin grids under GRID_DIR.

set -e

SET="${1:-mp}"
MODE="${2:-csv}"
case "$SET" in
    mp)    ID_COL=MP_ID;    OOD_ARGS="" ;;
    gnome) ID_COL=GNOME_ID; OOD_ARGS="--ood_path /path/to/gnome_ecd --ood_name gnome_ecd" ;;
    *) echo "unknown set: $SET (expected mp or gnome)" >&2; exit 2 ;;
esac
case "$MODE" in
    csv|oracle|none) ;;
    *) echo "unknown mode: $MODE (expected csv, oracle or none)" >&2; exit 2 ;;
esac
CKPT="${3:-trained_models/spin_density/constrained_spin_electrafi.ckpt}"
GRID_DIR="spin_eval_grids/${SET}_${MODE}"

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
CFG="eval_runs/configs/eval_spin_${SET}_${MODE}.yaml"

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

# --- spin channel, matching the full_spin_nmae_nocap training config ---
spin_type: total_diff
spin_loss_weight: 0.2
spin_mag_min: 0.1
spin_warmup_steps: 500
sigma_trace_max: 0.0
wsum_rel_floor: 0.001
model_name: mpfull2025_spin_nmae_nocap
EOF

srun python eval_spin_constrained.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --constraint "$MODE" \
    --constraint_csv "eval_runs/chgnet_magmom_constraints_${SET}.csv" \
    --id_col "$ID_COL" \
    $OOD_ARGS \
    --out_csv "spin_eval_csv/spin_eval_${SET}_${MODE}.csv" \
    --save_grids "$GRID_DIR"

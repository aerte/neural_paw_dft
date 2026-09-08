#!/bin/sh
#SBATCH --partition=<gpu-partition>
#SBATCH --job-name=full_spin_nmae_nocap
#SBATCH -N 1-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=full_spin_nmae_nocap.log
#SBATCH --error=full_spin_nmae_nocap.err
#SBATCH --mem=240G
# Auto-requeue: SIGUSR1 (only delivered under srun) makes Lightning checkpoint and requeue;
# train.py resumes from the newest checkpoint.
#SBATCH --signal=SIGUSR1@600
#SBATCH --requeue

# Full mpfull2025 run of the NMAE-trained spin model at production width (2160/120).
# Do not change the code on disk while the job is live: --requeue restarts against it.

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

nvidia-smi

mkdir -p runs/configs
# max_time accumulates across requeues (Timer state is checkpointed): it is the total budget.
sed -e 's/^data_split:.*/data_split: mpfull2025/' \
    -e 's/^max_epochs:.*/max_epochs: 5/' \
    -e "s/^max_time:.*/max_time: '14:00:00:00'/" \
    -e 's/^save_memory:.*/save_memory: True/' \
    hpc_conf.yaml > runs/configs/hpc_full_spin.yaml
cat >> runs/configs/hpc_full_spin.yaml <<'EOF'

# --- spin channel (NMAE-trained; loss_type stays "normal" = NMAE for charge) ---
spin_type: total_diff
spin_loss_weight: 0.2
spin_mag_min: 0.1
spin_warmup_steps: 500

# --- divergence guards ---
sigma_trace_max: 0.0          # width cap off (a cap only pins the runaway at the cap)
skip_nonfinite_grads: True    # skip the step on any non-finite gradient
skip_loss_spike_factor: 50.0  # skip steps whose charge NMAE exceeds 50x a 0.99-EMA baseline
wsum_rel_floor: 0.001         # electron-renorm floor relative to sum|w|

# model_name isolates the checkpoint dir and wandb id from other mpfull2025 runs.
model_name: mpfull2025_spin_nmae_nocap
run_name: full_spin_nmae_nocap
EOF

srun python train.py --config runs/configs/hpc_full_spin.yaml

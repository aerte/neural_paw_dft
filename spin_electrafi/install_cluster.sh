# Load your cluster's Python 3.11 / CUDA 12.1 / cuDNN / cuTENSOR modules here, e.g.
# module load <python> <cuda> <cudnn> <cutensor>
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install uv
uv pip install -e .

#!/bin/bash
#SBATCH --job-name=vggface2-dino
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-dino-%j.out
#SBATCH --error=slurm-dino-%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Start    : $(date)"
echo "========================================"

# ── SIGTERM handler — checkpoint then requeue ─────────────────────────────────
PYTHON_PID=""
_slurm_term() {
    echo "[dino] SIGTERM — forwarding to Python (PID=$PYTHON_PID)..."
    [[ -n "$PYTHON_PID" ]] && kill -TERM "$PYTHON_PID" 2>/dev/null && wait "$PYTHON_PID" 2>/dev/null
    if [[ -n "${OUT_DIR:-}" && ! -f "$OUT_DIR/run_summary.json" ]]; then
        echo "[dino] Resubmitting job for next slice..."
        sbatch "$(realpath "${BASH_SOURCE[0]}")"
    fi
    exit 0
}
trap _slurm_term SIGTERM

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda

set +u
source activate rlft
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_DISABLE_ADDR2LINE=1

# Cache pretrained DINO weights on scratch to avoid re-downloading on requeue
export TORCH_HOME=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}/.cache/torch

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}
PROJECT_DIR=$HOME/vggface2_face_recognition/face_recognition

# VGGFace2 dataset from Kaggle:
#   https://www.kaggle.com/datasets/hearfool/vggface2
#
# Download steps (run once before submitting):
#   pip install kaggle
#   kaggle datasets download -d hearfool/vggface2 -p $SCRATCH/vggface2
#   unzip $SCRATCH/vggface2/vggface2.zip -d $SCRATCH/vggface2/
#   OR: sbatch unzip_vggface2.sh
#
# Expected layout:
#   $DATA_ROOT/train/n000001/*.jpg
#   $DATA_ROOT/test/n009131/*.jpg
DATA_ROOT=${VGGFACE2_DATA:-$SCRATCH/vggface2}

# Output directory
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${VGGFACE2_DINO_OUT:-$PROJECT_DIR/runs/dino_${TS}}"

# Resume latest incomplete run if one exists
if [[ -z "${VGGFACE2_DINO_OUT:-}" ]]; then
    LAST_RUN=$(ls -dt "$PROJECT_DIR/runs"/dino_[0-9]* 2>/dev/null | head -1 || true)
    if [[ -n "$LAST_RUN" && -f "$LAST_RUN/last.ckpt" && ! -f "$LAST_RUN/run_summary.json" ]]; then
        OUT_DIR="$LAST_RUN"
        echo "[dino] Resuming incomplete run: $OUT_DIR"
    fi
fi

export OUT_DIR
mkdir -p "$OUT_DIR"

# ── Validate data ─────────────────────────────────────────────────────────────
if [[ ! -d "$DATA_ROOT/train" ]]; then
    echo "[error] VGGFace2 train folder not found at: $DATA_ROOT/train"
    echo "  Download from: https://www.kaggle.com/datasets/hearfool/vggface2"
    echo "  Then run: sbatch unzip_vggface2.sh"
    exit 1
fi

echo "[dino] data  : $DATA_ROOT"
echo "[dino] output: $OUT_DIR"
echo "[dino] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"

# ── Dependency check ──────────────────────────────────────────────────────────
python - << 'PYEOF'
import importlib, importlib.util, sys, subprocess

needed = {
    "torch":                   "torch",
    "torchvision":             "torchvision",
    "pytorch_lightning":       "pytorch-lightning",
    "sklearn":                 "scikit-learn",
    "pytorch_metric_learning": "pytorch-metric-learning",
}
missing = [pkg for mod, pkg in needed.items() if not importlib.util.find_spec(mod)]
if missing:
    print(f"[deps] Installing: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
else:
    import torch
    print(f"[deps] torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"GPU: {torch.cuda.is_available()}")
PYEOF

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Training start: $(date)"
echo "========================================"

python -u "$PROJECT_DIR/dino_vggface2.py"   \
    --data           "$DATA_ROOT"           \
    --out            "$OUT_DIR"             \
    --backbone       dino_vitb16            \
    --backbone-dim   768                    \
    --embed-dim      256                    \
    --epochs         30                     \
    --warmup-epochs  2                      \
    --batch-size     32                     \
    --lr             3e-5                   \
    --workers        "$SLURM_CPUS_PER_TASK" \
    --save-every     3                      \
    --precision      16                     \
    --arcface                               \
    --arcface-epochs 10                     \
    --arcface-lr     1e-4                   \
    &

PYTHON_PID=$!
wait $PYTHON_PID
PYTHON_PID=""

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Job complete: $(date)"
if [[ -f "$OUT_DIR/run_summary.json" ]]; then
    echo "Summary:"
    cat "$OUT_DIR/run_summary.json"
else
    echo "[warn] run_summary.json not found — training may have been interrupted."
fi
echo "========================================"

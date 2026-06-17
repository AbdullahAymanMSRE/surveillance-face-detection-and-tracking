#!/bin/bash
#SBATCH --job-name=vggface2-arcface-v2
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-arcface-v2-%j.out
#SBATCH --error=slurm-arcface-v2-%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Start    : $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda

set +u
source activate rlft
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_HOME=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}/.cache/torch

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}
PROJECT_DIR=$HOME/vggface2_face_recognition/face_recognition
DATA_ROOT=${VGGFACE2_DATA:-$SCRATCH/vggface2}

# Continue from the post-DINO checkpoint of the existing run, write a fresh
# ArcFace stage into its own directory so the original baseline (rank-1 0.803,
# AUC 0.933) is left untouched for comparison.
BASE_RUN="$PROJECT_DIR/runs/dino_20260616_203019"
INIT_CKPT="$BASE_RUN/dino_epoch=epoch=29.ckpt"
OUT_DIR="$BASE_RUN/arcface_v2"
mkdir -p "$OUT_DIR"

if [[ ! -f "$INIT_CKPT" ]]; then
    echo "[error] init checkpoint not found: $INIT_CKPT"
    exit 1
fi

echo "[arcface-v2] data  : $DATA_ROOT"
echo "[arcface-v2] init  : $INIT_CKPT"
echo "[arcface-v2] output: $OUT_DIR"
echo "[arcface-v2] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"

echo ""
echo "========================================"
echo "Training start: $(date)"
echo "========================================"

python -u "$PROJECT_DIR/dino_vggface2.py"        \
    --data                  "$DATA_ROOT"         \
    --out                   "$OUT_DIR"           \
    --backbone               dino_vitb16         \
    --backbone-dim            768                \
    --embed-dim               256                \
    --batch-size               32                \
    --workers          "$SLURM_CPUS_PER_TASK"    \
    --arcface-only                                \
    --init-checkpoint        "$INIT_CKPT"        \
    --arcface-epochs           25                \
    --arcface-lr              1e-4               \
    --arcface-unfreeze-blocks   2

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

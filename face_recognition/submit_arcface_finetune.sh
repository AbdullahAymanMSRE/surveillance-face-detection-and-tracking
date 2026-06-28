#!/bin/bash
#SBATCH --job-name=arcface-onnx-finetune
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-arcface-onnx-%j.out
#SBATCH --error=slurm-arcface-onnx-%j.err

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

# ── Dependency check (onnx2torch stack is not part of the rlft env yet) ───────
python - << 'PYEOF'
import importlib.util, sys, subprocess

needed = {
    "onnx":        "onnx",
    "onnx2torch":  "onnx2torch",
    "onnxruntime": "onnxruntime",
    "onnxscript":  "onnxscript",
    "cv2":         "opencv-python-headless",
}
missing = [pkg for mod, pkg in needed.items() if not importlib.util.find_spec(mod)]
if missing:
    print(f"[deps] Installing: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
else:
    print("[deps] all present")
PYEOF

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}
PROJECT_DIR=$HOME/vggface2_face_recognition/face_recognition

# Production ONNX model to fine-tune (deployed surveillance recognizer).
ONNX_IN=${ARCFACE_ONNX_IN:-$PROJECT_DIR/models/w600k_mbf.onnx}

# VGGFace2 identity-folder dataset already on scratch (480 identities,
# same data used for the DINO runs). The notebook's default CASIA-WebFace
# path doesn't exist on this cluster, so we fine-tune on VGGFace2 instead —
# same ImageFolder layout (DATA_DIR/<identity>/<img>.jpg), so no code changes
# are needed, just a different --data root.
DATA_ROOT=${VGGFACE2_DATA:-$SCRATCH/vggface2}/train

# Optional: dir containing lfw.bin / cfp_fp.bin / agedb_30.bin for the
# standard verification benchmark. Leave unset to skip (not downloaded here).
VAL_BIN_DIR=${ARCFACE_VAL_BIN_DIR:-}

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ARCFACE_ONNX_OUT:-$PROJECT_DIR/runs/arcface_onnx_${TS}}"
mkdir -p "$OUT_DIR"
ONNX_OUT="$OUT_DIR/w600k_mbf_finetuned.onnx"

if [[ ! -f "$ONNX_IN" ]]; then
    echo "[error] production model not found at: $ONNX_IN"
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[error] training data not found at: $DATA_ROOT"
    exit 1
fi

echo "[arcface-onnx] onnx-in : $ONNX_IN"
echo "[arcface-onnx] data    : $DATA_ROOT"
echo "[arcface-onnx] out-dir : $OUT_DIR"
echo "[arcface-onnx] GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"

VAL_ARGS=()
if [[ -n "$VAL_BIN_DIR" ]]; then
    VAL_ARGS=(--val-bin-dir "$VAL_BIN_DIR")
fi

echo ""
echo "========================================"
echo "Training start: $(date)"
echo "========================================"

python -u "$PROJECT_DIR/arcface_finetune.py" \
    --onnx-in         "$ONNX_IN"             \
    --onnx-out        "$ONNX_OUT"            \
    --data            "$DATA_ROOT"           \
    --out             "$OUT_DIR"             \
    --epochs          60                     \
    --warmup-epochs   3                      \
    --batch-size      128                    \
    --lr              3e-5                   \
    --unfreeze-last   9999                   \
    --workers         "$SLURM_CPUS_PER_TASK" \
    "${VAL_ARGS[@]}"

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

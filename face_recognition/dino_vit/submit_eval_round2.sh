#!/bin/bash
#SBATCH --job-name=vggface2-eval-r2
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-eval-r2-%j.out
#SBATCH --error=slurm-eval-r2-%j.err

set -euo pipefail

PROJECT_DIR=$HOME/vggface2_face_recognition/face_recognition
DATA_ROOT=${VGGFACE2_DATA:-/scratch/gilbreth/$USER/vggface2}
RUN_DIR=$PROJECT_DIR/runs/dino_20260617_175717

/scratch/gilbreth/aelmersa/conda_envs/2025.06-py313/rlft/bin/python -u "$PROJECT_DIR/eval_dino_vggface2.py" \
    --data "$DATA_ROOT" \
    --run-dir "$RUN_DIR" \
    --checkpoint arcface_finetuned.pt

echo "=== eval_results.json ==="
cat "$RUN_DIR/eval_results.json"

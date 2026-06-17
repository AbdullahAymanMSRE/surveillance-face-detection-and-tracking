#!/bin/bash
#SBATCH --job-name=unzip-vggface2
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-unzip-%j.out
#SBATCH --error=slurm-unzip-%j.err

set -euo pipefail

echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "========================================"

SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}
ZIP="$SCRATCH/vggface2/vggface2.zip"
DEST="$SCRATCH/vggface2"

if [[ ! -f "$ZIP" ]]; then
    echo "[error] zip not found: $ZIP"
    exit 1
fi

echo "[unzip] Extracting $ZIP → $DEST"
unzip -o "$ZIP" -d "$DEST"

echo ""
echo "[unzip] Done: $(date)"
echo "Identities extracted: $(ls "$DEST/train/" | wc -l)"

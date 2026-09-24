#!/bin/bash
#SBATCH --job-name=preprocess_afrl
#SBATCH --output=/workspace/logs/preprocess_%j.out
#SBATCH --error=/workspace/logs/preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Load modules if your HPC uses them (uncomment and adjust)
# module load python/3.10

# Create and activate venv in job scratch/workdir
VENV_DIR="/venv"
python3 -m venv ""
source "/bin/activate"
python -m pip install --upgrade pip
pip install numpy pandas

# Move to project directory (adjust path on HPC if different)
cd ""

# Run preprocessing
python -u src/preprocessing.py \
  --data-root "/data03/home/smirzaei/AFRL/AFRL_datadin" \
  --out-root "/data03/home/smirzaei/AFRL/AFRL_datadin_out" \
  --all \
  --time 0.1

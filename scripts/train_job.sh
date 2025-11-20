#!/usr/bin/env bash
set -e

# ---- 1) Module laden (falls nötig) ----
module purge
module load gcc/9.4.0-pe5.34

# ---- 2) Virtuelle Umgebung aktivieren ----
VENV_PATH="/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/venv"
source $VENV_PATH/bin/activate

# ---- 3) Debug Info ----
echo "Python executable:"
which python

echo "Python version:"
python --version

echo "GPU Info:"
nvidia-smi

# ---- 4) Training starten ----
python /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/ADL_multimodel_retrieval.py

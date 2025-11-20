#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err

module purge
module load USS/2022       # zuerst dieses Modul
module load gcc/9.4.0-pe5.34

# virtuellen Environment aktivieren
source /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/.venv/bin/activate

# ---- 3) Debug Info ----
echo "Python executable:"
which python

echo "Python version:"
python --version

echo "GPU Info:"
nvidia-smi

# ---- 4) Training starten ----
python /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/ADL_multimodel_retrieval.py


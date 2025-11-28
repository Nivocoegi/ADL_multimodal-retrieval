#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err

# Module laden
module load gcc/9.4.0-pe5.34
module load USS/2022
module load miniconda3/4.12.0

# ---- Debug Info ----
echo "Python executable:"
/cfs/earth/scratch/vognic01/.conda/envs/ADL_multimodal-retrieval/bin/python --version

echo "GPU Info:"
nvidia-smi

# ---- Training starten ----
/cfs/earth/scratch/vognic01/.conda/envs/ADL_multimodal-retrieval/bin/python \
/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/251128_ADL_multimodal_training.py

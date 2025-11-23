#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err

# Module laden
module load gcc/9.4.0-pe5.34
module load USS/2022
module load miniconda3/4.12.0






# virtuellen Environment aktivieren
source /cfs/software/uss/2022/spack/linux-rocky8-x86_64/gcc-9.4.0/miniconda3-4.12.0-7riiqd3uthjsmxqkabjrfhfxhyh5epcl/etc/profile.d/conda.sh
conda activate /cfs/earth/scratch/vognic01/.conda/envs/ADL_multimodal-retrieval


# ---- 3) Debug Info ----
echo "Python executable:"
which python

echo "Python version:"
python --version

echo "GPU Info:"
nvidia-smi

# ---- 4) Training starten ----
python /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/251123_ADL_multimodel_retrieval.py


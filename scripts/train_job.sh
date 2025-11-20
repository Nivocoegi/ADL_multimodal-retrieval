#!/bin/bash

# ----------------------------
# Module laden
# ----------------------------
module load gcc/9.4.0-pe5.34
module load python3
# module load cuda/12.1

# ----------------------------
# Conda initialisieren
# ----------------------------
source ~/miniconda3/etc/profile.d/conda.sh

# ----------------------------
# Environment erstellen (falls noch nicht vorhanden)
# ----------------------------
ENV_PATH="/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/.conda_env"
ENV_YML="/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/environment.yml"

if [ ! -d "$ENV_PATH" ]; then
    conda env create -f "$ENV_YML" -p "$ENV_PATH"
fi

# ----------------------------
# Environment aktivieren
# ----------------------------
conda activate "$ENV_PATH"

# ----------------------------
# Training starten
# ----------------------------
python /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/notebooks/ADL_multimodel_retrieval.py

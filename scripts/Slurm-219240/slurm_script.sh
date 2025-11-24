#!/bin/bash
#SBATCH --job-name=adl_training
#SBATCH --output=/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/out_%j.txt
#SBATCH --error=/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=earth-4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ----------------------------
# Job starten
# ----------------------------
bash /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/scripts/train_job.sh

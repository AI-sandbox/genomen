#!/bin/bash
#SBATCH --job-name=genomen_eval
#SBATCH --partition=long
#SBATCH --mail-user=christho@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5500:0
#SBATCH --output=logs/tools/multi_phenotype/train/stdout-%A-%a.log
#SBATCH --time=3-00:00:00
#SBATCH --array=21
pwd; hostname; date
export PYTHONPATH=$(pwd)

echo "Running evaluation for array task ${SLURM_ARRAY_TASK_ID}"

uv run python genomen/tools/multi_phenotype/train/train_multiple_phenotypes.py --task_id "${SLURM_ARRAY_TASK_ID}" --use_phenotype_config False --backend "cpu"

date

echo "Done"
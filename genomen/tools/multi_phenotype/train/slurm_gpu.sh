#!/bin/bash
#SBATCH --job-name=metaprs_eval
#SBATCH --partition=gpu
#SBATCH --mail-user=christho@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem=178gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A5500:1
#SBATCH --output=logs/tools/multi_phenotype/train/stdout-%A-%a.log
#SBATCH --time=1-00:00:00
#SBATCH --array=4,14,18,21

pwd; hostname; date
export PYTHONPATH=$(pwd)
uv run python genomen/tools/multi_phenotype/train/train_multiple_phenotypes.py --task_id "${SLURM_ARRAY_TASK_ID}" --use_phenotype_config True --backend "gpu"

date

echo "Done"

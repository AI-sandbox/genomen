#!/bin/bash
#SBATCH --job-name=genomen_bootstrap_eval
#SBATCH --partition=long
#SBATCH --mail-user=christho@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5500:0
#SBATCH --output=logs/tools/multi_phenotype/eval/bootstrap_stdout-%A-%a.log
#SBATCH --time=14-00:00:00
#SBATCH --array=1-21
pwd; hostname; date
export PYTHONPATH=$(pwd)

echo "Running evaluation for array task ${SLURM_ARRAY_TASK_ID}"

uv run python genomen/multi_phenotype/eval/bootstrap_eval_on_phenotypes.py --task_id "${SLURM_ARRAY_TASK_ID}" --use_phenotype_config True --backend "cpu"

date

echo "Done"
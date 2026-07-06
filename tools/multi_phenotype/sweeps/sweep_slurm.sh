#!/bin/bash
#SBATCH --job-name=genomen_sweep
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=384gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/tools/multi_phenotype/sweeps/sweep-%A-%a.log
#SBATCH --time=7-00:00:00
#SBATCH --array=3

MODEL_TYPE=${1}
export MODEL_TYPE=$MODEL_TYPE
export TASK_ID=$SLURM_ARRAY_TASK_ID

echo "Running sweep with model type: $MODEL_TYPE and task ID: $TASK_ID"

# Run the sweep using the Python script
uv run python genomen/tools/multi_phenotype/sweeps/run_sweep.py --task_id=$TASK_ID --model_type=$MODEL_TYPE

echo "Done"
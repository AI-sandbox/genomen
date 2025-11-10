#!/bin/bash
#SBATCH --job-name=genomen_test
#SBATCH --partition=long
#SBATCH --mail-user=christho@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5500:0
#SBATCH --output=test2.log
#SBATCH --time=14-00:00:00
pwd; hostname; date
export PYTHONPATH=$(pwd)

uv run python main.py

echo "Done"
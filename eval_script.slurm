#!/bin/bash

#SBATCH --job-name=cse562

#SBATCH --account=socialrl
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=24:00:00 # Max runtime in HH:MM:SS format

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/cse562-final-project/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/cse562-final-project/logs/%x.%j.err
#SBATCH --open-mode=append

# Add requeue flag
#SBATCH --requeue

cd /mmfs1/gscratch/socialrl/kjha/cse562-final-project

source ~/.bashrc
conda activate cmarl
module load cuda/12.6

thinking_mode=$1
main_model=$2
small_model=${3:-"none"}  # Optional parameter, defaults to "none"
prompt_mode=${4:-"seconds"}  # Optional parameter, defaults to "seconds"

echo "thinking_mode: $thinking_mode"
echo "main_model: $main_model"
echo "small_model: $small_model"
echo "prompt_mode: $prompt_mode"

# Construct command based on thinking mode
if [ "$thinking_mode" -eq 2 ]; then
    python eval.py --thinking_mode $thinking_mode \
                  --main_model "$main_model" \
                  --small_model "$small_model" \
                  --prompt_mode "$prompt_mode"
else
    python eval.py --thinking_mode $thinking_mode \
                  --main_model "$main_model" \
                  --prompt_mode "$prompt_mode"
fi
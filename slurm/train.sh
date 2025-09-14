#!/bin/bash
#SBATCH -J DQN
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --save_postfix steps=100 algo_config.algortihm=RANDOM

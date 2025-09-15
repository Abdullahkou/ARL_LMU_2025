#!/bin/bash
#SBATCH -J DQN
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --save_postfix="1qh" steps=1000 algo_config.algortihm= algo_config.hyper_params.n_q_heads=1

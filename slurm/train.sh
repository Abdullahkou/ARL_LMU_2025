#!/bin/bash
#SBATCH -J DQN
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --save_postfix="3qh" algo_config.hyper_params.n_q_heads=3 env_id=Taxi-v3  algo_config.algorithm= steps=

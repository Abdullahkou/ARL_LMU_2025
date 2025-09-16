#!/bin/bash
#SBATCH -J DQN_1
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --save_postfix="1qh" seeds=[0,1,2,3,4,5,6,7,8,9] algo_config.hyper_params.n_q_heads=1 env_id=LunarLander-v3 algo_config.algorithm=Custom_DQN steps=1000000

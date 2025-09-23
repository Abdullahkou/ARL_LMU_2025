#!/bin/bash
#SBATCH -J DQN_3
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

# dont forget to adjust the postfix!! 3qh
# [0,1,2,3,4,5,6,7,8,9]
uv run src/train.py --save_postfix="3qh" \
 algo_config.hyper_params.n_q_heads=3 \
 seeds=[0,1,2,3,4,5,6,7,8,9] \
 algo_config.hyper_params.hidden_sizes=[64,64] \
 algo_config.hyper_params.bootstrap_prob=1.0 \
 algo_config.hyper_params.independent_heads=False \
 env_id=LunarLander-v3 \
 algo_config.algorithm=Custom_DQN \
 steps=1000000

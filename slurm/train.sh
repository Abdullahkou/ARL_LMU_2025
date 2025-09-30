#!/bin/bash
#SBATCH -J test_dqn
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

# dont forget to adjust the postfix!! 3qh
# [0,1,2,3,4,5,6,7,8,9]
uv run src/train.py --save_postfix="5qh_16x16_bp0.5" \
 algo_config.hyper_params.n_q_heads=5 \
 seeds=[3] \
 algo_config.hyper_params.hidden_sizes=[16,16] \
 algo_config.hyper_params.bootstrap_prob=0.5 \
 algo_config.hyper_params.independent_heads=False \
 env_id=MountainCar-v0 \
 algo_config.algorithm=Custom_DQN \
 steps=1000000

#!/bin/bash
#SBATCH -J DQN_3_auto
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

# dont forget to adjust the postfix!! 3qh_bpauto
uv run src/train.py --save_postfix="3qh_bpauto" algo_config.hyper_params.n_q_heads=3 algo_config.algorithm=Custom_DQN env_id=LunarLander-v3 seeds=[0,1,2,3,4,5,6,7,8,9] steps=1000000

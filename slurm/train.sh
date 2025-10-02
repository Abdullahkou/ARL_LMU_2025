#!/bin/bash
#SBATCH -J tx_1q_b1
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

# dont forget to adjust the postfix!! 3qh_32x32_s_sh
# [0,1,2,3,4,5,6,7,8,9]
uv run src/train.py --save_postfix="1qh_64x64_sh_bp1.0" \
 algo_config.hyper_params.n_q_heads=5 \
 seeds=[0,1,2,3,4,5,6,7,8,9] \
 algo_config.hyper_params.hidden_sizes=[64,64] \
 algo_config.hyper_params.bootstrap_prob=1.0 \
 env_id=Taxi-v3 \
 algo_config.algorithm=Custom_DQN \
 steps=200000 \
 algo_config.hyper_params.ebql_strict=False \
 record_heads=False \
 algo_config.hyper_params.independent_heads=False \
 algo_config.hyper_params.single_head_action_select=True \

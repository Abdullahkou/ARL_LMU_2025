#!/bin/bash
#SBATCH -J FL3q0532
#SBATCH -D ./
#SBATCH --partition=All
#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

# ==========================
# Variablen #SBATCH -J GreenNameGene
# ==========================
HEADS=3
ENV="FrozenLake-v1"
ALGO="Custom_DQN"
SEEDS="[0,1,2,3,4,5,6,7,8,9]"
STEPS=1000000

# Env-Kürzel automatisch ziehen (alles vor dem Bindestrich nehmen)
ENV_SHORT=$(echo $ENV | cut -d'-' -f1)

SAVE_POSTFIX="${ENV_SHORT}_05bp_32_ebql_${HEADS}qh"

# ==========================
# Training starten
# ==========================
cd ..

uv version

uv run src/train.py \
  --save_postfix="$SAVE_POSTFIX" \
  algo_config.hyper_params.n_q_heads=$HEADS \
  algo_config.algorithm=$ALGO \
  env_id=$ENV \
  seeds=$SEEDS \
  steps=$STEPS


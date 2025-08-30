#!/bin/bash
#SBATCH -J RANDOM
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --algos RANDOM --env_id Pendulum-v1
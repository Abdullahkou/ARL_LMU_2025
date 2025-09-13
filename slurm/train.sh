#!/bin/bash
#SBATCH -J RANDOM
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/train.py --set steps=100 --set eval_episodes=3 --set eval_phases=3

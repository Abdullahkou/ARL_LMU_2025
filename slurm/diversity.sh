#!/bin/bash
#SBATCH -J ensembles_diversity
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

cd ..

uv version

uv run src/measure_diversity.py
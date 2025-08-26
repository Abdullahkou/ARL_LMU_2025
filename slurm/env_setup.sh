#!/bin/bash
#SBATCH -J env_install
#SBATCH -D ./
#SBATCH --partition=All

#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=ALL

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

# install environment
uv sync
# ARL_LMU_2025 Ensemble Trainer

## Overview

ARL_LMU_2025 is an ensemble training project designed for machine learning research and experimentation. It enables users to train, evaluate, and compare multiple models as an ensemble, improving predictive performance and robustness.

## Features

- Support for various ensemble methods (bagging, boosting, stacking)
- Automated training, validation, and testing pipelines
- Configurable hyperparameter tuning

## Installation

```bash
git clone <repository_url> 
cd ARL_LMU_2025
uv sync
```

## Local Usage

1. Configure your train config in [src/training_config.yaml](src/training_config.yaml).
2. Configure the trainers results dir within ``main`` of [src/train.py](src/train.py).
3. Run the trainer:

    ```bash
    uv run src/train.py --algos <algorithm> --env_id <gym_env_id>
    ```

4. View the results in the your specified results directory.

## Slurm Usage

### General

- login to the cluster using:

    ```bash
    ssh <cip_username>@remote.cip.ifi.lmu.de
    ```

    this will get you into the login node of the cluster.

- to see the available nodes and their status use:

    ```bash
    sinfo
    ```

- we can always manually get onto one of the available nodes using and do our work directly (not recommended for long running jobs):

    ```bash
    ssh <node_name>
    ```

- to submit a job use:

    ```bash
    sbatch <script_name>.sh
    ```

    where we specify at the top of the script as comments (cf. [slurm/slurm_test.sh](slurm/slurm_test.sh)) the setup we need, e.g. the partition, ...
    this will schedule your job onto an available node automatically.

- to see the jobs you have running or queued use:

    ```bash
    squeue -u <cip_username>
    ```

- to cancel a job use:

    ```bash
    scancel <job_id>
    ```

### Env Setup and Training

1. run the setup script to install `uv` and the `environment`:

    ```bash
    cd slurm
    sbatch ./env_setup.sh
    ```

    the `sbatch` command will schedule your install job onto a node, as we are not supposed to run compute intensive tasks on the login node we are on by default!

2. to check if the installation was successful you can periodically check the output file `*.out` that is created upon submission

3. then to train your model you can submit the training script:

    ```bash
    cd slurm
    sbatch train.sh
    ```

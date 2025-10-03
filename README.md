# ARL_LMU_2025 Ensemble Trainer

## Setup

This Project uses ``uv`` to manage dependencies.

run ```uv sync``` to install the ``.venv``. In case theres a build problem with `gymnasium[box2d]` (This may or may not happen), remove the dependency entry first from the [pyproject.toml](pyproject.toml) file, run the command again and activate the venv. Then revert your change to [pyproject.toml](pyproject.toml) and run the install one more time. (NOTE: The activated ``.venv`` seems necessary to have the pip package ``swig`` for available for the install!)

## Overview

ARL_LMU_2025 is an ensemble training project designed for machine learning research and experimentation. It enables users to train, evaluate, and compare multiple ensemble configurations of ``EBQL`` (An adaptation to a Qlearning Learner), seeking to improving predictive performance.

## Features

- Support for various ``ensemble configs`` -> [training_config.yaml](src/training_config.yaml)
- Automated training, validation, and testing ``pipeline`` -> [train.py](src/train.py)
- Configurable ``plotter`` -> [plot_results.py](src/plot_results.py)
- ``slurm`` ready-to-run script -> [train.sh](slurm/train.sh)

## Usage

### Local Usage

1. Configure your ``train config`` in [training_config.yaml](src/training_config.yaml).
2. Configure the trainers ``results base dir`` within ``main`` of [train.py](src/train.py).
3. Run the trainer. By default you shouldnt need to pass any args to the script. But if desired, you may also specify any config params from [training_config.yaml](src/training_config.yaml) as args. See the [train.sh](slurm/train.sh) for reference.

    ```bash
    uv run src/train.py
    ```

4. View the results in the your specified ``results directory``.

### Slurm Usage

#### General

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

#### Env Setup and Training

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

#### Retrieving saved Checkpoints

By default any saved model files ending in `.zip` or ``.pt`` will be not be tracked by git. To upload them to your local machine you can use the provided scripts: [file_transfer.ps1](slurm/file_transfer.ps1) or [file_transfer.sh](slurm/file_transfer.sh). If asked for the ``src`` path, enter the absolute path to the directory or file you want to transfer from. By default it will then transfer into your current directory. But you may also specify a different destination path via the `-dest` argument.

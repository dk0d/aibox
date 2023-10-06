#!/bin/bash

{HEADER}
#SBATCH -o {LOG_DIR}/{NAME}.%J.log
#SBATCH -J {NAME}

{BASH_SETUP}

echo "Date      : $(date)"
echo "host      : $(hostname -s)"
echo "Directory : $(pwd)"

module purge
{MODULES}

echo "$(nvidia-smi | grep Version)"
echo "Running $SLURM_JOB_NAME on $SLURM_CPUS_ON_NODE CPU cores"
{EXPORTS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PATH=$PATH:{ENV_PATH}/bin

{RAY_TUNE}

source activate {ENV_PATH}

{COMMAND}

echo "End Date    : $(date)"

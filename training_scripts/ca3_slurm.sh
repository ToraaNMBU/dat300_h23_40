#!/bin/bash
#SBATCH --ntasks=16          # 1 core (CPU)
#SBATCH --nodes=1            # Use 1 node
#SBATCH --job-name=training  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gres=gpu:1         # Use one GPUs
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=./output_logs/ca3_training%j.out # Stdout and stderr file       ## CHANGE

## Script commands
module load singularity

SIFFILE="$COURSES/DAT300_H23/singularity_container/container_dat300_h23.sif"


## RUN THE PYTHON SCRIPT
# Using a singularity container named container_u_net.sif
singularity exec --nv $SIFFILE python ca03.py                             ## CHANGE


#!/bin/bash

#SBATCH --mail-user=hsdhillon@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH --account=rbe549
#SBATCH -p academic

#SBATCH -J train
#SBATCH --output=train.out
#SBATCH --error=train.err

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A30

#SBATCH -t 9:00:00

python3 Train.py


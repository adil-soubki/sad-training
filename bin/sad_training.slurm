#!/bin/bash
#SBATCH --job-name=sad_training
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

PYTHON=/home/asoubki/.miniconda3/envs/mmcg/bin/python
BIN=/home/asoubki/dev/sad-training/bin/sad_training.py
CONFIG=/home/asoubki/dev/sad-training/configs/sad_training/early-fusion.json
$PYTHON $BIN $CONFIG -t commitment_bank

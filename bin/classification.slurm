#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

PYTHON=/home/asoubki/.miniconda3/envs/face/bin/python
BIN=/home/asoubki/dev/face/bin/classification.py
CONFIG=/home/asoubki/dev/face/configs/llama3.json
$PYTHON $BIN $CONFIG

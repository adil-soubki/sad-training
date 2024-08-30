#!/bin/bash
#SBATCH --job-name=seq2seq
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

PYTHON=/home/asoubki/.miniconda3/envs/face/bin/python
BIN=/home/asoubki/dev/calm/bin/seq2seq.py
CONFIG=/home/asoubki/dev/calm/configs/seq2seq/default.json
$PYTHON $BIN $CONFIG

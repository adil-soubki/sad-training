#!/bin/bash

while getopts ":m:t:" opt; do
  case "$opt" in
    m) modality="$OPTARG" ;;
    t) task="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

shift $((OPTIND-1))

# Check if options are set, otherwise use defaults.
if [ -z "$modality" ]; then
    modality=("text-only" "audio-only" "early-fusion" "late-fusion");
fi
if [ -z $task ]; then
    task=(
        "boolq" "commitment_bank" "commitment_bank_text_only"
        "fact_bank" "goemotions" "wsc" "wic"
    );
fi

# Iterate over arrays.
for t in "${task[@]}"; do
    for m in "${modality[@]}"; do
        echo "Task: $t -- Modality: $m"
        PYTHON=/home/asoubki/.miniconda3/envs/mmcg/bin/python
        BIN=/home/asoubki/dev/sad-training/bin/sad_training.py
        CONFIG=/home/asoubki/dev/sad-training/configs/sad_training/$m.json

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=sad_training
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

$PYTHON $BIN $CONFIG -t $t
EOT
    done
done


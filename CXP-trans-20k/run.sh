#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
source "/h/sinaakb/TEEEEESSSSST/Tensorboard-Pytorch/bin/activate"
python main.py



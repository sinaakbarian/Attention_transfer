#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
pip list | grep -F tensorflow
pip list | grep -F tensorboard
source "/h/sinaakb/TEEEEESSSSST/Tensorboard-Pytorch/bin/activate"
pip list | grep -F tensorboard
pip list | grep -F tensorflow 

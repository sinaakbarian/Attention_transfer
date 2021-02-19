#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p  gpu
#SBATCH --mem=1GB
#SBATCH -o Tensorboard.out
tensorboard --logdir=runs

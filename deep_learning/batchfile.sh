#!/bin/bash
#SBATCH -J bear_analysis
#SBATCH -o outputfile%a.txt
#SBATCH -e errorfile%a.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=25G
#SBATCH -t 60
#SBATCH --array 0-29

/users/shawd/miniconda3/envs/capstone/bin/python /users/shawd/repos/MSDS_Capstone/deep_learning/multivar_lstm.py $1
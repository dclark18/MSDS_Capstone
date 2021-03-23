#!/bin/bash
#SBATCH -J bear_analysis
#SBATCH -o outputfile%a.txt
#SBATCH -e errorfile-%a.txt
#SBATCH -p fp-gpgpu-3
#SBATCH -t 60
#SBATCH -D $1
#SBATCH --array 0-29

/users/shawd/miniconda3/envs/capstone/bin/python attention.py
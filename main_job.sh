#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH -c 16
#SBATCH -o slurm-%j.out  # %j = job ID

conda activate bpd
python non_hrl_fits.py

#!/bin/bash
#SBATCH -c 20  # Number of Cores per Task
#SBATCH --mem=100g  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -G 0  # Number of GPUs
#SBATCH --time 1-00:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
source ~/miniconda/bin/activate
conda activate transformers_env
export HF_HOME=/project/pi_rrahimi_umass_edu/vishalg/
python Datasets.py
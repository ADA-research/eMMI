#!/bin/sh
#SBATCH --job-name=test
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-1
#SBATCH --output=/scratch/arp/slurm_output/IPQ_test_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_test_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/test.py
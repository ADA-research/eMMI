#!/bin/sh
#SBATCH --job-name=est_eps
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-20
#SBATCH --output=/scratch/arp/slurm_output/IPQ_est_eps_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_est_eps_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/estimate_epsilon.py $SLURM_ARRAY_TASK_ID
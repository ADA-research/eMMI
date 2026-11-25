#!/bin/sh
#SBATCH --job-name=IPQ_budget
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-5
#SBATCH --output=/scratch/arp/slurm_output/IPQ_budget_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_budget_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/budget_experiment.py $SLURM_ARRAY_TASK_ID
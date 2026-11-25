#!/bin/sh
#SBATCH --job-name=exp_uo
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-45
#SBATCH --exclude=ethnode[01-05]
#SBATCH --output=/scratch/arp/slurm_output/IPQ_exp_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_exp_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/run_experiments.py $SLURM_ARRAY_TASK_ID uniform own
#!/bin/sh
#SBATCH --job-name=val_eM
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-14
#SBATCH --output=/scratch/arp/slurm_output/IPQ_val_eM_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_val_eM_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/validate_manifolds.py $SLURM_ARRAY_TASK_ID
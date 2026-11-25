#!/bin/sh
#SBATCH --job-name=slc
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-150
#SBATCH --output=/scratch/arp/slurm_output/IPQ_slc_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_slc_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/shift_loss_correlation.py $SLURM_ARRAY_TASK_ID
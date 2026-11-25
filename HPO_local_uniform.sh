#!/bin/sh
#SBATCH --job-name=HPO_lu
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-28
#SBATCH --output=/scratch/arp/slurm_output/IPQ_HPO_lu_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_HPO_lu_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/HPO.py $SLURM_ARRAY_TASK_ID uniform all
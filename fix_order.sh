#!/bin/sh
#SBATCH --job-name=order
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --output=/scratch/arp/slurm_output/IPQ_order.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_order.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/fix_order.py
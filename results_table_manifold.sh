#!/bin/sh
#SBATCH --job-name=res_tab
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --output=/scratch/arp/slurm_output/IPQ_res_tab_manifold_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_res_tab_manifold_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/results_table_manifold.py
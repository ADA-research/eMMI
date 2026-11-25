#!/bin/sh
#SBATCH --job-name=gls
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-34
#SBATCH --exclude=ethnode[01-02]
#SBATCH --output=/scratch/arp/slurm_output/IPQ_gls_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_gls_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/generate_local_samples.py $SLURM_ARRAY_TASK_ID
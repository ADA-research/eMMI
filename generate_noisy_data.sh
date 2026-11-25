#!/bin/sh
#SBATCH --job-name=gen_data_noise
#SBATCH --partition=graceCPU
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=1-30
#SBATCH --exclude=ethnode[01-05]
#SBATCH --output=/scratch/arp/slurm_output/IPQ_gen_data_noise_out.txt
#SBATCH --error=/scratch/arp/slurm_output/IPQ_gen_data_noise_err.txt

source activate RTM
srun ~/anaconda3/envs/RTM/bin/python ~/AutoIPQ/generate_noisy_data.py $SLURM_ARRAY_TASK_ID
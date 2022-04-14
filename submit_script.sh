#!/bin/bash
#SBATCH --account KIETZMANN-SL2-GPU
#SBATCH --partition ampere
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module purge
module load rhel7/default-amp
module unload cuda/8.0
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/hpcgies1/rds/hpc-work/tf24/bin/activate
python /home/hpcgies1/rds/hpc-work/NIC/Masters-Thesis/AttemptFour/main.py

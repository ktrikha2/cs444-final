#!/bin/bash

#SBATCH --job-name=swin_detr       # updated job name
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g

#SBATCH --mail-user=dsingh102@illinois.edu
#SBATCH --mail-type=NONE

#SBATCH --output=/work/nvme/bfdu/dsingh10/logs/swin_run1/swin-%x-%j.out
#SBATCH --error=/work/nvme/bfdu/dsingh10/logs/swin_run1/swin-%x-%j.err

# Load required modules
module load python/3.10.14 cuda/12.4.0 gcc-native/12.3

# Activate your virtual environment
source /work/nvme/bfdu/dsingh10/code/cs444-final/env_swin/bin/activate

# Navigate to project root
cd /work/nvme/bfdu/dsingh10/code/cs444-final

# Environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $SLURM_NODELIST"
nvidia-smi || true

# Run the training script for the new SwinDETR model
python src/Training/train_CNN_Swin.py --config Configs/swin_cnn.yaml

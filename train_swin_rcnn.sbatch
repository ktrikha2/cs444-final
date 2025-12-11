#!/bin/bash

#SBATCH --job-name=swin_detr
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g


#SBATCH --mail-user=dipalis2@illinois.edu # replace with your email
#SBATCH --mail-type=NONE


#SBATCH --output=/work/nvme/bfdu/dsingh10/logs/swindetr_final_150ep_night/swin-%x-%j.out
#SBATCH --error=/work/nvme/bfdu/dsingh10/logs/swindetr_final_150ep_night/swin-%x-%j.err

module load python/3.10.14 cuda/12.4.0 gcc-native/12.3
source /work/nvme/bfdu/dsingh10/code/cs444-final/env_swin/bin/activate

cd /work/nvme/bfdu/dsingh10/code/cs444-final-fin

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0



echo "Running on node: $SLURM_NODELIST"
nvidia-smi || true


cd /work/nvme/bfdu/dsingh10/code/cs444-final-fin
python src/Training/train_CNN_Swin.py \
  --config Configs/swin_cnn_night.yaml
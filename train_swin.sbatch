#!/bin/bash

#SBATCH --job-name=swin_detr
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g


#SBATCH --mail-user=ktrikha2@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE


#SBATCH --output=/work/nvme/bfdu/ktrikha/logs/swindetr_full/swin-%x-%j.out
#SBATCH --error=/work/nvme/bfdu/ktrikha/logs/swindetr_full/swin-%x-%j.err

module load python/3.10.14 cuda/12.4.0 gcc-native/12.3
source /work/nvme/bfdu/ktrikha/code/cs444-final/swin_env/bin/activate

cd /work/nvme/bfdu/ktrikha/code/swin_debug

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0



echo "Running on node: $SLURM_NODELIST"
nvidia-smi || true


cd /work/nvme/bfdu/ktrikha/code/swin_debug
python src/Training/train_CNN_Swin.py \
  --config Configs/swin_cnn.yaml
#!/bin/bash
#SBATCH --job-name=swin-debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfdu/ktrikha/logs/swin_frcnn/swin_debug_%j.out
#SBATCH --error=/work/nvme/bfdu/ktrikha/logs/swin_frcnn/swin_debug_%j.err

cd /work/nvme/bfdu/ktrikha/code/cs444-final

source swin_env/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6

python src/Training/Train.py \
  --config Configs/swin_detection_delta.yaml

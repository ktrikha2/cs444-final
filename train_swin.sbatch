#!/bin/bash

#SBATCH --job-name=swin-debug
#SBATCH --account=bfdu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00

#SBATCH --mail-user=ktrikha2@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE


#SBATCH --output=/work/nvme/bfdu/ktrikha/logs/swin_frcnn/swin-%x-%j.out
#SBATCH --error=/work/nvme/bfdu/ktrikha/logs/swin_frcnn/swin-%x-%j.err

module load python/3.10.14 cuda/12.4.0 gcc-native/12.3
source /work/nvme/bfdu/ktrikha/code/cs444-final/swin_env/bin/activate

cd /work/nvme/bfdu/ktrikha/code/cs444-final

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0



echo "Running on node: $SLURM_NODELIST"
nvidia-smi || true


cd /work/nvme/bfdu/ktrikha/code/cs444-final
python src/Training/Train.py \
  --config Configs/swin_detection_delta.yaml

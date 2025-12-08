#!/bin/bash

#SBATCH --job-name=swin_train_eval
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g

#SBATCH --mail-user=dsingh102@illinois.edu
#SBATCH --mail-type=NONE

#SBATCH --output=/work/nvme/bfdu/dsingh10/logs/swin_run1/swin-%x-%j.out
#SBATCH --error=/work/nvme/bfdu/dsingh10/logs/swin_run1/swin-%x-%j.err

# Load modules
module load python/3.10.14 cuda/12.4.0 gcc-native/12.3

# Activate environment
source /work/nvme/bfdu/dsingh10/code/cs444-final/env_swin/bin/activate

# Move to project root
cd /work/nvme/bfdu/dsingh10/code/cs444-final

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $SLURM_NODELIST"
nvidia-smi || true

#######################
#        TRAIN
#######################
echo "Starting training..."
python src/Training/train_CNN_Swin.py --config Configs/swin_cnn.yaml

TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi


#######################
#        INFER
#######################
# Adjust these paths if your model lands somewhere else
CKPT_PATH="/work/nvme/bfdu/dsingh10/checkpoints/swin_cnn/swin_detr_epoch20.pth"
OUT_PATH="/work/nvme/bfdu/dsingh10/output/val_preds1.json"

echo "Starting inference..."
python src/Training/Eval_swin_detr.py \
    --config Configs/swin_cnn.yaml \
    --ckpt "$CKPT_PATH" \
    --out "$OUT_PATH"

INFER_EXIT=$?
echo "Inference exit code: $INFER_EXIT"

exit $INFER_EXIT

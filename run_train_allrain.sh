#!/bin/bash
#SBATCH --gpus=8
#SBATCH --gres=gpu:a100:8
#SBATCH -p a100x
#SBATCH --job-name=restormer_allrain
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# ============================================================
# Restormer 训练脚本 - 多源去雨数据集 (8x A100)
# ============================================================
# 使用前请先运行数据准备脚本:
#   python prepare_training_data.py
#
# 然后提交此任务:
#   sbatch run_train_allrain.sh
#
# 或直接运行 (非SLURM环境):
#   bash run_train_allrain.sh
# ============================================================

# 卸载所有已加载的CUDA模块，避免冲突
module purge 2>/dev/null

# 加载CUDA环境 (需要11.8以支持A100 GPU)
module load CUDA/11.8 2>/dev/null

# 初始化conda并激活环境
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate pytorch181 2>/dev/null

# 打印环境信息
echo "============================================================"
echo "Restormer Training - AllRain Dataset (8x A100)"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>&1)"
echo ""

# 检查数据是否已准备
TRAIN_DIR="./Deraining/Datasets/train/AllRain/input"
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "[错误] 训练数据未准备! 请先运行:"
    echo "  python prepare_training_data.py"
    exit 1
fi

TRAIN_COUNT=$(ls "$TRAIN_DIR" | wc -l)
echo "训练图像数: $TRAIN_COUNT"
echo "配置文件: Deraining/Options/Deraining_AllRain_8xA100.yml"
echo "============================================================"
echo ""

# 开始训练
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=4321 \
    basicsr/train.py \
    -opt Deraining/Options/Deraining_AllRain_8xA100.yml \
    --launcher pytorch

echo ""
echo "============================================================"
echo "Training completed at $(date)"
echo "============================================================"

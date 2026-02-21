#!/bin/bash
#SBATCH --gpus=8
#SBATCH -p a100x
#SBATCH --job-name=restormer_oracle
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================
# Oracle UpperBound 实验 - FiLM 条件注入 (8x A100)
# ============================================================
# 全程使用 ground truth 7 维天气标签作为 condition
# 不训练 Router，只有 L1 像素 loss
#
# 使用前请确保:
#   1. 数据集已准备 (python prepare_training_data.py)
#   2. 标签文件已在 offline_features_v2/ 下 (随代码提交)
#
# 提交: sbatch run_train_oracle.sh
# 直接运行: bash run_train_oracle.sh
# ============================================================

# 切换到项目目录
cd /HOME/pxyai/pxyai_0009/Restormer-oracle

# [关键] 将当前目录设为 PYTHONPATH 最高优先级，确保 import basicsr 使用本地 Oracle 版本
# 而非 pip 安装的原始 Restormer 版本 (否则 condition_mlp 不会被加载!)
export PYTHONPATH="/HOME/pxyai/pxyai_0009/Restormer-oracle:$PYTHONPATH"

# 卸载所有已加载的CUDA模块，避免冲突
module purge 2>/dev/null

# 加载CUDA环境 (需要11.7以匹配pytorch181环境)
module load CUDA/11.7 2>/dev/null

# 初始化conda并激活环境
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate pytorch181 2>/dev/null

# 打印环境信息
echo "============================================================"
echo "Restormer Oracle UpperBound - AllRain Dataset (8x A100)"
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

# 检查标签文件
TRAIN_LABELS="./offline_features_v2/train_target_labels_7c.pt"
VAL_LABELS="./offline_features_v2/val_target_labels_7c.pt"
if [ ! -f "$TRAIN_LABELS" ]; then
    echo "[错误] 训练集标签文件不存在: $TRAIN_LABELS"
    exit 1
fi
if [ ! -f "$VAL_LABELS" ]; then
    echo "[警告] 验证集标签文件不存在: $VAL_LABELS"
    echo "  验证时 FiLM 不生效，PSNR 不代表 Oracle 性能!"
fi

TRAIN_COUNT=$(ls "$TRAIN_DIR" | wc -l)
echo "训练图像数: $TRAIN_COUNT"
echo "配置文件: Deraining/Options/Deraining_Oracle_8xA100.yml"
echo "实验类型: Oracle UpperBound (全程 GT 标签, 纯 L1 loss)"
echo "============================================================"
echo ""

# [关键验证] 确认导入的 Restormer 包含 condition_mlp (FiLM 模块)
echo "验证 basicsr 导入路径..."
python -c "
from basicsr.models.archs.restormer_arch import Restormer
m = Restormer(inp_channels=3, out_channels=3, dim=48,
              num_blocks=[4,6,6,8], num_refinement_blocks=4,
              heads=[1,2,4,8], ffn_expansion_factor=2.66,
              bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
assert hasattr(m, 'condition_mlp'), \
    'FATAL: condition_mlp NOT found! 导入了错误的 basicsr (原始 Restormer)，请检查 PYTHONPATH!'
import basicsr.utils.options as opts
print(f'  basicsr 路径: {opts.__file__}')
print(f'  condition_mlp 参数数: {sum(p.numel() for p in m.condition_mlp.parameters())}')
print('  [OK] condition_mlp 已确认存在')
"
if [ $? -ne 0 ]; then
    echo "[FATAL] basicsr 验证失败! 训练中止。"
    exit 1
fi
echo ""

# 开始训练
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=4321 \
    basicsr/train.py \
    -opt Deraining/Options/Deraining_Oracle_8xA100.yml \
    --launcher pytorch

echo ""
echo "============================================================"
echo "Oracle training completed at $(date)"
echo "============================================================"

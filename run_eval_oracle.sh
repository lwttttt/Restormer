#!/bin/bash
#SBATCH -G 4
#SBATCH -p a100x
#SBATCH --job-name=oracle_eval
#SBATCH --output=eval_oracle_%j.out
#SBATCH --error=eval_oracle_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 确保导入本地 Oracle basicsr
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 加载环境
for mod in $(module list 2>&1 | grep -oP 'CUDA/\S+'); do
    module unload "$mod" 2>/dev/null
done
module load CUDA/11.7
source activate pytorch181

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "Oracle Restormer Evaluation - All Checkpoints x All Test Datasets"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

python test_all_checkpoints_oracle.py \
    --ckpt_dir ./experiments/Deraining_Oracle_Restormer/models/ \
    --data_dir ./Deraining/Datasets/ \
    --yaml_file ./Deraining/Options/Deraining_Oracle_8xA100.yml \
    --output ./oracle_eval_results.csv \
    --tile 720 \
    "$@"

echo ""
echo "============================================================"
echo "Evaluation completed at $(date)"
echo "============================================================"

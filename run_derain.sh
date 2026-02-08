#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p a100x
#SBATCH --job-name=restormer_derain
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 卸载所有已加载的CUDA模块，避免冲突
module purge

# 加载CUDA环境 (需要11.8以支持A100 GPU)
module load CUDA/11.8

# 初始化conda并激活环境
eval "$(conda shell.bash hook)"
conda activate pytorch181

# 运行Restormer demo - 去雨任务
echo "Starting Restormer demo..."
echo "Task: Deraining"
echo "Input directory: ./demo/degraded/"
echo "Output directory: ./demo/restored/"
echo ""

python demo.py \
    --task Deraining \
    --input_dir './demo/degraded/' \
    --result_dir './demo/restored/'

echo ""
echo "Demo completed! Check results in ./demo/restored/"

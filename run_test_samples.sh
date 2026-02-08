#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p a100x
#SBATCH --job-name=restormer_test_samples
#SBATCH --output=test_samples_%j.out
#SBATCH --error=test_samples_%j.err

# 卸载所有已加载的CUDA模块，避免冲突
module purge

# 加载CUDA环境
module load CUDA/11.8

# 激活虚拟环境
source /XYAIFS00/HOME/pxyai/pxyai_0009/Restormer/restormer_env/bin/activate

# 打印环境信息
echo "CUDA Version:"
nvcc --version
echo ""
echo "GPU Info:"
nvidia-smi
echo ""
echo "Python version:"
python --version
echo ""

# 运行Restormer demo - 处理test_samples中的图片
echo "Processing Rain100H test samples..."
echo "Input directory: ./test_samples/"
echo "Output directory: ./test_samples_restored/"
echo ""

python demo.py \
    --task Deraining \
    --input_dir './test_samples/' \
    --result_dir './test_samples_restored/'

echo ""
echo "Processing completed! Results saved in ./test_samples_restored/Deraining/"
echo ""
echo "Result files:"
ls -lh ./test_samples_restored/Deraining/

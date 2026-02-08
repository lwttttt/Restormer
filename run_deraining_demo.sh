#!/bin/bash
#SBATCH --job-name=restormer_derain
#SBATCH --output=derain_demo_%j.out
#SBATCH --error=derain_demo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# 加载CUDA模块 (A100适配CUDA 11.8或更高)
module purge
module load CUDA/11.8

# 打印CUDA信息
echo "CUDA Version:"
nvcc --version
echo ""
nvidia-smi
echo ""

# 设置工作目录
cd /XYAIFS00/HOME/pxyai/pxyai_0009/Restormer

# 激活虚拟环境
source restormer_env/bin/activate

# 打印Python和PyTorch信息
echo "Python version:"
python --version
echo ""
echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
echo ""

# 运行去雨任务demo
echo "Running Deraining demo on test images..."
python demo.py --task Deraining --input_dir './demo/degraded/' --result_dir './demo/restored/'

echo ""
echo "Demo completed! Results saved in ./demo/restored/Deraining/"

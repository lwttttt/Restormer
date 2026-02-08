# Restormer 多源去雨训练指南

## 概述

使用 `sampled_dataset_balanced`（7个去雨数据集的平衡采样，共9144对图像）在 8x A100 上训练 Restormer。

## 文件说明

| 文件 | 作用 |
|------|------|
| `prepare_training_data.py` | 数据准备脚本，用符号链接将7个子数据集合并为Restormer需要的单一 input/target 结构 |
| `Deraining/Options/Deraining_AllRain_8xA100.yml` | 8x A100 训练配置（200k迭代，渐进式训练） |
| `run_train_allrain.sh` | SLURM 提交脚本 |

## 服务器部署步骤

### 1. 拉取代码

```bash
cd /path/to/Restormer
git pull
```

### 2. 修改数据路径

编辑 `prepare_training_data.py` 顶部的两个路径变量，改为服务器上的实际路径：

```python
BALANCED_DATASET_ROOT = "/服务器上/sampled_dataset_balanced/的路径"
RESTORMER_ROOT = "/服务器上/Restormer/的路径"
```

### 3. 运行数据准备

```bash
python prepare_training_data.py
```

脚本会通过符号链接将所有子数据集合并到：
- `Deraining/Datasets/train/AllRain/input/` （7361张训练输入）
- `Deraining/Datasets/train/AllRain/target/` （7361张训练GT）
- `Deraining/Datasets/test/<DatasetName>/input/` + `target/` （各测试集）

**不会修改或复制原始 `sampled_dataset_balanced` 数据。**

### 4. 开始训练

**SLURM 环境：**
```bash
sbatch run_train_allrain.sh
```

**非 SLURM 环境（直接多卡运行）：**
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt Deraining/Options/Deraining_AllRain_8xA100.yml --launcher pytorch
```

### 5. 注意事项

- `run_train_allrain.sh` 中的 conda 环境名（`pytorch181`）和 CUDA 模块（`CUDA/11.8`）需根据服务器实际情况修改
- 训练会自动保存 checkpoint 到 `experiments/Deraining_AllRain_Restormer/`，支持断点续训
- 验证集使用 RainDS-Syn（267对合成图像），每4000迭代验证一次

## 训练配置详情

- **总迭代**: 200k（原版Rain13K用300k，按数据量比例调整）
- **渐进式训练**: patch从128逐步增大到384，batch自动递减
- **优化器**: AdamW, lr=3e-4, CosineAnnealing调度
- **损失函数**: L1Loss

| 迭代区间 | Patch Size | Batch/GPU |
|----------|-----------|-----------|
| 0-60k | 128x128 | 8 |
| 60k-104k | 160x160 | 5 |
| 104k-136k | 192x192 | 4 |
| 136k-160k | 256x256 | 2 |
| 160k-184k | 320x320 | 1 |
| 184k-200k | 384x384 | 1 |

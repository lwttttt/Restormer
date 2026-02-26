# Restormer + Weather-Conditioned Deraining

## 项目概述

基于 Restormer 的多源去雨项目。在原始 Restormer 基础上，引入**天气形态感知**机制：
通过离线聚类产生 7 类雨纹软标签，用 **FiLM（Feature-wise Linear Modulation）** 多级条件注入
引导网络针对不同雨型做自适应恢复。

## 分支说明：`feature/rain-soft-label`

本分支包含三个实验方案（共用同一套代码，通过 YAML 配置切换）：

| 实验 | YAML 配置 | 说明 |
|------|-----------|------|
| **Baseline** | 不配 `target_labels_path` | 原始 Restormer + 内置 RainPredictor（无监督信号，FiLM 零初始化等价无效果） |
| **Oracle UpperBound** | `Deraining_Oracle_8xA100.yml` | GT 标签通过 CE loss 引导内置 RainPredictor 学正确分类，L1 + CE loss |
| **RainPredictor 方案** | `Deraining_RainPredictor_8xA100.yml` | 同 Oracle，但验证时不依赖 GT 标签 |

## 关键架构改动

### 1. 内置 RainPredictor + 多级 FiLM 条件注入 (`restormer_arch.py`)

**RainPredictor**（行 193-208）：轻量 CNN，从输入雨图直接预测 7 类雨纹分布：
- `AvgPool2d(4) → Conv(3→32,3x3,s=2) → GELU → Conv(32→64,3x3,s=2) → GELU → GAP → Linear(64→7)`
- 输出 `rain_logits [B, 7]`，经 softmax 得到 `soft_label [B, 7]`

**FiLM MLP x3**（行 268-278）：3 个独立 MLP 分别在 Encoder L1/L2/L3 层做 channel-wise 条件缩放：
- `film_mlp_L1`: `Linear(7→48) → GELU → Linear(48→48)`，零初始化
- `film_mlp_L2`: `Linear(7→96) → GELU → Linear(96→96)`，零初始化
- `film_mlp_L3`: `Linear(7→192) → GELU → Linear(192→192)`，零初始化

零初始化确保训练初期 FiLM 缩放为 `(1 + 0) = 1`（恒等变换），等价原始 Restormer。

**`forward(inp_img)` 不再接受外部 `condition` 参数**，改为内置 RainPredictor 自动预测：

```
inp_img [B,3,H,W]
    │
    ├──→ RainPredictor → rain_logits [B,7] → softmax → soft_label [B,7]
    │
    ├──→ patch_embed → feat [B,48,H,W]
    │         │
    │    FiLM L1: feat × (1 + film_mlp_L1(soft_label))
    │         ↓
    │    Encoder L1 (4 blocks, dim=48) → Downsample
    │         │
    │    FiLM L2: feat × (1 + film_mlp_L2(soft_label))
    │         ↓
    │    Encoder L2 (6 blocks, dim=96) → Downsample
    │         │
    │    FiLM L3: feat × (1 + film_mlp_L3(soft_label))
    │         ↓
    │    Encoder L3 (6 blocks, dim=192) → Downsample
    │         ↓
    │    Latent (8 blocks, dim=384)
    │         ↓
    │    Decoder L3 → L2 → L1 (skip connections)
    │         ↓
    │    Refinement (4 blocks) → Output Conv
    │         ↓
    │    残差图 + inp_img
    ↓
返回 (output [B,3,H,W], rain_logits [B,7])
```

### 2. 训练/验证逻辑 (`image_restoration_model.py`)

- `feed_data()` / `feed_train_data()`：存储 `target_label`（离线 GT 软标签）
- `optimize_parameters()`：
  - `self.output, self.pred_rain_logits = self.net_g(self.lq)` — 自动使用内置 RainPredictor
  - 当 `target_label` 存在时，计算 `F.cross_entropy(pred_rain_logits, target_label)` 作为分类损失
  - 总损失 = L1 像素损失 + `lambda_rain` × CE 分类损失（`lambda_rain` 默认 0.1）
- `nonpad_test()` / `tile_test()`：适配二元组输出 `isinstance(pred, tuple)`

### 3. 数据集 (`paired_image_dataset.py`)

- `Dataset_PairedImage` 支持 `target_labels_path` 配置项
- 从 `.pt` 文件按文件名 basename 查表返回 7 维标签
- 训练集和验证集都需要配置各自的 `target_labels_path`

## 快速开始

### 环境安装

```bash
conda create -n pytorch181 python=3.8
conda activate pytorch181
pip install torch==1.8.1 torchvision==0.9.1
cd Restormer && pip install -e .
pip install einops lmdb scikit-image opencv-python pyyaml tqdm tensorboard
```

### 数据集准备

```bash
python prepare_training_data.py
```

目录结构：
```
Deraining/Datasets/
├── train/AllRain/
│   ├── input/    # ~7361 张雨图
│   └── target/   # ~7361 张干净图
└── test/
    ├── RainDS-Syn/     # 267 对 (验证集)
    ├── LHP-RAIN/       # 450 对
    ├── RainDrop/       # 180 对
    ├── RainDS-Real/    # 293 对
    ├── RealRain-1k/    # 450 对
    ├── SynRain-13k/    # 121 对
    └── WeatherBench/   # 21 对
```

### 标签文件（已随代码提交）

仓库 `offline_features_v2/` 下已包含三个标签文件：

| 文件 | 图片数 | 用途 |
|------|--------|------|
| `train_target_labels_7c.pt` | 7,361 | 训练集标签 |
| `val_target_labels_7c.pt` | 267 | 验证集标签 (RainDS-Syn) |
| `test_target_labels_7c.pt` | 1,782 | 全部测试集标签 (7个数据集) |

格式：`dict[str, Tensor]`，key 是图片 basename（如 `RainDS-Syn_007503`），value 是 `[7]` 维概率分布。

### 训练

```bash
# [关键] 必须设置 PYTHONPATH!
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 单机 8 卡 Oracle 实验
python -m torch.distributed.launch --nproc_per_node=8 basicsr/train.py \
  -opt Deraining/Options/Deraining_Oracle_8xA100.yml --launcher pytorch

# 单卡调试
python basicsr/train.py -opt Deraining/Options/Deraining_Oracle_8xA100.yml --force_yml num_gpu=1

# 或使用 SLURM
sbatch run_train_oracle.sh
```

### 评估

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

python test_all_checkpoints_oracle.py \
    --ckpt_dir ./experiments/Deraining_Oracle_Restormer/models/ \
    --data_dir ./Deraining/Datasets/ \
    --yaml_file ./Deraining/Options/Deraining_Oracle_8xA100.yml \
    --output ./oracle_eval_results.csv \
    --tile 720
```

## 重要注意事项

1. **[血泪教训] PYTHONPATH 必须显式设置！**

   **问题**：服务器上若有多个 Restormer 版本，`import basicsr` 可能加载错误版本（原始 Restormer 没有 FiLM 模块），导致：
   - 训练全程没有 FiLM 条件注入，等于又训了一遍 Baseline
   - checkpoint 中不含 `film_mlp` / `rain_predictor` 权重
   - 浪费数小时 GPU 时间

   **修复**：所有训练/测试脚本中必须：
   ```bash
   export PYTHONPATH="/path/to/Restormer:$PYTHONPATH"
   ```

   **如何检查 checkpoint 是否正确**：
   ```python
   ckpt = torch.load('net_g_XXXX.pth', map_location='cpu')
   keys = list(ckpt['params'].keys())
   has_film = any('film_mlp' in k for k in keys)
   has_predictor = any('rain_predictor' in k for k in keys)
   print(f"参数数: {len(keys)}, film_mlp: {has_film}, rain_predictor: {has_predictor}")
   # 正确: 498 个 key, film_mlp=True, rain_predictor=True
   # 错误: 494 个 key (实际是 vanilla Restormer)
   ```

2. **Oracle 实验的 CE 损失**：Oracle 配置有 `target_labels_path`，因此 CE 损失会自动生效
   （`lambda_rain` 默认 0.1），引导内置 RainPredictor 学习正确的雨纹分类。

3. **`forward()` 返回二元组**：`(output_image, rain_logits)`。所有调用处（demo、test、model）
   都已适配 `isinstance(pred, tuple)` 检查。加载旧版 checkpoint 时使用 `strict=False`。

## 文件结构

```
Restormer/
├── basicsr/
│   ├── data/
│   │   └── paired_image_dataset.py      # Dataset_PairedImage + target_label 支持
│   ├── models/
│   │   ├── archs/
│   │   │   ├── restormer_arch.py        # Restormer + RainPredictor + film_mlp_L1/L2/L3
│   │   │   └── weather_router.py        # [DEPRECATED] WeatherRouter CNN
│   │   └── image_restoration_model.py   # 训练/验证逻辑 + CE loss + 二元组输出适配
│   ├── train.py                         # 训练入口，target_label 从 Dataset 透传
│   └── utils/
│       └── rain_label.py                # [DEPRECATED] 在线标签生成，已迁移到离线
├── offline_features_v2/
│   ├── train_target_labels_7c.pt        # 训练集 7 维软标签 (7361 张)
│   ├── val_target_labels_7c.pt          # 验证集 7 维软标签 (RainDS-Syn 267 张)
│   └── test_target_labels_7c.pt         # 测试集 7 维软标签 (全部 1782 张)
├── Deraining/Options/
│   ├── Deraining_Oracle_8xA100.yml      # Oracle UpperBound 配置
│   ├── Deraining_RainPredictor_8xA100.yml  # RainPredictor 方案配置
│   └── Deraining_AllRain_8xA100.yml     # AllRain 配置 (旧 Router 方案)
├── test_all_checkpoints_oracle.py       # 批量评估所有 checkpoint
├── prepare_training_data.py             # 数据准备脚本 (创建 symlink)
├── run_train_oracle.sh                  # Oracle 训练 SLURM 脚本
├── run_train_allrain.sh                 # AllRain 训练 SLURM 脚本
├── run_eval_oracle.sh                   # Oracle 评估 SLURM 脚本
├── demo.py                              # 单图/目录推理 demo
└── CLAUDE.md                            # ← 本文件
```

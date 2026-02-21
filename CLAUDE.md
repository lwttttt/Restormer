# Restormer + Weather-Conditioned Deraining

## 项目概述

基于 Restormer 的多源去雨项目。在原始 Restormer 基础上，引入**天气形态感知**机制：
通过离线聚类产生 7 类雨纹软标签，用 FiLM 条件注入引导网络针对不同雨型做自适应恢复。

## 分支说明：`feature/rain-soft-label`

本分支包含三个实验方案（共用同一套代码，通过 YAML 配置切换）：

| 实验 | YAML 配置 | 说明 |
|------|-----------|------|
| **Baseline** | 注释掉 `network_router`，不配 `target_labels_path` | 原始 Restormer，无 condition |
| **Oracle UpperBound** | `Deraining_Oracle_8xA100.yml` | 全程用 ground truth 7 维标签作 FiLM condition，不训练 Router，只有 L1 loss |
| **Router 方案（后续）** | `Deraining_AllRain_8xA100.yml` | WeatherRouter CNN 预测标签 + 蒸馏 loss |

## 关键架构改动

### 1. FiLM 条件注入 (`restormer_arch.py`)

- `Restormer.__init__` 新增 `condition_mlp`：`Linear(7→48) → GELU → Linear(48→48)`，最后一层零初始化
- `Restormer.forward(inp_img, condition=None)`：在 `patch_embed` 之后、`encoder_level1` 之前做 channel-wise 缩放
- `condition=None` 时行为等价原始 Restormer（零初始化也保证训练初期无影响）

```
inp_img → patch_embed → feat [B,48,H,W]
                              ↓
                    condition [B,7] → MLP → scale [B,48,1,1]
                              ↓
                    feat * (1 + scale)  ← FiLM 调制点
                              ↓
                    encoder_level1 → ... → output
```

### 2. 训练/验证逻辑 (`image_restoration_model.py`)

- `feed_data()`：同时存储 `target_label`（验证用）
- `feed_train_data()`：同时存储 `target_label`（训练用）
- `optimize_parameters()`：`self.net_g(self.lq, condition=self.target_label)`
- `nonpad_test()`：推理时也传入 `condition`
- Router loss 由 `self.weather_router is not None` 守卫，Oracle 实验不启用

### 3. 数据集 (`paired_image_dataset.py`)

- `Dataset_PairedImage` 支持 `target_labels_path` 配置项
- 从 `.pt` 文件按文件名 basename 查表返回 7 维标签
- 训练集和验证集都需要配置各自的 `target_labels_path`

## 快速开始 (服务器端)

### 拉取代码

```bash
git clone -b feature/rain-soft-label git@github.com:lwttttt/Restormer.git Restormer-oracle
cd Restormer-oracle
```

### 数据集准备

训练集和测试集需要放到如下位置（已有则跳过）：

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

### 标签文件（已随代码提交，无需额外操作）

仓库 `offline_features_v2/` 下已包含三个标签文件：

| 文件 | 图片数 | 用途 |
|------|--------|------|
| `train_target_labels_7c.pt` | 7,361 | 训练集标签 |
| `val_target_labels_7c.pt` | 267 | 验证集标签 (RainDS-Syn) |
| `test_target_labels_7c.pt` | 1,782 | 全部测试集标签 (7个数据集) |

格式：`dict[str, Tensor]`，key 是图片 basename（如 `RainDS-Syn_007503`），value 是 `[7]` 维概率分布。

### 运行 Oracle 实验

```bash
# 单机 8 卡
python -m torch.distributed.launch --nproc_per_node=8 basicsr/train.py \
  -opt Deraining/Options/Deraining_Oracle_8xA100.yml --launcher pytorch

# 如果用 SLURM
srun --gres=gpu:8 python -m torch.distributed.launch --nproc_per_node=8 \
  basicsr/train.py -opt Deraining/Options/Deraining_Oracle_8xA100.yml --launcher pytorch
```

### 运行 Baseline 实验（对照组）

用 `Deraining_AllRain_8xA100.yml`，但需要注释掉 `network_router` 段和 `target_labels_path`，使网络不接收任何 condition：

```yaml
# 注释掉这些：
# target_labels_path: ...
# network_router:
#   type: WeatherRouter
#   ...
```

## Oracle 实验注意事项

1. **YAML 配置**：使用 `Deraining_Oracle_8xA100.yml`
   - `network_router` 段已注释掉 → `self.weather_router = None` → 无 Router loss
   - 训练集和验证集都配了 `target_labels_path`
   - 只有 L1 像素 loss

2. **验证集标签不能缺**：如果验证集 YAML 没配 `target_labels_path`，`feed_data()` 中 `target_label=None`，FiLM 不生效，验证 PSNR 不代表 Oracle 性能。网络训练时学到了依赖 condition 的特征缩放，验证时不传 condition 会导致特征分布不一致。

3. **对照实验解读**：`Oracle PSNR - Baseline PSNR = 天气形态引导的理论增益上限`

## 文件结构

```
Restormer/
├── basicsr/
│   ├── data/
│   │   └── paired_image_dataset.py      # Dataset_PairedImage + target_label 支持
│   ├── models/
│   │   ├── archs/
│   │   │   ├── restormer_arch.py        # Restormer + FiLM condition_mlp
│   │   │   └── weather_router.py        # WeatherRouter CNN (Oracle 实验不用)
│   │   └── image_restoration_model.py   # 训练/验证逻辑 + condition 传入
│   ├── train.py                         # 训练入口，target_label 从 Dataset 透传
│   └── utils/
│       └── rain_label.py                # [DEPRECATED] 在线标签生成，已迁移到离线
├── offline_features_v2/
│   ├── train_target_labels_7c.pt        # 训练集 7 维软标签 (7361 张)
│   ├── val_target_labels_7c.pt          # 验证集 7 维软标签 (RainDS-Syn 267 张)
│   └── test_target_labels_7c.pt         # 测试集 7 维软标签 (全部 1782 张)
├── Deraining/Options/
│   ├── Deraining_AllRain_8xA100.yml     # Router 方案配置
│   └── Deraining_Oracle_8xA100.yml      # Oracle UpperBound 配置
└── CLAUDE.md                            # ← 本文件
```

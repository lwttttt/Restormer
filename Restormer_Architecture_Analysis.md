# Restormer 网络结构详解 & 分类标签预测注入点分析

> 用于同行评审讨论：WeatherRouter / Predictor Head 应插入在哪里

---

## 1. 整体架构概览

Restormer 采用 **4 级 U-Net 编码器-解码器 + Refinement** 结构，每级由若干 Transformer Block 组成。核心算子是 MDTA（Multi-DConv Head Transposed Self-Attention）和 GDFN（Gated-Dconv Feed-Forward Network）。

```
输入图像 [B, 3, H, W]
│
▼
┌─────────────────────────────────────────────────────────┐
│  patch_embed: Conv2d(3→48, k3s1p1)                      │
│  输出: [B, 48, H, W]                                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ ◀── [当前 FiLM 注入点] condition_mlp(7→48)
                         │     feat = feat * (1 + scale)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  encoder_level1: 4× TransformerBlock(dim=48, heads=1)   │
│  输出: out_enc_L1 [B, 48, H, W]  ──────────────────┐   │
└────────────────────────┬────────────────────────────│───┘
                         ▼                            │ skip
┌────────────────────────────────┐                    │
│  down1_2: Conv(48→24)+PixUnSh  │                    │
│  输出: [B, 96, H/2, W/2]      │                    │
└────────────────────────┬───────┘                    │
                         ▼                            │
┌─────────────────────────────────────────────────────│───┐
│  encoder_level2: 6× TransformerBlock(dim=96, heads=2)   │
│  输出: out_enc_L2 [B, 96, H/2, W/2]  ─────────┐   │   │
└────────────────────────┬───────────────────────│───│───┘
                         ▼                       │   │
┌────────────────────────────────┐               │   │
│  down2_3: Conv(96→48)+PixUnSh  │               │   │
│  输出: [B, 192, H/4, W/4]     │               │   │
└────────────────────────┬───────┘               │   │
                         ▼                       │   │
┌─────────────────────────────────────────────────│───│───┐
│  encoder_level3: 6× TransformerBlock(dim=192, heads=4)  │
│  输出: out_enc_L3 [B, 192, H/4, W/4]  ────┐   │   │   │
└────────────────────────┬───────────────────│───│───│───┘
                         ▼                   │   │   │
┌────────────────────────────────┐           │   │   │
│  down3_4: Conv(192→96)+PixUnSh │           │   │   │
│  输出: [B, 384, H/8, W/8]     │           │   │   │
└────────────────────────┬───────┘           │   │   │
                         ▼                   │   │   │
┌─────────────────────────────────────────────────────────┐
│  latent (bottleneck):                                   │
│  8× TransformerBlock(dim=384, heads=8)                  │
│  输出: [B, 384, H/8, W/8]                               │
└────────────────────────┬────────────────────────────────┘
                         │                   │   │   │
        ─ ─ ─ ─ 解码器开始 ─ ─ ─ ─          │   │   │
                         ▼                   │   │   │
┌────────────────────────────────┐           │   │   │
│  up4_3: Conv(384→768)+PixSh    │           │   │   │
│  输出: [B, 192, H/4, W/4]     │           │   │   │
└────────────────────────┬───────┘           │   │   │
                         ▼                   │   │   │
                    cat(↑, out_enc_L3) ◀─────┘   │   │
                    → [B, 384, H/4, W/4]         │   │
                         ▼                       │   │
              reduce_chan_level3: Conv(384→192)   │   │
                         ▼                       │   │
┌─────────────────────────────────────────────────────────┐
│  decoder_level3: 6× TransformerBlock(dim=192, heads=4)  │
│  输出: [B, 192, H/4, W/4]                               │
└────────────────────────┬────────────────────────────────┘
                         ▼                       │   │
┌────────────────────────────────┐               │   │
│  up3_2: Conv(192→384)+PixSh    │               │   │
│  输出: [B, 96, H/2, W/2]      │               │   │
└────────────────────────┬───────┘               │   │
                         ▼                       │   │
                    cat(↑, out_enc_L2) ◀─────────┘   │
                    → [B, 192, H/2, W/2]             │
                         ▼                           │
              reduce_chan_level2: Conv(192→96)        │
                         ▼                           │
┌─────────────────────────────────────────────────────────┐
│  decoder_level2: 6× TransformerBlock(dim=96, heads=2)   │
│  输出: [B, 96, H/2, W/2]                                │
└────────────────────────┬────────────────────────────────┘
                         ▼                           │
┌────────────────────────────────┐                   │
│  up2_1: Conv(96→192)+PixSh     │                   │
│  输出: [B, 48, H, W]          │                   │
└────────────────────────┬───────┘                   │
                         ▼                           │
                    cat(↑, out_enc_L1) ◀─────────────┘
                    → [B, 96, H, W]
                         ▼
┌─────────────────────────────────────────────────────────┐
│  decoder_level1: 4× TransformerBlock(dim=96, heads=1)   │
│  输出: [B, 96, H, W]                                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  refinement: 4× TransformerBlock(dim=96, heads=1)       │
│  输出: [B, 96, H, W]                                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
              output: Conv(96→3, k3s1p1)
                         ▼
              out + inp_img  (全局残差)
                         ▼
              输出 [B, 3, H, W]
```

---

## 2. 各组件维度汇总

| 组件 | 输入维度 | 输出维度 | Transformer Blocks | Heads | 备注 |
|------|---------|---------|-------------------|-------|------|
| `patch_embed` | [B,3,H,W] | [B,48,H,W] | - | - | 3×3 Conv |
| **FiLM 注入** | condition [B,7] | scale [B,48,1,1] | - | - | MLP(7→48→48), 零初始化 |
| `encoder_level1` | [B,48,H,W] | [B,48,H,W] | 4 | 1 | |
| `down1_2` | [B,48,H,W] | [B,96,H/2,W/2] | - | - | Conv+PixelUnshuffle |
| `encoder_level2` | [B,96,H/2,W/2] | [B,96,H/2,W/2] | 6 | 2 | |
| `down2_3` | [B,96,H/2,W/2] | [B,192,H/4,W/4] | - | - | Conv+PixelUnshuffle |
| `encoder_level3` | [B,192,H/4,W/4] | [B,192,H/4,W/4] | 6 | 4 | |
| `down3_4` | [B,192,H/4,W/4] | [B,384,H/8,W/8] | - | - | Conv+PixelUnshuffle |
| `latent` | [B,384,H/8,W/8] | [B,384,H/8,W/8] | 8 | 8 | bottleneck |
| `up4_3` | [B,384,H/8,W/8] | [B,192,H/4,W/4] | - | - | Conv+PixelShuffle |
| `decoder_level3` | [B,192,H/4,W/4] | [B,192,H/4,W/4] | 6 | 4 | skip cat 后先 1×1 降通道 |
| `up3_2` | [B,192,H/4,W/4] | [B,96,H/2,W/2] | - | - | |
| `decoder_level2` | [B,96,H/2,W/2] | [B,96,H/2,W/2] | 6 | 2 | |
| `up2_1` | [B,96,H/2,W/2] | [B,48,H,W] | - | - | |
| `decoder_level1` | [B,96,H,W] | [B,96,H,W] | 4 | 1 | skip cat 后不降通道(48+48=96) |
| `refinement` | [B,96,H,W] | [B,96,H,W] | 4 | 1 | |
| `output` | [B,96,H,W] | [B,3,H,W] | - | - | 3×3 Conv + 全局残差 |

**Transformer Block 结构**（每个 block 一致）:
```
x → LayerNorm → MDTA(Attention) → + → LayerNorm → GDFN(FeedForward) → + → out
  └─────────────────────────────┘    └─────────────────────────────────┘
           残差连接                              残差连接
```

- **MDTA**: 1×1 Conv → 3×3 DWConv → split QKV → transpose attention (C×C 而非 HW×HW) → 1×1 Conv
- **GDFN**: 1×1 Conv(dim→hidden×2) → 3×3 DWConv → Gated(GELU(x1)×x2) → 1×1 Conv(hidden→dim)
- `hidden = int(dim × 2.66)`

---

## 3. 当前条件注入方案 (FiLM)

```python
# restormer_arch.py 第 246-263 行
self.condition_mlp = nn.Sequential(
    nn.Linear(7, 48), nn.GELU(), nn.Linear(48, 48)  # 零初始化末层
)

def forward(self, inp_img, condition=None):
    feat = self.patch_embed(inp_img)          # [B, 48, H, W]
    if condition is not None:
        scale = self.condition_mlp(condition)  # [B, 48]
        scale = scale[:, :, None, None]        # [B, 48, 1, 1]
        feat = feat * (1 + scale)              # FiLM channel-wise scaling
    out = self.encoder_level1(feat)
    ...
```

**特点**:
- 注入在**最浅层**（patch_embed 之后、encoder_level1 之前）
- 仅做 channel-wise **乘性缩放**，不改变空间结构
- 零初始化保证训练初期等价于无条件 Restormer

---

## 4. 分类标签预测（Predictor/Router）候选注入位置

核心矛盾：**如果用 Restormer 内部特征预测标签 → 预测结果又要反过来条件化 Restormer → 因果环路**。以下方案按是否引入环路分两大类。

### 方案 A: 独立网络，无环路 (当前 WeatherRouter 方案)

```
输入图像 ──┬──→ Restormer(condition=pred) ──→ 去雨结果
           │
           └──→ WeatherRouter(CNN) ──→ pred [B, 7]
                  (3层Conv + GAP, ~11K params)
```

| 优势 | 劣势 |
|------|------|
| 结构最简单，无因果依赖 | 不共享 Restormer 特征，表达能力受限于小 CNN |
| 可单独预训练 Router | 需额外前向传播（虽然极轻量 <0.5ms） |
| 推理时可并行/提前计算 | 标签预测质量取决于小网络容量 |
| 不修改 Restormer 主干 | |

**适合场景**: 标签语义简单（粗粒度雨型分类），小模型即可胜任。

---

### 方案 B: 从 Encoder 分支预测，条件化 Decoder (无环路)

```
                        ┌───────────────────────────────────────┐
输入 → patch_embed      │           encoder (无 condition)       │
         ↓              │                                       │
    encoder_level1 ─────│─── skip ──────────────────────────┐   │
         ↓              │                                   │   │
    encoder_level2 ─────│─── skip ─────────────────────┐   │   │
         ↓              │                              │   │   │
    encoder_level3 ─────│─── skip ────────────────┐   │   │   │
         ↓              │                         │   │   │   │
    latent (bottleneck) │                         │   │   │   │
         │              └─────────────────────────│───│───│───┘
         │                                        │   │   │
         ├──→ Predictor Head ──→ pred [B, 7]     │   │   │
         │         (GAP + Linear)                 │   │   │
         ▼                                        │   │   │
    ┌────────── decoder (接收 condition=pred) ─────────────────┐
    │  decoder_level3 ◀── FiLM(pred→192) ── skip cat L3       │
    │  decoder_level2 ◀── FiLM(pred→96)  ── skip cat L2       │
    │  decoder_level1 ◀── FiLM(pred→96)  ── skip cat L1       │
    │  refinement                                              │
    │  output + residual                                       │
    └──────────────────────────────────────────────────────────┘
```

**Predictor Head 设计**: 在 latent 输出 [B, 384, H/8, W/8] 上接:
```python
self.predictor = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # [B, 384, 1, 1]
    nn.Flatten(),              # [B, 384]
    nn.Linear(384, 7)          # [B, 7]
)
```

| 优势 | 劣势 |
|------|------|
| 利用 Restormer 最深层语义特征，预测能力强 | 只能条件化 decoder，encoder 无法受益 |
| 无因果环路，前向传播自然串行 | latent 到 predictor 的梯度路径可能干扰主干训练 |
| 参数量极少 (~2.7K: 384×7+7) | 需多级 FiLM，注入逻辑更复杂 |
| 与现有 FiLM 机制兼容，只需改注入位置 | |

**适合场景**: 希望预测器充分利用深层语义，且认为 decoder 条件化已足够。

---

### 方案 C: 从中间层分支预测，条件化后续层 (无环路)

```
输入 → patch_embed → encoder_level1 → encoder_level2
                                           │
                    ┌──────────────────────┤
                    │                      ▼
             Predictor Head         encoder_level3 ◀── FiLM(pred→192)
           (GAP on L2 features)          ▼
            [B,96,H/2,W/2]→[B,7]   latent ◀── FiLM(pred→384)
                    │                    ▼
                    │              decoder (各级 FiLM)
                    │                    ▼
                    └──────────────→  输出
```

从 `out_enc_level2` [B, 96, H/2, W/2] 分支:
```python
self.predictor = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # [B, 96, 1, 1]
    nn.Flatten(),
    nn.Linear(96, 7)
)
```

| 优势 | 劣势 |
|------|------|
| encoder L3 / L4 / 全部 decoder 都能接收 condition | encoder L1, L2 无法条件化 |
| 中间层特征已有一定语义（96ch, 6个 Transformer Block） | L2 语义不如 latent 丰富 |
| 折中方案：较早预测 → 更多层受益 | 预测精度可能不如方案 B |
| 不增加推理延迟（单次前向） | |

**变体**: 也可从 `out_enc_level3` 分支，让 latent + decoder 受益。

---

### 方案 D: 两阶段前向 (可条件化全部层，但增加计算)

```
第 1 阶段 (轻量): 输入 → Restormer 子集 or 小网络 → pred [B, 7]
第 2 阶段 (完整): 输入 + pred → Restormer(全部层 FiLM) → 输出
```

| 优势 | 劣势 |
|------|------|
| 所有层都能接收 condition，最大化利用标签信息 | 两次前向，计算量接近翻倍 |
| 理论上限最高 | 实现复杂 |
| 第 1 阶段可复用方案 A 的 WeatherRouter | 训练不稳定风险（第2阶段依赖第1阶段输出）|

---

### 方案 E: Encoder 特征聚合预测 (多尺度)

```
encoder_level1 ──→ GAP → [B, 48]  ──┐
encoder_level2 ──→ GAP → [B, 96]  ──┼──→ concat [B, 720] → MLP → pred [B, 7]
encoder_level3 ──→ GAP → [B, 192] ──┤
latent         ──→ GAP → [B, 384] ──┘

pred 仅注入 decoder (同方案 B)
```

| 优势 | 劣势 |
|------|------|
| 多尺度特征聚合，信息最丰富 | 只能条件化 decoder |
| 低+中+高层语义互补 | 聚合模块略增复杂度 |
| 参数量适中 (~5K: 720×7+7) | |

---

## 5. 各方案对比总结

| 方案 | 预测来源 | 条件化范围 | 额外参数 | 推理开销 | 实现复杂度 | 预测质量 |
|------|---------|-----------|---------|---------|-----------|---------|
| **A: 独立 CNN** | 输入图像 | 全部层 (encoder+decoder) | ~11K | +<0.5ms | 低 | 一般 |
| **B: Bottleneck 分支** | latent (384ch) | 仅 decoder | ~2.7K | 无额外 | 中 | 高 |
| **C: 中间层分支** | enc_L2/L3 | enc 后半 + decoder | ~0.7K/~1.3K | 无额外 | 中 | 中 |
| **D: 两阶段** | 第1阶段网络 | 全部层 | ~11K | +1次前向 | 高 | 取决于第1阶段 |
| **E: 多尺度聚合** | 全部 encoder 层 | 仅 decoder | ~5K | 无额外 | 中高 | 高 |

---

## 6. 建议与讨论点

### 6.1 推荐优先验证方案

**建议先做方案 B (Bottleneck 分支)**，理由：
1. Oracle 实验已证明 FiLM 条件化有增益，方案 B 能最大化利用 Restormer 深层语义来预测标签
2. 实现改动最小：只需在 `forward()` 的 latent 输出后加一个 GAP+Linear head
3. 与当前 Oracle 实验的差异**仅在于标签来源**（GT → predicted），方便做消融对比
4. 不增加推理开销

如果方案 B 的预测精度足够（接近 GT 标签），则 decoder 条件化的增益已接近 Oracle 上限，无需复杂化。

### 6.2 若 decoder-only 条件化不够

如果实验表明 encoder 也需要条件化，再升级到：
- **方案 A + 当前 FiLM**：最稳妥，独立 CNN 预测 → 全层条件化（当前代码已支持）
- **方案 C (L2 分支)**：折中，L1-L2 无 condition 但 L3/latent/decoder 有 condition

### 6.3 评审讨论要点

1. **条件化位置 vs 预测质量的 tradeoff**: 越早分支预测 → 越多层能条件化，但预测越粗糙
2. **是否需要 stop-gradient**: Predictor 的梯度是否应该回传到 Restormer 主干？如果回传，主干可能为了让标签更好预测而牺牲去雨质量
3. **当前 FiLM 只在最浅层注入是否足够**: Oracle 实验的 PSNR 增益大小决定了这个问题——如果增益小，说明浅层 FiLM 已足够；如果增益大但实际 Router 方案达不到，可能需要多级注入
4. **Predictor 训练信号**: 用 cross-entropy 对离线标签蒸馏，还是用辅助任务（如对比学习）让特征自发聚类？

---

## 7. 附录: 关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `basicsr/models/archs/restormer_arch.py` | 193-301 | Restormer 主类定义与 forward |
| `basicsr/models/archs/restormer_arch.py` | 246-253 | FiLM condition_mlp 定义 |
| `basicsr/models/archs/restormer_arch.py` | 255-263 | FiLM 注入逻辑 (forward) |
| `basicsr/models/archs/weather_router.py` | 15-31 | WeatherRouter CNN 定义 |
| `basicsr/models/image_restoration_model.py` | 76-89 | WeatherRouter 初始化 |
| `basicsr/models/image_restoration_model.py` | 180-210 | 训练 optimize_parameters (含 Router loss) |
| `basicsr/models/image_restoration_model.py` | 262-281 | 推理 nonpad_test (传入 condition) |
| `Deraining/Options/Deraining_Oracle_8xA100.yml` | - | Oracle 实验配置 |
| `Deraining/Options/Deraining_AllRain_8xA100.yml` | - | Router 方案配置 |

# MedViT 网络结构介绍

> **MedViTV2**: A Robust Vision Transformer for Generalized Medical Image Classification

## 📋 概述

MedViT是一个专为医学图像分类设计的混合架构Vision Transformer，结合了卷积神经网络（CNN）和Transformer的优势，通过分层混合策略平衡局部和全局特征感知。

## 🏗️ 整体架构

MedViT采用**分层混合架构**，从输入到输出包含三个主要部分：

### 1. Stem层（初始特征提取）

```python
# 4层ConvBNReLU卷积
ConvBNReLU(3 → 64, stride=2)    # 第一次下采样
ConvBNReLU(64 → 32, stride=1)
ConvBNReLU(32 → 64, stride=1)
ConvBNReLU(64 → 64, stride=2)    # 第二次下采样
```

- **功能**: 将3通道RGB图像转换为64通道特征图
- **下采样**: 通过两次stride=2，将空间分辨率降低到原来的1/4
- **输出**: `(B, 64, H/4, W/4)`

### 2. 特征提取阶段（4个Stage）

网络包含4个特征提取阶段，每个阶段使用不同组合的LFP和GFP块：

| Stage | 块类型组合 | 通道数 | 空间分辨率 |
|-------|-----------|--------|-----------|
| Stage 1 | LFP × depths[0] | dims[0] | H/4 × W/4 |
| Stage 2 | LFP × (depths[1]-1) + GFP × 1 | dims[1] | H/8 × W/8 |
| Stage 3 | [LFP, LFP, GFP] × (depths[2]//3) | dims[2] | H/16 × W/16 |
| Stage 4 | GFP × depths[3] | dims[3] | H/32 × W/32 |

**设计理念**:
- **早期Stage（1-2）**: 以LFP为主，捕获局部细节特征
- **中期Stage（3）**: LFP和GFP混合，平衡局部和全局特征
- **后期Stage（4）**: 以GFP为主，捕获全局语义特征

### 3. 分类头

```python
BatchNorm2d → AdaptiveAvgPool2d(1,1) → Linear(num_classes)
```

- **归一化**: BatchNorm2d稳定特征分布
- **全局池化**: 将空间特征图压缩为1×1
- **分类**: Linear层输出类别logits

## 🔧 核心组件

### 1. LFP (Local Feature Perception) 块

**功能**: 局部特征感知，高效捕获邻域信息

**结构**:
```
输入 x
  ↓
PatchEmbed (特征嵌入/下采样)
  ↓
Norm1 (BatchNorm2d)
  ↓
NeighborhoodAttention (邻域注意力, kernel_size=7)
  ↓ (残差连接)
  ↓
Norm2 (BatchNorm2d)
  ↓
LocalityFeedForward (局部前馈网络)
  ↓ (残差连接)
输出
```

**关键特性**:
- **NeighborhoodAttention**: 使用natten库实现，只关注7×7邻域，计算复杂度O(N)
- **LocalityFeedForward**: 使用深度卷积（Depthwise Conv）进行局部特征变换
- **优势**: 计算效率高，适合早期特征提取阶段

**代码位置**: `MedViT.py` 第290-348行

### 2. GFP (Global Feature Perception) 块

**功能**: 全局特征感知，捕获长距离依赖关系

**结构**:
```
输入 x
  ↓
PatchEmbed (特征嵌入/下采样)
  ↓
┌─────────────────────────────────┐
│ 分支1: E_MHSA (全局注意力)      │
│   - 支持sr_ratio空间缩减        │
│   - 捕获长距离依赖              │
└─────────────────────────────────┘
  ↓ (残差连接)
  ↓
┌─────────────────────────────────┐
│ 分支2: MHCA (卷积注意力)        │
│   - 3×3分组卷积                 │
│   - 捕获局部模式                │
└─────────────────────────────────┘
  ↓ (特征拼接)
  ↓
Norm2 (BatchNorm2d)
  ↓
KAN (Kolmogorov-Arnold Network)
  ↓ (残差连接)
输出
```

**关键特性**:
- **E_MHSA**: 高效多头自注意力，支持空间缩减（sr_ratio）降低计算量
- **MHCA**: 多头卷积注意力，通过分组卷积捕获局部模式
- **KAN**: 使用Kolmogorov-Arnold Network替代传统MLP，提高表达能力
- **混合设计**: 同时使用全局注意力和卷积注意力，平衡全局和局部特征

**代码位置**: `MedViT.py` 第417-483行

### 3. KAN (Kolmogorov-Arnold Network)

**位置**: 在GFP块中替代传统MLP

**原理**:
- 使用可学习的样条函数（Spline）作为激活函数
- 通过ReflectionalSwitchFunction生成基函数
- 相比传统MLP，具有更强的表达能力

**实现细节**:
```python
# 在GFP块中的使用
hidden_dim = int(out_channels * mlp_ratio)
self.kan = KAN([out_channels, hidden_dim, out_channels])
```

**优势**:
- 可学习的激活函数，适应不同数据分布
- 提高模型表达能力
- 在医学图像分类任务中表现优异

**代码位置**: `fasterkan.py` 第106-137行

### 4. Neighborhood Attention

**来源**: natten库（Neighborhood Attention Transformer）

**特点**:
- **局部注意力**: 只关注kernel_size×kernel_size邻域内的像素
- **计算复杂度**: O(N×k²)，其中k是kernel_size，远小于全局注意力的O(N²)
- **感受野**: 通过dilation参数可以扩展感受野

**在LFP中的使用**:
```python
self.attn = NeighborhoodAttention(
    out_channels,
    kernel_size=7,  # 7×7邻域
    num_heads=out_channels // head_dim,
    ...
)
```

**优势**: 在保持局部特征提取能力的同时，大幅降低计算复杂度

## 🎯 设计特点

### 1. 分层混合策略

网络采用渐进式混合策略，从局部到全局逐步过渡：

```
Stage 1: [LFP, LFP]           # 全部局部特征
Stage 2: [LFP, GFP]           # 开始引入全局特征
Stage 3: [LFP, LFP, GFP, ...] # 混合策略
Stage 4: [GFP, GFP]           # 全部全局特征
```

这种设计符合视觉特征提取的层次性：底层关注局部细节，高层关注全局语义。

### 2. 渐进式特征提取

**通道数变化**:
- Stem: 3 → 64
- Stage 1: 64
- Stage 2: 128
- Stage 3: 256/320/384/512
- Stage 4: 512/768/1024

**空间分辨率变化**:
- 输入: H × W
- Stem后: H/4 × W/4
- Stage 2后: H/8 × W/8
- Stage 3后: H/16 × W/16
- Stage 4后: H/32 × W/32

### 3. 效率优化

1. **Neighborhood Attention**: 降低注意力计算复杂度
2. **空间缩减注意力**: E_MHSA支持sr_ratio，减少key/value计算量
3. **梯度检查点**: 支持use_checkpoint，节省训练内存
4. **BN融合**: 支持merge_bn，加速推理

### 4. 模型变体

MedViT提供4种不同规模的模型变体：

| 模型 | depths | dims | 参数量 | 适用场景 |
|------|--------|------|--------|----------|
| **MedViT_tiny** | [2,2,6,1] | [64,128,192,384] | 最小 | 快速实验、资源受限 |
| **MedViT_small** | [2,2,6,2] | [64,128,256,512] | 小 | 平衡性能和效率 |
| **MedViT_base** | [2,2,6,2] | [96,192,384,768] | 中等 | 标准配置 |
| **MedViT_large** | [2,2,6,2] | [96,256,512,1024] | 大 | 最佳性能 |

## 📊 数据流示意

```
输入图像
  (B, 3, H, W)
    ↓
┌─────────────────┐
│   Stem层        │
│ ConvBNReLU × 4  │
└─────────────────┘
    ↓
  (B, 64, H/4, W/4)
    ↓
┌─────────────────┐
│   Stage 1       │
│   LFP × 2       │
└─────────────────┘
    ↓
  (B, 64, H/4, W/4)
    ↓
┌─────────────────┐
│   Stage 2       │
│ LFP × 1, GFP × 1│
└─────────────────┘
    ↓
  (B, 128, H/8, W/8)
    ↓
┌─────────────────┐
│   Stage 3       │
│ [LFP,LFP,GFP]×2 │
└─────────────────┘
    ↓
  (B, 256, H/16, W/16)
    ↓
┌─────────────────┐
│   Stage 4       │
│   GFP × 2       │
└─────────────────┘
    ↓
  (B, 512, H/32, W/32)
    ↓
┌─────────────────┐
│   分类头        │
│ BN → Pool → FC  │
└─────────────────┘
    ↓
输出logits
  (B, num_classes)
```

## 📍 关键代码位置

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| **主网络类** | `MedViT.py` | 486-583 | MedViT主类定义 |
| **LFP块** | `MedViT.py` | 290-348 | Local Feature Perception块 |
| **GFP块** | `MedViT.py` | 417-483 | Global Feature Perception块 |
| **E_MHSA** | `MedViT.py` | 351-414 | 高效多头自注意力 |
| **LocalityFeedForward** | `MedViT.py` | 206-265 | 局部前馈网络 |
| **KAN实现** | `fasterkan.py` | 106-137 | FasterKAN网络 |
| **模型注册** | `MedViT.py` | 585-617 | tiny/small/base/large变体 |

## 🔍 关键设计决策

1. **为什么使用混合架构？**
   - CNN擅长捕获局部特征，Transformer擅长捕获全局依赖
   - 混合架构结合两者优势，在效率和性能间取得平衡

2. **为什么LFP在前，GFP在后？**
   - 符合视觉特征层次：底层需要局部细节，高层需要全局语义
   - 早期使用LFP降低计算量，后期使用GFP提升表达能力

3. **为什么使用KAN替代MLP？**
   - KAN使用可学习的激活函数，适应性强
   - 在医学图像分类任务中，KAN表现优于传统MLP

4. **为什么使用Neighborhood Attention？**
   - 全局注意力计算复杂度O(N²)，Neighborhood Attention为O(N×k²)
   - 在保持局部特征提取能力的同时，大幅降低计算成本

## 📚 相关资源

- **论文**: MedViTV2: A Robust Vision Transformer for Generalized Medical Image Classification
- **代码仓库**: 当前项目目录
- **依赖库**: 
  - `natten`: Neighborhood Attention实现
  - `timm`: PyTorch Image Models
  - `einops`: 张量操作工具

---

**最后更新**: 2026-01-28
**版本**: MedViTV2

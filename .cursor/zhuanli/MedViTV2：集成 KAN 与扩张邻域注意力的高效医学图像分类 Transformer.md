
# MedViTV2：集成 KAN 与扩张邻域注意力的高效医学图像分类 Transformer

## 一、研究背景与动机

现有多数深度学习方法主要针对**干净图像**设计，但现实临床数据常存在因多中心研究及成像设备差异导致的**图像损坏**（corruptions）和伪影。本文提出 **Medical Vision Transformer V2 (MedViTV2)**，旨在构建面向真实世界、兼具鲁棒性与可扩展性的通用医学图像分类模型。

**MedViTV1 的局限**：随着模型规模扩大，准确率反而下降（特征崩溃问题），且缺乏对损坏数据的鲁棒性。

---

## 二、核心贡献

1. **KAN 集成创新**  
   首次将 Kolmogorov-Arnold Network (KAN) 引入 Transformer 的前馈路径，替代传统 MLP，在减少 44% 计算量的同时显著提升性能。

2. **扩张邻域注意力 (DiNA)**  
   提出高效的稀疏全局注意力机制，解决 MedViTV1 中的特征崩溃问题，实现线性复杂度的全局上下文捕获。

3. **分层混合策略**  
   设计 `(LFP × N + GFP × 1) × L` 的层级结构，平衡局部特征感知（CNN）与全局特征感知（Transformer）。

4. **SOTA 性能**  
   在 **29 项实验中的 27 项**达到当前最佳水平，在 MedMNIST-C 鲁棒性基准上准确率提升 **13.4%**。

---

## 三、方法架构

### 3.1 整体结构
MedViTV2 采用四阶段分层设计：

| 阶段 | 操作 | 输出尺寸 | 核心组件 |
|------|------|----------|----------|
| Stem | 卷积层 | H/4 × W/4 | Conv 3×3 |
| Stage 1 | Patch Embedding | H/4 × W/4 | **LFP** × 2 |
| Stage 2 | 下采样 | H/8 × W/8 | **LFP** × 1 + **GFP** × 1 |
| Stage 3 | 下采样 | H/16 × W/16 | LFP × 2 + GFP × 1 (×3 重复) |
| Stage 4 | 下采样 | H/32 × W/32 | GFP × 1 (×2 重复) |

### 3.2 关键模块

**局部特征感知块 (LFP)**
- 采用 **DiNA (Dilated Neighborhood Attention)**：像素级滑动窗口注意力，保持线性复杂度，通过扩张率 δ 捕获多尺度邻域关系
- 配合 **LFFN (Local Feed-Forward Network)** 提取局部细节

**全局特征感知块 (GFP)**
- **E-MHSA**: 高效多头自注意力
- **MHCA**: 多头卷积注意力（保持局部归纳偏置）
- **KAN 层**: 使用 Reflectional Switch Activation Functions (RSWAF) 替代 B-splines，降低计算开销

### 3.3 KAN 实现
采用 FastKAN 的径向基函数近似思想，将激活函数定义为：
```
ϕ(x) = w_b · silu(x) + w_s · spline(x)
```
其中 spline 使用高斯 RBF 近似，参数效率显著优于传统 MLP。

---

## 四、实验验证

### 4.1 数据集
- **17 个医学数据集**：涵盖 X光、CT、MRI、超声、病理切片、皮肤镜等 9 种成像模态
- **MedMNIST-C**：12 个数据集的损坏版本（噪声、模糊、颜色偏移、数字伪影等）

### 4.2 关键结果

**准确率比较 (MedMNIST 2D)**：
| 模型 | PathMNIST | ChestMNIST | DermaMNIST | OCTMNIST |
|------|-----------|------------|------------|----------|
| MedViTV1-T | 93.8% | 78.6% | 91.4% | 76.7% |
| MedMamba-T | 95.3% | - | 91.7% | 91.8% |
| **MedViTV2-T** | **95.9%** | **79.1%** | **93.1%** | **92.7%** |

**鲁棒性评估 (MedMNIST-C)**：
- **bACC (clean)**: 84.1% (MedViTV2-S)
- **bACC (corrupted)**: 75.2% (远高于 ViT-B 的 72.0%)
- **相对平衡误差 (rBE)**: 89.2%（最佳）

**非 MNIST 数据集 (PAD-UFES-20, CPN X-ray, Kvasir 等)**：
- **PAD-UFES-20**: 准确率 63.6%，F1-score 61.2%（超越 Swin-T、ConvNeXt）
- **CPN X-ray**: 准确率 98.2%，F1-score 98.2%（SOTA）
- **Fetal-Planes-DB**: 准确率 95.3%，优于 MedMamba-B 0.9%

### 4.3 消融实验 (TissueMNIST-C)

| 配置 | LFP (DiNA) | GFP (KAN) | bACC (干净) | bACC (损坏) |
|------|------------|-----------|-------------|-------------|
| MedViTV1-T | ✓ (MHCA) | ✗ (MLP) | 68.6% | 50.1% |
| MedViTV2-T | ✓ (DiNA) | ✓ (KAN) | **72.7%** | **56.9%** |
| MedViTV1-L | ✓ (MHCA) | ✗ (MLP) | 67.9% | 53.3% |
| MedViTV2-L | ✓ (DiNA) | ✓ (KAN) | **74.1%** | **59.1%** |

**关键发现**：
- KAN 显著提升干净图像准确率
- DiNA 解决大模型特征崩溃问题（MedViTV1-L 准确率低于 MedViTV1-T）
- LFFN 增强对损坏特征的鲁棒性

---

## 五、可视化分析

**Grad-CAM 热图分析**：
- MedViTV1-L 存在**特征崩溃**（feature collapse）：大量特征图饱和或失效
- MedViTV2 各层注意力聚焦于病灶区域，深层特征呈现类似分割掩码的精细化定位

---

## 六、结论

MedViTV2 通过 KAN 与 DiNA 的协同设计，首次在医学图像领域实现了：
1. **计算效率与准确率的双重提升**（44% 计算量减少 + 4.6-13.4% 准确率提升）
2. **优异的鲁棒性**（对抗多中心数据损坏）
3. **可扩展性**（模型规模增大时性能持续提升，无崩溃现象）

代码开源地址：https://github.com/Omid-Nejati/MedViTV2.git

---

## 关键术语对照

| 英文 | 中文 |
|------|------|
| Kolmogorov-Arnold Networks (KAN) | 柯尔莫果洛夫-阿诺德网络 |
| Dilated Neighborhood Attention (DiNA) | 扩张邻域注意力 |
| Local Feature Perception (LFP) | 局部特征感知 |
| Global Feature Perception (GFP) | 全局特征感知 |
| Reflectional Switch Activation Function (RSWAF) | 反射切换激活函数 |
| Feature Collapse | 特征崩溃 |

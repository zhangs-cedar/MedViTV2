[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.13693)
[![Paper](https://img.shields.io/badge/Elsevier-ASOC-blue)](https://doi.org/10.1016/j.asoc.2025.114045)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omid-Nejati/MedViTV2/blob/main/Tutorials/Evaluation.ipynb)

<div align="center">
  <h1 style="font-family: Arial;">MedViT</h1>
  <h3>MedViTV2: 基于 KAN 集成 Transformer 和扩张邻域注意力的医学图像分类</h3>
</div>

## 简介

MedViTV2 是首个将 Kolmogorov-Arnold Network (KAN) 层集成到 Transformer 中的医学图像分类架构。通过高效的 KAN 模块和扩张邻域注意力 (DiNA) 机制，在 17 个分类数据集和 12 个损坏数据集上取得了优异的性能。

**注意：** 本代码库支持训练所有 TIMM 模型。

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Omid-Nejati/MedViTV2.git
cd MedViTV2
```

### 2. 安装 PyTorch 2.5

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装 natten 0.17.3

```bash
pip install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/
```

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 训练

在单个 GPU 上训练 MedViT-small 模型，使用 breastMNIST 数据集，训练 100 个 epoch：

```bash
python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained False
```

```bash
python main.py --model_name 'MedViT_small' --dataset 'fabric' --pretrained False
```

### 主要参数说明

- `--model_name`: 模型名称，可选 `MedViT_tiny`, `MedViT_small`, `MedViT_base`, `MedViT_large`
- `--dataset`: 数据集名称，支持 MedMNIST 系列（如 `breastmnist`, `chestmnist`, `pathmnist` 等）和其他医学图像数据集
- `--pretrained`: 是否使用预训练权重（`True`/`False`）
- `--batch_size`: 批次大小（默认 24）
- `--lr`: 学习率（默认 0.0001）
- `--epochs`: 训练轮数（默认 100）

### 支持的 MedMNIST 数据集

- `breastmnist`, `chestmnist`, `dermamnist`, `octmnist`
- `pneumoniamnist`, `retinamnist`, `bloodmnist`, `tissuemnist`
- `pathmnist`, `organamnist`, `organcmnist`, `organsmnist`

## 评估

在 17 个医学数据集上训练或评估 MedViT 模型，请参考 [评估教程](https://github.com/Omid-Nejati/MedViTV2/blob/main/Tutorials/Evaluation.ipynb)。

## 可视化

可视化 MedViT 的 Grad-CAM 热图，请参考 [可视化教程](https://github.com/Omid-Nejati/MedViTV2/blob/main/Tutorials/Visualization.ipynb)。

## 性能概览

MedViT 在多个医学图像数据集上的性能表现。预训练模型权重已发布。

| 数据集 | 任务 | MedViTV2-tiny (%) | MedViTV2-small (%) | MedViTV2-base (%) | MedViTV2-large (%) |
|:------:|:----:|:-----------------:|:-------------------:|:------------------:|:-------------------:|
| ChestMNIST | 多分类 (14) | 96.3 | 96.4 | 96.4 | 96.7 |
| PathMNIST | 多分类 (9) | 95.9 | 96.5 | 97.0 | 97.7 |
| DermaMNIST | 多分类 (7) | 78.1 | 79.2 | 80.8 | 81.7 |
| OCTMNIST | 多分类 (4) | 92.7 | 94.2 | 94.4 | 95.2 |
| PneumoniaMNIST | 多分类 (2) | 95.1 | 96.5 | 96.9 | 97.3 |
| BreastMNIST | 多分类 (2) | 88.2 | 89.5 | 90.4 | 91.0 |
| BloodMNIST | 多分类 (8) | 97.9 | 98.5 | 98.5 | 98.7 |

更多数据集和预训练模型下载链接，请参考原始仓库。

## 许可证

MedViT 采用 [MIT License](LICENSE) 许可证。

## 引用

如果本仓库对您有帮助，请考虑给个 Star！

```bibtex
@article{manzari2025medical,
  title={Medical image classification with kan-integrated transformers and dilated neighborhood attention},
  author={Manzari, Omid Nejati and Asgariandehkordi, Hojat and Koleilat, Taha and Xiao, Yiming and Rivaz, Hassan},
  journal={arXiv preprint arXiv:2502.13693},
  year={2025}
}
```

## 参考

* [FasterKAN](https://github.com/AthanasiosDelis/faster-kan)
* [Natten](https://github.com/SHI-Labs/NATTEN)
* [MedViTV1](https://github.com/Omid-Nejati/MedViT)

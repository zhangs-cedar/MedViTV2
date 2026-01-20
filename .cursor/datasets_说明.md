# 数据集处理文件详解（通俗版）

## 📚 这个文件是做什么的？

想象一下，你要训练一个 AI 模型来识别图片：
- **第一步**：需要很多图片数据
- **第二步**：需要把这些图片整理好（分类、标注）
- **第三步**：需要把图片处理成模型能理解的格式

这个 `datasets.py` 文件就是帮你完成这些工作的！

---

## 🎯 文件结构总览

这个文件主要做了三件事：

1. **下载数据集** - 从网上下载医学图片数据
2. **整理数据集** - 把图片按类别分类好
3. **准备训练数据** - 把图片处理成模型能用的格式

---

## 📦 第一部分：数据集下载器（就像"数据搬运工"）

### 什么是数据集下载器？

就像你从网上下载电影一样，这些类帮你从网上下载医学图片数据集。

### 1. PADatasetDownloader（皮肤病变数据集）

**作用**：下载和整理皮肤病变的图片

**工作流程**：
```
1. 从网上下载压缩包
2. 解压压缩包
3. 读取 metadata.csv（一个表格，记录了每张图片的类别）
4. 把图片按照类别（比如：正常、病变等）分类到不同文件夹
```

**文件夹结构**：
```
PAD-Dataset/
  ├── 正常/
  │   ├── 图片1.jpg
  │   └── 图片2.jpg
  ├── 病变类型1/
  │   ├── 图片3.jpg
  │   └── 图片4.jpg
  └── 病变类型2/
      └── ...
```

### 2. FetalDatasetDownloader（胎儿超声数据集）

**作用**：下载和整理胎儿超声图片

**工作流程**：
```
1. 下载压缩包
2. 解压
3. 读取 Excel 文件（记录了每张图片对应的胎儿平面类型）
4. 按平面类型分类图片
```

### 3. ISICDatasetManager（皮肤癌数据集）

**作用**：下载 ISIC 2018 皮肤癌挑战赛的数据

**特点**：
- 有训练集和测试集
- 图片需要根据 CSV 文件中的标签分类
- 支持多标签（一张图片可能属于多个类别）

### 4. CPNDatasetDownloader（胸部 X 光数据集）

**作用**：下载 COVID-19、肺炎、正常胸部 X 光图片

### 5. KvasirDatasetDownloader（胃肠道数据集）

**作用**：下载胃肠道内窥镜图片

---

## 🔧 第二部分：核心函数

### build_dataset(args) - 构建数据集

**作用**：根据你指定的数据集名称，自动下载、整理并返回训练和测试数据

**工作流程**：
```python
def build_dataset(args):
    # 1. 根据数据集名称选择对应的下载器
    if args.dataset == 'Kvasir':
        downloader = KvasirDatasetDownloader()
        data_dir = downloader.get_dataset()  # 下载并整理数据
    
    # 2. 创建数据变换（把图片处理成模型能用的格式）
    train_transform, test_transform = build_transform(args)
    
    # 3. 加载图片并应用变换
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # 4. 返回训练集、测试集和类别数量
    return train_dataset, test_dataset, nb_classes
```

**关键概念**：
- **训练集**：用来训练模型的数据（就像学生用的教材）
- **测试集**：用来测试模型效果的数据（就像考试题）
- **类别数量**：有多少种不同的分类（比如：猫、狗、鸟 = 3 类）

### build_transform(args) - 构建数据变换

**作用**：把原始图片处理成模型能理解的格式

**训练时的变换**（让模型学习时更"聪明"）：
```python
t_train = [
    transforms.RandomResizedCrop(224),      # 随机裁剪并缩放到 224x224
    transforms.AugMix(alpha=0.4),           # 数据增强（让图片有点变化）
    transforms.RandomHorizontalFlip(p=0.4), # 随机水平翻转（40% 概率）
    transforms.ToTensor(),                  # 转换成张量（模型能理解的数字格式）
    transforms.Normalize(mean=[.5], std=[.5]) # 归一化（把数值调整到合适范围）
]
```

**测试时的变换**（简单处理）：
```python
t_test = [
    transforms.Resize((224, 224)),          # 缩放到 224x224
    transforms.ToTensor(),                  # 转换成张量
    transforms.Normalize(mean=[.5], std=[.5]) # 归一化
]
```

**为什么训练和测试的变换不一样？**
- **训练时**：需要数据增强，让模型看到更多变化的图片，学得更全面
- **测试时**：只需要简单处理，看看模型真实的表现

---

## 🎓 如何创建自己的分类数据集？

### 第一步：准备图片

1. **收集图片**
   - 每个类别至少准备 100-200 张图片（越多越好）
   - 图片格式：JPG、PNG 都可以
   - 图片大小：建议统一大小（比如 224x224 或更大）

2. **整理文件夹结构**
   ```
   my_dataset/
     ├── 类别1/
     │   ├── 图片1.jpg
     │   ├── 图片2.jpg
     │   └── ...
     ├── 类别2/
     │   ├── 图片1.jpg
     │   └── ...
     └── 类别3/
         └── ...
   ```

   **重要**：文件夹名称就是类别名称！

### 第二步：修改代码

#### 方法 1：直接使用 ImageFolder（最简单）

如果你的数据集已经是上面的文件夹结构，可以直接使用：

```python
from torchvision import datasets, transforms

# 创建数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = datasets.ImageFolder(
    root='my_dataset/train',  # 训练集路径
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root='my_dataset/test',   # 测试集路径
    transform=test_transform
)

# 获取类别数量
nb_classes = len(train_dataset.classes)
print(f"类别数量: {nb_classes}")
print(f"类别名称: {train_dataset.classes}")
```

#### 方法 2：添加到 build_dataset 函数中

在 `datasets.py` 的 `build_dataset` 函数中添加你的数据集：

```python
def build_dataset(args):
    train_transform, test_transform = build_transform(args)
    
    # ... 其他数据集的处理 ...
    
    elif args.dataset == 'my_dataset':  # 你的数据集名称
        # 设置类别数量
        nb_classes = 3  # 根据你的类别数量修改
        
        # 直接使用 ImageFolder 加载
        train_dataset = datasets.ImageFolder(
            root='./data/my_dataset/train',  # 训练集路径
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root='./data/my_dataset/test',   # 测试集路径
            transform=test_transform
        )
        
        return train_dataset, test_dataset, nb_classes
```

### 第三步：划分训练集和测试集

如果你的所有图片都在一个文件夹里，需要划分：

```python
from torch.utils.data import random_split

# 加载所有数据
full_dataset = datasets.ImageFolder(root='my_dataset', transform=train_transform)

# 计算划分大小（80% 训练，20% 测试）
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 随机划分
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
```

### 第四步：使用你的数据集训练

```bash
python main.py --dataset 'my_dataset' --model_name 'MedViT_small' --epochs 100
```

---

## 📝 完整示例：创建一个"水果分类"数据集

### 步骤 1：准备数据

```
fruit_dataset/
  ├── train/
  │   ├── apple/
  │   │   ├── apple1.jpg
  │   │   ├── apple2.jpg
  │   │   └── ...
  │   ├── banana/
  │   │   ├── banana1.jpg
  │   │   └── ...
  │   └── orange/
  │       └── ...
  └── test/
      ├── apple/
      ├── banana/
      └── orange/
```

### 步骤 2：修改代码

在 `datasets.py` 中添加：

```python
elif args.dataset == 'fruit':
    nb_classes = 3  # 苹果、香蕉、橙子
    
    train_dataset = datasets.ImageFolder(
        root='./data/fruit_dataset/train',
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root='./data/fruit_dataset/test',
        transform=test_transform
    )
    
    return train_dataset, test_dataset, nb_classes
```

### 步骤 3：训练

```bash
python main.py --dataset 'fruit' --model_name 'MedViT_tiny' --epochs 50 --batch_size 16
```

---

## 🔍 代码详解

### 每个下载器类都有这些方法：

1. **`__init__`** - 初始化（设置路径、URL 等）
2. **`download_dataset`** - 下载数据集
3. **`extract_dataset`** - 解压数据集
4. **`organize_images`** - 整理图片（按类别分类）
5. **`get_dataset`** - 主方法（自动完成所有步骤）

### 关键代码片段解释：

#### 1. 下载文件
```python
with requests.get(self.dataset_url, stream=True) as r:
    r.raise_for_status()
    with open(self.dataset_zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
```
**解释**：从网上下载文件，分块写入本地（避免内存溢出）

#### 2. 解压文件
```python
with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(self.root_dir)
```
**解释**：解压 ZIP 文件到指定目录

#### 3. 读取 CSV 并分类
```python
metadata = pd.read_csv(self.metadata_file_path)
for _, row in metadata.iterrows():
    img_id = row['img_id']
    diagnostic = row['diagnostic']  # 类别
    # 移动文件到对应类别文件夹
    shutil.move(source_path, destination_path)
```
**解释**：读取表格，根据类别把图片移动到对应文件夹

---

## 💡 常见问题

### Q1: 我的图片大小不一样怎么办？
**A**: 不用担心！`transforms.Resize((224, 224))` 会自动把所有图片缩放到统一大小。

### Q2: 需要多少张图片？
**A**: 
- **最少**：每个类别 50-100 张
- **推荐**：每个类别 200-500 张
- **理想**：每个类别 1000+ 张

### Q3: 图片格式有要求吗？
**A**: JPG、PNG 都可以，ImageFolder 会自动处理。

### Q4: 如何知道我的数据集有多少类别？
**A**: 
```python
dataset = datasets.ImageFolder(root='my_dataset')
print(f"类别数量: {len(dataset.classes)}")
print(f"类别名称: {dataset.classes}")
```

### Q5: 训练集和测试集的比例？
**A**: 通常 80% 训练，20% 测试。如果数据少，可以用 70%-30%。

---

## 🎯 快速开始清单

创建自己的数据集：

- [ ] 收集图片（每个类别至少 100 张）
- [ ] 按类别整理到文件夹
- [ ] 划分训练集和测试集
- [ ] 在 `build_dataset` 函数中添加你的数据集
- [ ] 运行训练命令

---

## 📚 总结

这个文件的核心思想：
1. **下载** - 从网上下载数据
2. **整理** - 按类别分类
3. **处理** - 转换成模型能用的格式
4. **返回** - 返回训练集和测试集

创建自己的数据集只需要：
1. 准备图片并按类别分类
2. 使用 `ImageFolder` 加载
3. 添加到 `build_dataset` 函数
4. 开始训练！

**记住**：文件夹名称 = 类别名称，这是关键！

---

**祝你训练顺利！如果遇到问题，可以查看代码中的 print 语句，它们会告诉你每一步在做什么。** 😊

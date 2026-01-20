"""
创建自己的分类数据集 - 完整示例
这个文件展示了如何创建和加载自己的图片分类数据集
"""

# 标准库
import os
import shutil
from pathlib import Path

# 第三方库
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 本地模块
from cedar.utils import print


def create_my_dataset_structure(base_dir='my_dataset'):
    """
    创建数据集文件夹结构
    
    参数:
        base_dir: 数据集根目录
    """
    print(f"[创建数据集] 开始创建数据集结构: {base_dir}")
    
    # 创建主目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建训练集和测试集目录
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 示例：创建类别文件夹（根据你的实际类别修改）
    classes = ['类别1', '类别2', '类别3']  # 修改为你的类别名称
    
    for class_name in classes:
        # 创建训练集类别文件夹
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        print(f"[创建数据集] 创建训练集类别文件夹: {train_class_dir}")
        
        # 创建测试集类别文件夹
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        print(f"[创建数据集] 创建测试集类别文件夹: {test_class_dir}")
    
    print(f"[创建数据集] 数据集结构创建完成！")
    print(f"[创建数据集] 请将图片放入对应的类别文件夹中")
    print(f"[创建数据集] 文件夹结构:")
    print(f"  {base_dir}/")
    print(f"    ├── train/")
    for class_name in classes:
        print(f"    │   ├── {class_name}/")
        print(f"    │   │   └── (放入训练图片)")
    print(f"    └── test/")
    for class_name in classes:
        print(f"        ├── {class_name}/")
        print(f"        │   └── (放入测试图片)")


def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8):
    """
    如果所有图片都在一个文件夹里，这个函数帮你划分训练集和测试集
    
    参数:
        source_dir: 源文件夹（包含所有类别文件夹）
        train_dir: 训练集目标文件夹
        test_dir: 测试集目标文件夹
        train_ratio: 训练集比例（默认 80%）
    """
    print(f"[划分数据集] 开始划分数据集")
    print(f"[划分数据集] 源目录: {source_dir}")
    print(f"[划分数据集] 训练集比例: {train_ratio*100}%")
    
    # 获取所有类别
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        print(f"[划分数据集] 处理类别: {class_name}")
        
        source_class_dir = os.path.join(source_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        # 创建目标文件夹
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 计算划分点
        train_count = int(len(image_files) * train_ratio)
        
        # 划分文件
        train_files = image_files[:train_count]
        test_files = image_files[train_count:]
        
        # 复制文件
        for f in train_files:
            shutil.copy(
                os.path.join(source_class_dir, f),
                os.path.join(train_class_dir, f)
            )
        
        for f in test_files:
            shutil.copy(
                os.path.join(source_class_dir, f),
                os.path.join(test_class_dir, f)
            )
        
        print(f"[划分数据集] {class_name}: 训练集 {len(train_files)} 张, 测试集 {len(test_files)} 张")
    
    print(f"[划分数据集] 数据集划分完成！")


def load_my_dataset(dataset_path, batch_size=32):
    """
    加载自己的数据集
    
    参数:
        dataset_path: 数据集路径
        batch_size: 批次大小
    """
    print(f"[加载数据集] 开始加载数据集: {dataset_path}")
    
    # 创建数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),           # 缩放到 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),           # 随机旋转 ±10 度
        transforms.ToTensor(),                   # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),           # 缩放到 224x224
        transforms.ToTensor(),                   # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])
    
    # 加载数据集
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    # 获取类别信息
    nb_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    print(f"[加载数据集] 类别数量: {nb_classes}")
    print(f"[加载数据集] 类别名称: {class_names}")
    print(f"[加载数据集] 训练集大小: {len(train_dataset)}")
    print(f"[加载数据集] 测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 训练时打乱顺序
        num_workers=2      # 使用 2 个进程加载数据
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # 测试时不打乱
        num_workers=2
    )
    
    print(f"[加载数据集] 数据加载器创建完成")
    
    return train_loader, test_loader, nb_classes, class_names


def add_to_build_dataset_example():
    """
    展示如何在 datasets.py 的 build_dataset 函数中添加你的数据集
    """
    example_code = '''
# 在 datasets.py 的 build_dataset 函数中添加：

elif args.dataset == 'my_dataset':  # 你的数据集名称
    nb_classes = 3  # 根据你的类别数量修改
    
    # 创建数据变换
    train_transform, test_transform = build_transform(args)
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root='./data/my_dataset/train',  # 训练集路径
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root='./data/my_dataset/test',   # 测试集路径
        transform=test_transform
    )
    
    print(f"Dataset is available at: ./data/my_dataset")
    print(f"Number of classes: {nb_classes}")
    print(f"Class names: {train_dataset.classes}")
    
    return train_dataset, test_dataset, nb_classes
    '''
    
    print("[示例代码] 如何在 build_dataset 中添加你的数据集:")
    print(example_code)


if __name__ == '__main__':
    print("=" * 60)
    print("创建自己的分类数据集 - 示例代码")
    print("=" * 60)
    print()
    
    # 示例 1: 创建数据集结构
    print("示例 1: 创建数据集文件夹结构")
    print("-" * 60)
    create_my_dataset_structure('my_dataset')
    print()
    
    # 示例 2: 展示如何划分数据集
    print("示例 2: 划分数据集（如果所有图片在一个文件夹）")
    print("-" * 60)
    print("如果所有图片都在一个文件夹里，可以使用 split_dataset 函数:")
    print("  split_dataset('all_images', 'my_dataset/train', 'my_dataset/test')")
    print()
    
    # 示例 3: 加载数据集
    print("示例 3: 加载数据集")
    print("-" * 60)
    print("如果数据集已经准备好，可以使用 load_my_dataset 函数:")
    print("  train_loader, test_loader, nb_classes, class_names = load_my_dataset('my_dataset')")
    print()
    
    # 示例 4: 添加到 build_dataset
    print("示例 4: 添加到 build_dataset 函数")
    print("-" * 60)
    add_to_build_dataset_example()
    print()
    
    print("=" * 60)
    print("使用步骤:")
    print("1. 准备图片，按类别分类到文件夹")
    print("2. 使用 create_my_dataset_structure 创建文件夹结构")
    print("3. 将图片放入对应文件夹")
    print("4. 在 datasets.py 的 build_dataset 函数中添加你的数据集")
    print("5. 运行: python main.py --dataset 'my_dataset' --model_name 'MedViT_tiny'")
    print("=" * 60)

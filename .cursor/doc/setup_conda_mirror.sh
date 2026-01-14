#!/bin/bash
# 配置Conda使用清华镜像源（推荐方案）

echo "=========================================="
echo "配置Conda使用清华镜像源"
echo "=========================================="

# 移除现有的defaults通道（它指向无法访问的repo.anaconda.com）
echo "1. 移除defaults通道..."
# 尝试多种方法移除defaults通道
conda config --remove channels defaults 2>/dev/null || true
conda config --remove channels 'defaults' 2>/dev/null || true

# 清除所有现有通道
echo "2. 清除所有现有通道..."
conda config --remove-key channels 2>/dev/null || echo "  (可能没有配置通道)"

# 添加清华镜像源（按优先级顺序）
echo "3. 添加清华镜像源..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2

# 关键：设置通道优先级为strict，确保只使用配置的通道，忽略defaults
echo "4. 设置通道优先级为strict（忽略defaults）..."
conda config --set channel_priority strict

# 设置显示通道URL
conda config --set show_channel_urls yes

# 再次尝试移除defaults（如果仍然存在）
echo "5. 再次检查并移除defaults通道..."
conda config --remove channels defaults 2>/dev/null || true

# 显示配置结果
echo ""
echo "=========================================="
echo "当前Conda通道配置:"
echo "=========================================="
conda config --show channels

echo ""
echo "=========================================="
echo "✓ 配置完成！"
echo "=========================================="
echo ""
echo "现在可以尝试创建环境:"
echo "  conda create -n py312 python=3.12.12 -y"
echo ""

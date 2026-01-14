# 环境设置指南

## 问题：Conda 网络连接失败

当遇到 `Network is unreachable` 或 `HTTP 000 CONNECTION FAILED` 错误时，可以使用以下解决方案。

---

## 解决方案

### 方案1：使用国内Conda镜像源

**根据网络测试结果，推荐使用清华镜像源（mirrors.tuna.tsinghua.edu.cn）**

#### 1.1 配置清华镜像源（⭐ 推荐，已验证可用）

```bash
# 重要：先移除defaults通道（它指向无法访问的repo.anaconda.com）
conda config --remove channels defaults

# 清除所有现有通道（可选，确保干净配置）
conda config --remove-key channels

# 添加清华镜像源（按优先级顺序）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes

# 验证配置
conda config --show channels
```

**或使用一键配置脚本：**
```bash
chmod +x setup_conda_mirror.sh
./setup_conda_mirror.sh
```

#### 1.2 配置中科大镜像源（⚠️ 不推荐，HTTP 403错误）

**注意：根据网络测试，中科大镜像源虽然TCP连接成功，但HTTP返回403 Forbidden，可能有限制。**

```bash
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
```

#### 1.3 配置阿里云镜像源

```bash
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge
```

#### 1.4 使用镜像源创建环境

```bash
# 创建Python 3.12环境
conda create -n py312 python=3.12.12 -y


# 激活环境
conda activate py312
```

#### 1.5 恢复默认源（如需要）

```bash
conda config --remove-key channels
```

---

### 方案2：使用Python venv（最简单，推荐）

如果conda网络问题无法解决，直接使用Python内置的venv模块，无需网络连接。

#### 2.1 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活环境（Linux/Mac）
source venv/bin/activate

# 激活环境（Windows）
venv\Scripts\activate
```

#### 2.2 升级pip并安装依赖

**根据网络测试结果：**
- ✅ PyTorch官方源（download.pytorch.org）可访问
- ✅ PyPI官方源（pypi.org）可访问
- ⚠️ natten源（shi-labs.com）SSL证书过期，需要特殊处理

```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch（使用PyTorch官方源，已验证可访问）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 安装natten（注意：shi-labs.com的SSL证书过期，需要跳过证书验证）
pip install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com

# 安装其他依赖（使用PyPI官方源，已验证可访问）
pip install -r requirements.txt
```

**或使用一键安装脚本：**
```bash
chmod +x setup_venv_recommended.sh
./setup_venv_recommended.sh
```

#### 2.3 使用国内pip镜像源（可选）

**注意：根据网络测试，清华PyPI镜像返回403 Forbidden，建议使用PyPI官方源。**

```bash
# PyPI官方源已验证可访问，推荐直接使用
# 如果确实需要使用镜像，可以尝试其他镜像源：

# 阿里云镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 中科大镜像
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
```

---

### 方案3：配置代理（如果有代理服务器）

#### 3.1 配置conda代理

```bash
# 设置HTTP代理
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https https://proxy.example.com:8080

# 或使用环境变量
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=https://proxy.example.com:8080
```

#### 3.2 配置pip代理

```bash
# 使用代理安装
pip install --proxy http://proxy.example.com:8080 package_name

# 或配置环境变量
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=https://proxy.example.com:8080
pip install package_name
```

---

### 方案4：离线安装（完全无网络环境）

#### 4.1 在有网络的机器上下载包

```bash
# 使用conda-pack打包环境
conda install conda-pack
conda pack -n py312 -o py312.tar.gz

# 或使用pip下载wheel文件
pip download -d ./packages -r requirements.txt
pip download -d ./packages torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
```

#### 4.2 在无网络机器上安装

```bash
# 解压conda环境
mkdir -p py312
tar -xzf py312.tar.gz -C py312
source py312/bin/activate

# 或安装wheel文件
pip install --no-index --find-links ./packages -r requirements.txt
```

---

## MedViTV2 完整环境设置步骤

### ⭐ 推荐方案1：使用venv（最简单可靠）

**根据网络测试，推荐使用venv + PyPI官方源，所有依赖源已验证可访问。**

```bash
# 使用一键安装脚本（推荐）
chmod +x setup_venv_recommended.sh
./setup_venv_recommended.sh

# 或手动执行：
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 安装PyTorch（使用官方源，已验证可访问）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装natten（跳过SSL证书验证）
pip install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com

# 5. 安装其他依赖（使用PyPI官方源，已验证可访问）
pip install -r requirements.txt

# 6. 验证安装
python -c "import torch; import natten; print('安装成功！')"
```

### 推荐方案2：使用conda + 清华镜像源

**根据网络测试，清华conda镜像源完全可用。**

```bash
# 使用一键配置脚本（推荐）
chmod +x setup_conda_mirror.sh
./setup_conda_mirror.sh

# 然后创建环境
conda create -n py312 python=3.12.12 -y
conda activate py312

# 安装依赖（使用pip，因为conda可能没有所有包）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com
pip install -r requirements.txt
```

---

## 常见问题排查

### Q1: conda创建环境时提示网络不可达

**解决方案**：
- 使用方案1配置国内镜像源
- 或使用方案2改用venv

### Q2: pip安装PyTorch时连接超时

**解决方案**：
- 检查网络连接
- 尝试使用代理（方案3）
- 或使用国内PyTorch镜像（注意版本可能不完全匹配）

### Q3: natten安装失败

**解决方案**：
- 确保已安装对应版本的PyTorch
- 检查natten版本是否与PyTorch版本匹配
- 访问 https://shi-labs.com/natten/ 查看支持的版本

### Q4: 在服务器上无法访问外网

**解决方案**：
- 使用方案4离线安装
- 或联系管理员配置代理服务器

---

## 环境验证

安装完成后，运行以下命令验证环境：

```bash
# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查natten
python -c "import natten; print('natten安装成功')"

# 检查其他依赖
python -c "import timm, medmnist, einops; print('依赖包安装成功')"
```

---

## 快速参考

### Conda镜像源配置（一键配置）

```bash
# 清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### Pip镜像源配置

```bash
# 清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里云镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 中科大镜像
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
```

---

**最后更新**: 2025-01-XX
**维护者**: 项目维护团队

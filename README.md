# AI算法课程项目

本项目包含AI算法课程中的模型训练、结果分析与参数调优相关内容。

## 项目结构

```
.
├── .venv/                          # Python 3.14 虚拟环境
├── 05_模型结果分析与参数调优.ipynb   # 主Notebook文件
└── README.md                       # 本文件
```

## 环境配置

本项目使用 Python 3.14 虚拟环境，包含以下主要依赖包：

| 包名 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.13.0.dev20260426+cpu | 深度学习框架 |
| torchvision | 0.27.0.dev20260425+cpu | 计算机视觉工具库 |
| torchaudio | 2.11.0.dev20260425+cpu | 音频处理工具库 |
| transformers | 5.6.2 | 预训练模型库 |
| datasets | 4.8.4 | 数据集加载工具 |
| trl | 1.2.0 | 强化学习训练库 |
| matplotlib | 3.10.9 | 数据可视化 |
| pandas | 3.0.2 | 数据处理分析 |
| torchinfo | 1.8.0 | 模型结构可视化 |
| torchmetrics | 1.9.0 | 模型评估指标 |
| mlxtend | 0.24.0 | 机器学习扩展工具 |
| jupyter | 1.1.1 | Jupyter Notebook支持 |
| ipykernel | 7.2.0 | Jupyter内核支持 |

## 快速开始

### 1. 激活虚拟环境

在 Windows PowerShell 中：
```powershell
.venv\Scripts\activate
```

### 2. 启动 Jupyter Notebook

```powershell
python -m jupyter notebook
```

或者在 VS Code 中直接打开 `.ipynb` 文件。

### 3. 选择内核

在 Notebook 中，点击右上角的 **"选择内核"** 按钮，选择 **"Python 3.14 (.venv)"**。

## Notebook 内容说明

### 05_模型结果分析与参数调优.ipynb

本 Notebook 涵盖以下内容：

1. **数据准备**
   - CIFAR-10 数据集加载与预处理
   - 数据增强技术（随机裁剪、翻转、旋转、颜色抖动、随机擦除）
   - 数据归一化处理

2. **模型构建**
   - 基础卷积神经网络（CNN）
   - 残差网络（ResNet）结构
   - 注意力机制（SE Block）
   - 深度可分离卷积（Depthwise Separable Convolution）

3. **训练与评估**
   - 训练循环实现
   - 测试评估函数
   - 准确率计算
   - 损失函数分析

4. **模型优化技术**
   - 残差连接（Residual Connections）
   - 批量归一化（Batch Normalization）
   - 通道注意力机制（Squeeze-and-Excitation）
   - 轻量化卷积设计

## 硬件要求

- **推荐**: NVIDIA GPU with CUDA 支持
- **最低**: CPU 运行（已配置）

代码会自动检测 CUDA 可用性：
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 数据集

本项目使用 CIFAR-10 数据集，包含：
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 训练集：50,000 张 32×32 彩色图像
- 测试集：10,000 张 32×32 彩色图像

数据集会自动下载到 `data/` 目录。

## 内核管理

### 查看已安装的内核
```powershell
python -m jupyter kernelspec list
```

### 当前可用内核
- `python3` - 默认 Python 3 内核
- `.venv` - Python 3.14 (.venv) 内核（推荐）

## 注意事项

1. 本项目使用 Python 3.14 最新版本，部分包使用 nightly 版本以确保兼容性
2. 首次运行时会自动下载 CIFAR-10 数据集（约 170MB）
3. 建议使用 GPU 进行训练以获得更好性能

## 许可证

仅供学习交流使用

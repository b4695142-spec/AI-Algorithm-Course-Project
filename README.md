# PyTorch深度学习实践教程

本项目是一套PyTorch深度学习入门教程，从基础的二次函数回归到卷积神经网络搭建与调优，循序渐进地帮助初学者掌握PyTorch的核心模块和深度学习工作流程。

## 项目简介

本项目包含三个递进式教程，覆盖了从PyTorch基础操作到CNN网络搭建与模型调优的完整学习路径。通过二次函数回归和CIFAR-10图像分类两个实战案例，您可以系统性地学习PyTorch的完整工作流程。

## 环境要求

- Python 3.x
- PyTorch 2.7.1+
- torchvision 0.27+
- matplotlib
- torchmetrics
- torchinfo
- tqdm

## 安装依赖

```bash
pip install torch torchvision matplotlib torchmetrics torchinfo tqdm
```

## 项目内容

### 文件说明

| 文件 | 说明 |
|------|------|
| `1.Pytorch模块实践.ipynb` | PyTorch基础教程，通过二次函数回归学习核心模块 |
| `2. 深度学习网络搭建.ipynb` | CNN卷积神经网络教程，基于CIFAR-10图像分类任务 |
| `3. 模型结果分析与参数调优.ipynb` | 模型调优教程，多种网络架构对比与数据增强实践 |

### 教程一：PyTorch模块实践

| 步骤 | 模块 | 内容 |
|------|------|------|
| **1. 数据准备** | `torch.Tensor`, `torch.utils.data` | 生成带噪声的二次函数数据，划分训练集和测试集 |
| **2. 构建模型** | `torch.nn` | 创建自定义神经网络模型（`nn.Module`、`nn.Parameter`），学习数据中的模式 |
| **3. 模型训练** | `Autograd`, `torch.optim` | 定义损失函数（MAE）和优化器（SGD），构建训练循环 |
| **4. 推理与评估** | `Eval`, `torch.inference_mode()` | 在测试数据上评估模型性能 |
| **5. 保存与加载** | `torch.save/load`, `state_dict()` | 保存和加载训练好的模型参数 |

### 教程二：深度学习网络搭建

| 步骤 | 内容 |
|------|------|
| **0. 计算机视觉库** | 了解 `torchvision`、`torchvision.datasets`、`torchvision.transforms` 等模块 |
| **1. 加载数据** | 使用CIFAR-10数据集（60000张32×32彩色图像，10个类别） |
| **2. 准备数据** | 使用 `DataLoader` 创建可迭代数据集 |
| **3. 基准模型** | 构建线性基准分类模型，选择损失函数和优化器 |
| **4. 预测与评估** | 使用基准模型进行预测和评估 |
| **5. 设备无关代码** | 编写CPU/GPU兼容代码 |
| **6. 非线性模型** | 添加非线性激活层改进基准模型 |
| **7. CNN模型** | 引入卷积神经网络（`nn.Conv2d`、`nn.MaxPool2d`） |
| **8. 模型对比** | 比较线性模型、非线性模型和CNN模型的性能 |
| **9. 评估最佳模型** | 在随机图像上进行预测和评估 |
| **10. 混淆矩阵** | 生成分类混淆矩阵，分析模型在各类别上的表现 |
| **11. 保存与加载** | 保存最佳模型并验证加载正确性 |

### 教程三：模型结果分析与参数调优

| 步骤 | 内容 |
|------|------|
| **数据增强** | RandomCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、RandomErasing |
| **多种网络架构** | 对比6种不同架构的CNN模型 |
| **训练与评估** | 封装 `train_step`、`test_step`、`eval_model` 函数 |
| **结果可视化** | 绘制训练/测试损失曲线和精度曲线 |
| **模型结构分析** | 使用 `torchinfo.summary` 查看模型参数量和计算量 |

#### 网络架构对比

| 模型 | 核心技术 | 特点 |
|------|----------|------|
| `CIFAR10ModelV2` | 基础CNN | Conv2d + ReLU + MaxPool2d |
| `ResidualCIFAR10Model` | 残差连接 | ResidualBlock + BatchNorm |
| `SECIFAR10Model` | SE注意力 | Squeeze-and-Excitation通道注意力 |
| `DWConvCIFAR10Model` | 深度可分离卷积 | Depthwise + Pointwise卷积 |
| `CombinedCIFAR10Model` | 残差+深度可分离+SE | 多技术融合 |
| `Combined2CIFAR10Model` | 残差+深度可分离+ECA+SiLU | ECA注意力替代SE，SiLU替代ReLU |

## 核心知识点

### PyTorch关键模块

- **torch.nn**: 包含神经网络的所有构建块
- **nn.Module**: 所有神经网络模块的基类
- **nn.Parameter**: 存储可学习的参数（权重和偏置）
- **forward()**: 定义前向传播计算
- **Autograd**: 自动微分，用于计算梯度
- **torch.optim**: 优化器，用于更新模型参数

### 计算机视觉关键模块

- **torchvision.datasets**: 常用视觉数据集
- **torchvision.transforms**: 图像变换与数据增强
- **torch.utils.data.DataLoader**: 批量数据加载
- **nn.Conv2d**: 二维卷积层
- **nn.MaxPool2d**: 最大池化层
- **nn.BatchNorm2d**: 批归一化层

### 数据划分

- **训练集 (60-80%)**: 用于模型学习
- **验证集 (10-20%)**: 用于模型调优
- **测试集 (10-20%)**: 用于评估模型泛化能力

## 使用方法

1. 克隆或下载本项目
2. 使用Jupyter Notebook按顺序打开教程文件
3. 按顺序运行代码单元格，逐步学习

```bash
jupyter notebook "1.Pytorch模块实践.ipynb"
jupyter notebook "2. 深度学习网络搭建.ipynb"
jupyter notebook "3. 模型结果分析与参数调优.ipynb"
```

## 学习目标

完成本项目后，您将能够：

- 理解PyTorch的张量操作和数据处理
- 掌握神经网络模型的构建方法（从线性模型到CNN）
- 学会定义损失函数和优化器
- 理解训练循环的工作原理
- 掌握模型的保存和加载方法
- 了解数据增强技术及其应用
- 理解残差连接、注意力机制、深度可分离卷积等现代网络技术
- 学会比较和评估不同模型架构的性能

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [torchvision文档](https://pytorch.org/vision/stable/index.html)

## 许可证

本项目仅供学习和教学使用。

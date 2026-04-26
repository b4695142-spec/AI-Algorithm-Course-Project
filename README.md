# PyTorch模块实践

本项目是一个PyTorch入门教程，通过构建一个简单的二次函数回归模型，帮助初学者快速了解PyTorch的核心模块和工作流程。

## 项目简介

本项目使用PyTorch构建了一个神经网络模型，用于学习二次函数曲线的参数。通过本项目，您可以学习到PyTorch的完整工作流程，包括数据准备、模型构建、训练、评估和保存等核心环节。

## 环境要求

- Python 3.x
- PyTorch 2.7.1+
- matplotlib

## 安装依赖

```bash
pip install torch matplotlib
```

## 项目内容

### 文件说明

| 文件 | 说明 |
|------|------|
| `1.Pytorch模块实践.ipynb` | 主教程文件，包含完整的PyTorch实践代码 |

### PyTorch工作流程

本项目涵盖以下核心模块和实践内容：

| 步骤 | 模块 | 内容 |
|------|------|------|
| **1. 数据准备** | `torch.Tensor`, `torch.utils.data` | 生成带噪声的二次函数数据，划分训练集和测试集 |
| **2. 构建模型** | `torch.nn` | 创建自定义神经网络模型，学习数据中的模式 |
| **3. 模型训练** | `Autograd`, `torch.optim` | 定义损失函数和优化器，构建训练循环 |
| **4. 推理与评估** | `Eval` | 在测试数据上评估模型性能 |
| **5. 保存与加载** | `torch.save/load` | 保存和加载训练好的模型 |

## 核心知识点

### PyTorch关键模块

- **torch.nn**: 包含神经网络的所有构建块
- **nn.Module**: 所有神经网络模块的基类
- **nn.Parameter**: 存储可学习的参数（权重和偏置）
- **forward()**: 定义前向传播计算
- **Autograd**: 自动微分，用于计算梯度
- **torch.optim**: 优化器，用于更新模型参数

### 数据划分

- **训练集 (60-80%)**: 用于模型学习
- **验证集 (10-20%)**: 用于模型调优
- **测试集 (10-20%)**: 用于评估模型泛化能力

## 使用方法

1. 克隆或下载本项目
2. 使用Jupyter Notebook打开 `1.Pytorch模块实践.ipynb`
3. 按顺序运行代码单元格，学习PyTorch的各个模块

```bash
jupyter notebook "1.Pytorch模块实践.ipynb"
```

## 学习目标

完成本项目后，您将能够：

- 理解PyTorch的张量操作和数据处理
- 掌握神经网络模型的构建方法
- 学会定义损失函数和优化器
- 理解训练循环的工作原理
- 掌握模型的保存和加载方法

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch教程](https://pytorch.org/tutorials/)

## 许可证

本项目仅供学习和教学使用。

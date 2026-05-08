# PyTorch深度学习与大模型实践教程

本项目是一套从PyTorch深度学习入门到Hugging Face大模型应用的递进式教程，从基础的二次函数回归到卷积神经网络搭建与调优，再到Transformer模型与NLP实战，循序渐进地帮助初学者掌握深度学习的核心技术和工作流程。

## 项目简介

本项目包含四个递进式教程，覆盖了从PyTorch基础操作到CNN网络搭建与模型调优，再到Hugging Face核心组件与NLP文本分类实战的完整学习路径。通过二次函数回归、CIFAR-10图像分类和中文文本分类三个实战案例，您可以系统性地学习深度学习的完整工作流程。

## 环境要求

- Python 3.x
- PyTorch 2.7.1+
- torchvision 0.27+
- matplotlib
- torchmetrics
- torchinfo
- tqdm
- transformers
- datasets
- evaluate

## 安装依赖

```bash
pip install torch torchvision matplotlib torchmetrics torchinfo tqdm transformers datasets evaluate
```

## 项目内容

### 文件说明

| 文件 | 说明 |
|------|------|
| `1. Pytorch模块实践.ipynb` | PyTorch基础教程，通过二次函数回归学习核心模块 |
| `2. 深度学习网络搭建.ipynb` | CNN卷积神经网络教程，基于CIFAR-10图像分类任务 |
| `3. 模型结果分析与参数调优.ipynb` | 模型调优教程，多种网络架构对比与数据增强实践 |
| `4. HF核心组件.ipynb` | Hugging Face核心组件教程，Tokenizer/Model/Datasets/Evaluate与文本分类实战 |

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

### 教程四：Hugging Face核心组件

#### 1. Tokenizer基本使用

| 步骤 | 内容 |
|------|------|
| **Step1 加载与保存** | 使用 `AutoTokenizer.from_pretrained` 加载中文RoBERTa分词器，`save_pretrained` 保存到本地 |
| **Step2 句子分词** | 使用 `tokenizer.tokenize` 进行子词切分 |
| **Step3 查看词典** | 通过 `tokenizer.vocab` 查看完整词表 |
| **Step4 索引转换** | token与id之间的相互转换（`convert_tokens_to_ids`、`convert_ids_to_tokens`） |
| **Step5 填充与截断** | 处理变长序列的padding与truncation策略 |
| **Step6 其他输入部分** | attention_mask、token_type_ids等附加输入 |
| **Step7 快速调用方式** | 直接调用 `tokenizer()` 完成编码全流程 |
| **Step8 处理batch数据** | 批量文本的高效编码处理 |

#### 2. 模型加载与保存

| 步骤 | 内容 |
|------|------|
| **在线加载** | `AutoModel.from_pretrained` 从Hub下载模型 |
| **模型保存** | `save_pretrained` 将模型权重保存到本地 |
| **本地加载** | 从本地路径加载已保存的模型 |
| **Config** | `AutoConfig` 查看和修改模型配置参数 |
| **带头部模型** | `AutoModelForSequenceClassification` 等带任务头的模型加载 |

#### 3. Datasets基本使用

| 步骤 | 内容 |
|------|------|
| **加载在线数据集** | `load_dataset` 从Hub加载数据集 |
| **加载本地数据集** | 从CSV、JSON、文本等本地文件加载数据 |
| **数据集操作** | 选择、过滤、映射、拆分等数据处理操作 |
| **数据集保存** | 保存处理后的数据集 |

#### 4. Evaluate使用指南

| 步骤 | 内容 |
|------|------|
| **查看评估函数** | `evaluate.list_evaluation_modules` 列出所有可用指标 |
| **加载评估指标** | `evaluate.load` 加载指定评估函数 |
| **计算评估结果** | 使用指标计算模型性能 |
| **结果可视化** | `radar_plot` 雷达图对比多个指标 |

#### 5. 文本分类实例

| 步骤 | 内容 |
|------|------|
| **Step1 导入相关包** | AutoTokenizer、AutoModelForSequenceClassification、Trainer、TrainingArguments |
| **Step2 加载数据集** | 使用 `load_dataset` 加载中文评论分类数据集 |
| **Step3 划分数据集** | 将训练集拆分为训练集和验证集 |
| **Step4 数据集预处理** | 使用 `map` 对数据集进行分词和编码 |
| **Step5 创建模型** | 加载带分类头的预训练模型 |
| **Step6 创建评估函数** | 定义评估指标计算函数 |
| **Step7 创建TrainingArguments** | 配置训练超参数（学习率、批大小、评估策略等） |
| **Step8 创建Trainer** | 封装训练、评估和预测流程 |
| **Step9 模型训练** | 调用 `trainer.train()` 执行训练 |
| **Step10 模型评估** | 在测试集上评估模型性能 |
| **Step11 模型预测** | 使用 `pipeline` 进行推理预测 |

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

### Hugging Face关键组件

- **AutoTokenizer**: 自动匹配并实例化分词器，文本与token ID互转
- **AutoModel**: 自动加载预训练Transformer基础模型
- **AutoConfig**: 查看和修改模型配置参数
- **AutoModelForSequenceClassification**: 加载带分类头的模型
- **datasets**: 高效数据集加载与处理库
- **evaluate**: 统一的模型评估指标库
- **Trainer & TrainingArguments**: 高层训练API，封装训练/评估/预测流程
- **pipeline**: 一行代码完成推理预测

### 数据划分

- **训练集 (60-80%)**: 用于模型学习
- **验证集 (10-20%)**: 用于模型调优
- **测试集 (10-20%)**: 用于评估模型泛化能力

## 使用方法

1. 克隆或下载本项目
2. 使用Jupyter Notebook按顺序打开教程文件
3. 按顺序运行代码单元格，逐步学习

```bash
jupyter notebook "1. Pytorch模块实践.ipynb"
jupyter notebook "2. 深度学习网络搭建.ipynb"
jupyter notebook "3. 模型结果分析与参数调优.ipynb"
jupyter notebook "4. HF核心组件.ipynb"
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
- 掌握Hugging Face Transformers的核心组件（Tokenizer、Model、Config）
- 学会使用datasets库加载和处理NLP数据集
- 学会使用evaluate库进行模型评估
- 掌握Trainer API完成文本分类任务的端到端训练流程

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [torchvision文档](https://pytorch.org/vision/stable/index.html)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets文档](https://huggingface.co/docs/datasets)
- [Hugging Face Evaluate文档](https://huggingface.co/docs/evaluate)

## 许可证

本项目仅供学习和教学使用。

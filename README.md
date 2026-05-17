# PyTorch深度学习与大模型实践教程

本项目是一套从PyTorch深度学习入门到大模型微调与量化的递进式教程，从基础的二次函数回归到卷积神经网络搭建与调优，再到Transformer模型、NLP实战、LoRA微调、SFT监督微调与模型量化，循序渐进地帮助初学者掌握深度学习与大模型的核心技术和工作流程。

## 项目简介

本项目包含七个递进式教程，覆盖了从PyTorch基础操作到CNN网络搭建与模型调优，再到Hugging Face核心组件、NLP文本分类、LoRA参数高效微调、SFT监督微调以及模型量化的完整学习路径。通过二次函数回归、CIFAR-10图像分类、中文文本分类、LLM微调与量化等多个实战案例，您可以系统性地学习深度学习与大模型的完整工作流程。

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
- trl
- peft

## 安装依赖

```bash
pip install torch torchvision matplotlib torchmetrics torchinfo tqdm transformers datasets evaluate trl peft
```

## 数据集下载

教程6-8所需的数据集较大，请自行下载：

链接: <https://pan.baidu.com/s/11p_zhvX4ttC0vJ1BbBZklA?pwd=4bbx> 提取码: 4bbx

## 项目内容

### 文件说明

| 文件                           | 说明                                                          |
| ---------------------------- | ----------------------------------------------------------- |
| `1. Pytorch模块实践.ipynb`       | PyTorch基础教程，通过二次函数回归学习核心模块                                  |
| `2. 深度学习网络搭建.ipynb`          | CNN卷积神经网络教程，基于CIFAR-10图像分类任务                                |
| `3. 模型结果分析与参数调优.ipynb`       | 模型调优教程，多种网络架构对比与数据增强实践                                      |
| `4. HF核心组件.ipynb`            | Hugging Face核心组件教程，Tokenizer/Model/Datasets/Evaluate与文本分类实战 |
| `5. LoRA.ipynb`              | LoRA参数高效微调教程，使用PEFT+TRL微调SmolLM2                            |
| `6. quant_lenet5_cifar10.py` | 模型量化教程，LeNet-5量化感知训练(QAT)与ONNX导出                            |
| `7. SFT.ipynb`               | SFT监督微调教程，使用SFTTrainer微调SmolLM2                             |
| `8. quant_lenet5_cifar10.py` | 模型量化教程（同教程6），LeNet-5量化感知训练与ONNX导出                           |

### 教程一：PyTorch模块实践

| 步骤           | 模块                                 | 内容                                               |
| ------------ | ---------------------------------- | ------------------------------------------------ |
| **1. 数据准备**  | `torch.Tensor`, `torch.utils.data` | 生成带噪声的二次函数数据，划分训练集和测试集                           |
| **2. 构建模型**  | `torch.nn`                         | 创建自定义神经网络模型（`nn.Module`、`nn.Parameter`），学习数据中的模式 |
| **3. 模型训练**  | `Autograd`, `torch.optim`          | 定义损失函数（MAE）和优化器（SGD），构建训练循环                      |
| **4. 推理与评估** | `Eval`, `torch.inference_mode()`   | 在测试数据上评估模型性能                                     |
| **5. 保存与加载** | `torch.save/load`, `state_dict()`  | 保存和加载训练好的模型参数                                    |

### 教程二：深度学习网络搭建

| 步骤            | 内容                                                                   |
| ------------- | -------------------------------------------------------------------- |
| **0. 计算机视觉库** | 了解 `torchvision`、`torchvision.datasets`、`torchvision.transforms` 等模块 |
| **1. 加载数据**   | 使用CIFAR-10数据集（60000张32×32彩色图像，10个类别）                                 |
| **2. 准备数据**   | 使用 `DataLoader` 创建可迭代数据集                                             |
| **3. 基准模型**   | 构建线性基准分类模型，选择损失函数和优化器                                                |
| **4. 预测与评估**  | 使用基准模型进行预测和评估                                                        |
| **5. 设备无关代码** | 编写CPU/GPU兼容代码                                                        |
| **6. 非线性模型**  | 添加非线性激活层改进基准模型                                                       |
| **7. CNN模型**  | 引入卷积神经网络（`nn.Conv2d`、`nn.MaxPool2d`）                                 |
| **8. 模型对比**   | 比较线性模型、非线性模型和CNN模型的性能                                                |
| **9. 评估最佳模型** | 在随机图像上进行预测和评估                                                        |
| **10. 混淆矩阵**  | 生成分类混淆矩阵，分析模型在各类别上的表现                                                |
| **11. 保存与加载** | 保存最佳模型并验证加载正确性                                                       |

### 教程三：模型结果分析与参数调优

| 步骤         | 内容                                                                       |
| ---------- | ------------------------------------------------------------------------ |
| **数据增强**   | RandomCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、RandomErasing |
| **多种网络架构** | 对比6种不同架构的CNN模型                                                           |
| **训练与评估**  | 封装 `train_step`、`test_step`、`eval_model` 函数                              |
| **结果可视化**  | 绘制训练/测试损失曲线和精度曲线                                                         |
| **模型结构分析** | 使用 `torchinfo.summary` 查看模型参数量和计算量                                       |

#### 网络架构对比

| 模型                      | 核心技术              | 特点                          |
| ----------------------- | ----------------- | --------------------------- |
| `CIFAR10ModelV2`        | 基础CNN             | Conv2d + ReLU + MaxPool2d   |
| `ResidualCIFAR10Model`  | 残差连接              | ResidualBlock + BatchNorm   |
| `SECIFAR10Model`        | SE注意力             | Squeeze-and-Excitation通道注意力 |
| `DWConvCIFAR10Model`    | 深度可分离卷积           | Depthwise + Pointwise卷积     |
| `CombinedCIFAR10Model`  | 残差+深度可分离+SE       | 多技术融合                       |
| `Combined2CIFAR10Model` | 残差+深度可分离+ECA+SiLU | ECA注意力替代SE，SiLU替代ReLU       |

### 教程四：Hugging Face核心组件

#### 1. Tokenizer基本使用

| 步骤                  | 内容                                                                        |
| ------------------- | ------------------------------------------------------------------------- |
| **Step1 加载与保存**     | 使用 `AutoTokenizer.from_pretrained` 加载中文RoBERTa分词器，`save_pretrained` 保存到本地 |
| **Step2 句子分词**      | 使用 `tokenizer.tokenize` 进行子词切分                                            |
| **Step3 查看词典**      | 通过 `tokenizer.vocab` 查看完整词表                                               |
| **Step4 索引转换**      | token与id之间的相互转换（`convert_tokens_to_ids`、`convert_ids_to_tokens`）          |
| **Step5 填充与截断**     | 处理变长序列的padding与truncation策略                                               |
| **Step6 其他输入部分**    | attention\_mask、token\_type\_ids等附加输入                                     |
| **Step7 快速调用方式**    | 直接调用 `tokenizer()` 完成编码全流程                                                |
| **Step8 处理batch数据** | 批量文本的高效编码处理                                                               |

#### 2. 模型加载与保存

| 步骤         | 内容                                              |
| ---------- | ----------------------------------------------- |
| **在线加载**   | `AutoModel.from_pretrained` 从Hub下载模型            |
| **模型保存**   | `save_pretrained` 将模型权重保存到本地                    |
| **本地加载**   | 从本地路径加载已保存的模型                                   |
| **Config** | `AutoConfig` 查看和修改模型配置参数                        |
| **带头部模型**  | `AutoModelForSequenceClassification` 等带任务头的模型加载 |

#### 3. Datasets基本使用

| 步骤          | 内容                       |
| ----------- | ------------------------ |
| **加载在线数据集** | `load_dataset` 从Hub加载数据集 |
| **加载本地数据集** | 从CSV、JSON、文本等本地文件加载数据    |
| **数据集操作**   | 选择、过滤、映射、拆分等数据处理操作       |
| **数据集保存**   | 保存处理后的数据集                |

#### 4. Evaluate使用指南

| 步骤         | 内容                                          |
| ---------- | ------------------------------------------- |
| **查看评估函数** | `evaluate.list_evaluation_modules` 列出所有可用指标 |
| **加载评估指标** | `evaluate.load` 加载指定评估函数                    |
| **计算评估结果** | 使用指标计算模型性能                                  |
| **结果可视化**  | `radar_plot` 雷达图对比多个指标                      |

#### 5. 文本分类实例

| 步骤                            | 内容                                                                         |
| ----------------------------- | -------------------------------------------------------------------------- |
| **Step1 导入相关包**               | AutoTokenizer、AutoModelForSequenceClassification、Trainer、TrainingArguments |
| **Step2 加载数据集**               | 使用 `load_dataset` 加载中文评论分类数据集                                              |
| **Step3 划分数据集**               | 将训练集拆分为训练集和验证集                                                             |
| **Step4 数据集预处理**              | 使用 `map` 对数据集进行分词和编码                                                       |
| **Step5 创建模型**                | 加载带分类头的预训练模型                                                               |
| **Step6 创建评估函数**              | 定义评估指标计算函数                                                                 |
| **Step7 创建TrainingArguments** | 配置训练超参数（学习率、批大小、评估策略等）                                                     |
| **Step8 创建Trainer**           | 封装训练、评估和预测流程                                                               |
| **Step9 模型训练**                | 调用 `trainer.train()` 执行训练                                                  |
| **Step10 模型评估**               | 在测试集上评估模型性能                                                                |
| **Step11 模型预测**               | 使用 `pipeline` 进行推理预测                                                       |

### 教程五：LoRA参数高效微调

本教程演示如何使用LoRA（Low-Rank Adaptation）适配器高效微调大语言模型。LoRA是一种参数高效微调技术，冻结预训练模型权重，在注意力层添加小型可训练的低秩分解矩阵，通常可减少约90%的可训练参数。

| 步骤                  | 内容                                               |
| ------------------- | ------------------------------------------------ |
| **1. 环境配置**         | 安装Hugging Face库（trl、transformers、datasets、peft）  |
| **2. 加载数据集**        | 使用 `HuggingFaceTB/smoltalk` 日常对话数据集              |
| **3. 加载模型与分词器**     | 加载 `SmolLM2-135M` 模型，设置对话格式（`setup_chat_format`） |
| **4. 配置LoRA参数**     | 使用 `LoraConfig` 配置秩维度(r)、缩放因子(alpha)、dropout等参数  |
| **5. 配置训练参数**       | 使用 `SFTConfig` 配置学习率、批大小、梯度检查点等                  |
| **6. 创建SFTTrainer** | 传入PEFT配置，创建带LoRA的训练器                             |
| **7. 模型训练**         | 调用 `trainer.train()` 执行LoRA微调                    |
| **8. 合并适配器**        | 使用 `merge_and_unload()` 将LoRA适配器合并回基础模型          |
| **9. 推理测试**         | 使用 `pipeline` 对多个提示进行推理测试                        |

#### LoRA核心参数

| 参数               | 说明                   | 典型值            |
| ---------------- | -------------------- | -------------- |
| `r`              | LoRA更新矩阵的秩维度，越小压缩越多  | 4-32           |
| `lora_alpha`     | LoRA层缩放因子，越高适配越强     | 通常为2×r         |
| `lora_dropout`   | LoRA层dropout概率，防止过拟合 | 0.05           |
| `target_modules` | 应用LoRA的目标模块          | `"all-linear"` |

### 教程六：模型量化（LeNet-5 QAT）

本教程演示PyTorch量化感知训练（Quantization-Aware Training, QAT）的完整流程，基于LeNet-5网络在CIFAR-10数据集（5类子集）上进行训练和量化。

| 步骤              | 内容                                                                             |
| --------------- | ------------------------------------------------------------------------------ |
| **1. 定义模型**     | 构建带量化占位符（`QuantStub`/`DeQuantStub`）的LeNet-5模型                                  |
| **2. 数据准备**     | 加载CIFAR-10数据集，筛选5个类别（鸟、猫、鹿、狗、马），配置数据增强                                         |
| **3. 训练原始模型**   | 使用SGD优化器训练20个epoch                                                             |
| **4. 导出ONNX模型** | 将浮点模型导出为 `lenet5_cifar10.onnx`                                                 |
| **5. 配置量化参数**   | 设置 `QConfig`，激活值使用quint8 + per\_tensor\_affine，权重使用qint8 + per\_tensor\_affine |
| **6. 量化感知训练**   | 使用 `prepare_qat` 插入伪量化节点，继续训练2个epoch                                           |
| **7. 转换量化模型**   | 使用 `convert` 将伪量化节点转换为实际量化操作                                                   |
| **8. 导出与保存**    | 导出量化ONNX模型（`lenet5_cifar10_quant.onnx`），保存权重（`lenet5_quantized.pth`）           |

#### 量化关键组件

| 组件            | 说明                |
| ------------- | ----------------- |
| `QuantStub`   | 量化入口，将浮点输入转换为量化张量 |
| `DeQuantStub` | 反量化出口，将量化张量转换回浮点  |
| `prepare_qat` | 在模型中插入伪量化观察节点     |
| `convert`     | 将观察节点转换为实际量化操作    |
| `QConfig`     | 配置激活值和权重的量化方案     |

### 教程七：SFT监督微调

本教程演示如何使用 `trl` 库中的 `SFTTrainer` 对 `SmolLM2-135M` 模型进行监督微调（Supervised Fine-Tuning）。

| 步骤                  | 内容                                            |
| ------------------- | --------------------------------------------- |
| **1. 加载模型与分词器**     | 加载 `SmolLM2-135M`，设置对话格式（`setup_chat_format`） |
| **2. 基础模型测试**       | 微调前使用基础模型生成文本，观察输出质量                          |
| **3. 加载数据集**        | 使用 `HuggingFaceTB/smoltalk` 日常对话数据集           |
| **4. 配置SFTTrainer** | 设置训练参数（步数、批大小、学习率、评估策略等）                      |
| **5. 模型训练**         | 调用 `trainer.train()` 执行监督微调                   |
| **6. 训练曲线可视化**      | 使用matplotlib绘制训练/验证损失曲线                       |
| **7. 微调后推理**        | 使用微调后模型生成文本，与微调前结果对比                          |

#### Trainer vs SFTTrainer 对比

| 特性        | Trainer           | SFTTrainer         |
| --------- | ----------------- | ------------------ |
| 适用任务      | 通用监督学习（分类、问答、摘要等） | 大语言模型监督式生成微调       |
| PEFT/LoRA | 需手动集成             | 原生集成               |
| 示例打包      | 需手动优化             | 支持packing，提高GPU利用率 |
| Prompt屏蔽  | 需手动设置labels=-100  | 自动屏蔽prompt部分的loss  |
| 聊天格式      | 需手动处理             | 自动添加和管理聊天特殊标记      |

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

### 大模型微调关键组件

- **PEFT**: 参数高效微调库，支持LoRA、Prefix Tuning等方法
- **LoraConfig**: LoRA适配器配置（秩、缩放因子、dropout、目标模块）
- **SFTTrainer & SFTConfig**: 监督微调训练器，原生集成PEFT、示例打包、prompt屏蔽
- **setup\_chat\_format**: 自动配置模型的聊天格式和特殊标记
- **merge\_and\_unload**: 将LoRA适配器权重合并回基础模型

### 模型量化关键组件

- **QuantStub / DeQuantStub**: 量化/反量化占位符
- **prepare\_qat**: 量化感知训练准备，插入伪量化节点
- **convert**: 将伪量化模型转换为真正的量化模型
- **QConfig**: 量化配置（激活值/权重的数据类型和量化方案）
- **torch.onnx.export**: 导出ONNX格式模型

### 数据划分

- **训练集 (60-80%)**: 用于模型学习
- **验证集 (10-20%)**: 用于模型调优
- **测试集 (10-20%)**: 用于评估模型泛化能力

## 使用方法

1. 克隆或下载本项目
2. 下载教程6-7所需的数据集（见上方数据集下载链接）
3. 使用Jupyter Notebook按顺序打开教程文件
4. 按顺序运行代码单元格，逐步学习

```bash
jupyter notebook "1. Pytorch模块实践.ipynb"
jupyter notebook "2. 深度学习网络搭建.ipynb"
jupyter notebook "3. 模型结果分析与参数调优.ipynb"
jupyter notebook "4. HF核心组件.ipynb"
jupyter notebook "5. LoRA.ipynb"
python "6. quant_lenet5_cifar10.py"
jupyter notebook "7. SFT.ipynb"
python "8. quant_lenet5_cifar10.py"
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
- 理解LoRA参数高效微调的原理与配置
- 掌握使用SFTTrainer进行大模型监督微调的完整流程
- 理解量化感知训练（QAT）的原理与实现
- 学会将模型导出为ONNX格式并保存量化权重

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [torchvision文档](https://pytorch.org/vision/stable/index.html)
- [PyTorch量化教程](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets文档](https://huggingface.co/docs/datasets)
- [Hugging Face Evaluate文档](https://huggingface.co/docs/evaluate)
- [PEFT文档](https://huggingface.co/docs/peft)
- [TRL文档](https://huggingface.co/docs/trl)

## 许可证

本项目仅供学习和教学使用。

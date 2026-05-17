import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert, QConfig, default_observer, default_weight_observer

# 修改1：定义5个目标类别并创建标签映射
CLASSES = [2, 3, 4, 5, 7]  # 选择前5个类别（鸟、猫、鹿、狗、马）
NUM_CLASSES = len(CLASSES)

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 量化占位符
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        # 量化输入
        x = self.quant(x)
        # 第一层卷积 + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 第二层卷积 + 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # 展平
        x = x.reshape(-1, 16 * 5 * 5)  # 使用 reshape 替代 view
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 反量化输出
        x = self.dequant(x)
        return x


# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

def filter_classes(dataset):
    idx = [i for i, label in enumerate(dataset.targets) if label in CLASSES]
    dataset.targets = [CLASSES.index(label) for label in dataset.targets if label in CLASSES]
    dataset.data = dataset.data[idx]
    return dataset

# 加载CIFAR10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
train_dataset = filter_classes(train_dataset)
test_dataset = filter_classes(test_dataset)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义训练和测试函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

# 设置设备并初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)      
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练原始模型
epochs = 20
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 将模型和数据移动到 CPU（量化操作需要在 CPU 上进行）
device = torch.device("cpu")
model = model.to(device)

inputs = torch.randn(1,3,32,32)
torch.onnx.export(model, inputs, "lenet5_cifar10.onnx",opset_version=16)

# 设置量化配置为 per_tensor_affine
qconfig = QConfig(
    activation=default_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=default_weight_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)

model.qconfig = qconfig

# 准备量化感知训练
model.train()  # 确保模型处于训练模式
model = prepare_qat(model)

# 继续训练（QAT）
for epoch in range(epochs + 1, epochs + 3):  # 再训练2个epoch
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 转换为量化模型
model.eval()  # 转换为量化模型前需要设置为评估模式
model = convert(model)

# 测试量化后的模型
print("Testing quantized model...")
# test(model, device, test_loader)

# 保存量化模型
print(model)
inputs = torch.randn(1,3,32,32)
torch.onnx.export(model, inputs, "lenet5_cifar10_quant.onnx",opset_version=17)

torch.save(model.state_dict(), "lenet5_quantized.pth")
print("Quantized model saved to lenet5_quantized.pth")
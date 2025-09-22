import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="./data", #下载数据集到哪里
    train=True, #获取训练集
    download=True,
    transform=ToTensor(), #转换成张量
)
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)
# print(len(training_data))

#创建数据加载器 --- 加载成包裹，方便分布式训练
train_loader = DataLoader(training_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 搭建CNN类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 将多层组合在一起
            nn.Conv2d( #Conv3d用于卷积视频
                in_channels=1, #图通道个数，1表示灰度图
                out_channels=16, #要得到多找个特征图
                kernel_size=5,#卷积核大小
                stride=1,#步长,第一次最好是1
                padding=2 #计算出来
            ),  # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ), # (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 7, 7)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5,1,2),
            nn.ReLU(), # (64, 7, 7)
        )
        self.out = nn.Linear( in_features=64*7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

model = CNN().to(device)
# print(model)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    batch_size_num = 1
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model.forward(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad() #梯度值清零
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"loss: {loss:>7f} [number: {batch_size_num}]")
        batch_size_num += 1

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            correct += (pred == y).type(torch.float).sum().item()
            a = (pred == y)
            b = (pred == y).type(torch.float).mean()
    test_loss /= num_batch
    correct /= size
    print(f"test result: Accuracy : {(100*correct)}%, Avg loss: {test_loss}")

loss_fn = nn.CrossEntropyLoss() # 创建损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #创建优化器

train(train_loader, model, loss_fn, optimizer)
test(test_loader, model, loss_fn)
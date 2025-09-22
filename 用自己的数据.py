import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms #数据转换

from torchvision.transforms import ToTensor
import os
import numpy as np
from PIL import Image

data_transformer = {
    'train': # 数据增强
        transforms.Compose([ #将操作组合到一起
            transforms.Resize([300, 300]),
            transforms.RandomRotation(45), # 随机旋转 -45度到 45度
            transforms.CenterCrop(256), # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p = 0.5), # 随机水平翻转
            transforms.RandomVerticalFlip(p = 0.5), #随机垂直翻转
            transforms.RandomGrayscale(p = 0.1), # 概率转换成灰度图
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

#自己的数据类
class food_dataset(Dataset):
    def __init__(self, file_path, transform = None):
        self.filepath = file_path
        self.transform = transform
        self.images = [] #路径
        self.labels = [] #标签
        with open(file_path, 'r') as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for img_path, label in samples:
                self.images.append(img_path)
                self.labels.append(label)

    def __getitem__(self, index): #为了兼容Dataloader
        image = Image.open(self.images[index]) #获取图像本身
        if self.transform is not None:
            image = self.transform(image) #转换成Tensor格式

        label = self.labels[index]
        label = torch.from_numpy(np.array(label, dtype=np.int64)) #转换成tensor类型
        return image, label

    def __len__(self):
        return len(self.images)

train_data = food_dataset('./train.txt', data_transformer['train'])
test_data = food_dataset('./test.txt', data_transformer['valid'])


device = "cuda" if torch.cuda.is_available() else "cpu"

# 搭建CNN类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 将多层组合在一起
            nn.Conv2d( #Conv3d用于卷积视频
                in_channels=3, #图通道个数，1表示灰度图
                out_channels=16, #要得到多找个特征图， 卷积核的个数
                kernel_size=5,#卷积核大小
                stride=1,#步长,第一次最好是1
                padding=2 #计算出来
            ),  # (16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (16, 128, 128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ), # (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 7, 7)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5,1,2),
            nn.ReLU(), # (64, 64, 64)
        )
        self.out = nn.Linear( in_features=64*64*64, out_features=20)

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

best_acc = 0
def test(dataloader, model, loss_fn, acc_s, loss_s):
    global best_acc
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
    acc_s.append(correct)
    loss_s.append(test_loss)

    # 保存最优模型
    if correct > best_acc:
        best_acc = correct
        # 保存参数 torch.save(model.state_dict(), path)
        torch.save(model.state_dict(), f'best_acc_{best_acc}.pt')
        # 保存完整模型
        # torch.save(model, 'best.pt')

loss_fn = nn.CrossEntropyLoss() # 创建损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #创建优化器
#调整学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 多轮训练
epoch_num = 3
acc_s = []
loss_s = []
for epoch in range(epoch_num):
    # 创建数据加载器 --- 加载成包裹，方便分布式训练
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # shuffle随机打包
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    print(f"epoch: {epoch+1}\n -----------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn, acc_s, loss_s)
    scheduler.step() #调整学习率
    



'''绘制效果曲线图'''
from matplotlib import pyplot as plt
plt.subplot(121)
plt.plot(range(0, epoch_num), acc_s)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.subplot(122)
plt.plot(range(0, epoch_num), loss_s)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



'''-----------------------------------------------------------------------------------------------------------------'''
# 提取模型
model = CNN().to(device)
# 1. 参数读取 (先创建一个model)
model.load_state_dict(torch.load(f'best_acc_{best_acc}.pt'))
# 2. 直接读取
# model = torch.load(f'best_acc_{best_acc}.pt')

# 进行验证
model.eval() # 固定参数，防止后面被修改
test_data = food_dataset('./test_true.txt', data_transformer['valid'])
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
result = []
labels = []

def test_true(dataloader, model):
    with torch.no_grad(): # 一个上下文管理器，关闭梯度计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model.forward(X)
            result.append(pred.argmax(dim=1).item())
            labels.append(y.item())

print(result)
print(labels)



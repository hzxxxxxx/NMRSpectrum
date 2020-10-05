from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch
from noise_reduction.readmat import readmat
from noise_reduction.network import model
import numpy as np
import matplotlib.pyplot as plt
'''
用于读取光谱文件进行训练
主要网络结构为残差网络

各文件使用说明
H_gen: 生成模拟光谱
main: 定义以及训练网络
readmat:  读取数据文件
load: 通过读取'resnet.ckpt'中的参数重构网络
resnet.ckpt: 训练好的网络中的参数

test/test.py: 用于测试单条光谱效果
test/test.m: 用于画出.py输出的图
test/pure.mat: 单条的纯净光谱
test/impure.mat: 单条的不纯净光谱
test/generate.mat: 单条的训练后光谱
'''

# 读取mat文件
path_y1 = "./data/data_impure.mat"
path_y2 = "./data/data_pure.mat"

# 选择cpu或者gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 保存迭代的loss
total_loss = []

# 迭代参数设置
num_epochs = 20
batch_size = 10
learning_rate = 0.001

# 读取数据集并进行处理

init_train, real_train, init_test, real_test = readmat(path_y1, path_y2)

init = torch.from_numpy(init_train).type(torch.FloatTensor)
real = torch.from_numpy(real_train).type(torch.FloatTensor)

init_test = torch.from_numpy(init_test).type(torch.FloatTensor)
real_test = torch.from_numpy(real_test).type(torch.FloatTensor)

dataset = TensorDataset(init, real)
dataset_test = TensorDataset(init_test, real_test)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=dataset_test,
                          batch_size=batch_size,
                          shuffle=True)

# 更新学习率函数
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 设定loss函数
def NMSE(y_pred, y_true):
    a = torch.sqrt(torch.sum(torch.square(y_true - y_pred)))
    b = torch.sqrt(torch.sum(torch.square(y_true)))
    return a / b

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#定义损失函数
criteria = torch.nn.L1Loss()

# 训练网络
total_step = len(train_loader)
curr_lr = learning_rate

# 通过epoch循环得到参数
for epoch in range(num_epochs):
    for i, (spectrums, labels) in enumerate(train_loader):
        spectrums = spectrums.to(device)
        labels = labels.to(device)

        # 输入数据到网络中得到输出
        outputs = model(spectrums)

        #计算损失
        loss = criteria(outputs, labels)

        # 反向传播计算参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 减少学习率
    if (epoch + 1) % 20 == 0:
        curr_lr /= 1.5
        update_lr(optimizer, curr_lr)

plt.plot(total_loss)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 1))
plt.show()

# 对网络进行评价
model.eval()
model_loss = 0
test_len = len(test_loader)
with torch.no_grad():
    for spectrums, labels in test_loader:
        spectrums = spectrums.to(device)
        labels = labels.to(device)
        outputs = model(spectrums)
        loss = criteria(outputs, labels)
        model_loss += loss.item()

avg_loss = model_loss/test_len
print(avg_loss)

# 保存本次网络的结构参数
torch.save(model.state_dict(), 'resnet.ckpt')
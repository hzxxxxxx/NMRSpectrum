from torch.utils.data import DataLoader,TensorDataset
import torch
from peak_find.indeedpack.readmat import readmat
from peak_find.network import model, criteria
import matplotlib.pyplot as plt
from peak_find.indeedpack.early_stopping import EarlyStopping
import numpy as np

# 读取mat文件
path_y1 = "./test/data_pure.mat"
path_y2 = "./test/data_peakarea.mat"

# 选择cpu或者gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 迭代参数设置
num_epochs = 10
batch_size = 1
learning_rate = 0.003
patience = 2
curr_lr = learning_rate

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
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 更新学习率函数
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_area(spe):
    area = torch.gt(spe, 0.005)
    flag = 0
    index = []
    for j in range(len(spe[0, 0])):
        if (area[0, 0, j] == True) & (flag == 0):
            l = j
            flag = 1
        if (area[0, 0, j] == False) & (flag == 1):
            r = j
            flag = 0
            index.append((l, r))
    print(index)

# 通过早停法训练模型
def train_model(model, batch_size, patience, n_epochs, curr_lr):

    # 储存训练损失
    train_losses = []
    # 储存评价损失
    valid_losses = []
    # 追踪每阶段的平均训练损失
    avg_train_losses = []
    # 追踪每阶段的平均评价损失
    avg_valid_losses = []
    # 计算长度
    total_step = len(train_loader)
    # 初始化early_stop对象
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # 储存每次的训练损失
    total_losses = []

    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        for i, (spectrums, labels) in enumerate(train_loader):
            spectrums = spectrums.to(device)
            labels = labels.to(device)
            find_area(spectrums)
    return model, avg_train_losses, avg_valid_losses

if __name__=="__main__":
    # 开始训练
    model, avg_train_losses, avg_valid_losses = train_model(model, batch_size, patience, num_epochs, curr_lr)
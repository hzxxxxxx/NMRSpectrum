from torch.utils.data import DataLoader,TensorDataset
import torch
from noise_reduction.indeedpack.readmat import readmat
from noise_reduction.network import model, criteria
import matplotlib.pyplot as plt
from noise_reduction.indeedpack.early_stopping import EarlyStopping
import numpy as np

'''
用于读取光谱文件进行训练
主要网络结构为残差网络

各文件使用说明
H_gen: 生成模拟光谱
One_of_H_gen: 生成单条模拟光谱 输出为data/testpure.mat以及data/testimpure.mat
main: 定义以及训练网络
readmat:  读取数据文件
network: 储存网络结构 输出为model
load: 通过读取'resnet_l2.ckpt'中的参数重构网络
resnet_l2.ckpt: 训练好的网络中的参数

test/test.py: 用于测试单条光谱效果
test/test.m: 用于画出.py输出的图
test/pure.mat: 单条的纯净光谱
test/impure.mat: 单条的不纯净光谱
test/generate.mat: 单条的训练后光谱
'''

# 读取mat文件
path_y1 = "./data/data_impure_1.mat"
path_y2 = "./data/data_pure_1.mat"

# 选择cpu或者gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 迭代参数设置
num_epochs = 150
batch_size = 10
learning_rate = 0.003
patience = 5
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

            # 输入数据到网络中得到输出
            outputs = model(spectrums)

            # 计算损失
            loss = criteria(outputs, labels)

            # 反向传播计算参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            total_losses.append(loss.item())

            # 输出每次batch的损失
            if (i + 1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if (epoch+1) % 10 == 0:
            # 评价模型
            model.eval()
            for spectrums, labels in test_loader:
                spectrums = spectrums.to(device)
                labels = labels.to(device)
                output = model(spectrums)
                loss = criteria(output, labels)
                valid_losses.append(loss.item())

            # 计算各阶段平均损失
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch+1:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # 为下一个epoch清除缓存
            train_losses = []
            valid_losses = []

            # early_stop需要验证丢失来检查它是否衰减，如果有的话，它将为当前模型设置一个检查点
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    # 减少学习率
        if (epoch + 1) % 20 == 0:
            curr_lr /= 1.5
            update_lr(optimizer, curr_lr)

    # 载入上一次的存档点
    model.load_state_dict(torch.load('noise_reduction/wirehouse/model/resnet_l2.ckpt'))

    # 画出损失函数的图像
    plt.plot(total_losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

    return model, avg_train_losses, avg_valid_losses

if __name__=="__main__":
    # 开始训练
    model, avg_train_losses, avg_valid_losses = train_model(model, batch_size, patience, num_epochs, curr_lr)
from torch.utils.data import DataLoader,TensorDataset
import torch
from peak_find_res.indeedpack.readmat import readmat
from peak_find_res.network import model, criteria

'''
读取已经训练好的网络参数进行测试
'''

# 测试文件路径
path_y1 = "./data/data_impure.mat"
path_y2 = "./data/data_pure.mat"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

# 创建测试集
init_train, real_train, init_test, real_test = readmat(path_y1, path_y2)

init = torch.from_numpy(init_train).type(torch.FloatTensor)
real = torch.from_numpy(real_train).type(torch.FloatTensor)

init_test = torch.from_numpy(init_test).type(torch.FloatTensor)
real_test = torch.from_numpy(real_test).type(torch.FloatTensor)

dataset_test = TensorDataset(init_test, real_test)

test_loader = DataLoader(dataset=dataset_test,
                          batch_size=batch_size,
                          shuffle=True)

model.load_state_dict(torch.load('resnet_l2.ckpt'))

#测试模型部分
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
print(test_len)
print(avg_loss)


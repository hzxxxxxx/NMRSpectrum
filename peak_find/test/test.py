import torch
import scipy.io as sio
import numpy as np
from peak_find.network import model

'''
读取单条光谱并且输出在网络中训练好的输出的mat文件
后续在matlab中画图
'''

# 选择处理器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取测试的mat文件
x = sio.loadmat('pure.mat')
x = x['pure']
x = np.expand_dims(x, 1).astype('float32')
x = torch.from_numpy(x).to(device)

# 读取网络参数
model.load_state_dict(torch.load('../wirehouse/model/resnet.ckpt'))
# 把参数放入网络
X = model(x)
# 降维
X = torch.squeeze(X, axis=1).cpu()
print(X.shape)
X = X.detach().numpy().astype('float64')
# 结果输出为mat文件
sio.savemat('generate.mat',mdict={'generate':X,})
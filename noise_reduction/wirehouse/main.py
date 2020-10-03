from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from readmat import readmat
import os
import torch
from torch import nn
from torch import functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np

path_y1 = "./data/y1.mat"
path_y2 = "./data/y2.mat"

readmat(path_y1, path_y2)#会返回一个longth,并且生成csv的文件列表以及对应的txt文件

class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,  path_file, transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        self.path_file = path_file
        self.transform = transform
        self.size = 0
        self.names_list = []
        self.init_array = []
        self.label_array = []

        if not os.path.isfile(self.path_file):
            print(self.path_file + 'does not exist!')
        file = open(self.path_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        init_path = self.names_list[idx].split(' ')[0]
        if not os.path.isfile(init_path):
            print(init_path + 'does not exist!')
            return None
        label_path = self.names_list[idx].split(' ')[1].strip()
        if not os.path.isfile(label_path):
            print(label_path + 'does not exist!')
            return None
        self.init_array.append(self.default_loader(init_path))
        self.label_array.append(self.default_loader(label_path))
        sample = {'init': self.init_array, 'label': self.label_array}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def default_loader(self, spe_path):
        file = np.array(pd.read_csv(spe_path)).squeeze()
        return file

class ToTensor(object):
    def __call__(self, sample):
        init = np.array(sample['init']).astype('float32')
        label = np.array(sample['label']).astype('float32')
        return {'init': torch.from_numpy(init),
                'label': torch.from_numpy(label)}

transformed_trainset = MyDataset(path_file="./data/pathfile.txt",
                          transform=transforms.Compose(
                              [
                               ToTensor()]
                          ))

trainset_dataloader = DataLoader(dataset=transformed_trainset,
                                 batch_size=100,
                                 shuffle=True,
                                 num_workers=10)

batch_size = 100

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(65536, 500)
        self.l2 = nn.Linear(500, 100)

    def forward(self, x):
        x = x.view(-1,65536) # Flattern
        x = F.relu(self.l1(x))

        return self.l2(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr= 0.01 , momentum= 0.5)

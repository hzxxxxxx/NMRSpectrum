from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 网络部分的开始

# 卷积核为16的卷积层
def conv16(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15,
                     stride=stride, padding=7, bias=False)

# 卷积核为1的卷积层
def conv1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

# 残差快的网络结构
class ResidualBlock(nn.Module):
    #定义各层网络结构
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv16(out_channels, out_channels, stride)
        self.conv2 = conv16(out_channels, out_channels)
        self.conv3 = conv1(in_channels, out_channels)
        self.conv4 = conv1(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

    #向前传播结构
    def forward(self, x):
        residual = x
        out = self.conv3(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.conv4(out)
        out = self.bn(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 构建残差网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        #定义网络中的各层
        super(ResNet, self).__init__()
        self.channels = 256
        self.conv = conv16(1, self.channels)
        self.conv2 = conv16(self.channels, 1)
        self.bn = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self.make_layer(block, self.channels, layers)
        self.layer2 = self.make_layer(block, self.channels, layers)
        self.layer3 = self.make_layer(block, self.channels, layers)
        self.layer4 = self.make_layer(block, self.channels, layers)
        self.layer5 = self.make_layer(block, self.channels, layers)

    # 定义残差层的结构
    def make_layer(self, block, out_channels, blocks, stride=1):
        # downsample用于改变通道数
        downsample = None
        if (stride != 1) or (self.channels != out_channels):
            downsample = nn.Sequential(
                conv16(self.channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels)
                )
        layers = []
        layers.append(block(self.channels, out_channels, stride, downsample))
        self.channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    # 向前传播的网络结构
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.bn2(out)
        out = self.conv2(out)


        return out

# 实例化网络
model = ResNet(ResidualBlock, 4).to(device)

## 网络部分的结束
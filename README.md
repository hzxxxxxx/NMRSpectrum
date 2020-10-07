# MriSpectrum
一款关于对MRI光谱进行处理的程序

## NoiseReduction

用于读取光谱文件进行光谱的去噪训练
主要网络结构为残差网络

#### 各文件使用说明

H_gen: 生成模拟光谱
One_of_H_gen: 生成单条模拟光谱 输出为data/testpure.mat以及data/testimpure.mat
main: 定义以及训练网络
readmat:  读取数据文件
network: 储存网络结构 输出为model
load: 通过读取'resnet.ckpt'中的参数重构网络
resnet.ckpt: 训练好的网络中的参数

test/test.py: 用于测试单条光谱效果
test/test.m: 用于画出.py输出的图
test/pure.mat: 单条的纯净光谱
test/impure.mat: 单条的不纯净光谱
test/generate.mat: 单条的训练后光谱
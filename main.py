import torch
from torch import nn
from torch.nn import init
from scipy import signal
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
current_module = sys.modules[__name__]
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

### 网络类定义
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class VarLayer(nn.Module):
    '''
    方差层
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    计算数据在dim 的std
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    取方差的对数
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    取dim维的平均值
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    取dim维的最大值
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FBCNet(nn.Module):
    '''
        输入格式： batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.Softmax(dim = 1))

    def __init__(self, nChan, nTime, nClass = 2, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 5, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

    
def initialize(net):
    '''
    网络实例参数（权重、bias）初始化
    :param net: 网络的实例
    :return:
    '''
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean = 0, std = 0.1)
        if 'bias' in name:
            init.constant_(param, val = 0)
    print('initialize done')

def train(net, train_X, train_y, test_X, test_y, optimizer, device, num_epochs, the_round):
    '''
    网络参数训练函数，返回各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率
    :param net: 网络实例
    :param train_iter: 训练样本数据集
    :param test_iter: 测试样本数据集
    :param batch_size: 批次大小
    :param optimizer: 梯度下降优化方法
    :param device: 在cpu还是gpu上训练
    :param num_epochs: 训练多少次
    :return:
    '''
    # 将网络实例加载到指定设备上
    
    net = net.to(device)
    print("training on ", device)

    #loss = torch.nn.CrossEntropyLoss()  # 定义损失函数
    loss = torch.nn.MSELoss()  # 定义损失函数
    #loss = torch.nn.NLLLoss()  # 定义损失函数
    #loss = torch.nn.BCELoss()  # 定义损失函数

    # 各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率的list
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    print('epochs = ' + str(num_epochs))
    best_test_acc = 0
    best_test_acc_epoch = 0
    now_epoch = 0
    # 开始训练
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()  # 参数初始化
        #train_X, train_y = prepare_a_new_epoch(train_X, train_y)
        now_epoch = now_epoch + 1
        for X, y in zip(train_X, train_y): 
            
            #print('done')
            net = net.to(device)
            X = X.to(device)
            y = y.to(device)
            if y == 0:
                y_ = torch.from_numpy(np.array([1,0])).to(torch.float32).to(device)
            else :
                y_ = torch.from_numpy(np.array([0,1])).to(torch.float32).to(device)
            y_hat = net(X)  # 在神经网络中得到输出值
            y_ = torch.reshape(y_,(2,1))
            y_hat = torch.reshape(y_hat, (2,1))
            
            l = loss(y_hat, y_)  # 得到损失值
            
            optimizer.zero_grad()  # 置零梯度
            l.backward()  # 得到反传梯度值
            optimizer.module.step()  # 用随机梯度下降法训练参数
            train_l_sum += l.cpu().item()  # 得到损失之和
            train_acc_sum += (y_hat.argmax(dim = 0) == y).sum().cpu().item()  # 得到模型在训练数据中的准确度
            n += 1
            batch_count += 1 
        test_acc = evaluate_accuracy(test_X, test_y, net, device)  # 得到当前网络参数在测试数据集上的结果
        #print(test_acc)

        # 各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率的list
        loss_list.extend([train_l_sum / n])
        train_acc_list.extend([train_acc_sum / n])
        test_acc_list.extend([test_acc])
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
        if test_acc > best_test_acc :
            if the_round == 0:
                torch.save(net.to('cpu'), 'net_1.pth')
            elif the_round == 1:
                torch.save(net.to('cpu'), 'net_2.pth')
            elif the_round == 2:
                torch.save(net.to('cpu'), 'net_3.pth')
            best_test_acc = test_acc
            best_test_acc_epoch = now_epoch

    return [loss_list, train_acc_list, test_acc_list, best_test_acc, best_test_acc_epoch, net]



def evaluate_accuracy(data_X, data_y, net, device=None):
    acc_sum, n = 0.0, 0
    net = net.to(device)

    for X, y in zip(data_X, data_y): 
    #print('done')
        X = X.to(device)
        y = y.to(device)
        
        if y == 0:
            y_ = torch.from_numpy(np.array([1,0])).to(torch.float32).to(device)
        else :
            y_ = torch.from_numpy(np.array([0,1])).to(torch.float32).to(device)
        y_hat = net(X)  # 在神经网络中得到输出值
        y_ = torch.reshape(y_,(2,1))
        y_hat = torch.reshape(y_hat, (2,1))
        
        acc_sum += (y_hat.argmax(dim = 0) == y).sum().cpu().item()  # 得到模型在训练数据中的准确度
        n += 1

    return acc_sum / n


def draw(loss_list, train_acc_list, test_acc_list):
    '''
    画出损失、训练样本集正确率、测试样本集正确率随epcoh的变化
    :param loss_list:
    :param train_acc_list:
    :param test_acc_list:
    :return:
    '''
    fig1 = plt.figure(1)
    # plot中参数的含义分别是横轴值，纵轴值，颜色，透明度和标签
    x = [i + 1 for i in range(10)]
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(np.array(train_acc_list), 'o-', color = 'blue', alpha = 0.8, label = 'train_acc')

    plt.plot(np.array(test_acc_list), 'o-', color = 'red', alpha = 0.8, label = 'test_acc')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, 'o-', color = 'blue', alpha = 0.8, label = 'loss')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    fig1.savefig("fig.png")

    plt.show()


def bandpass(data):
    '''
    实现带通滤波：
    给的数据采样率是250hz，共750个采样点。频率在8hz到32hz之间，得到的滤波数据是8-12hz，12-16hz...28-32hz共6个带通波形
    输入：[num_of_trials, 13, 750]
    输出：[num_of_trials, 1， 6, 13, 750]
    '''
    f_unit = 2 * 4 / 250   #一个 unit 是 4 hz
    b, a = signal.butter(6, [f_unit * 2, f_unit * 3], 'bandpass')  #这里的 6 是指带通滤波器为 6 阶滤波器
    
    data_bandpass = []   # 一个 data_bandpass 的形状为 [6, num_of_trials, 13, 750]
    for k in range(6):
        data_single_band = []  # 一个 data_single_band 的形状为 [num_of_trials, 13, 750]
        for i in range(data.shape[0]):
            trial = []  # 一个 filtedData 的形状是 750，一个 trail 的形状是 [13*750]
            for j in range(data.shape[1]):
                b, a = signal.butter(6, [f_unit * (2 + k - 0.5), f_unit * (3 + k + 0.5)], 'bandpass')
                filtedData = signal.filtfilt(b, a, data[i][j][:]) #data为要过滤的信号
                trial.append(filtedData)
            data_single_band.append(trial)
        data_bandpass.append(data_single_band)
    
    data_bandpass = np.array(data_bandpass)
    data_bandpass = torch.from_numpy(data_bandpass).to(torch.float32)
    data_bandpass = data_bandpass.permute(1,0,2,3)
    data_bandpass = data_bandpass.view(data.shape[0], 1, 6, 13, 750)
    return data_bandpass



time_begin = time.time()

device_ids = [0, 1, 2, 3]
#device_ids = [1, 0, 2, 3]
#device_ids = [2, 1, 0, 3]
#device_ids = [3, 1, 2, 1]

##################################### 数据的准备  ###################################
## 训练数据的准备
print('begin preparing the data')
S1 = np.load('data/train/S4.npz')
S2 = np.load('data/train/S2.npz')
S3 = np.load('data/train/S3.npz')
S_train_X = np.concatenate((S1['X'], S2['X'], S3['X']))
S_train_y = np.concatenate((S1['y'], S2['y'], S3['y']))

train_X = bandpass(S_train_X)
train_y = torch.from_numpy(S_train_y).to(torch.float32)

print(train_X.shape)

## 验证数据的准备
S_valid = np.load('data/train/S1.npz')
S_valid_X = S_valid['X']
S_valid_y = S_valid['y']

valid_X = bandpass(S_valid_X)
valid_y = torch.from_numpy(S_valid_y).to(torch.float32)

print("data prepared done")

#######################################  网络初始化  ###########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = FBCNet(13, 750, 2, 6, m = 48)
net = torch.nn.DataParallel(net, device_ids=device_ids)
initialize(net)  # 权重、bias初始化

print(device)
acc = evaluate_accuracy(train_X, train_y, net, device)
print('the initial accuracy is ' + str(acc))


######################################  超参数的设置  ##########################################
lr, num_epochs = 0.000001, 100  # 学习率和训练次数的指定  0.00001
optimizer = torch.optim.Adam(net.parameters(), lr = lr)  # 梯度下降的优化方法的选择
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

######################################  开始训练  ############################################

the_round = 0
[loss_list0, train_acc_list0, test_acc_list0, best_test_acc, best_test_acc_epoch, net] = train(net, train_X, train_y, valid_X, valid_y, optimizer, device, num_epochs, the_round)  # 训练+得到各次训练的损失值，准确度
loss_list = loss_list0
train_acc_list = train_acc_list0
test_acc_list = test_acc_list0
print("in the " + str(best_test_acc_epoch) + "epoch, the net get the best test accuracy: " + str(best_test_acc))
print("and it is saved as 'net0.pth' ")

lr = lr / 10
the_round = 1
[loss_list1, train_acc_list1, test_acc_list1, best_test_acc, best_test_acc_epoch, net] = train(net, train_X, train_y, valid_X, valid_y, optimizer, device, num_epochs, the_round)  # 训练+得到各次训练的损失值，准确度
loss_list = loss_list + loss_list1
train_acc_list = train_acc_list + train_acc_list1
test_acc_list = test_acc_list + test_acc_list1
print("in the " + str(best_test_acc_epoch) + "epoch, the net get the best test accuracy: " + str(best_test_acc))
print("and it is saved as 'net1.pth' ")

###################################### 结果可视化  ###########################################
print("the total time is " + str(time.time() - time_begin) + " sec ")


draw(loss_list, train_acc_list, test_acc_list)
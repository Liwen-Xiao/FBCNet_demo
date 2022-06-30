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
import d2lzh_pytorch as d2l
from networks import *
import pandas as pd

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


def bandpass(data):
    '''
    实现带通滤波：
    给的数据采样率是250hz，共750个采样点。频率在8hz到32hz之间，得到的滤波数据是8-12hz，12-16hz...28-32hz共6个带通波形
    输入：[num_of_trials, 13, 750]
    输出：[num_of_trials, 6, 13, 750]
            not [num_of_trials, 1, 13, 750, 6]
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

# print('begin preparing the valid data')


# ## 验证集准备 
# S_valid = np.load('data/train/S1.npz')
# S_valid_X = S_valid['X']
# S_valid_y = S_valid['y']
# valid_X = bandpass(S_valid_X)
# valid_y = torch.from_numpy(S_valid_y).to(torch.float32)





# print(valid_X.shape)

# print('the test data prepared down')


net = torch.load('net11_2.pth')

device_ids = [0, 1, 2, 3]
device = 'cuda:2'
# valid_acc = evaluate_accuracy(valid_X, valid_y, net, device)
# print('the accuracy on the valid data is ' + str(valid_acc))


print('begin preparing the test data')
#S5 = np.load('data/test/S5.npz')
#S6 = np.load('data/test/S6.npz')
#S7 = np.load('data/test/S7.npz')
S8 = np.load('data/test/S8.npz')
#print(S5['X'].shape)
#S_test_X = np.concatenate((S5['X'], S6['X'], S7['X'], S8['X']))
S_test_X = S8['X']

test_X = bandpass(S_test_X)
y_list = []

net = net.to(device)
for X in test_X:
    X = X.to(device)
    y = net(X)
    y_list.append(y.cpu().detach().numpy())

test_y = []
for y in y_list:
    if y[0][0] > y[0][1] :
        test_y.append(0)
    else:
        test_y.append(1)
print(test_y)
test_y = np.array(test_y).reshape(200,1)
print(test_y)
test_y = torch.from_numpy(np.array(test_y)).to(torch.float32)
test_acc = evaluate_accuracy(test_X, test_y, net, device)
print('the accuracy on the test data is ' + str(test_acc))



# 下面这行代码运行报错
# list.to_csv('e:/testcsv.csv',encoding='utf-8')
test=pd.DataFrame(data=test_y)#数据有三列，列名分别为one,two,three
print(test)
test.to_csv('S8.csv',encoding='utf-8')















# import time
# tic = time.time()





# import torch
# from torch import nn
# from torch.nn import init
# import numpy as np
# import sys
# import time
# import matplotlib.pyplot as plt
# import random
# import torch.nn.functional as F
# import d2lzh_pytorch as d2l
# from scipy import signal

# A = torch.tensor([[1,2,3],
#                   [4,5,6],
#                   [7,8,9]])


# S1 = np.load('data/train/S1.npz')
# S2 = np.load('data/train/S2.npz')
# S3 = np.load('data/train/S3.npz')
# S_train_X = np.concatenate((S1['X'], S2['X'], S3['X']))
# S_train_y = np.concatenate((S1['y'], S2['y'], S3['y']))
# x = S_train_X[150][0][:]
# print(x.shape)

# # for a in S_train_X[:][:]:
# #     print(a.shape)

# a=[1,2,3]
# print(a)
# print(type(a))
# b=[4,5,6]
# A = [a]
# print(A)
# print(type(A))
# A.append(b)
# print(A)
# A = np.array(A)
# print(A.shape)
# print(type(A))


# print()
# C = [[1,2,3],
#      [4,5,6]]

# C = np.array(C)
# print(C)
# print(C.shape)
# print(type(C))

# A = time.time()
# while time.time() - A < 10:
#      print("A")

# #print("B")

# print("C")



# f_unit = 2 * 4 / 250   #一个 unit 是 4 hz

# b, a = signal.butter(6, [f_unit * 2, f_unit * 3], 'bandpass')

# filtedData = signal.filtfilt(b, a, x)#data为要过滤的信号
# #print(filtedData)

# t = np.linspace( 0, 1 , 750)

# fig1 = plt.figure()
# plt.plot(filtedData)

# fig1.savefig("f1.png")

# plt.show()

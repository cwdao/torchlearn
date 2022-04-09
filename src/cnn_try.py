

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary

x = torch.rand(2,2)
print(x)
print(torch.cuda.is_available())

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
])

class EMGDataset(Dataset):
 
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transforms = transform
 
    def __getitem__(self, index):
        emgData = self.data[index,...]
        emgData = np.squeeze(emgData)#似乎不应该压缩了
        # emgData =emgData.unsqueeze(0)
        # emglabel = self.label[index,0]

        emglabel = self.label[index]
        # while emglabel>=100:
        #     emglabel = emglabel-100

        emglabel = emglabel.astype(np.int16)
        # emglabel = emglabel/1.0
        emgData = self.transforms(emgData)
        # emglabel = self.transforms(emglabel)#.long()
        # self.data= emgData
        # self.label = emglabel
        
        return emgData,emglabel
 
    def __len__(self):
        return len(self.label)
 
 
# if __name__ == '__main__':
traindata = np.load('../data/trainX.npy')
trainlabel = np.load('../data/trainY.npy')
print(trainlabel[:,0])

trainlabel = trainlabel[:,0]
print(type(trainlabel))
train_set = EMGDataset(traindata, trainlabel)

train_loader1 = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)#, pin_memory=True,
                                            #num_workers=3)

# for x, y in train_loader:
#     print(x, y)
sample = next(iter(train_set))

batch = next(iter(train_loader1))
images, labels = batch
print(images.shape)
traindata = np.load('../data/trainX.npy')
print(traindata.shape)
print(type(traindata))
print(traindata[1,:,:,:].shape)

testX = iter(traindata)
testXX = next(testX)
print(testXX.shape)
#coding=utf8
# collect some functions and classes which may useful
#####################################################

import numpy as np
import torch 
# from torch.utils.data import Dataset

#####################################################
# compute how many samples are predicted correctly
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#####################################################
# 计算数据集的网络预测输出并构建新的特征向量四维 array
# 注意迭代器只能建立一次，然后以此前进，所以这里单列一行，避免重复建设
# 达成类似目标的做法有很多种，例如 append() 等，这里先完成功能，优化待后来同学了
def emgdata_to_net_preds(data_set,net_vector):
    batchl = iter(data_set)
    emg_vec = [torch.tensor([],requires_grad=False) \
        for i in range(len(data_set.label))]
    for idx, _ in enumerate(emg_vec):
        sample_data,sample_label = next(batchl)
        sample_data = sample_data.to(torch.float32).unsqueeze(0)
        sample_label = torch.as_tensor(sample_label).long()
        emg_vec[idx] = net_vector(sample_data).detach().numpy()
    emg_vec_np = np.array(emg_vec)
    emg_vec_np = emg_vec_np[:,np.newaxis,:,:]
    return emg_vec_np
#####################################################
# 计算对某一数据集的网络预测值，并返回一个列表
def emgdata_to_net_preds_sigmoid(data_set,net_vector):
    batchl = iter(data_set)
    emg_vec = [torch.tensor([],requires_grad=False) \
        for i in range(len(data_set.label))]
    for idx, _ in enumerate(emg_vec):
        sample_data,sample_label = next(batchl)
        sample_data = sample_data.to(torch.float32).unsqueeze(0)
        sample_label = torch.as_tensor(sample_label).long()
        emg_vec[idx] = net_vector(sample_data).sigmoid().detach().numpy()
    emg_vec_np = np.array(emg_vec)
    emg_vec_np = emg_vec_np[:,np.newaxis,:,:]
    return emg_vec_np
#####################################################
# EMG data class,first valiate on Ninapro
# class EMGDataset(Dataset):
 
#     def __init__(self, data, label):
#         self.data = data
#         self.label = label
#         self.transforms = transforms.ToTensor()
 
#     def __getitem__(self, index):
#         emgData = self.data[index,:,:,:]
#         # why here need condense?
#         emgData = np.squeeze(emgData)
#         emglabel = self.label[index]
#         emglabel = emglabel.astype(np.int16)
#         emgData = self.transforms(emgData)      
#         return emgData,emglabel
 
#     def __len__(self):
#         return len(self.label)
#####################################################
# 利用约登指数 求ROC最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold): 
    y = TPR - FPR
    Youden_index = np.argmax(y) 
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold,point
#####################################################
# if __name__ == '__main__':
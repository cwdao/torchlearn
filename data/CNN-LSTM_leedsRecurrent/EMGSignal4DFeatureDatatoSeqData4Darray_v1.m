function [X_Train,Y_Train,...
    X_Test,Y_Test] = ...
    EMGSignal4DFeatureDatatoSeqData4Darray_v1(...
    TimeStep,...
    ParentDatasetPath)
% EMGSignal4DFeatureDatatoSeqData4Darray, 实现原始信号数据集转神经网络训练用数据集功能
% 版本说明：按 Ninapro 官方推荐方式分割数据集，取消随机模式
% 数据集：Ninapro DB1;
% 功能：把由CNN提取的特征重新组织为LSTM需要的序列数据集
% 举例：由CNN提取的特征为1*52向量，本代码将10（时间步，这个自己设定）个组成一个样本，第十个的标签作为一个样本的标签...、
% 当出现10个中同时有两种标签时，如1-7为动作1，8-10为动作2，本函数会复值第七个3次凑成1-7，7，7，7样本，同时第七个标签作为
% 本样本标签，然后下一个样本从刚才的第八个开始。
% 初版可用日期：2021.7.25
% 变量说明：
% 调试说明：
% 1.将待处理数据集测试集和训练集目录赋给 ParentDatasetPath ，
%   需要自行调整细节保证文件名可对应;
% 2.设定时间步，也就是一个样本的长度。
% 3.最终返回所需测试集和训练集
% 作者：王骋
% 程序版本：v1.0（在原 SignaltoDataset 上继续迭代）
% 函数版本：v1.0
% 时间版本：210725(用于不同程序间同步)
%% 

% 本段预设了大部分需要调整的变量
% 有如下预设变量：

%% 
% clc;clear;
% load('CNNfet_XTrainrelu_620210723T210236.mat');
% load('CNNfet_XTestrelu_620210723T210236.mat');
% load('CNNfet_YTestrelu_620210723T210236.mat');
% load('CNNfet_YTrainrelu_620210723T210236.mat');
load(ParentDatasetPath);
fex = FeatureofDataset_XTrain;
fey = Y_TrainC;
fex2 = FeatureofDataset_XTest;
fey2 = Y_TestC;
m1 = size(fex);
length1 = m1(4);
m2 = size(fex2);
length2 = m2(4);
%% 
% 温馨提示：本代码首先只为实现功能，若有后来者有更好实现方式，大可修改之。
% 原始功能只需要提取时间步数的个数特征，取最后一个时间步的标签做本样本的标签就行。
% 理论上讲不需要再添加其他功能了，但是截至目前（210724），数据集在提取的时候就没有考虑到时序连续性，如果直接一路排下去，
% 必然会造成处于动作更换处的时间步们没有合适的标签，因为在时间上他们并不是连续的，因此需要在交界处判断不一致位置，然后
% 复制分界前的特征和数据至满时间步，接着从新开始下一个时间步的存储。
%% 
% ts = 10;
% 时间步
ts = TimeStep;
% 每个样本内的序列位置
c1 = 1;
% 整个新序列的当前位置
j = 1;
% 两个循环功能及内容一致，只是分别处理测试集和训练集，故注释不再多写一份
% 基本思想是一个个从原数据集提取特征向量，然后凑足10（时间步）个做一个样本，放到新序列数组去。
% adr是待处理的数据集实时坐标，fe21，lb21，fe22,lb22分别是特征暂存变量和标签暂存变量，
% Seq_Slip是用于放置一个样本大小的数组，存满之后放入序列数据集数组 Seq_X,Seq_Xce,二者只是存储格式不同，前者是四维数组，
% 后者是元胞数组。我最后使用的是元胞数组，因此所有 Xce 被返回变量和存储。
for adr = 1:length1
%     判断是否是样本中第一个数据
    if(c1 == 1)
%         若是，直接存入
        fe21 = permute(fex(1,1,:,adr),[2 3 1 4]);
        lb21 = fey(adr,1);
        SeqSlip(:,c1) = fe21;
        c1 = c1+1;
    else
%         否则先用22系列变量暂存本次的，准备与上一次21系列变量比对标签是否一致
        fe22 = permute(fex(1,1,:,adr),[2 3 1 4]);
        lb22 = fey(adr,1); 
        if(lb21 == lb22)
%             如果标签一致，放入样本数组，同时本次的再赋给21系列变量，准备和下下次比较
            fe21 = fe22;
            SeqSlip(:,c1) = fe21;
            lb21 = lb22;
            if(c1 == ts)
%                 凑足时间步数后，样本就可以放入样本序列数组去了
                Seq_X(:,:,1,j) = SeqSlip();
                Seq_Y(j,1) = lb21;
                Seq_Xce{j,1} = SeqSlip();
%                 样本序列坐标移动一位，样本数组坐标归1
                j = j+1;
                c1 = 1;
            else
%                 还没到时间步，那就继续存，移动样本数组的坐标
                c1 = c1+1;
            end
        else
%             标签不一致，需要暂停，把存在21系列的前一个标签复制直到凑够时间步
            while (c1<=ts)
                SeqSlip(:,c1) = fe21;
                c1 =c1+1;
            end
%             凑够了本样本再度放入样本序列中去
            Seq_X(:,:,1,j) = SeqSlip();
            Seq_Xce{j,1} = SeqSlip();
            Seq_Y(j,1) = lb21;
%             样本序列坐标移位
            j = j+1;
%             暂停位置的内容，本次的也要给下一个样本了，不然就丢失了
            SeqSlip(:,1) = fe22;
            lb21 = lb22;
%             第一个是刚才暂停的数据，下一回自然要从样本数组内的第二个开始
            c1 = 2;
        end
    end
end
size(Seq_X)
size(Seq_Xce)
size(Seq_Y)
% 赋值，清理变量
X_Train = Seq_Xce;
Y_Train = Seq_Y;
clear fe21 fe22 lb21 lb22
% make test
% 时间步
ts = TimeStep;
% 每个样本内的序列位置
c1 = 1;
% 整个新序列的当前位置
j = 1;
for adr = 1:length2
    if(c1 == 1)
        fe21 = permute(fex2(1,1,:,adr),[2 3 1 4]);
        lb21 = fey2(adr,1);
        SeqSlip(:,c1) = fe21;
        c1 = c1+1;
    else
        fe22 = permute(fex2(1,1,:,adr),[2 3 1 4]);
        lb22 = fey2(adr,1); 
        if(lb21 == lb22)
            fe21 = fe22;
            SeqSlip(:,c1) = fe21;
            lb21 = lb22;
            if(c1 == ts)
                Seq_X2(:,:,1,j) = SeqSlip();
                Seq_Y2(j,1) = lb21;
                Seq_Xce2{j,1} = SeqSlip();
                j = j+1;
                c1 = 1;
            else
                c1 = c1+1;
            end
        else
%             SeqSlip(c1,:) = fe21;
            while (c1<=ts)
                SeqSlip(:,c1) = fe21;
                c1 =c1+1;
            end
            Seq_X2(:,:,1,j) = SeqSlip();
            Seq_Xce2{j,1} = SeqSlip();
            Seq_Y2(j,1) = lb21;
            j = j+1;
%             本次的也要给下一个样本了，不然就丢失了
            SeqSlip(:,1) = fe22;
            lb21 = lb22;
            c1 = 2;
        end
    end
end
size(Seq_X2)
size(Seq_Xce2)
size(Seq_Y2)
X_Test = Seq_Xce2;
Y_Test = Seq_Y2;
%% 
% save
ts = string(ts);
DatemarktoNN = datestr(now,30);
FetName =...
    strcat('LSTMfex_Seq_ts_',ts,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'X_Train','X_Test','Y_Train','Y_Test');
end

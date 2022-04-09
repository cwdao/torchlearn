% 完整 LSTM 流程，于210726验证有效。
%% Load Dataset 
%本文档前两节来自于 CNN_EMG_v1，方便加载旧数据，执行新的数据提取，从第三节 make dataset 开始是正式文档部分
load('1Xtrain-20210721T193346.mat');
load('1Xtest-20210721T193346.mat');
load('1Ytrain-20210721T193346.mat');
load('1Ytest-20210721T193346.mat');
X_TrainC = X_Train;
Y_TrainC = Y_Train;
X_TestC = X_Test;
Y_TestC = Y_Test;
Y_TrainC = categorical(Y_TrainC);
Y_TestC = categorical(Y_TestC);
clear X_Train X_Test Y_Train Y_Test
cn3 = load('./20210721T204826/net_checkpoint__67200__2021_07_21__21_28_27.mat','net');
PredictLabels = classify(cn3.net,X_TestC);
TestLabels = Y_TestC;
Accuracy = sum(PredictLabels == TestLabels)/numel(PredictLabels);
disp(['accuracy:',num2str(Accuracy)]); 
% 作混淆矩阵图
figure
confusionchart(TestLabels,PredictLabels,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for CNN');
%% Extract Feature
% 从神经网络的某一层提取数据并与标签组成新数据集，提取后的 xtain、xtest与之前的y依然一一对应，因此存在一起 
% exTrain
exLayer = 'relu_6';
FeatureofDataset_XTrain = activations(cn3.net,X_TrainC,exLayer);
% exTest
FeatureofDataset_XTest = activations(cn3.net,X_TestC,exLayer);
size(FeatureofDataset_XTrain)
DatemarktoNN = datestr(now,30);
FetName =...
    strcat('CNNfet_',exLayer,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'FeatureofDataset_XTrain','Y_TrainC','FeatureofDataset_XTest','Y_TestC');
%% 
% 本节清除无用变量
clc
clear
%% make Dataset
% 加载保存文件，获得用于构建数据集的原始特征序列
% 加载已提取特征的数据集
DatasetParentPath = 'CNNfet_relu_6_20210726T171831.mat';
% 制作并加载新数据集，第一项输入参数为样本时间步
 [X_TrainR,Y_TrainR,...
    X_TestR,Y_TestR] = ...
    EMGSignal4DFeatureDatatoSeqData4Darray_v1(...
    10,...
    DatasetParentPath);
% 确保为分类数组
Y_TestR = categorical(Y_TestR);
Y_TrainR = categorical(Y_TrainR);
% 清除无用变量
clear DatasetParentPath
%% Pretarin step 1
% 搭建LSTM网络，序列特征维52，可将此节先运行一遍，变量加载入工作区后使用 
% DeepNetwork Designer 更方便快捷，同时减少出错，分析完导出代码复制至此即可
layersLSTM = [
    sequenceInputLayer(52,"Name","sequence")
    lstmLayer(128,"Name","lstm_1","OutputMode","last")
    dropoutLayer(0.1,"Name","dropout_1")
    fullyConnectedLayer(52,"Name","fc")
    dropoutLayer(0.1,"Name","dropout")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
%% Pretrain step 2
% 设置训练选项，建立检查点方便续训练，同时设定验证集为 X_Test
CheckpointPath = pwd;
DatemarktoNN = datestr(now,30);
NNCheckpointPath =...
    strcat(CheckpointPath,'\',DatemarktoNN);
mkdir(NNCheckpointPath);
CheckpointPath = string(NNCheckpointPath);
% 神经网络训练设置，梯度限制 2，学习率初始为0.001，每5 epoch 减半
LSTMoptions = trainingOptions('adam',...
    'Plots', 'training-progress', ...
    'Maxepochs',20,...
    'GradientThreshold',2, ...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.5, ...
    'ValidationData',{X_TestR,Y_TestR},...
    'MaxEpochs', 200,...
    'MiniBatchSize',10, ...
    'CheckpointPath',CheckpointPath);
%% Train step 1
% 第一次训练,后续会有调整，且会有暂存点保存，效果等同save，视情况启用保存功能，不过
% 启用检查点后只需要记住时间就行，这里保存更多是为了方便查找
LSTMnetwork = trainNetwork(X_TrainR,Y_TrainR,layersLSTM,LSTMoptions);
save('LSTM_trydata.mat','LSTMnetwork');
%% Train step 2
% 再训练，这时要加载保存点数据，从自选的 mat 中提取 net 变量作为续练网络，
% 顺便复制一份训练设置在这里，修改更方便，如无意外，在本节可持续修改直到满意结果。
ResumePath = strcat(CheckpointPath,'/net_checkpoint__11490__2021_07_26__17_47_11.mat');
LSTMnetworkOld = load(ResumePath,'net');

%% Train step 2.1
% 若重启了软件，变量被清理，此节就是必要的，否则可跳过 
load('LSTMfex_Seq_ts_10_20210726T174216.mat');
X_TrainR = X_Train;
Y_TrainR = Y_Train;
X_TestR = X_Test;
Y_TestR = Y_Test;
% 确保为分类数组
Y_TestR = categorical(Y_TestR);
Y_TrainR = categorical(Y_TrainR);
% 清除无用变量
clear X_Train Y_Train X_Test Y_Test
%% Train step 2.2
% 继续训练,这里就默认保存了,检查点文件夹也带时间戳方便归档
CheckpointPath = pwd;
DatemarktoNN = datestr(now,30);
NNCheckpointPath =...
    strcat(CheckpointPath,'\',DatemarktoNN);
mkdir(NNCheckpointPath);
CheckpointPath = string(NNCheckpointPath);
% 神经网络训练设置，梯度限制 2，学习率初始为0.001，每5 epoch 减半
LSTMoptions = trainingOptions('adam',...
    'Plots', 'training-progress', ...
    'Maxepochs',20,...
    'GradientThreshold',2, ...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.5, ...
    'ValidationData',{X_TestR,Y_TestR},...
    'MaxEpochs', 200,...
    'MiniBatchSize',10, ...
    'CheckpointPath',CheckpointPath);
% 默认保存神经网络时带时间戳，方便归档
LSTMnetwork = trainNetwork(X_TrainR,Y_TrainR,LSTMnetworkOld.net.Layers,LSTMoptions);
NNName =...
    strcat('LSTMtry',DatemarktoNN,'.mat' );
NNName = string(NNName);
save(NNName,'LSTMetwork');
clear DatemarktoNN NNCheckpointPath
%% Predict
% 本节预测汇总测试集所有数据并输出总正确率
PredictLabels = classify(LSTMnetwork,X_TestR);
TestLabels = Y_TestR;
% 计算正确率
Accuracy = sum(PredictLabels == TestLabels)/numel(PredictLabels);
disp(['accuracy:',num2str(Accuracy)]); 
% 作混淆矩阵图
figure
confusionchart(TestLabels,PredictLabels,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

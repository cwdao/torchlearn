clear;
clc;
% 建议使用 Run Section（光标选中当前节并Ctrl+Enter）按节运行
%% Dataset Input Pretreatment 
DatasetParentPath = 'C:\Users\cwdbo\Downloads\ninapro\db1\';
% 更细致的文件名调整需要在EMGSignalto4DArray内部完成，若需要，也可注释掉相关部分自行输入完整路径
[X_TrainC,Y_TrainC,...
    X_TestC,Y_TestC,...
    Porprotion] = ...
    EMGSignalto4DArray_v1d1(...
    DatasetParentPath,...
    1,...
    3,...
    23,...
    100,...
    50,...
    400);
% 分类任务需要Y数据是分类数组（categorical array）
Y_TrainC = categorical(Y_TrainC);
Y_TestC = categorical(Y_TestC);
%% Pretrain step 1 
% 首先需要建立网络，可将此节先运行一遍，变量加载入工作区后使用 
% DeepNetwork Designer 更方便快捷，同时减少出错
layers = [
    imageInputLayer([40 10 1],"Name","imageinput")
    convolution2dLayer([3 3],30,"Name","conv_1")
    reluLayer("Name","relu_1")
    crossChannelNormalizationLayer(5,"Name","crossnorm_1")
    maxPooling2dLayer([3 3],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_2")
    reluLayer("Name","relu_2")
    crossChannelNormalizationLayer(5,"Name","crossnorm_2")
    maxPooling2dLayer([3 3],"Name","maxpool_2","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_3")
    reluLayer("Name","relu_3")
    crossChannelNormalizationLayer(5,"Name","crossnorm_4")
    maxPooling2dLayer([3 3],"Name","maxpool_3")
    fullyConnectedLayer(128,"Name","fc_1")
    reluLayer("Name","relu_5")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(52,"Name","fc_2")
    reluLayer("Name","relu_6")
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
% 神经网络训练设置
CNNoptions = trainingOptions('sgdm',...
    'Plots', 'training-progress', ...
    'Maxepochs',20,...
    'InitialLearnRate',0.001,...
    'MaxEpochs', 400,...
    'ValidationData',{X_TestC,Y_TestC},...
    'CheckpointPath',CheckpointPath);
%% Train step 1
% 第一次训练,后续会有调整，且会有暂存点保存，效果等同save，视情况启用保存功能
CNNetwork = trainNetwork(X_TrainC,Y_TrainC,layers,CNNoptions);
% save('CNN_trydata','network');
%% Train step 2
% 再训练，这时要加载保存点数据，从自选的 mat 中提取 net
% 变量作为续练网络，顺便复制一份训练设置在这里，修改更方便，如无意外，在本节可持续修改直到满意结果。
CNNetwork2 = load('net_checkpoint__10080__2021_07_21__20_54_38.mat','net');
%% Train step 2.1
% 若重启了软件，变量被清理，此节就是必要的，否则可跳过
load('1Xtrain-227134.mat');
load('1Xtest-227134.mat');
load('1Ytrain-227134.mat');
load('1Ytest-227134.mat');
X_TrainC = X_Train;
Y_TrainC = Y_Train;
X_TestC = X_Test;
Y_TestC = Y_Test;
Y_TrainC = categorical(Y_TrainC);
Y_TestC = categorical(Y_TestC);
clear X_Train X_Test Y_Train Y_Test
%% Train step 2.2
% 继续训练,这里就默认保存了,检查点文件夹也带时间戳方便归档
CheckpointPath = pwd;
DatemarktoNN = datestr(now,30);
NNCheckpointPath =...
    strcat(CheckpointPath,'\',DatemarktoNN);
mkdir(NNCheckpointPath);
CheckpointPath = string(NNCheckpointPath);

CNNoptions = trainingOptions('sgdm',...
    'Plots', 'training-progress', ...
    'Maxepochs',20,...
    'InitialLearnRate',0.0001,...
    'MaxEpochs', 1600,...
    'ValidationData',{X_TestC,Y_TestC},...
    'CheckpointPath',CheckpointPath);
% 默认保存神经网络时带时间戳，方便归档
CNNetwork = trainNetwork(X_TrainC,Y_TrainC,CNNetwork2.net.Layers,CNNoptions);
NNName =...
    strcat('CNNtry',DatemarktoNN,'.mat' );
NNName = string(NNName);
save(NNName,'CNNetwork');
clear DatemarktoNN NNCheckpointPath
%% Predict
% 本节预测汇总测试集所有数据并输出总正确率
PredictLabels = classify(CNNetwork,X_TestC);
TestLabels = Y_TestC;

Accuracy = sum(PredictLabels == TestLabels)/numel(PredictLabels);
disp(['accuracy:',num2str(Accuracy)]); 
% 作混淆矩阵图
figure
confusionchart(TestLabels,PredictLabels,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for CNN');
%% 临时使用
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
% cn3 = load('CNN_trydata.mat','CNNetwork');
% net_checkpoint__67200__2021_07_21__21_28_27.mat
PredictLabels = classify(cn3.net,X_TestC);
% PredictLabels = classify(cn3.CNNetwork,X_TestC);
TestLabels = Y_TestC;

Accuracy = sum(PredictLabels == TestLabels)/numel(PredictLabels);
disp(['accuracy:',num2str(Accuracy)]); 
% 作混淆矩阵图
figure
confusionchart(TestLabels,PredictLabels,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for CNN');
%% Extract Feature
% 从神经网络的某一层提取数据并与标签组成新数据集
% exTrain
exLayer = 'relu_6';
FeatureofDataset_XTrain = activations(cn3.net,X_TrainC,exLayer);
size(FeatureofDataset_XTrain)
DatemarktoNN = datestr(now,30);
FetName =...
    strcat('CNNfet_XTrain_',exLayer,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'FeatureofDataset_XTrain');
FetName =...
    strcat('CNNfet_YTrain_',exLayer,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'Y_TrainC');
% exTest
FeatureofDataset_XTest = activations(cn3.net,X_TestC,exLayer);
size(FeatureofDataset_XTest)
% DatemarktoNN = datestr(now,30);
FetName =...
    strcat('CNNfet_XTest_',exLayer,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'FeatureofDataset_XTest');
FetName =...
    strcat('CNNfet_YTest_',exLayer,'_',DatemarktoNN,'.mat' );
FetName = string(FetName);
save(FetName,'Y_TestC');
 

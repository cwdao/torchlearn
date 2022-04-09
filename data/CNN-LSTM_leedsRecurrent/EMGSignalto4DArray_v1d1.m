function [X_Train,Y_Train,...
    X_Test,Y_Test,...
    Proportion] = ...
    EMGSignalto4DArray_v1d1(...
    ParentDatasetPath,...
    DataSetSubject,...
    DataSetSubjectExerciseCounts,...
    ExerciseMaximumMovementCounts,...
    SignalFrenquency,...
    Rec_StepLength,...
    Rec_WindowWidth)
% EMGSignalto4DArray, 实现原始信号数据集转神经网络训练用数据集功能
% 版本说明：按 Ninapro 官方推荐方式分割数据集，取消随机模式
% 数据集：Ninapro DB1;
% 功能：EMG信号的分段切割，存储为四维数组，如果把每段数据比作图片，
% 维度分别表示：数据长、宽、颜色通道数（二维就是1）、总图片个数...
% 初版可用日期：2021.7.20
% 变量说明：
%     DataSetSubject,...受试者个数
%     DataSetSubjectExerciseCounts,...每个受试者动作组组数
%     ExerciseMaximumMovementCounts,...所有动作组的最大动作个数
%     SignalFrenquency,...原始信号频率，单位 Hz，ex: 100
%     Rec_StepLength,...窗口步长设定，单位 ms，ex:50
%     Rec_WindowWidth...窗口窗宽设定，单位 ms,ex:400
% 调试说明：
% 1.下载数据集并解压至所有mat文件处于同一目录，将目录赋给 DatasetPath ，
%   需要自行调整细节保证文件名可对应;
% 2.启动脚本后，切割好的信号片段会放入数组，最终传出所需测试集、验证集、训练集，且文件也会单独存入当前目录下各 mat 文件中。
% 作者：王骋，冯子阳
% 程序版本：v2.1（在原 SignaltoDataset 上继续迭代）
% 函数版本：v1.1
% 时间版本：210721(用于不同程序间同步)
%% 

% 本段预设了大部分需要调整的变量
% 有如下预设变量：
% 
% 受试者个数,ex:1
% DataSetSubject = 1;
% 每个受试者动作组组数
% DataSetSubjectExerciseCounts = 3;
% 所有动作组的最大动作个数
% ExerciseMaximumMovementCounts = 23;
% 原始信号通道个数
% MaxEmgChannel = 12;
% 原始信号频率
% SignalFrenquency = 100;
% 窗口步长设定为50ms,这里根据频率自动计算循环所需步长
Rec_StepLength = Rec_StepLength*0.001/(1/SignalFrenquency);
% 窗口窗宽预设为400ms，此变量根据频率计算循环实际窗宽
Rec_WindowWidth = Rec_WindowWidth*0.001/(1/SignalFrenquency);
% 频谱计算时需设置的时间分辨率，暂时弃用
% TimeResolution = 0.01;
% 
% 预设占位变量，实际内容仍在代码位置修改：
% 
% 当前文件受试者编号
SubjectNumberString = '';
% 当前动作组组号
ExerciseNumberString = '';
% 当前动作编号
MovementsNumberString = '';
% 当前动作的预创文件夹目录
FolderNameOfEachExercise = '';
% 预设标签名变量
LabelNameOfEachExercise = '';
% 用于指定片段在四维数组坐标的变量
Rec_CurrentSignalSegmentNumb_Train = 1;
Rec_CurrentSignalSegmentNumb_Test = 1;
% 用于指定划分训练集和测试集
Rec_DecideWheretoPutSegment = 1;
% 图片名称变量，暂时弃用
% ImageName = '';
% 切割后的当前段暂存变量，暂时弃用
% SignalStack = [];
% 本字符串数组用于建立文件夹标注动作组名
StringAlphabet={'A','B','C','D','E','F','G',...
        'H','I','J','K','L','M','N','O',...
        'P','Q','R','S','T','U','V','W','X','Y','Z'};
%% 
% 此部分为代码主循环，注意所有用到的自定义函数(如果有)需要放在与本文档同目录下，或将其路径加入搜索 path。
% 
% 逻辑代码：
% 
% 循环测试对象
for a=1:DataSetSubject
%     循环测试练习组号
    for b=1:DataSetSubjectExerciseCounts
%         读取当前待分析mat文件
        SubjectNumberString=(num2str(a,'%01d'));
        ExerciseNumberString=(num2str(b,'%01d'));
        FolderFirstAlphabet = StringAlphabet(b);
%         DatasetPath=...
%             ['C:\Users\cwdbo\Downloads\ninapro\db1\S' SubjectNumberString '_E' ExerciseNumberString '_A1' '.mat'];
            DatasetPath=...
        [ParentDatasetPath 'S' SubjectNumberString '_A1' '_E' ExerciseNumberString '.mat'];
        load(DatasetPath);
%         获取emg信号长度，准备执行分割
        EMGSignalLength=size(emg,1);
%         分段循环中止标志置0
        Flag_EndCheck = 0;
%         数据集的littlebug，原始的Ninapro db1在这组数据中...
%         restimulus 比 EMG 信号数据少一个，要补齐
        if(b == 3)
            restimulus(877073,1) = 0;
        end
%         遍历每个练习组所有动作标签
        for j=1:ExerciseMaximumMovementCounts
%             针对当前标签创建（组号-动作号）文件夹
                 MovementsNumberString=(num2str(j,'%01d'));
%                  建标签名，字母转为字符串后接每组具体动作序号
                 FolderFirstAlphabet = char(FolderFirstAlphabet);
                 LabelofExerciseFirstName = abs(FolderFirstAlphabet);
                 LabelNameOfEachExercise = LabelofExerciseFirstName*100 + j;
%                 开始遍历标签
            for i=1:EMGSignalLength
%                 寻找标签起始点，设立起始flag
                 if(restimulus(i,1) == j && restimulus(i-1,1) == 0)
                    startnum=i;
                 end
%                  寻找标签中止点，设立中止flag
                 if(restimulus(i,1) == j && restimulus(i+1,1) == 0)
                     endnum=i;
                     Flag_EndCheck = 1;
                 end
%                  起始终止位置已经找到后，开始分段
                 if(Flag_EndCheck == 1)
%                      步长50ms,直到结束点前一个片段长度为止，开始切割

                     for start=startnum:Rec_StepLength:endnum-Rec_WindowWidth
%                          开始划分段                         
                        SegmentData=emg(start:start+Rec_WindowWidth-1,:);
%                         存入数组，坐标下移
                        if (Rec_DecideWheretoPutSegment == 2 ||...
                                Rec_DecideWheretoPutSegment == 5 ||...
                                Rec_DecideWheretoPutSegment == 7)
                            SigSeg_Xtest(:,:,1,Rec_CurrentSignalSegmentNumb_Test) = SegmentData;
                            SigSeg_Ytest(Rec_CurrentSignalSegmentNumb_Test,1) = LabelNameOfEachExercise;
                            Rec_CurrentSignalSegmentNumb_Test = Rec_CurrentSignalSegmentNumb_Test + 1;

                        else
                            SigSeg_Xtrain(:,:,1,Rec_CurrentSignalSegmentNumb_Train) = SegmentData;
                            SigSeg_Ytrain(Rec_CurrentSignalSegmentNumb_Train,1) = LabelNameOfEachExercise;
                            Rec_CurrentSignalSegmentNumb_Train = Rec_CurrentSignalSegmentNumb_Train + 1;
                        end
                     end
%                      当前标签分段结束后，重置起始终止flag
                        Rec_DecideWheretoPutSegment = Rec_DecideWheretoPutSegment+1;
%                      本次终值位留给下一次起始位，节省时间
                     Flag_EndCheck = 0;
%                      size(SigSeg_Xtrain)
%                      size(SigSeg_Ytrain)
%                      size(SigSeg_Xtest)
%                      size(SigSeg_Ytest)
%                      CheckP = Rec_CurrentSignalSegmentNumb_Train/...
%                          (Rec_CurrentSignalSegmentNumb_Test + Rec_CurrentSignalSegmentNumb_Train);
%                    数据集一个动作10次，到次数重置之
                    if (Rec_DecideWheretoPutSegment == 10)
                        Rec_DecideWheretoPutSegment = 1;
                    end
                 end
            end
        end
    end
end
clear i j Flag_EndCheck Rec_CurrentSignalSegmentNumb...
    MovementsNumberString FolderNameOfEachExercise...
    MaxEmgChannel
% 打印验证一下
size(SigSeg_Xtrain)
size(SigSeg_Ytrain)
size(SigSeg_Xtest)
size(SigSeg_Ytest)
CheckP = Rec_CurrentSignalSegmentNumb_Train/...
 (Rec_CurrentSignalSegmentNumb_Test + Rec_CurrentSignalSegmentNumb_Train);
% 此时数组已经可以输出了
X_Train = SigSeg_Xtrain;
Y_Train = SigSeg_Ytrain;
X_Test = SigSeg_Xtest;
Y_Test = SigSeg_Ytest;
Proportion = CheckP;
% SigSeg_X是信号分段后组成的四维数组，按照 Matlab 的默认方式，前两维表示数据的长和宽，
% 第三维表示层数（如彩图此维为3）,数据在第四维叠加，显示了总的数据数。这里把数据和其标签分别保存到两个mat里。
DatemarktoDataset = datestr(now,30);
DataSegmentPathandName =...
    strcat(SubjectNumberString,'Xtrain-',DatemarktoDataset,'.mat' );
DataSegmentPathandName_X = ...
    string(DataSegmentPathandName);
save(DataSegmentPathandName_X,'X_Train');
DataSegmentPathandName =...
    strcat(SubjectNumberString,'Ytrain-',DatemarktoDataset,'.mat' );
DataSegmentPathandName_Y = ...
    string(DataSegmentPathandName);
save(DataSegmentPathandName_Y,'Y_Train');                        
DataSegmentPathandName =...
    strcat(SubjectNumberString,'Xtest-',DatemarktoDataset,'.mat' );
DataSegmentPathandName_Y = ...
    string(DataSegmentPathandName);
save(DataSegmentPathandName_Y,'X_Test');   
DataSegmentPathandName =...
    strcat(SubjectNumberString,'Ytest-',DatemarktoDataset,'.mat' );
DataSegmentPathandName_Y = ...
    string(DataSegmentPathandName);
save(DataSegmentPathandName_Y,'Y_Test');   

%% 
clear emg acc stimulus glove inclin subject exercise repetition restimulus rerepetition str3
end

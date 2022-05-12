clear;
clc;
% load('./redact_S4_WFE_T1-new.mat');
% load('./redact_S4_WFE_T2-new.mat');
% load('redact_S4_WFE_T1.mat');
load('./S1/redact_S1_WFE_T1.mat');
%% 
% 本次数据开头处有部分变量法值缺失，因此选择从第12行开始；同时角度和时间有前面的
% 缓冲，速度和加速度可以覆盖完整

% cyc = Gait{:,1};
% time = Gait{:,2};
% emg_rf_l = Gait{:,3};
% emg_lh_l = Gait{:,4};
% mf_rf_l = Gait{:,5};
% mf_bm_l = Gait{:,6};
% ka_l = Gait{:,8};

% data = wristData{8:14218,:};
data = wristData;

%% 不用的数据数据删除
Gait(:,{'LKneeMoment'})=[];
%% 
data = Gait;
% 计算角速度
for i=2:101
    data{i,'A_v'} = ((data{i,7}-data{i-1,7})/360*2*pi)/(data{i,2}-data{i-1,2});
end
% 计算角加速度   
M_knee = 0.054596 *5.74*(0.174^2);
for i=3:101
    data{i,'A_a'} = (data{i,8}-data{i-1,8})/(data{i,2}-data{i-1,2});
    data{i,'Ma'} = M_knee*data{i,'A_a'};
end
%% 其他变量

for i=1:101
    data{i,'C_knee'} = 5.74*0.174*0.4*sin(data{i,7}/360*2*pi);
    data{i,'Cv'} = data{i,'C_knee'}*data{i,'A_v'};
end
for i=1:101
    data{i,'G_theta'} = 5.74*9.8*(0.174^2)*cos(data{i,7}/360*2*pi);
end

%% MArf & MAbif
for i=1:101
    x_1 = 0.032*data{i,1}/360*2*pi;
    data{i,'M_Arf'} = 0.032-0.0028*cos(x_1)+0.0021*sin(x_1)+2.4*exp(1)...
        -5*cos(2*x_1) +0.0013*sin(2*x_1)+0.00029*cos(3*x_1)...
        -0.000024*sin(3*x_1);
    x_2 = 0.036*data{i,1}/360*2*pi;
    data{i,'M_Abif'} = 0.037+0.0059*cos(x_2)-0.0029*sin(x_2)+0.00073*cos(2*x_2)...
        -0.00086*sin(2*x_2)-0.00075*sin(3*x_2)+0.00094*sin(3*x_2);
end
%% 

% time = data{:,1};
% fcr = data{:,2};
% fcu = data{:,3};
% ecrl = data{:,4};
% ecrb = data{:,5};
% ecu = data{:,6};
% angle = data{:,7};
% mf_fcr = data{:,8};
% mf_fcu = data{:,9};
% mf_ecrl = data{:,10};
% mf_ecrb = data{:,11};
% mf_ecu = data{:,12};
time = data{:,1}(1:2000);
fcr = data{:,2}(1:2000);
fcu = data{:,3}(1:2000);
ecrl = data{:,4}(1:2000);
ecrb = data{:,5}(1:2000);
ecu = data{:,6}(1:2000);
angle = data{:,7}(1:2000);
mf_fcr = data{:,8}(1:2000);
mf_fcu = data{:,9}(1:2000);
mf_ecrl = data{:,10}(1:2000);
mf_ecrb = data{:,11}(1:2000);
mf_ecu = data{:,12}(1:2000);


%%
sample  = 1800;
fcr = resample(fcr, sample, 2000);
time = resample(time, sample, 2000);
ecu = resample(ecu, sample, 2000);
fcu = resample(fcu, sample, 2000);
ecrl = resample(ecrl, sample, 2000);
ecrb = resample(ecrb, sample, 2000);
mf_fcr = resample(mf_fcr, sample, 2000);
mf_fcu = resample(mf_fcu, sample, 2000);
mf_ecrl = resample(mf_ecrl, sample, 2000);
mf_ecrb = resample(mf_ecrb, sample, 2000);
mf_ecu = resample(mf_ecu, sample, 2000);
angle = resample(angle, sample, 2000);
%% 

DataPathandName =...
    strcat('EMGSKdata-220426-s4t2-slim_2000.mat' );
save(DataPathandName,'time','fcr','fcu','ecrl',...
    'ecrb','ecu','angle','mf_fcr'...
    ,'mf_fcu','mf_ecrl','mf_ecrb','mf_ecu');


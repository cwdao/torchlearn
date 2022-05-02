clear;
clc;
% load('redacted_GIL01_Free4.mat');

load('redacted_GIL03_Free4.mat');

%% 

cyc = Gait{:,1};
time = Gait{:,2};
emg_rf_l = Gait{:,3};
emg_lh_l = Gait{:,4};
mf_rf_l = Gait{:,5};
mf_bm_l = Gait{:,6};
ka_l = Gait{:,8};
%% 
DataPathandName =...
    strcat('EMGSKdata-220426_s3f4.mat' );
save(DataPathandName,'cyc','time','emg_rf_l','emg_lh_l','mf_rf_l',...
    'mf_bm_l','ka_l');

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

cyc = data{:,1};
time = data{:,2};
emg_rf_l = data{:,3};
emg_lh_l = data{:,4};
mf_rf_l = data{:,5};
mf_bm_l = data{:,6};
ka_l = data{:,7};
A_v = data{:,8};
A_a = data{:,9};
Ma = data{:,10};
C_knee = data{:,11};
Cv = data{:,12};
G_theta = data{:,13};
M_Arf = data{:,14};
M_Abif = data{:,15};
%%

DataPathandName =...
    strcat('EMGSKdata-220323.mat' );
save(DataPathandName,'cyc','time','emg_rf_l','emg_lh_l','mf_rf_l',...
    'mf_bm_l','ka_l','A_v','A_a'...
    ,'Ma','C_knee','Cv','G_theta','M_Arf','M_Abif');


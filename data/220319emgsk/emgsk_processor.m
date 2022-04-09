clear;
clc;
% load('redacted_GIL01_XSlow2.mat');

%% 
cyc = Gait{:,1};
emg_rf_l = Gait{:,2};
emg_lh_l = Gait{:,3};
mf_rf_l = Gait{:,4};
mf_bm_l = Gait{:,5};
ka_l = Gait{:,7};
%% 合并数组
% data_K = [time_k,jk_hf_r,jk_kf_r,jk_af_r,jk_hf_l,jk_kf_l,jk_af_l];
% data_Mv = [time_m,jm_hm_r,jm_km_r,jm_am_r,jm_hm_l,jm_km_l,jm_am_l];
% data_MF = [time_ms,ms_rf_r,ms_s_r,ms_ta_r,ms_rf_l,ms_s_l,ms_ta_l];
%% 时间重整
% % 对齐K，MV
% data_Mv = data_Mv(3:126,:);
% %% 
% % 绝对时间轴上，JK 在 MF 之后的值，没有实际意义了。
% data_K = data_K(1:118,:);
% data_Mv = data_Mv(1:118,:);
% % data_MF = data_K(1:118,:)
% %% 
% % 绝对时间轴上，JK 在 MF 之前的值，应该也没啥意义。先对齐再做打算
% data_K = data_K(8:118,:);
% data_Mv = data_Mv(8:118,:);
% %% 降采样 MF 
% data_MFF_t = resample(data_MF(:,1), 111, 919);
% data_MFF = resample(data_MF, 111, 919);
%% 

DataPathandName =...
    strcat('EMGSKdata-220319.mat' );
save(DataPathandName,'cyc','emg_rf_l','emg_lh_l','mf_rf_l','mf_rf_l','ka_l');




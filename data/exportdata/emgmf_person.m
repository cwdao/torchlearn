clc;
clear;
%% 
% 上肢MSE
% load('./ul_rf_slim-omse.mat')
% 上肢
% load('./ul_rf_slim.mat')
% load('./ul_rf_slim_nl_s4t2.mat')
load('./ul_rf_slim_s4t2.mat')
% 下肢MSE
% load('./dl_rf_-omse_lr001.mat')
% 下肢
% load('./dl_rf.mat')
% load('./low_intj_s1f4_s3f4.mat')
% load('./low_nl_intj_s1f4_s3f4.mat')
%% 上肢，各变量
GT_angle = variables{1,1}.GT(:,1);
PD_angle = variables{1,1}.Preds(:,:,1);
GT_fcr = variables{1,1}.GT(:,2);
PD_fcr = variables{1,1}.Preds(:,:,2);
plot(PD_fcr)
plot(GT_fcr)
GT_fcu = variables{1,1}.GT(:,3);
PD_fcu = variables{1,1}.Preds(:,:,3);
GT_ecrl = variables{1,1}.GT(:,4);
PD_ecrl = variables{1,1}.Preds(:,:,4);
GT_ecrb = variables{1,1}.GT(:,5);
PD_ecrb = variables{1,1}.Preds(:,:,5);
GT_ecu = variables{1,1}.GT(:,6);
PD_ecu = variables{1,1}.Preds(:,:,6);
%% 
pearson(1,:) = corr(PD_angle,GT_angle);
pearson(2,:)  = corr(PD_fcr,GT_fcr);
pearson(3,:)  = corr(PD_fcu,GT_fcu);
pearson(4,:)  = corr(PD_ecrl,GT_ecrl);
pearson(5,:)  = corr(PD_ecrb,GT_ecrb);
pearson(6,:)  = corr(PD_ecu,GT_ecu);
%% 下肢，各变量测试集展示，GD:ground truth,PD:predicts
GT_ka = variables{1,1}.GT(:,3);
PD_ka = variables{1,1}.Preds(:,:,3);
plot(PD_ka)
plot(GT_ka)
GT_rf = variables{1,1}.GT(:,1);
PD_rf = variables{1,1}.Preds(:,:,1);
GT_bm = variables{1,1}.GT(:,2);
PD_bm = variables{1,1}.Preds(:,:,2);

% GT_ka = variables{1,1}.rf_GT(:,3);
% PD_ka = variables{1,1}.rf_Preds(:,:,3);
% plot(PD_ka)
% plot(GT_ka)
% GT_rf = variables{1,1}.rf_GT(:,1);
% PD_rf = variables{1,1}.rf_Preds(:,:,1);
% GT_bm = variables{1,1}.rf_GT(:,2);
% PD_bm = variables{1,1}.rf_Preds(:,:,2);
%% 
pearson(1,:) = corr(PD_rf,GT_rf);
pearson(2,:)  = corr(PD_bm,GT_bm);
pearson(3,:)  = corr(PD_ka,GT_ka);

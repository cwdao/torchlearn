clc;
clear;
%% 
% 上肢MSE
% load('./ul_rf_slim-omse.mat')
% 上肢
% load('./ul_rf_slim.mat')
% load('./ul_rf_slim_nl_s4t2.mat')
% load('./ul_rf_slim_s4t2.mat')
% 一个临时需处理的文件，要计算一些相关系数
% load('./unkn_personprocess_testS2.mat')
load('./unkn_personprocess_testS7.mat')
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
pearson_1(1,:) = corr(PD_angle,GT_angle);
pearson_1(2,:)  = corr(PD_fcr,GT_fcr);
pearson_1(3,:)  = corr(PD_fcu,GT_fcu);
pearson_1(4,:)  = corr(PD_ecrl,GT_ecrl);
pearson_1(5,:)  = corr(PD_ecrb,GT_ecrb);
pearson_1(6,:)  = corr(PD_ecu,GT_ecu);
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
pearson_1(1,:) = corr(PD_rf,GT_rf);
pearson_1(2,:)  = corr(PD_bm,GT_bm);
pearson_1(3,:)  = corr(PD_ka,GT_ka);
%% 临时文件unkn_personprocess_testS2所需
GD_angle = permute(angle,[2 1]);
GD_ecrb = permute(ecrb,[2,1]);
GD_ecrl = permute(ecrl,[2,1]);
GD_ecu = permute(ecu,[2 1]);
GD_fcr = permute(fcr,[2 1]);
GD_fcu = permute(fcu,[2 1]);
%% 这pred顺序还和样本不一样……
pearson_1(1,:) = corr(predict_show_1(:,:,1),GD_angle);
pearson_1(2,:) = corr(predict_show_2(:,:,1),GD_angle);
pearson_1(3,:) = corr(predict_show_3(:,:,1),GD_angle);
pearson_1(4,:) = corr(predict_show_4(:,:,1),GD_angle);
pearson_1(5,:) = corr(predict_show_5(:,:,1),GD_angle);
pearson_1(6,:) = corr(predict_show_6(:,:,1),GD_angle);

pearson_2(1,:) = corr(predict_show_1(:,:,2),GD_fcr);
pearson_2(2,:) = corr(predict_show_2(:,:,2),GD_fcr);
pearson_2(3,:) = corr(predict_show_3(:,:,2),GD_fcr);
pearson_2(4,:) = corr(predict_show_4(:,:,2),GD_fcr);
pearson_2(5,:) = corr(predict_show_5(:,:,2),GD_fcr);
pearson_2(6,:) = corr(predict_show_6(:,:,2),GD_fcr);

pearson_3(1,:) = corr(predict_show_1(:,:,3),GD_fcu);
pearson_3(2,:) = corr(predict_show_2(:,:,3),GD_fcu);
pearson_3(3,:) = corr(predict_show_3(:,:,3),GD_fcu);
pearson_3(4,:) = corr(predict_show_4(:,:,3),GD_fcu);
pearson_3(5,:) = corr(predict_show_5(:,:,3),GD_fcu);
pearson_3(6,:) = corr(predict_show_6(:,:,3),GD_fcu);

pearson_4(1,:) = corr(predict_show_1(:,:,4),GD_ecrl);
pearson_4(2,:) = corr(predict_show_2(:,:,4),GD_ecrl);
pearson_4(3,:) = corr(predict_show_3(:,:,4),GD_ecrl);
pearson_4(4,:) = corr(predict_show_4(:,:,4),GD_ecrl);
pearson_4(5,:) = corr(predict_show_5(:,:,4),GD_ecrl);
pearson_4(6,:) = corr(predict_show_6(:,:,4),GD_ecrl);

pearson_5(1,:) = corr(predict_show_1(:,:,5),GD_ecrb);
pearson_5(2,:) = corr(predict_show_2(:,:,5),GD_ecrb);
pearson_5(3,:) = corr(predict_show_3(:,:,5),GD_ecrb);
pearson_5(4,:) = corr(predict_show_4(:,:,5),GD_ecrb);
pearson_5(5,:) = corr(predict_show_5(:,:,5),GD_ecrb);
pearson_5(6,:) = corr(predict_show_6(:,:,5),GD_ecrb);

pearson_6(1,:) = corr(predict_show_1(:,:,6),GD_ecu);
pearson_6(2,:) = corr(predict_show_2(:,:,6),GD_ecu);
pearson_6(3,:) = corr(predict_show_3(:,:,6),GD_ecu);
pearson_6(4,:) = corr(predict_show_4(:,:,6),GD_ecu);
pearson_6(5,:) = corr(predict_show_5(:,:,6),GD_ecu);
pearson_6(6,:) = corr(predict_show_6(:,:,6),GD_ecu);
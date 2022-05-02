% 加载下肢数据
clear;
clc;
load('.\exportdata\dl_lr001.mat')
% load('.\exportdata\loss_rmse_dn.mat')
%% 
% 下肢，预测关节角度k_a、mf_rf、mf_bm三种变量
k_a = RMSE(:,3);
mf_rf = RMSE(:,1);
mf_bm = RMSE(:,2);
plot(RMSE)
%% 下肢，loss
% 包含MSE损失和总损失两变量
total_loss = loss(:,1);
mse_loss = loss(:,2);
plot(loss)
%% 下肢，各变量测试集展示，GD:ground truth,PD:predicts
GT_ka = variables.GT(:,3);
PD_ka = variables.Preds(:,:,3);
plot(PD_ka)
plot(GT_ka)
GT_rf = variables.GT(:,1);
PD_rf = variables.Preds(:,:,1);
GT_bm = variables.GT(:,2);
PD_bm = variables.Preds(:,:,2);
%% 
% 加载上肢数据
clear;
load('.\exportdata\ul_rf_slim-omse.mat')
% load('.\exportdata\loss_rmse_up_v2slim.mat')
%% 上肢，各肌肉力 RMSE
angle = nRMSE(:,1);
fcr = nRMSE(:,2);
fcu = nRMSE(:,3);
ecrl = nRMSE(:,4);
ecrb = nRMSE(:,5);
ecu = nRMSE(:,6);
plot(nRMSE)
%% 上肢，loss
% total_loss = loss(:,1);
mse_loss = loss(:,1);
plot(loss)
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
% save('.\exportdata\loss_rmse_up_v2slim-omse.mat','nRMSE','loss')
save('.\exportdata\loss_rmse_low-omse_lr001.mat','RMSE','loss')
%% 绘图

figure(1)
plot(k_a(1:1000),'LineWidth',1);hold on;
plot(mf_bm(1:1000),'LineWidth',1);hold on;
plot(mf_rf(1:1000),'LineWidth',1);hold on;
title('RMSE for Lower Limbs');
xlabel('Epoch');
ylabel('RMSE');
legend('knee angle','mf-bm','mf-rf');
%% 

figure(2)
plot(total_loss,'LineWidth',1);hold on;
plot(mse_loss,'LineWidth',1);hold on;
% plot(mf_rf(1:1000),'LineWidth',1);hold on;
title('Train Loss');
xlabel('Epoch');
ylabel('loss');
legend('total loss','mse loss');
% legend('mse loss');
%% 
figure(3)
plot(GT_ka,'LineWidth',1);hold on;
plot(PD_ka,'LineWidth',1);hold on;
title('Knee Angle Predicted on Testset(RMSE:1.6858)');
xlabel('num');
ylabel('Rad');
legend('Ground Truth','Prediction');
%% 
figure(5)
plot(GT_rf,'LineWidth',1);hold on;
plot(PD_rf,'LineWidth',1);hold on;
title('MuscleForce_{retus femoris l} Predicted on Testset(RMSE:16.064)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 
figure(7)
plot(GT_bm,'LineWidth',1);hold on;
plot(PD_bm,'LineWidth',1);hold on;
title('MuscleForce_{bifemsh l} Predicted on Testset(RMSE:14.168)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 绘图

figure(1)
plot(angle,'LineWidth',1);hold on;
plot(fcr,'LineWidth',1);hold on;
plot(fcu,'LineWidth',1);hold on;
plot(ecrl,'LineWidth',1);hold on;
plot(ecrb,'LineWidth',1);hold on;
plot(ecu,'LineWidth',1);hold on;

title('RMSE for Upper Limbs');
xlabel('Epoch');
ylabel('RMSE');
legend('angle','FCR','FCU','ECRL','ECRB','ECU');
%% 

figure(3)
plot(GT_angle,'LineWidth',1);hold on;
plot(PD_angle,'LineWidth',1);hold on;
title('MuscleForce_{angle} Predicted on Testset(RMSE:1.0404)');
xlabel('num');
ylabel('Degree');
legend('Ground Truth','Prediction');
%% 

figure(2)
% plot(total_loss,'LineWidth',1);hold on;
plot(mse_loss,'LineWidth',1);hold on;
% plot(mf_rf(1:1000),'LineWidth',1);hold on;
title('Train Loss');
xlabel('Epoch');
ylabel('loss');
% legend('total loss','mse loss');
legend('mse loss');
%% 
figure(3)
plot(GT_fcr,'LineWidth',1);hold on;
plot(PD_fcr,'LineWidth',1);hold on;
title('MuscleForce_{FCR} Predicted on Testset(RMSE:0.80461)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 
figure(4)
plot(GT_fcu,'LineWidth',1);hold on;
plot(PD_fcu,'LineWidth',1);hold on;
title('MuscleForce_{FCU} Predicted on Testset(RMSE:0.97196)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 
figure(5)
plot(GT_ecrl,'LineWidth',1);hold on;
plot(PD_ecrl,'LineWidth',1);hold on;
title('MuscleForce_{ECRL} Predicted on Testset(RMSE:0.90445)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 
figure(6)
plot(GT_ecrb,'LineWidth',1);hold on;
plot(PD_ecrb,'LineWidth',1);hold on;
title('MuscleForce_{ECRB} Predicted on Testset(RMSE:0.72806)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
%% 
figure(7)
plot(GT_ecu,'LineWidth',1);hold on;
plot(PD_ecu,'LineWidth',1);hold on;
title('MuscleForce_{ECU} Predicted on Testset(RMSE:0.15699)');
xlabel('num');
ylabel('N');
legend('Ground Truth','Prediction');
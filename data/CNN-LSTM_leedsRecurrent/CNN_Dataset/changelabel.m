% Y_Train(1,1);
% for i = 1:21520
%     if Y_Train(i,1)/100 <66
%         Y_TrainP(i,1) = Y_Train(i,1)-6501;
%     elseif Y_Train(i,1)/100 <67
%         Y_TrainP(i,1) = Y_Train(i,1)-6601+12;
%     elseif Y_Train(i,1)/100 <68
%         Y_TrainP(i,1) = Y_Train(i,1)-6701+12+17;
%     end
% end
save('Y_TrainP.mat','Y_TrainP');
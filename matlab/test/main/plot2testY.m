%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
fileName = ['data/', 'data_testLogisticReg0_20190328210451.mat'];
load(fileName);

posFlag = 1;
negFlag = 0;

%% 画出数据图
% 原始数据图
figure(1);
posOrigin = find(YOrigin == posFlag); 
negOrigin = find(YOrigin == negFlag);

plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 训练集图
figure(2);
posTrain = find(YTrain == posFlag); 
negTrain = find(YTrain == negFlag);

plot(XTrain(posTrain, 1), XTrain(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XTrain(negTrain, 1), XTrain(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
title('训练集图');
fprintf('训练集图\n');
hold off;

%% 交叉验证集图
figure(3);
posVal = find(YVal == posFlag); 
negVal = find(YVal == negFlag);

plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
title('交叉验证集图');
fprintf('交叉验证集图\n');
hold off;

%% 在各个图上画分割结果
for i=1:3
    figure(i);
    hold on;
    contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
    hold off;
end

%% 学习曲线
figure(4);
plot(realSplitVecLearn, errorTrainLearn, realSplitVecLearn, errorValLearn);
title('学习曲线');
legend('训练集', '交叉验证集');
xlabel('数量');
ylabel('误差');
fprintf('学习曲线\n');
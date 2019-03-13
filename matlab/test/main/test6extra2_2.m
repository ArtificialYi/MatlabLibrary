%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
data = load('../resource/ex6data1.mat');
XOrigin = data.X;
YOrigin = data.y;

m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

%% 基础训练模型


%% 画出数据图
% 原始数据图
figure(1);
posOrigin = find(YOrigin == 1); 
negOrigin = find(YOrigin == 0);

plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
% contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

% 训练集
figure(2);
posTrain = find(YTrain == 1); 
negTrain = find(YTrain == 0);

plot(XTrain(posTrain, 1), XTrain(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XTrain(negTrain, 1), XTrain(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
%contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('训练集');
fprintf('训练集图\n');
hold off;

% 交叉验证集
figure(3);
posVal = find(YVal == 1); 
negVal = find(YVal == 0);

plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
%contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('交叉验证集');
fprintf('交叉验证集\n');
hold off;
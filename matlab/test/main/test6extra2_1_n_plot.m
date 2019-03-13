%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
load data/data_test6extra2_1_n.mat;
% 额外数据准备
n = size(XOriginNorm, 2);

% 画图用的属性
height = 20;
weight = 20;
pad = 1;
imgVec = [-1 1];

% 利用第一次训练结果预测原始训练集的结果
thetaOriginMatrix = zeros(n+1, maxClass);
for i=1:maxClass
    thetaOriginMatrix(:, i) = [modelOriginMatrix(i).cpu.b;modelOriginMatrix(i).cpu.w];
end

predOriginMatrixTmp = [ones(m, 1) XOriginNorm] * thetaOriginMatrix;
predOriginMatrix = zeros(m, maxClass);
predOriginMatrix(predOriginMatrixTmp >= 0) = 1;
predOrigin = predOriginMatrix * (2.^(0:maxClass-1)');

%% 开始画图
% 原始数据图
figure(1);
colormap(gray);
plotImage(XOrigin, height, weight, pad, pad, imgVec);
title('原始数据集');

% 训练集图
figure(2);
colormap(gray);
plotImage(XTrain, height, weight, pad, pad, imgVec);
title('训练集');

% 交叉验证集图
figure(3);
colormap(gray);
plotImage(XVal, height, weight, pad, pad, imgVec);
title('交叉验证集');

% 原始数据正确的训练结果
figure(4);
colormap(gray);
plotImage(XOrigin(YOrigin==predOrigin,:), height, weight, pad, pad, imgVec);
title('原始数据-正确结果集');

% 原始数据的错误训练结果
figure(5);
colormap(gray);
plotImage(XOrigin(YOrigin~=predOrigin,:), height, weight, pad, pad, imgVec);
title('原始数据-错误结果集');

% 画出所有学习曲线
for i=1:maxClass
    figure(5+i);
    plot(realSplitVecLearn(:,i), errorTrainLearn(:,i), realSplitVecLearn(:,i), errorValLearn(:,i));
end
%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
load data/data_test7base0n_20190326134039.mat;

%% 画出数据图
% 原始数据图
figure(1);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
%contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 训练集图
figure(2);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
%contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('训练集图');

%% 交叉验证集图
figure(3);

plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
%contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('交叉验证集图');
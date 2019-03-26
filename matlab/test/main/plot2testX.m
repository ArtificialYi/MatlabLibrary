%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
load data/data_test7base0n_20190326180306.mat;

mK = size(centroidsOrigin, 1);
contourVec = mean(vec2subMatrix(2.^(1:mK), 2),2);
if mK == 1
    contourVec = [2 2];
end
m1 = size(vecX1, 1);
pred2D = reshape(2.^YTest, m1, m1);

%% 画出数据图
% 原始数据图
figure(1);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
contour(vecX1, vecX2, pred2D, contourVec);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 训练集图
figure(2);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
hold on;
contour(vecX1, vecX2, pred2D, contourVec);
title('训练集图');
hold off;

%% 交叉验证集图
figure(3);

plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
hold on;
contour(vecX1, vecX2, pred2D, contourVec);
hold off;
title('交叉验证集图');
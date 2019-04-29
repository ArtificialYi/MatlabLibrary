%% 初始化环境
clear; close all; clc;

%% 计算工作日和休息日的学习时间分配
data = load('resource/ex7data2.mat');
XOrigin = data.X;

% plot(XOrigin(:, 1), XOrigin(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);

KMax = 3;
plotVec = ["r+", "b+", "g+"];
[centroids, predY, errorMin, KReal] = kMedoidsTrain(XOrigin, KMax);

hold on;
for i=1:KReal
    plot(XOrigin(predY==i, 1), XOrigin(predY==i, 2), plotVec(i),'LineWidth', 1, 'MarkerSize', 5);
    plot(centroids(i, 1), centroids(i, 2), 'ko', 'MarkerSize', 7);
end
hold off;
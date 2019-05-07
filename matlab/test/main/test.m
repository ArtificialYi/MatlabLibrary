%% 初始化环境
clear; close all; clc;

%% 查看数据
data = load('resource/ex7data2.mat');
XOrigin = data.X;

% plot(XOrigin(:, 1), XOrigin(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);

%% 数据渲染-1
KMax = 6;
plotVec = ["r+", "b+", "g+", "k+", "y+", "c+"];
[centroids, predY, errorMin, KReal] = kMedoidsTrain(XOrigin, KMax);

figure(1)
hold on;
for i=1:KReal
    plot(XOrigin(predY==i, 1), XOrigin(predY==i, 2), plotVec(i),'LineWidth', 1, 'MarkerSize', 5);
    plot(centroids(i, 1), centroids(i, 2), 'ko', 'MarkerSize', 7);
end
hold off;

%% 初始化环境
clear; clc;
%% 读取无监督数据
data = load('resource/ex7data2.mat');
XOrigin = data.X;

maxIter = 1e8;

%% 使用means将数据特征化
% 特征归一化
[XOriginNorm, data2normFuncOrigin] = data2featureWithNormalize(XOrigin, 1);

% 特征离散化
kMeansFunc = @(paramX, paramK) kMeansTrainRandGPU(paramX, paramK, maxIter);
kMeansPredFunc = @(paramX, paramCentroids) kMeansTrainGPU(paramX, paramCentroids, 1);
KMax = 30;
p = 1;
% means-离散化函数
[XOriginNormMeansP1, data2binaryP1] = binaryFeature(XOriginNorm, KMax, p, kMeansFunc, kMeansPredFunc);

% 数据渲染-2
KPlot = 6;
plotVec = ["r+", "b+", "g+", "k+", "y+", "c+"];
[centroids, predY, errorMin, KReal] = kMedoidsTrain(XOrigin, KPlot);

figure(2)
hold on;
for i=1:KReal
    plot(XOrigin(predY==i, 1), XOrigin(predY==i, 2), plotVec(i),'LineWidth', 1, 'MarkerSize', 5);
    plot(centroids(i, 1), centroids(i, 2), 'ko', 'MarkerSize', 7);
end
hold off;

%% 使用medoids将数据特征化


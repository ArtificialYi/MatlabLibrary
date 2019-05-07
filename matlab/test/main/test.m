%% 初始化环境
clear; close all; clc;

%% 查看数据
data = load('resource/ex7data2.mat');
XOrigin = data.X;

% plot(XOrigin(:, 1), XOrigin(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);

%% 数据渲染-1
KMax = 3;
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
figure(3);
hist(XOrigin(:, 1), 32);
figure(4);
hist(XOrigin(:, 2), 32);

%% 使用means将数据特征化
% 特征归一化
[XOriginNorm, data2normFuncOrigin] = data2featureWithNormalize(XOrigin, 1);

% 特征离散化-means
kMeansFunc = @(paramX, paramK) kMeansTrainRandCPU(paramX, paramK, maxIter);
kMeansPredFunc = @(paramX, paramCentroids) kMeansTrainCPU(paramX, paramCentroids, 1);
KMax = 18;
p = 1;
% 离散化函数-means
[XOriginNormMeansP1, data2binaryP1] = binaryFeature(XOriginNorm, KMax, 1, kMeansFunc, kMeansPredFunc);
[XOriginNormMeansP2, data2binaryP2] = binaryFeature(XOriginNorm, KMax, 2, kMeansFunc, kMeansPredFunc);
% 01化函数-means
XOriginNormMeansP1_01 = K201(XOriginNormMeansP1);
XOriginNormMeansP2_01 = K201(XOriginNormMeansP2);

% 特征离散化-medoids
kMedoidsTrainFunc = @(paramX, paramK) kMedoidsTrain(paramX, paramK);
kMedoidsPredFunc = @(paramX, paramCentroids) kMedoidsPred(paramX, paramCentroids);

% 离散化函数-medoids
[XOriginNormMedoidsP1, data2binaryP1] = binaryFeature(XOriginNorm, KMax, 1, kMedoidsTrainFunc, kMedoidsPredFunc);
[XOriginNormMedoidsP2, data2binaryP2] = binaryFeature(XOriginNorm, KMax, 2, kMedoidsTrainFunc, kMedoidsPredFunc);
% 01化函数-medoids
XOriginNormMedoidsP1_01 = K201(XOriginNormMedoidsP1);
XOriginNormMedoidsP2_01 = K201(XOriginNormMedoidsP2);

% 数据渲染-2
KPlot = max(XOriginNormMeansP2);
plotVec = ["r+", "b+", "g+", "k+", "y+", "c+"];
[centroids, predY, errorMin, KReal] = kMedoidsTrain(XOrigin, KPlot);

figure(2)
hold on;
for i=1:KReal
    plot(XOrigin(predY==i, 1), XOrigin(predY==i, 2), plotVec(i),'LineWidth', 1, 'MarkerSize', 5);
    plot(centroids(i, 1), centroids(i, 2), 'ko', 'MarkerSize', 7);
end
hold off;

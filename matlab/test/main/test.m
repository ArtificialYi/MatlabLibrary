%% 初始化环境
clear; close all; clc;

%% 先读取数据
data = load('resource/ex7data2.mat');
X = data.X;
maxIter = 1e8;

kMeansFunc = @(paramX, paramK) kMeansTrainRandGPU(paramX, paramK, maxIter);

[YGPU, K] = unsupervisedSplit(X, kMeansFunc);
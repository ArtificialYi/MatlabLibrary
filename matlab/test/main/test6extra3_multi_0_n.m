%% 初始化环境
clear; close all; clc;
% 开启gpu
gpuDevice(1);

%% 读取数据
% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:,1:2);
YOrigin = data(:,3);
YOrigin(YOrigin==0)=-1;
m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

% 获取核结果
l = 1;
s = 1;
p = 2;
kernelFunc = @(X1, X2) svmKernelPolynomial(X1, X2, l, s, p);
KOrigin = kernelFunc(XOriginNorm, XOriginNorm);

% 边界线数据准备
minX1 = min(XOrigin(:,1));
maxX1 = max(XOrigin(:,1));
minX2 = min(XOrigin(:,2));
maxX2 = max(XOrigin(:,2));

splitTrain = 51;
vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

%% 基础训练模型
CTrain = 1;
tolTrain = 1e-15;
maxIterTrain = 50000;
alphaTrain = zeros(m, 1);

modelOrigin = ...
    svmTrainGPU(KOrigin, YOrigin, CTrain, alphaTrain, tolTrain, maxIterTrain);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
KTestTmp = kernelFunc(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOrigin.alpha .* YOrigin)'*KTestTmp+modelOrigin.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

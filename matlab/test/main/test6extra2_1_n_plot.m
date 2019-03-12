%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;
m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 交叉验证集的大小
mTrain = size(XTrain, 1);
mVal = size(XVal, 1);

% 二分切割
classNum = 10;
maxClass = ceil(log2(classNum));
YOriginMatrix = zeros(m, maxClass);
YTrainMatrix = zeros(mTrain, maxClass);
YValMatrix = zeros(mVal, maxClass);

YOriginTmp = YOrigin;
YTrainTmp = YTrain;
YValTmp = YVal;
for i=1:maxClass
    YOriginMatrix(:,i) = mod(YOriginTmp, 2);
    YOriginTmp = (YOriginTmp - YOriginMatrix(:,i))/2;
    YTrainMatrix(:,i) = mod(YTrainTmp, 2);
    YTrainTmp = (YTrainTmp - YTrainMatrix(:,i))/2;
    YValMatrix(:,i) = mod(YValTmp, 2);
    YValTmp = (YValTmp - YValMatrix(:,i))/2;
end

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

%% 开始画图
figure(1);
colormap(gray);
plotImage(XOrigin, 20, 20, 1, 1, [-0.01, 0.01]);

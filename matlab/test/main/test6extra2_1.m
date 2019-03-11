%% 清空工作区
clear; close all; clc;

%% 读取原始数据-抽象出测试集、训练集
% 读取原始数据
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;
m = size(XOrigin, 1);

% 二值分类器
classNum = 10;
maxClass = ceil(log2(classNum));
YOriginMatrix = zeros(m, maxClass);

YOriginTmp = YOrigin;
for i=1:maxClass
    YOriginMatrix(:,i) = mod(YOriginTmp, 2);
    YOriginTmp = (YOriginTmp - YOriginMatrix(:,i))/2; 
end

trainPoint = 0.7;
valPoint = 0.3;

% 将原始数据转化为随机的三个集合
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 交叉验证集的数据
mVal = size(XVal, 1);

% 特征归一化
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);
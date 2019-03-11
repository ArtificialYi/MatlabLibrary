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

%% 特征归一化-训练模型-40组／2min 80组／5min 160组／15min 320组/30min
CTrain = 1;
tolTrain = 1e-5;
maxIterTrain = 1;
alphaTrain = zeros(m, 1);

modelOriginTmp = ...
    svmTrainGPU(XOriginNorm, YOriginMatrix(:,1), CTrain, alphaTrain, tolTrain, 1);
modelOriginMatrix = repmat(modelOriginTmp, maxClass, 1);
for i=1:maxClass
    [modelOriginMatrix(i)] = svmTrain(XOriginNorm, YOriginMatrix(:,i), CTrain, modelOriginMatrix(i).alpha, tolTrain, maxIterTrain);
    fprintf('第%d组的%d次训练完毕.\n', i, maxIterTrain);
end

%% 训练结果展示
for i=1:maxClass
    fprintf('alpha的最大值为:%f\n', max(modelOriginMatrix(i).alpha));
    fprintf('w的结果为:\n');
    fprintf('b的结果为:%f\n', modelOriginMatrix(i).b);
    fprintf('point的和为:%f\n总的错误的点数为:%d\n', sum(modelOriginMatrix(i).point), sum(modelOriginMatrix(i).point>tolTrain));
    fprintf('向量误差为：%.20f\n', modelOriginMatrix(i).error);
    fprintf('真实精度为:%.20f\n', modelOriginMatrix(i).tol);
    fprintf('浮点误差为:%.20f\n', modelOriginMatrix(i).floatError);
end

%% 归一后的数据处理
XTestTmp = [vecX1Repeat vecX2Multi];
nTestTmp = size(XTestTmp, 2);

XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, noneIndex, mu, sigma);

predYTestTmp = XTestTmpNorm*modelOrigin.w+modelOrigin.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);
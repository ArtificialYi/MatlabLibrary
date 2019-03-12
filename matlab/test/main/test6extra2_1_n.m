%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;
m = size(XOrigin, 1);

% 二分
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

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 交叉验证集的大小
mVal = size(XVal, 1);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

%% 第一次训练数据
CTrain = 1;
tolTrain = 1e-5;
maxIterTrain = 100;
alphaTrain = zeros(m, 1);
gpuNum = 1;

modelOriginTmp = ...
    svmTrainGPU(XOriginNorm, YOriginMatrix(:,1), CTrain, alphaTrain, tolTrain, 1, gpuNum);
modelOriginMatrix = repmat(modelOriginTmp, maxClass, 1);
for i=1:maxClass
    [modelOriginMatrix(i)] = svmTrainGPU(XOriginNorm, YOriginMatrix(:,i), CTrain, modelOriginMatrix(i).alpha, tolTrain, maxIterTrain, gpuNum);
    fprintf('第%d组%d次运算结束.\n', i, maxIterTrain);
end

%% 训练结果展示
for i=1:maxClass
    fprintf('alpha:%f\n', max(modelOriginMatrix(i).alpha));
    fprintf('w:\n');
    fprintf('b:%f\n', modelOriginMatrix(i).b);
    fprintf('point:%f\n错误点数:%d\n', sum(modelOriginMatrix(i).point), sum(modelOriginMatrix(i).point>tolTrain));
    fprintf('误差值%.20f\n', modelOriginMatrix(i).error);
    fprintf('精度:%.20f\n', modelOriginMatrix(i).tol);
    fprintf('精度误差:%.20f\n', modelOriginMatrix(i).floatError);
end

%% 学习曲线
CLearn = 1;
tolLearn = 1e-5;
maxIterLearn = 100;
splitLearn = 51;

% 学习曲线参数
[errorTrainLearn, errorValLearn, realSplitVecLearn] = ...
    svmLearningCurveGPU(XTrainNorm, YTrain, ...
        XValNorm, YVal, CLearn, ...
        tolLearn, maxIterLearn, splitLearn, [1 2]);
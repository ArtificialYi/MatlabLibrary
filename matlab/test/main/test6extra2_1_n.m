%% 初始化环境
clear; close all; clc;
% 开启gpu
gpuDevice(1);

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

%% 第一次训练数据
% CPU->GPU
XOriginNormGPU = gpuArray(XOriginNorm);
YOriginMatrixGPU = gpuArray(YOriginMatrix);
CTrainGPU = gpuArray(1);
alphaTrainGPU = gpuArray.zeros(m, 1);
tolTrainGPU = gpuArray(1e-5);
maxIterTrainGPU = gpuArray(100);

modelOriginTmp = ...
    gather(svmTrainGPU(XOriginNormGPU, YOriginMatrixGPU(:,1), CTrainGPU, alphaTrainGPU, tolTrainGPU, 1));
modelOriginMatrix = repmat(modelOriginTmp, maxClass, 1);
for i=1:maxClass
    [modelOriginMatrix(i)] = gather(svmTrainGPU(XOriginNormGPU, YOriginMatrixGPU(:,i), CTrainGPU, gpuArray(modelOriginMatrix(i).alpha), tolTrainGPU, maxIterTrainGPU));
    fprintf('第%d组%d次运算结束.\n', i, maxIterTrainGPU);
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
for i=1:maxClass
    [errorTrainLearn(:, i), errorValLearn(:, i), realSplitVecLearn(:, i)] = ...
        svmLearningCurveGPU(XTrainNorm, YTrainMatrix(:, i), ...
            XValNorm, YValMatrix(:, i), CLearn, ...
            tolLearn, maxIterLearn, splitLearn, [1 2]);
end

%% 保存工作区变量
save data_test6extra2_1_n.mat;
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

% CPU可用变量
modelOriginTmp = ...
    gather(svmTrainGPU(XOriginNormGPU, YOriginMatrixGPU(:,1), CTrainGPU, alphaTrainGPU, tolTrainGPU, 1));
modelOriginTmp.alpha(:) = gather(alphaTrainGPU);
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
    fprintf('point:%f\n错误点数:%d\n', sum(modelOriginMatrix(i).point), sum(modelOriginMatrix(i).point>tolTrainGPU));
    fprintf('误差值%.20f\n', modelOriginMatrix(i).error);
    fprintf('精度:%.20f\n', modelOriginMatrix(i).tol);
    fprintf('精度误差:%.20f\n', modelOriginMatrix(i).floatError);
end

%% 学习曲线
% 生成GPU变量
XTrainNormGPU = gpuArray(XTrainNorm);
YTrainMatrixGPU = gpuArray(YTrainMatrix);
XValNormGPU = gpuArray(XValNorm);
YValMatrixGPU = gpuArray(YValMatrix);
CLearnGPU = gpuArray(1);
tolLearnGPU = gpuArray(1e-5);
maxIterLearnGPU = gpuArray(100);
splitLearnGPU = gpuArray(51);

% 学习曲线参数
errorTrainLearn = zeros(splitLearnGPU, maxClass);
errorValLearn = zeros(splitLearnGPU, maxClass);
realSplitVecLearn = zeros(splitLearnGPU, maxClass);
for i=1:maxClass
    [errorTrainLearnGPU, errorValLearnGPU, realSplitVecLearnGPU] = ...
        svmLearningCurveGPU(XTrainNormGPU, YTrainMatrixGPU(:, i), ...
            XValNormGPU, YValMatrixGPU(:, i), CLearnGPU, ...
            tolLearnGPU, maxIterLearnGPU, splitLearnGPU);
    errorTrainLearn(:, i) = gather(errorTrainLearnGPU);
    errorValLearn(:, i) = gather(errorValLearnGPU);
    realSplitVecLearn(:, i) = gather(realSplitVecLearnGPU);
end

%% 保存工作区变量
save data_test6extra2_1_n.mat;
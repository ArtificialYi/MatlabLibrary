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
% CPU->GPU
KOriginGPU = gpuArray(KOrigin);
YOriginGPU = gpuArray(YOrigin);
CTrainGPU = gpuArray(1);
tolTrainGPU = gpuArray(1e-15);
maxIterTrainGPU = gpuArray(50000);
alphaTrainGPU = gpuArray.zeros(m, 1);

modelOriginGPU = ...
    svmTrainGPU(KOriginGPU, YOriginGPU, CTrainGPU, alphaTrainGPU, tolTrainGPU, maxIterTrainGPU);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
KTestTmp = kernelFunc(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOriginGPU.cpu.alpha .* YOrigin)'*KTestTmp+modelOriginGPU.cpu.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线训练
%CPU->GPU
XTrainNormGPU = gpuArray(XTrainNorm);
YTrainGPU = gpuArray(YTrain);
XValNormGPU = gpuArray(XValNorm);
YValGPU = gpuArray(YVal);
CLearnGPU = gpuArray(3.26);
tolLearnGPU = gpuArray(1e-15);
maxIterLearnGPU = gpuArray(50000);
splitLearnGPU = gpuArray(51);

[errorTrainLearnGPU, errorValLearnGPU, realSplitVecLearnGPU] = ...
    svmLearningCurveGPU(XTrainNormGPU, YTrainGPU, ...
        XValNormGPU, YValGPU, CLearnGPU, ...
        tolLearnGPU, maxIterLearnGPU, splitLearnGPU, kernelFunc);

%% 尝试找到最优C
% CPU->GPU
splitCCurrentGPU = gpuArray(11);
predCurrentGPU = gpuArray(1e-3);
CLeftCurrentGPU = gpuArray(1e-6); % 精度的一半
CRightCurrentGPU = gpuArray(1e4);
tolCurrentGPU = gpuArray(1e-15);
maxIterCurrentGPU = gpuArray(50000);

KTrainGPU = kernelFunc(XTrainNormGPU, XTrainNormGPU);
KValGPU = kernelFunc(XTrainNormGPU, XValNormGPU);

% 先用等比数列找到最优数值
CVecCurrentGPU = logspace(log10(CLeftCurrentGPU), log10(CRightCurrentGPU), splitCCurrentGPU);
[errorTrainCurrentTmp, errorValCurrentTmp] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU, tolCurrentGPU, maxIterCurrentGPU);
indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU);
if length(indexCurrentGPU) > 1
    indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
end
[indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
    getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);

% 再开始用等差数列做循环
while CRightCurrentGPU - CLeftCurrentGPU > predCurrentGPU
    CVecCurrentGPU = linspace(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU);
    [errorTrainCurrentTmpGPU, errorValCurrentTmpGPU] = ...
        svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU, tolCurrentGPU, maxIterCurrentGPU);
    indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU);
    if length(indexCurrentGPU) > 1
        indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
    end
    [indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
        getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
    CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
    CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);
end

% 将当前最优C打印出来
CCurrentGPU = CVecCurrentGPU(indexCurrentGPU);
errorMinCurrentGPU = errorValCurrentTmpGPU(indexCurrentGPU);
fprintf('当前最优C是:%.15f\n', CCurrentGPU);
fprintf('当前最小误差是:%.15f\n', errorMinCurrent);

%% 变量存储
% 训练结果预测
modelOriginCpuRes = modelOriginGPU.cpu;
% 学习曲线
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitVecLearn = gather(realSplitVecLearnGPU);
% 当前最优C
CCurrent = gather(CCurrentGPU);
errorMinCurrent = gather(errorMinCurrentGPU);

save data/data_test6extra3_multi_0_n.mat ...
    XOrigin YOrigin vecX1 vecX2 predYTestTmp_2D ...
    realSplitVecLearn errorTrainLearn errorValLearn ...
    XTrain YTrain XVal YVal ...
    CCurrent errorMinCurrent;
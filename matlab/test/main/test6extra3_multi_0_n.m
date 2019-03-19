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
% 计算最优C
splitCCurrent = 11;
predCurrent = 1e-3;
CLeftCurrent = 1e-6; % 精度的一半
CRightCurrent = 1e4;
tolCurrent = 1e-15;
maxIterCurrent = 50000;

KTrain = kernelFunc(XTrainNorm, XTrainNorm);
KVal = kernelFunc(XTrainNorm, XValNorm);

% 先用等比数列找到最优数值
CVecCurrent = logspace(log10(CLeftCurrent), log10(CRightCurrent), splitCCurrent);
[errorTrainCurrentTmp, errorValCurrentTmp] = ...
    svmTrainForCVec(KTrain, YTrain, KVal, YVal, CVecCurrent, tolCurrent, maxIterCurrent);
indexCurrent = indexMinForVec(errorValCurrentTmp);
if length(indexCurrent) > 1
    indexCurrent = indexCurrent(length(indexCurrent));
end
[indexCurrentLeftTmp, indexCurrentRightTmp] = ...
    getLeftAndRightIndex(indexCurrent, 1, splitCCurrent);
CLeftCurrent = CVecCurrent(indexCurrentLeftTmp);
CRightCurrent = CVecCurrent(indexCurrentRightTmp);

% 再开始用等差数列做循环
while CRightCurrent - CLeftCurrent > predCurrent
    CVecCurrent = linspace(CLeftCurrent, CRightCurrent, splitCCurrent);
    [errorTrainCurrentTmp, errorValCurrentTmp] = ...
        svmTrainForCVec(KTrain, YTrain, KVal, YVal, CVecCurrent, tolCurrent, maxIterCurrent);
    indexCurrent = indexMinForVec(errorValCurrentTmp);
    if length(indexCurrent) > 1
        indexCurrent = indexCurrent(length(indexCurrent));
    end
    [indexCurrentLeftTmp, indexCurrentRightTmp] = ...
        getLeftAndRightIndex(indexCurrent, 1, splitCCurrent);
    CLeftCurrent = CVecCurrent(indexCurrentLeftTmp);
    CRightCurrent = CVecCurrent(indexCurrentRightTmp);
end

% 将当前最优C打印出来
CCurrent = CVecCurrent(indexCurrent);
errorMinCurrent = errorValCurrentTmp(indexCurrent);
fprintf('当前最优C是:%.15f\n', CCurrent);
fprintf('当前最小误差是:%.15f\n', errorMinCurrent);

%% 变量存储
% 训练结果预测
modelOriginCpuRes = modelOriginGPU.cpu;
% 学习曲线
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitVecLearn = gather(realSplitVecLearnGPU);

save data/data_test6extra3_multi_0_n.mat ...
    XOrigin YOrigin vecX1 vecX2 predYTestTmp_2D ...
    realSplitVecLearn errorTrainLearn errorValLearn ...
    XTrain YTrain XVal YVal;
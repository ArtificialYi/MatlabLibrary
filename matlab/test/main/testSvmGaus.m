function [outputArg1,outputArg2] = testSvmGaus(C, gu, guLeft, guRight, maxIter, isTrain)
%testComp 比赛用的函数

% 初始化数据
gu = str2double(gu);
C = str2double(C);
maxIter = str2double(maxIter);
guLeft = str2double(guLeft);
guRight = str2double(guRight);
isTrain = str2double(isTrain);

tol = 1e-8;

%% 先读取数据
data = load('resource/pfm_data.mat');

% 获取原始数据
XOrigin = data.XOrigin;
YOrigin = data.YOrigin;
YOrigin(YOrigin==0)=-1;
XTestOrigin = data.XTest;

%% 开始SVM基础
[mOrigin, nOrigin] = size(XOrigin);

trainPoint = 0.7;
valPoint = 0.3;
pred = 1e-16;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(mOrigin);
[XTrainSplit, XValSplit, ~] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);
[YTrain, YVal, ~] = ...
    splitData(YOrigin, indexVecRand, trainPoint, valPoint);

%% 特征扩充
% 将所有枚举型特征扩充为2进制特征
lenMax = 30;
[XTrainBinary, data2binaryFunc] = binaryFeature(XTrainSplit, lenMax);
XOriginBinary = data2binaryFunc(XOrigin);
XValBinary = data2binaryFunc(XValSplit);
XTestBinary = data2binaryFunc(XTestOrigin);

%% 最终计算数据准备
XOrigin = XOriginBinary;
XTrain = XTrainBinary;
XVal = XValBinary;
XTest = XTestBinary;

%% 获取基础数据
mTrain = size(XTrain, 1);
mVal = size(XVal, 1);
mTest = size(XTest, 1);

% 归一化数据
[XTrainNorm, data2normFunc] = data2featureWithNormalize(XTrain, 1);
nTrain = size(XTrainNorm, 2);
XOriginNorm = data2normFunc(XOrigin);
XValNorm = data2normFunc(XVal);
XTestNorm = data2normFunc(XTest);

%% GPU数据
% 归一化数据
XOriginNormGPU = gpuArray(XOriginNorm);
XTrainNormGPU = gpuArray(XTrainNorm);
XValNormGPU = gpuArray(XValNorm);
XTestNormGPU = gpuArray(XTestNorm);
nTrainGPU = gpuArray(nTrain);
predGPU = gpuArray(pred);

% 真实数据
XOriginNormRealGPU = [ones(mOrigin, 1) XOriginNormGPU];
XTrainNormRealGPU = [ones(mTrain, 1) XTrainNormGPU];
XValNormRealGPU = [ones(mVal, 1) XValNormGPU];
XTestNormRealGPU = [ones(mTest, 1) XTestNormGPU];

% 结果数据
YOriginGPU = gpuArray(YOrigin);
YTrainGPU = gpuArray(YTrain);
YValGPU = gpuArray(YVal);

% 默认参数
%lambdaGPU = gpuArray(lambda);
maxIterGPU = gpuArray(maxIter);
thetaInitGPU = gpuArray.zeros(nTrainGPU+1, 1);

% 学习曲线
splitLearningCurve = 50;
splitLearningCurveGPU = gpuArray(splitLearningCurve);

%% 基础训练模型
% 获取核结果
kernelFunc = @(X1, X2) svmKernelGaussian(X1, X2, gu);

KOriginGPU = kernelFunc(XOriginNormGPU, XOriginNormGPU);
CTrainGPU = gpuArray(C);
alphaTrainGPU = gpuArray.zeros(mTrain, 1);
tolTrainGPU = gpuArray(tol);
maxIterTrainGPU = gpuArray(maxIter);

modelOriginGPU = ...
    svmTrainGPU(KOriginGPU, YOriginGPU, CTrainGPU, alphaTrainGPU, tolTrainGPU, maxIterTrainGPU);

% 原始模型结果
KOrigin = gather(KOriginGPU);
predYOrigin = (modelOriginGPU.cpu.alpha .* YOrigin)'*KOrigin'+modelOriginGPU.cpu.b;

% 测试集预测
KTestOriginGPU = kernelFunc(XOriginNormGPU, XTestNormGPU);
KTestOrigin = gather(KTestOriginGPU);
predYTest = (modelOriginGPU.cpu.alpha .* YOrigin)'*KTestOrigin'+modelOriginGPU.cpu.b;

% 训练的模型结果
modelOriginCPU = modelOriginGPU.cpu;

%% save
% 获取文件名
fileName = sprintf('data/data_testSvmGaus_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'XTest', ...
    'YOrigin', 'YTrain', 'YVal', 'predYOrigin', 'predYTest', 'modelOriginCPU');
fprintf('保存完毕\n');
end
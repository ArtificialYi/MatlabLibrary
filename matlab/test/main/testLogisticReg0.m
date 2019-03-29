function [logisticRes] = testLogisticReg0(maxIter)
%testLogisticReg0 逻辑回归测试函数

%% str2double
maxIter = str2double(maxIter);

%% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:, 1:2);
YOrigin = data(:, 3);

[m, n] = size(XOrigin);

trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, XVal, XTest] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);
[YTrain, YVal, YTest] = ...
    splitData(YOrigin, indexVecRand, trainPoint, valPoint);

% 获取数据
mTrain = size(XTrain, 1);
mVal = size(XVal, 1);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

%% GPU数据准备
% 归一化数据
XOriginNormGPU = gpuArray(XOriginNorm);
XTrainNormGPU = gpuArray(XTrainNorm);
XValNormGPU = gpuArray(XValNorm);
nGPU = gpuArray(n);

% 真实数据
XOriginNormRealGPU = [ones(m, 1) XOriginNormGPU];
XTrainNormRealGPU = [ones(mTrain, 1) XTrainNormGPU];
XValNormRealGPU = [ones(mVal, 1) XValNormGPU];

% 结果数据
YOriginGPU = gpuArray(YOrigin);
YTrainGPU = gpuArray(YTrain);
YValGPU = gpuArray(YVal);

% 默认参数
thetaInitGPU = gpuArray.zeros(n+1, 1);
maxIterGPU = gpuArray(maxIter);

% 学习曲线
splitLearningCurveGPU = gpuArray(50);

%% pca提取
[UTrainGPU, STrainGPU] = pcaTrainGPU(XTrainNormGPU);

% pca-gpu
XOriginNormPcaGPU = data2pca(XOriginNormGPU, UTrainGPU, nGPU);
XTrainNormPcaGPU = data2pca(XTrainNormGPU, UTrainGPU, nGPU);
XValNormPcaGPU = data2pca(XValNormGPU, UTrainGPU, nGPU);

% pca-cpu
XOriginNormPca = gather(XOriginNormPcaGPU);
XTrainNormPca = gather(XTrainNormPcaGPU);
XValNormPca = gather(XValNormPcaGPU);

% 真实数据
XOriginNormPcaRealGPU = [ones(m, 1) XOriginNormPcaGPU];
XTrainNormPcaRealGPU = [ones(mTrain, 1) XTrainNormPcaGPU];
XValNormPcaRealGPU = [ones(mVal, 1) XValNormPcaGPU];

%% 边界线数据准备
splitTrain = 101;
% pca
minX1Pca = min(XOriginNormPca(:,1));
maxX1Pca = max(XOriginNormPca(:,1));
minX2Pca = min(XOriginNormPca(:,2));
maxX2Pca = max(XOriginNormPca(:,2));

vecX1Pca = linspace(minX1Pca, maxX1Pca, splitTrain)';
vecX2Pca = linspace(minX2Pca, maxX2Pca, splitTrain)';
vecX1RepeatPca = repeatMatrix(vecX1Pca, splitTrain);
vecX2MultiPca = multiMatrix(vecX2Pca, splitTrain);

% pca-训练结果预测
XTestTmpPca = [vecX1RepeatPca vecX2MultiPca];
mTestTmpPca = size(XTestTmpPca, 1);
XTestTmpPcaGPU = gpuArray(XTestTmpPca);
XTestTmpPcaRealGPU = [ones(mTestTmpPca, 1) XTestTmpPcaGPU];

% data
minX1 = min(XOrigin(:,1));
maxX1 = max(XOrigin(:,1));
minX2 = min(XOrigin(:,2));
maxX2 = max(XOrigin(:,2));

vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

% data结果预测
XDataTmp = [vecX1Repeat vecX2Multi];
mDataTmp = size(XDataTmp, 1);
XDataTmpNorm = ...
    mapFeatureWithParam(XDataTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XDataTmpNormGPU = gpuArray(XDataTmpNorm);
XDataTmpNormPcaGPU = data2pca(XDataTmpNormGPU, UTrainGPU, nGPU);
XDataTmpNormPcaRealGPU = [ones(mDataTmp, 1) XDataTmpNormPcaGPU];

%% 基础训练模型
[thetaOriginGPU, ~] = ...
    logisticRegTrainGPU(XOriginNormPcaRealGPU, YOriginGPU, thetaInitGPU, maxIterGPU);

% pca预测-预测结果
predYPcaTmpGPU = logisticHypothesis(XTestTmpPcaRealGPU, thetaOriginGPU);
predYPcaTmp = gather(predYPcaTmpGPU);
predYPcaTmp_2D = reshape(predYPcaTmp, splitTrain, splitTrain);

% pca2data预测
predYDataTmpGPU = logisticHypothesis(XDataTmpNormPcaRealGPU, thetaOriginGPU);
predYDataTmp = gather(predYDataTmpGPU);
predYDataTmp_2D = reshape(predYDataTmp, splitTrain, splitTrain);

%% 学习曲线
[errorTrainGPU, errorValGPU, realSplitVecGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormPcaRealGPU, YTrainGPU, XValNormPcaRealGPU, YValGPU, ...
        thetaInitGPU, maxIterGPU, splitLearningCurveGPU);

% 画图
errorTrain = gather(errorTrainGPU);
errorVal = gather(errorValGPU);
realSplitVec = gather(realSplitVecGPU);
    
%% save
% 获取文件名
fileName = sprintf('data/data_testLogisticReg0_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'XTest', ...
    'YOrigin', 'YTrain', 'YVal', 'YTest', ...
    'XOriginNormPca', 'XTrainNormPca', 'XValNormPca', ...
    'vecX1Pca', 'vecX2Pca', 'predYPcaTmp_2D', ...
    'vecX1', 'vecX2', 'predYDataTmp_2D', ...
    'errorTrain', 'errorVal', 'realSplitVec');
fprintf('保存完毕\n');
end


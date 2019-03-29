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

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

% 边界线数据准备
minX1 = min(XOrigin(:,1));
maxX1 = max(XOrigin(:,1));
minX2 = min(XOrigin(:,2));
maxX2 = max(XOrigin(:,2));

splitTrain = 101;
vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);

%% GPU数据准备
% pca
XOriginNormGPU = gpuArray(XOriginNorm);
XTrainNormGPU = gpuArray(XTrainNorm);
XValNormGPU = gpuArray(XValNorm);
nGPU = gpuArray(n);

% 基础训练模型
XOriginNormRealGPU = [ones(m, 1) XOriginNormGPU];
YOriginGPU = gpuArray(YOrigin);
thetaInitGPU = gpuArray.zeros(n+1, 1);
maxIterGPU = gpuArray(maxIter);
XTestTmpNormGPU = gpuArray(XTestTmpNorm);

% 学习曲线
YTrainGPU = gpuArray(YTrain);
YValGPU = gpuArray(YVal);
splitLearningCurveGPU = gpuArray(50);

%% pca提取
[UTrainGPU, STrainGPU] = pcaTrainGPU(XTrainNormGPU);

XOriginNormPcaGPU = data2pca(XOriginNormGPU, UTrainGPU, nGPU);
XTrainNormPcaGPU = data2pca(XTrainNormGPU, UTrainGPU, nGPU);
XValNormPcaGPU = data2pca(XValNormGPU, UTrainGPU, nGPU);

XOriginNormPca = gather(XOriginNormPcaGPU);
XTrainNormPca = gather(XTrainNormPcaGPU);
XValNormPca = gather(XValNormPcaGPU);

%% 基础训练模型
[thetaOriginGPU, costOriginGPU] = ...
    logisticRegTrainGPU(XOriginNormRealGPU, YOriginGPU, thetaInitGPU, maxIterGPU);

% 预测结果
predYTestTmpGPU = logisticHypothesis(XTestTmpNormGPU, thetaOriginGPU);
predYTestTmp = gather(predYTestTmpGPU);
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线
[errorTrainGPU, errorValGPU, realSplitVecGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormGPU, YTrainGPU, XValNormGPU, YValGPU, ...
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
    'XOrigin', 'XTrain', 'XVal', 'YOrigin', 'YTrain', 'YVal', 'YTest', ...
    'XOriginNormPca', 'XTrainNormPca', 'XValNormPca', ...
    'vecX1', 'vecX2', 'predYTestTmp_2D', 'errorTrain', 'errorVal', 'realSplitVec');
fprintf('保存完毕\n');
end


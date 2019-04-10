function [outputArg1,outputArg2] = testComp(p, lambda, maxIter)
%testComp 比赛用的函数

%% str2double
maxIter = str2double(maxIter);
p = str2double(p);
lambda = str2double(lambda);
%pLeft = str2double(pLeft);
%pRight = str2double(pRight);

%% 先读取数据
data = load('resource/pfm_data.mat');

% 获取原始数据
XOrigin = data.XOrigin;
YOrigin = data.YOrigin;
XTest = data.XTest;

[mOrigin, nOrigin] = size(XOrigin);

trainPoint = 0.7;
valPoint = 0.3;
pred = 1e-16;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(mOrigin);
[XTrain, XVal, ~] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);
[YTrain, YVal, ~] = ...
    splitData(YOrigin, indexVecRand, trainPoint, valPoint);

% 获取基础数据
mTrain = size(XTrain, 1);
mVal = size(XVal, 1);
mTest = size(XTest, 1);

% 归一化数据
[XTrainNorm, data2normFunc] = data2featureWithNormalize(XTrain, p);
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
thetaInitGPU = gpuArray.zeros(nTrain+1, 1);
lambdaGPU = gpuArray(lambda);
maxIterGPU = gpuArray(maxIter);

% 学习曲线
splitLearningCurve = 50;
splitLearningCurveGPU = gpuArray(splitLearningCurve);

%% pca提取
[UTrainGPU, STrainGPU] = pcaTrainGPU(XTrainNormGPU);

% pca-gpu
XOriginNormPcaGPU = data2pca(XOriginNormGPU, UTrainGPU, nGPU);
XTrainNormPcaGPU = data2pca(XTrainNormGPU, UTrainGPU, nGPU);
XValNormPcaGPU = data2pca(XValNormGPU, UTrainGPU, nGPU);
XTestNormPcaGPU = data2pca(XTestNormGPU, UTrainGPU, nGPU);

% pca-cpu
XOriginNormPca = gather(XOriginNormPcaGPU);
XTrainNormPca = gather(XTrainNormPcaGPU);
XValNormPca = gather(XValNormPcaGPU);
XTestNormPca = gather(XTestNormPcaGPU);

% 真实数据
XOriginNormPcaRealGPU = [ones(m, 1) XOriginNormPcaGPU];
XTrainNormPcaRealGPU = [ones(mTrain, 1) XTrainNormPcaGPU];
XValNormPcaRealGPU = [ones(mVal, 1) XValNormPcaGPU];
XTestNormPcaRealGPU = [ones(mTest, 1) XTestNormPcaGPU];

% pcaVec
pcaVecGPU = diag(STrainGPU);
pcaSumVecGPU = pcaVecGPU;
for i=2:length(pcaSumVecGPU)
    pcaSumVecGPU(i) = pcaSumVecGPU(i-1)+pcaVecGPU(i);
end

pcaVec = gather(pcaVecGPU);
pcaSumVec = gather(pcaSumVecGPU);

%% 基础训练模型
[thetaOriginGPU, ~] = ...
    logisticRegTrainGPU(XOriginNormPcaRealGPU, YOriginGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU);

% 测试集预测
predYTestGPU = logisticHypothesis(XTestNormPcaRealGPU, thetaOriginGPU, predGPU);
predYTest = gather(predYTestGPU);

%% 学习曲线
[errorTrainLearnGPU, errorValLearnGPU, realSplitLearnVecGPU, thetaMatrixLearnGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormPcaRealGPU, YTrainGPU, XValNormPcaRealGPU, YValGPU, ...
        thetaInitGPU, lambdaGPU, maxIterGPU, predGPU, splitLearningCurveGPU);
% 学习曲线的结果
predYLearnTmpGPU = logisticHypothesis(XDataTmpNormPcaRealGPU, thetaMatrixLearnGPU, predGPU);
predYLearnDataTmp = gather(predYLearnTmpGPU);
showHy(predYLearnDataTmp, 'predYLearnDataTmp');
showHy(splitTrain, 'splitTrain');
showHy(splitLearningCurve, 'splitLearningCurve');
predYLearnDataTmp_3D = reshape(predYLearnDataTmp, splitTrain, splitTrain, splitLearningCurve);

% 画图
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitLearnVec = gather(realSplitLearnVecGPU);
thetaMatrixLearn = gather(thetaMatrixLearnGPU);
    
    
%% save
% 获取文件名
fileName = sprintf('data/data_testComp_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'XTest', ...
    'YOrigin', 'YTrain', 'YVal', ...
    'pcaVec', 'pcaSumVec', 'predYTest', ...
    'errorTrainLearn', 'errorValLearn', 'realSplitLearnVec', 'predYLearnDataTmp_3D');
fprintf('保存完毕\n');
end


function [logisticRes] = testLogisticReg0(p, lambda, pLeft, pRight, maxIter)
%testLogisticReg0 逻辑回归测试函数

%% str2double
maxIter = str2double(maxIter);
p = str2double(p);
lambda = str2double(lambda);
pLeft = str2double(pLeft);
pRight = str2double(pRight);

%% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:, 1:2);
YOrigin = data(:, 3);

m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;
pred = 1e-16;

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
[XTrainNorm, data2normFunc] = data2featureWithNormalize(XTrain, p);
n = size(XTrainNorm, 2);
XOriginNorm = data2normFunc(XOrigin);
XValNorm = data2normFunc(XVal);

%% GPU数据准备
% 归一化数据
XOriginNormGPU = gpuArray(XOriginNorm);
XTrainNormGPU = gpuArray(XTrainNorm);
XValNormGPU = gpuArray(XValNorm);
nGPU = gpuArray(n);
predGPU = gpuArray(pred);

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

% pca-cpu
XOriginNormPca = gather(XOriginNormPcaGPU);
XTrainNormPca = gather(XTrainNormPcaGPU);
XValNormPca = gather(XValNormPcaGPU);

% 真实数据
XOriginNormPcaRealGPU = [ones(m, 1) XOriginNormPcaGPU];
XTrainNormPcaRealGPU = [ones(mTrain, 1) XTrainNormPcaGPU];
XValNormPcaRealGPU = [ones(mVal, 1) XValNormPcaGPU];

% pcaVec
pcaVecGPU = diag(STrainGPU);
pcaSumVecGPU = pcaVecGPU;
for i=2:length(pcaSumVecGPU)
    pcaSumVecGPU(i) = pcaSumVecGPU(i-1)+pcaVecGPU(i);
end

pcaVec = gather(pcaVecGPU);
pcaSumVec = gather(pcaSumVecGPU);

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
dataExtra = gpuArray.zeros(mTestTmpPca, n-2);
XTestTmpPcaGPU = gpuArray(XTestTmpPca);
XTestTmpPcaRealGPU = [ones(mTestTmpPca, 1) XTestTmpPcaGPU dataExtra];

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
XDataTmpNorm = data2normFunc(XDataTmp);
XDataTmpNormGPU = gpuArray(XDataTmpNorm);
XDataTmpNormPcaGPU = data2pca(XDataTmpNormGPU, UTrainGPU, nGPU);
XDataTmpNormPcaRealGPU = [ones(mDataTmp, 1) XDataTmpNormPcaGPU];

%% 基础训练模型
[thetaOriginGPU, ~] = ...
    logisticRegTrainGPU(XOriginNormPcaRealGPU, YOriginGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU);

% pca预测-预测结果
predYPcaTmpGPU = logisticHypothesis(XTestTmpPcaRealGPU, thetaOriginGPU, predGPU);
predYPcaTmp = gather(predYPcaTmpGPU);
predYPcaTmp_2D = reshape(predYPcaTmp, splitTrain, splitTrain);

% pca2data预测
predYDataTmpGPU = logisticHypothesis(XDataTmpNormPcaRealGPU, thetaOriginGPU, predGPU);
predYDataTmp = gather(predYDataTmpGPU);
predYDataTmp_2D = reshape(predYDataTmp, splitTrain, splitTrain);

%% 学习曲线
[errorTrainLearnGPU, errorValLearnGPU, realSplitLearnVecGPU, thetaMatrixLearnGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormPcaRealGPU, YTrainGPU, XValNormPcaRealGPU, YValGPU, ...
        thetaInitGPU, lambdaGPU, maxIterGPU, predGPU, splitLearningCurveGPU);
% 学习曲线的结果
predYLearnTmpGPU = logisticHypothesis(XDataTmpNormPcaRealGPU, thetaMatrixLearnGPU, predGPU);
predYLearnDataTmp = gather(predYLearnTmpGPU);
predYLearnDataTmp_3D = reshape(predYLearnDataTmp, splitTrain, splitTrain, splitLearningCurve);

% 画图
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitLearnVec = gather(realSplitLearnVecGPU);
thetaMatrixLearn = gather(thetaMatrixLearnGPU);

%% 最优化
pVec = pLeft:pRight;
predLambdaGPU = gpuArray(1e-3);

pErrorVecGPU = gpuArray(pVec);
pLambdaVecGPU = gpuArray(pVec);

for i=1:length(pVec)
    fprintf('开始多项式最优化:%d\n', pVec(i));
    % 多项式&归一化数据
    [XTrainNormTmp, data2normFunc] = data2featureWithNormalize(XTrain, pVec(i));
    nTmp = size(XTrainNormTmp, 2);
    XValNormTmp = data2normFunc(XVal);
    
    % 转GPU
    XTrainNormTmpGPU = gpuArray(XTrainNormTmp);
    XValNormTmpGPU = gpuArray(XValNormTmp);
    nTmpGPU = gpuArray(nTmp);
    
    % pca化
    [UTrainTmpGPU, ~] = pcaTrainGPU(XTrainNormTmpGPU);
    XTrainNormTmpPcaGPU = data2pca(XTrainNormTmpGPU, UTrainTmpGPU, nTmpGPU);
    XValNormTmpPcaGPU = data2pca(XValNormTmpGPU, UTrainTmpGPU, nTmpGPU);
    
    % 添加常量数据
    XTrainNormTmpPcaRealGPU = [ones(mTrain, 1) XTrainNormTmpPcaGPU];
    XValNormTmpPcaRealGPU = [ones(mVal, 1) XValNormTmpPcaGPU];
    thetaInitTmpGPU = gpuArray.zeros(nTmpGPU+1, 1);
    
    % 开始计算
    [lambdaCurrentGPU, errorCurrentGPU] = ...
        logisticRegFindCurrentMinLambda(XTrainNormTmpPcaRealGPU, YTrainGPU, ...
        XValNormTmpPcaRealGPU, YValGPU, ...
        thetaInitTmpGPU, maxIterGPU, predGPU, predLambdaGPU);
    
    % 储存结果
    pLambdaVecGPU(i) = lambdaCurrentGPU;
    pErrorVecGPU(i) = errorCurrentGPU;
end
% 最小值所在索引
indexMinVecGPU = indexMinForMulti(pErrorVecGPU);
indexMinGPU = indexMinVecGPU(1);

% 最优解
lambdaMinGPU = pLambdaVecGPU(indexMinGPU);
errorMinGPU = pErrorVecGPU(indexMinGPU);

% 所有解-CPU
pLambdaVec = gather(pLambdaVecGPU);
pErrorVec = gather(pErrorVecGPU);

% 最优解-CPU
lambdaMin = gather(lambdaMinGPU);
errorMin = gather(errorMinGPU);
pMin = pVec(gather(indexMinGPU));

%% save
% 获取文件名
fileName = sprintf('data/data_testLogisticReg0_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'XTest', ...
    'YOrigin', 'YTrain', 'YVal', 'YTest', ...
    'XOriginNormPca', 'XTrainNormPca', 'XValNormPca', ...
    'pcaVec', 'pcaSumVec', ...
    'vecX1Pca', 'vecX2Pca', 'predYPcaTmp_2D', ...
    'vecX1', 'vecX2', 'predYDataTmp_2D', ...
    'errorTrainLearn', 'errorValLearn', 'realSplitLearnVec', 'predYLearnDataTmp_3D', ...
    'lambdaMin', 'errorMin', 'pMin', 'pLambdaVec', 'pErrorVec');
fprintf('保存完毕\n');
end


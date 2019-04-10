function [logisticRes] = testLogisticReg0(p, lambda, pLeft, pRight, maxIter)
%testLogisticReg0 逻辑回归测试函数

%% str2double
maxIter = str2double(maxIter);
p = str2double(p);
lambda = str2double(lambda);
pLeft = str2double(pLeft);
pRight = str2double(pRight);

%% 读取数据
data = load('resource/pfm_data.mat');
%%
XOrigin = data.XOrigin;
YOrigin = data.YOrigin;
XTest = data.XTest;

[m, nOrigin] = size(XOrigin);

trainPoint = 0.7;
valPoint = 0.3;
pred = 1e-16;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, XVal, ~] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);
[YTrain, YVal, ~] = ...
    splitData(YOrigin, indexVecRand, trainPoint, valPoint);

% 获取数据
mTrain = size(XTrain, 1);
mVal = size(XVal, 1);
mTest = size(XTest, 1);

% 归一化数据
[XTrainNorm, data2normFunc] = data2featureWithNormalize(XTrain, p);
n = size(XTrainNorm, 2);
XOriginNorm = data2normFunc(XOrigin);
XValNorm = data2normFunc(XVal);
XTestNorm = data2normFunc(XTest);

%% GPU数据准备
% 归一化数据
XOriginNormGPU = gpuArray(XOriginNorm);
XTrainNormGPU = gpuArray(XTrainNorm);
XValNormGPU = gpuArray(XValNorm);
XTestNormGPU = gpuArray(XTestNorm);
nGPU = gpuArray(n);
predGPU = gpuArray(pred);

% 真实数据
XOriginNormRealGPU = [ones(m, 1) XOriginNormGPU];
XTrainNormRealGPU = [ones(mTrain, 1) XTrainNormGPU];
XValNormRealGPU = [ones(mVal, 1) XValNormGPU];
XTestNormRealGPU = [ones(mTest, 1) XTestNormGPU];

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

%% 边界线数据准备
splitTrain = 101;
%% pca边界
minXPca = min(XOriginNormPca(:, 1:min(end,2)));
maxXPca = max(XOriginNormPca(:, 1:min(end,2)));

lenDataPca = length(minXPca);
splitTrainPcaVec = zeros(1, lenDataPca)+splitTrain;
matrixXPcaGPU = gpuArray.zeros(splitTrain, lenDataPca);

% 初始化轴向量
for i=1:lenDataPca
    matrixXPcaGPU(:, i) = linspace(minXPca(i), maxXPca(i), splitTrain)';
end

% 初始化结果集
mTestTmpPca = splitTrain^lenDataPca;
XTestTmpPcaRealGPU = gpuArray.ones(mTestTmpPca, n+1);
XTestTmpPcaRealGPU(1:splitTrain, 2) = matrixXPcaGPU(:, 1);
for i=2:lenDataPca
    XTestTmpPcaRealGPU(1:splitTrain^i, 2:i+1) = [
        repeatMatrix(XTestTmpPcaRealGPU(1:splitTrain^(i-1), 2:i), splitTrain) ...
        multiMatrix(matrixXPcaGPU(:,i), splitTrain^(i-1)) ...
        ];
end
matrixXPca = gather(matrixXPcaGPU);

%% data边界
minX = min(XOrigin(:, 1:min(end,2)));
maxX = max(XOrigin(:, 1:min(end,2)));

lenData = size(minX, 2);
splitTrainDataVec = zeros(1, lenData)+splitTrain;
matrixXGPU = gpuArray.zeros(splitTrain, lenData);

% 初始化轴向量
for i=1:lenData
    matrixXGPU(:, i) = linspace(minX(i), maxX(i), splitTrain)';
end

% 初始化结果集
mDataTmp = splitTrain^lenData;
XDataTmpNormPcaRealGPU = gpuArray.ones(mDataTmp, n+1);
XDataTmpNormPcaRealGPU(1:splitTrain, 2) = matrixXGPU(:, 1);
for i=2:lenData
    XDataTmpNormPcaRealGPU(1:splitTrain^i, 2:i+1) = [
        repeatMatrix(XDataTmpNormPcaRealGPU(1:splitTrain^(i-1), 2:i), splitTrain) ...
        multiMatrix(matrixXGPU(:,i), splitTrain^(i-1)) ...
    ];
end
% 多项式&特征归一
% 获取data的原始数据集
XDataTmpNormPcaRealGPU(:, 2:end) = data2normFunc([XDataTmpNormPcaRealGPU(:, 2:lenData+1) ones(mDataTmp, nOrigin-lenData)]);

% 转pca
XDataTmpNormPcaRealGPU(:,2:end) = data2pca(XDataTmpNormPcaRealGPU(:,2:end), UTrainGPU, nGPU);
matrixX = gather(matrixXGPU);

%% 基础训练模型
[thetaOriginGPU, ~] = ...
    logisticRegTrainGPU(XOriginNormPcaRealGPU, YOriginGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU);

% pca预测-预测结果
predYPcaTmpGPU = logisticHypothesis(XTestTmpPcaRealGPU, thetaOriginGPU, predGPU);
predYPcaTmp = gather(predYPcaTmpGPU);
predYPcaTmp_DMulti = reshape(predYPcaTmp, splitTrainPcaVec);

% pca2data预测
predYDataTmpGPU = logisticHypothesis(XDataTmpNormPcaRealGPU, thetaOriginGPU, predGPU);
predYDataTmp = gather(predYDataTmpGPU);
predYDataTmp_DMulti = reshape(predYDataTmp, splitTrainDataVec);

% 测试集预测
predYTestGPU = logisticHypothesis(XTestNormPcaRealGPU, thetaOriginGPU, predGPU);

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
    'XOriginNormPca', 'XTrainNormPca', 'XValNormPca', 'XTestNormPca', ...
    'pcaVec', 'pcaSumVec', ...
    'matrixXPca', 'predYPcaTmp_DMulti', 'matrixX', 'predYDataTmp_DMulti', 'predYTestGPU', ...
    'errorTrainLearn', 'errorValLearn', 'realSplitLearnVec', 'predYLearnDataTmp_3D', ...
    'lambdaMin', 'errorMin', 'pMin', 'pLambdaVec', 'pErrorVec');
fprintf('保存完毕\n');
end


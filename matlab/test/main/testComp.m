function [outputArg1,outputArg2] = testComp(p, lambda, K, pLeft, pRight, maxIter)
%testComp 比赛用的函数

%% str2double
maxIter = str2double(maxIter);
p = str2double(p);
lambda = str2double(lambda);
K = str2double(K);
pLeft = str2double(pLeft);
pRight = str2double(pRight);

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
KGPU = gpuArray(K);

% 学习曲线
splitLearningCurve = 50;
splitLearningCurveGPU = gpuArray(splitLearningCurve);

%% pca提取
[UTrainGPU, STrainGPU] = pcaTrainGPU(XTrainNormGPU);

% pca-gpu
KGPU = length(STrainGPU);
for j=1:length(STrainGPU)
    if STrainGPU(j) == 0
        KGPU = j - 1;
        break;
    end
end
XOriginNormPcaGPU = data2pca(XOriginNormGPU, UTrainGPU, KGPU);
XTrainNormPcaGPU = data2pca(XTrainNormGPU, UTrainGPU, KGPU);
XValNormPcaGPU = data2pca(XValNormGPU, UTrainGPU, KGPU);
XTestNormPcaGPU = data2pca(XTestNormGPU, UTrainGPU, KGPU);

% pca-cpu
XOriginNormPca = gather(XOriginNormPcaGPU);
XTrainNormPca = gather(XTrainNormPcaGPU);
XValNormPca = gather(XValNormPcaGPU);
XTestNormPca = gather(XTestNormPcaGPU);

% 真实数据
XOriginNormPcaRealGPU = [ones(mOrigin, 1) XOriginNormPcaGPU];
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

% 原始模型结果
predYOriginGPU = logisticHypothesis(XOriginNormPcaRealGPU, thetaOriginGPU, predGPU);
predYOrigin = gather(predYOriginGPU);

% 测试集预测
predYTestGPU = logisticHypothesis(XTestNormPcaRealGPU, thetaOriginGPU, predGPU);
predYTest = gather(predYTestGPU);

%% 学习曲线
[errorTrainLearnGPU, errorValLearnGPU, realSplitLearnVecGPU, thetaMatrixLearnGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormPcaRealGPU, YTrainGPU, XValNormPcaRealGPU, YValGPU, ...
        thetaInitGPU, lambdaGPU, maxIterGPU, predGPU, splitLearningCurveGPU);

% 画图
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitLearnVec = gather(realSplitLearnVecGPU);
thetaMatrixLearn = gather(thetaMatrixLearnGPU);

%% 找到当前最优解
pVec = pLeft:pRight;
predLambdaGPU = gpuArray(1e-3);

pErrorVecGPU = gpuArray(pVec);
pLambdaVecGPU = gpuArray(pVec);
KVecGPU = gpuArray(pVec);

for i=1:length(pVec)
    fprintf('开始多项式最优化:%d\n', pVec(i));
    
    % 多项式&归一化数据
    [XTrainNormTmp, data2normFunc] = data2featureWithNormalize(XTrain, pVec(i));
    nTmp = size(XTrainNormTmp, 2);
    XValNormTmp = data2normFunc(XVal);
    fprintf('归一化完毕\n');
    
    % 转GPU
    XTrainNormTmpGPU = gpuArray(XTrainNormTmp);
    XValNormTmpGPU = gpuArray(XValNormTmp);
    nTmpGPU = gpuArray(nTmp);
    fprintf('转GPU完毕\n');
    
    % 数据pca化
    [UTrainTmpGPU, STrainTmpGPU] = pcaTrainGPU(XTrainNormTmpGPU);
    KGPU = nTmpGPU;
    for j=1:length(STrainTmpGPU)
        if STrainTmpGPU(j) == 0
            KGPU = j - 1;
            break;
        end
    end
    XTrainNormTmpPcaGPU = data2pca(XTrainNormTmpGPU, UTrainTmpGPU, KGPU);
    XValNormTmpPcaGPU = data2pca(XValNormTmpGPU, UTrainTmpGPU, KGPU);
    fprintf('数据PCA完毕:%d->%d\n', nTmpGPU, KGPU);
    
    % 添加常量数据
    XTrainNormTmpPcaRealGPU = [ones(mTrain, 1) XTrainNormTmpPcaGPU];
    XValNormTmpPcaRealGPU = [ones(mVal, 1) XValNormTmpPcaGPU];
    thetaInitTmpGPU = gpuArray.zeros(KGPU+1, 1);
    fprintf('添加常量变量，开始计算\n');
    
    % 开始计算
    [lambdaCurrentGPU, errorCurrentGPU] = ...
        logisticRegFindCurrentMinLambda(XTrainNormTmpPcaRealGPU, YTrainGPU, ...
        XValNormTmpPcaRealGPU, YValGPU, ...
        thetaInitTmpGPU, maxIterGPU, predGPU, predLambdaGPU);
    fprintf('已找到最优组合:%f, %f\n', lambdaCurrentGPU, errorCurrentGPU);
    
    % 储存结果
    pLambdaVecGPU(i) = lambdaCurrentGPU;
    pErrorVecGPU(i) = errorCurrentGPU;
    KVecGPU(i) = KGPU;
    fprintf('数据已存储\n');
end
%% 得到最优解
% 最小值所在索引
indexMinVecGPU = indexMinForMulti(pErrorVecGPU);
indexMinGPU = indexMinVecGPU(1);

% 最优解
lambdaMinGPU = pLambdaVecGPU(indexMinGPU);
errorMinGPU = pErrorVecGPU(indexMinGPU);
KVecGPU = KVecGPU(indexMinGPU);

% 所有解-CPU
pLambdaVec = gather(pLambdaVecGPU);
pErrorVec = gather(pErrorVecGPU);
KVec = gather(KVecGPU);

% 最优解-CPU
lambdaMin = gather(lambdaMinGPU);
errorMin = gather(errorMinGPU);
KMin = gather(KVecGPU);
pMin = pVec(gather(indexMinGPU));

%% save
% 获取文件名
fileName = sprintf('data/data_testComp_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'XTest', ...
    'YOrigin', 'YTrain', 'YVal', ...
    'pcaVec', 'pcaSumVec', 'predYTest', 'predYOrigin', ...
    'errorTrainLearn', 'errorValLearn', 'realSplitLearnVec', ...
    'lambdaMin', 'errorMin', 'KMin', 'pMin', 'pLambdaVec', 'pErrorVec');
fprintf('保存完毕\n');
end


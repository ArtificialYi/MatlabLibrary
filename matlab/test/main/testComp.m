function [outputArg1,outputArg2] = testComp(p, lambda, pLeft, pRight, maxIter, isTrain)
%testComp 比赛用的函数

%% str2double
maxIter = str2double(maxIter);
p = str2double(p);
lambda = str2double(lambda);
pLeft = str2double(pLeft);
pRight = str2double(pRight);
isTrain = str2double(isTrain);

%% 先读取数据
data = load('resource/pfm_data.mat');

% 获取原始数据
XOrigin = data.XOrigin;
YOrigin = data.YOrigin;
XTestOrigin = data.XTest;

%% 开始逻辑回归
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
lambdaGPU = gpuArray(lambda);
maxIterGPU = gpuArray(maxIter);
thetaInitGPU = gpuArray.zeros(nTrainGPU+1, 1);

% 学习曲线
splitLearningCurve = 50;
splitLearningCurveGPU = gpuArray(splitLearningCurve);

%% 基础训练模型
[thetaOriginGPU, ~] = ...
    logisticRegTrainGPU(XOriginNormRealGPU, YOriginGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU);

% 原始模型结果
predYOriginGPU = logisticHypothesis(XOriginNormRealGPU, thetaOriginGPU, predGPU);
predYOrigin = gather(predYOriginGPU);

% 测试集预测
predYTestGPU = logisticHypothesis(XTestNormRealGPU, thetaOriginGPU, predGPU);
predYTest = gather(predYTestGPU);

%% 学习曲线
[errorTrainLearnGPU, errorValLearnGPU, realSplitLearnVecGPU, thetaMatrixLearnGPU] = ...
    logisticRegLearningCurveGPU(XTrainNormRealGPU, YTrainGPU, XValNormRealGPU, YValGPU, ...
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

if isTrain
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

        % 添加常量数据
        XTrainNormTmpRealGPU = [ones(mTrain, 1) XTrainNormTmpGPU];
        XValNormTmpRealGPU = [ones(mVal, 1) XValNormTmpGPU];
        thetaInitTmpGPU = gpuArray.zeros(nTmpGPU+1, 1);
        fprintf('添加常量变量，开始计算\n');

        % 开始计算
        [lambdaCurrentGPU, errorCurrentGPU] = ...
            logisticRegFindCurrentMinLambda(XTrainNormTmpRealGPU, YTrainGPU, ...
            XValNormTmpRealGPU, YValGPU, ...
            thetaInitTmpGPU, maxIterGPU, predGPU, predLambdaGPU);
        fprintf('已找到最优组合:%d, %f, %f\n', pVec(i), lambdaCurrentGPU, errorCurrentGPU);

        % 储存结果
        pLambdaVecGPU(i) = lambdaCurrentGPU;
        pErrorVecGPU(i) = errorCurrentGPU;
        KVecGPU(i) = KGPU;
        fprintf('数据已存储\n');
    end
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
    'YOrigin', 'YTrain', 'YVal', 'predYOrigin', 'predYTest', ...
    'errorTrainLearn', 'errorValLearn', 'realSplitLearnVec', ...
    'lambdaMin', 'errorMin', 'KMin', 'pMin', 'pLambdaVec', 'pErrorVec');
fprintf('保存完毕\n');
end


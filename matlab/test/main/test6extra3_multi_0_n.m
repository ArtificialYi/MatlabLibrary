function [tmp] = test6extra3_multi_0_n(p, l, s, C, isTrain)
%% 测试函数
% p 多项式的值
% l 高阶参数
% s 低阶参数
% C svm训练参数

p = str2double(p);
l = str2double(l);
s = str2double(s);
C = str2double(C);
isTrain = str2double(isTrain);

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
CTrainGPU = gpuArray(C);
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
CLearnGPU = gpuArray(C);
tolLearnGPU = gpuArray(1e-15);
maxIterLearnGPU = gpuArray(50000);
splitLearnGPU = gpuArray(11);

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

function [CCurrentGPU, errorMinCurrentGPU] = ...
        findCurrentMinCFunc(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU, ...
        KTrainGPU, YTrainGPU, KValGPU, YValGPU)

    % 先用等比数列找到最优数值
    CVecCurrentGPU = logspace(log10(CLeftCurrentGPU), log10(CRightCurrentGPU), splitCCurrentGPU);
    [errorTrainCurrentTmpGPU, errorValCurrentTmpGPU] = ...
        svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU, tolCurrentGPU, maxIterCurrentGPU);
    indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU(:, 3));
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
        indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU(:, 3));
        if length(indexCurrentGPU) > 1
            indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
        end
        [indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
            getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
        CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
        CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);
    end
    
    CCurrentGPU = CVecCurrentGPU(indexCurrentGPU);
    errorMinCurrentGPU = errorValCurrentTmpGPU(indexCurrentGPU, :);
end

%% 寻找全局最优C
%% 将当前最优各种参数打印出来
pVec = 2:10;
lVec = linspace(0.1, 1.9, 3);
sVec = linspace(0.1, 1.9, 3);

function [errorMinMatrix3, CMinMatrix3] = ...
        findGlobalMinPLSCFunc(pVec, lVec, sVec, ...
            XTrainNorm, YTrainGPU, XValNorm, YValGPU, findCurrentFunc, ...
            CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
            predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU)
    errorMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));
    CMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));
    for i=1:length(pVec)
        for j=1:length(lVec)
            for k=1:length(sVec)
                pCurrent = pVec(i);
                lCurrent = lVec(j);
                sCurrent = sVec(k);
                lCurrentReal = sqrt(lCurrent/sCurrent);
                sCurrentReal = sqrt(sCurrent/lCurrent);

                kernelFuncTmp = @(X1, X2) svmKernelPolynomial(X1, X2, lCurrentReal, sCurrentReal, pCurrent);
                KTrainTmpGPU = gpuArray(kernelFuncTmp(XTrainNorm, XTrainNorm));
                KValTmpGPU = gpuArray(kernelFuncTmp(XTrainNorm, XValNorm));

                [CCurrentTmpGPU, errorMinCurrentTmpGPU] = ...
                    findCurrentFunc(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
                        predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU, ...
                        KTrainTmpGPU, YTrainGPU, KValTmpGPU, YValGPU);

                errorMinMatrix3(i, j, k) = gather(errorMinCurrentTmpGPU);
                CMinMatrix3(i, j, k) = gather(CCurrentTmpGPU);
            end
        end
    end
end

%% 如果是训练模式,找到最优的C
CCurrent = 0;
errorMinCurrent = 1;

pMin = p;
lMin = l;
sMin = s;
lMinReal = sqrt(lMin/sMin);
sMinReal = sqrt(sMin/lMin);
CMin = 0;
errorMin = 1;

if isTrain
    [CCurrentGPU, errorMinCurrentGPU] = ...
        findCurrentMinCFunc(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
            predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU, ...
            KTrainGPU, YTrainGPU, KValGPU, YValGPU);
        
    findGlobalFunc = @(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
            predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU, ...
            KTrainGPU, YTrainGPU, KValGPU, YValGPU) findCurrentMinCFunc(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
            predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU, ...
            KTrainGPU, YTrainGPU, KValGPU, YValGPU);
    findGlobalMinPLSCFunc(pVec, lVec, sVec, ...
            XTrainNorm, YTrainGPU, XValNorm, YValGPU, findGlobalFunc, ...
            CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU, ...
            predCurrentGPU, tolCurrentGPU, maxIterCurrentGPU)
    % 将当前最优C打印出来
    fprintf('当前最优C是:%.15f\n', CCurrentGPU);
    fprintf('当前最小误差是:%.15f\n', errorMinCurrentGPU);
    
    % 当前最优C
    CCurrent = gather(CCurrentGPU);
    errorMinCurrent = gather(errorMinCurrentGPU);
    
    % 找到最优的p、l、s、C
    [iMin, indexMin] = find(min(min(min(errorMinMatrix3))));
    kMin = mod(indexMin, length(sVec));
    kMin(kMin==0) = length(sVec);
    jMin = (indexMin-kMin)/length(lVec)+1;

    pMin = pVec(iMin);
    lMin = lVec(jMin);
    sMin = sVec(kMin);
    lMinReal = sqrt(lMin/sMin);
    sMinReal = sqrt(sMin/lMin);
    CMin = CMinMatrix3(iMin, jMin, kMin);
    errorMin = errorMinMatrix3(iMin, jMin, kMin);
end

%% 变量存储
% 学习曲线
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitVecLearn = gather(realSplitVecLearnGPU);

% 获取文件名
fileName = sprintf('data/data_test6extra3_multi_0_n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
save(fileName, ...
    'XOrigin', 'YOrigin', 'vecX1', 'vecX2', 'predYTestTmp_2D', ...
    'realSplitVecLearn', 'errorTrainLearn', 'errorValLearn', ...
    'XTrain', 'YTrain', 'XVal', 'YVal', 'CCurrent', 'errorMinCurrent', ...
    'pMin', 'lMin', 'lMinReal', 'sMin', 'sMinReal', 'CMin', 'errorMin');

end
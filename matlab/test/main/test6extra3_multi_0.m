%% 初始化环境
clear; close all; clc;

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
l = 0.1;
s = 0.1;
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
CTrain = 4239.306;
tolTrain = 1e-15;
maxIterTrain = 50000;
alphaTrain = zeros(m, 1);

modelOrigin = ...
    svmTrain(KOrigin, YOrigin, CTrain, alphaTrain, tolTrain, maxIterTrain);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
KTestTmp = kernelFunc(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOrigin.alpha .* YOrigin)'*KTestTmp+modelOrigin.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线训练
CLearn = 3.26;
tolLearn = 1e-15;
maxIterLearn = 50000;
splitLearn = 51;

[errorTrainLearn, errorValLearn, realSplitVecLearn] = ...
    svmLearningCurve(XTrainNorm, YTrain, ...
        XValNorm, YVal, CLearn, ...
        tolLearn, maxIterLearn, splitLearn, kernelFunc);
    
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

%% 将当前最优各种参数打印出来
pVec = 2:10;
lVec = linspace(0.1, 1.9, 3);
sVec = linspace(0.1, 1.9, 3);

errorMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));
CMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));

for i=1:length(pVec)
    for j=1:length(lVec)
        for k=1:length(sVec)
            pCurrent = pVec(i);
            lCurrent = lVec(j);
            sCurrent = sVec(k);
            
            splitCCurrent = 11;
            predCurrent = 1e-3;
            CLeftCurrent = 1e-6; % 精度的一半
            CRightCurrent = 1e4;
            tolCurrent = 1e-8;
            maxIterCurrent = 50000;
            
            kernelFuncTmp = @(X1, X2) svmKernelPolynomial(X1, X2, lCurrent, sCurrent, pCurrent);
            KTrain = kernelFuncTmp(XTrainNorm, XTrainNorm);
            KVal = kernelFuncTmp(XTrainNorm, XValNorm);

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
            
            errorMinMatrix3(i, j, k) = errorMinCurrent;
            CMinMatrix3(i, j, k) = CCurrent;
        end
    end
end

%% 找到最优的p、l、s、C
[iMin, indexMin] = find(min(min(min(errorMinCurrent))));
kMin = mod(indexMin, length(sVec));
kMin(kMin==0) = length(sVec);
jMin = (indexMin-kMin)/length(lVec)+1;

pMin = pVec(iMin);
lMin = lVec(jMin);
sMin = sVec(kMin);
CMin = CMinMatrix3(iMin, jMin, kMin);
errorMin = errorMinMatrix3(iMin, jMin, kMin);

fprintf('最优多项式:%d\n', pMin);
fprintf('最优高阶系数:%d\n', lMin);
fprintf('最优低阶系数:%d\n', sMin);
fprintf('最优SVM系数:%d\n', CMin);
fprintf('最优误差值:%d\n', errorMin);

%% 画出数据图
% 原始数据图
figure(1);
posOrigin = find(YOrigin == 1); 
negOrigin = find(YOrigin == -1);

plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 学习曲线
figure(2);
plot(realSplitVecLearn, errorTrainLearn, realSplitVecLearn, errorValLearn);
title('学习曲线');
legend('训练集', '交叉验证集');
xlabel('数量');
ylabel('误差');
fprintf('学习曲线\n');

%% 训练集图
figure(3);
posTrain = find(YTrain == 1); 
negTrain = find(YTrain == -1);

plot(XTrain(posTrain, 1), XTrain(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XTrain(negTrain, 1), XTrain(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 交叉验证集图
figure(4);
posVal = find(YVal == 1); 
negVal = find(YVal == -1);

plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

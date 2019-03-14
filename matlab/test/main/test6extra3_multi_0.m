%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:,1:2);
YOrigin = data(:,3);
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
KOrigin = svmKernelLinear(XOriginNorm, XOriginNorm);

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
CTrain = 1;
tolTrain = 1e-8;
maxIterTrain = 10000;
alphaTrain = zeros(m, 1);

modelOrigin = ...
    svmTrain(KOrigin, YOrigin, CTrain, alphaTrain, tolTrain, maxIterTrain);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, noneIndex, mu, sigma);
KTestTmp = svmKernelLinear(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOrigin.alpha .* YOrigin)'*KTestTmp+modelOrigin.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线训练
CLearn = 1;
tolLearn = 1e-15;
maxIterLearn = 1000;
splitLearn = 51;

[errorTrainLearn, errorValLearn, realSplitVecLearn] = ...
    svmLearningCurve(XTrainNorm, YTrain, ...
        XValNorm, YVal, CLearn, ...
        tolLearn, maxIterLearn, splitLearn);
    
%% 尝试找到最优C
% 计算最优C
splitCCurrent = 5;
predCurrent = 1e-3;
CLeftCurrent = 1e-6; % 精度的一半
CRightCurrent = 1e3;
tolCurrent = 1e-6;
maxIterCurrent = 1;

while CRightCurrent - CLeftCurrent > predCurrent
    CVecCurrent = linspace(CLeftCurrent, CRightCurrent, splitCCurrent);
    [errorTrainCurrentTmp, errorValCurrentTmp] = ...
        svmTrainForCVec(XTrainNorm, YTrain, XValNorm, YVal, CVecCurrent, tolCurrent, maxIterCurrent);
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
fprintf('当前最优C是:%.15f\n', CCurrent);

%% 画出数据图
% 原始数据图
figure(1);
posOrigin = find(YOrigin == 1); 
negOrigin = find(YOrigin == 0);

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
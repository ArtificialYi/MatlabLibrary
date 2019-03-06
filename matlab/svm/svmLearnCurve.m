function [errorTrain, errorVal, realSplitVec] = ...
    svmLearningCurve(X, y, XVal, yVal, C, tol, maxIter, split)
%svmLearningCurve SVM的学习曲线
% X 训练集
% y 训练集结果
% XVal 交叉验证集
% yVal 交叉验证集结果
% C 正则化参数
% tol 精度
% maxIter 最大迭代次数

% 训练集大小
m = size(X, 1);

if m < split
    realSplit = m;
else
    realSplit = split;
end

% 初始化结果数组
errorTrain = zeros(realSplit, 1);
errorVal = zeros(realSplit, 1);
realSplitVec = zeros(realSplit, 1);

for i=1:realSplit
    currentIndex = floor(m * i / realSplit);
    XTmp = X(1:currentIndex, :);
    yTmp = y(1:currentIndex);
    
    modelTmp = svmTrain(XTmp, yTmp, C, tol, maxIter);
    
    realSplitVec(i) = currentIndex;
    
    errorTrain(i) = svmCost(XTmp, yTmp, modelTmp.w, modelTmp.b, 0);
    errorVal(i) = svmCost(XVal, yVal, modelTmp.w, modelTmp.b, 0);
end


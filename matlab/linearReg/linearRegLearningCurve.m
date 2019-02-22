function [errorTrain, errorVal, realSplitVec] = ...
    linearRegLearningCurve(X, y, XVal, yVal, lambda, initTheta, split)
%linearRegLearningCurve 线性回归正则学习曲线
% X 训练集
% y 训练集的结果
% XVal 交叉验证集
% yVal 交叉验证集的结果
% lambda 正则化参数
% initTheta 初始化theta
% split 学习曲线分割段数

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
    
    thetaTmp = linearRegTrain(XTmp, yTmp, lambda, initTheta);
    
    realSplitVec(i) = currentIndex;
    errorTrain(i) = linearRegCost(XTmp, yTmp, thetaTmp, 0);
    errorVal(i) = linearRegCost(XVal, yVal, thetaTmp, 0);
end


end

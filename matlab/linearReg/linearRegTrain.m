function [theta, maxIter, Jval] = linearRegTrain(X, y, lambda, initTheta)
%linearRegTrain 使用线性回归正则化来训练X
% X 原始数据
% y 原始结果
% lambda 正则化参数
% initTheta 初始化theta

% 定义代价函数
costFunction = @(t) linearRegCost(X, y, t, lambda);

% 训练参数
maxIter = 1024;
options = optimset('MaxIter', maxIter, 'GradObj', 'on', 'Algorithm', 'trust-region');

% 开始训练
exitFlag = 0;
tmpTheta = initTheta;
while exitFlag == 0
    [theta, Jval, exitFlag] = ...
        fminunc(costFunction, tmpTheta, options);
    tmpTheta = theta;
    maxIter = maxIter * 2;
    options = optimset(options, 'MaxIter', maxIter);
end
end

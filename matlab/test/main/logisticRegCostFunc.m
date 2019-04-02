function [J, grad] = logisticRegCostFunc(X, Y, theta, lambda, pred)
%logisticCostFunc 逻辑回归的代价函数
% X 数据集
% Y 结果集
% theta theta向量
% lambda 正则化参数
% pred 精度

m = size(X, 1);

lambdaVec = theta * lambda;
lambdaVec(1) = 0;

h = logisticHypothesis(X, theta, pred);
J = (-Y'*log(h) - (1-Y')*log(1-h) + theta'*lambdaVec/2)/m;
grad = (X'*(h-Y) + lambdaVec)*(1-2*pred)/m;
end


function [J, grad] = logisticRegCostFunc(X, Y, theta, lambda, pred)
%logisticCostFunc 逻辑回归的代价函数
% X 数据集
% Y 结果集
% theta theta向量
% lambda 正则化参数
% pred 精度

m = size(X, 1);

h = logisticHypothesis(X, theta, pred);

thetaNone = theta(2:end);

J = (-Y'*log(h) - (1-Y')*log(1-h) + thetaNone'*thetaNone*lambda/2)/m;
grad = (X'*(h-Y) + thetaNone*lambda)*(1-2*pred)/m;
end


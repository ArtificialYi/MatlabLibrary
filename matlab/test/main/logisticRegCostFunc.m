function [J, grad] = logisticRegCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

m = size(X, 1);
pred = 1e-100;

h = logisticHypothesis(X, theta, pred);

J = (-Y'*log(h) - ( 1-Y')*log(1-h))/m;
grad = (X'*(h-Y)/m)*(1-2*pred);
end


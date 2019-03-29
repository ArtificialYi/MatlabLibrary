function [J, grad] = logisticCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

m = size(X, 1);

h = logisticHypothesis(X, theta);
J = sum(-Y.*log(h) - (1-Y).*log(1-h))/m;
grad = sum((h-Y).*X)/m;

end


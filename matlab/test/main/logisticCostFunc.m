function [J, grad] = logisticCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

m = size(X, 1);

h = 1./exp(-X*theta)+1;
J = sum(-y.*log(h) - (1-y).*log(1-h))/m;
grad = sum((h-y).*X)/m;
end


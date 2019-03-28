function [J, grad] = logisticCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

m = size(X, 1);

h = 1./exp(-X*theta)+1;
J = sum(-Y.*log(h) - (1-Y).*log(1-h))/m;
grad = sum((h-Y).*X)/m;
end


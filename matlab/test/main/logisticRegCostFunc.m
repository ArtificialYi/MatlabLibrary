function [J, grad] = logisticRegCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

m = size(X, 1);
h = logisticHypothesis(X, theta);
J = (-Y'*log(h) - (1-Y')*log(1-h))/m;
grad = X'*(h-Y)/m;

showHy(m, 'm');
showHy(h, 'h');
showHy(J, 'J');
showHy(grad, 'grad');
pause;
end


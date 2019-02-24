function [J, grad] = linearRegCost(X, y, theta, lambda)
%linearRegCost 线性回归正则化代价函数

% 数据的大小
[m, n] = size(X);

% 模型函数计算结果
h = X * theta;

% 正则化参数
lambdaArr = zeros(n, 1) + lambda;
lambdaArr(1) = 0;

% 计算代价函数
J = ((h-y)' * (h-y) + lambdaArr' * theta.^2) / (m*2);
grad = ((h-y)' * X + lambdaArr' .* theta') ./ m;
grad = grad(:);
end
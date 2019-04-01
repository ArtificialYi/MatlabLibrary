function [h] = logisticHypothesis(X, theta)
%logisticHypothesis 逻辑回归假设函数

hUnit = 1 ./ (exp(-X*theta)+1);

% 计算精度
pred = 1e-100;

h = (1-2*pred)*(2*hUnit-1)/2;

end


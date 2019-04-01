function [h] = logisticHypothesis(X, theta, pred)
%logisticHypothesis 逻辑回归假设函数

hUnit = 1 ./ (exp(-X*theta)+1);

h = (1-2*pred)*hUnit + pred;

end


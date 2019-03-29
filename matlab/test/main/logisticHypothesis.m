function [h] = logisticHypothesis(X, theta)
%logisticHypothesis 逻辑回归假设函数

h = 1 ./ (exp(-X*theta)+1);

end


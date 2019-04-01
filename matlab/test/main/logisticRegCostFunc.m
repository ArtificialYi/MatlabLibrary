function [J, grad] = logisticRegCostFunc(X, Y, theta)
%logisticCostFunc 逻辑回归的代价函数

% log计算精度
pred = 1e-100;

m = size(X, 1);
h = logisticHypothesis(X, theta);
J = (-Y'*log(h+pred) - (1-Y')*log((1-h+pred)))/m;
grad = X'*(h-Y)/m;

if isnan(J)
    disp(X);
    disp(Y);
    disp(theta);
    disp(h);
    
    showHy(X, 'X');
    showHy(Y, 'Y');
    showHy(theta, 'theta');
    showHy(m, 'm');
    showHy(h, 'h');
    showHy(J, 'J');
    showHy(grad, 'grad');
    pause;
end

end


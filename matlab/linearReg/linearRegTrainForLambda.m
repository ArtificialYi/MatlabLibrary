function [errorTrain, errorVal] = ...
    linearRegTrainForLambda(X, y, XVal, yVal, initTheta, lambdaVec)
%linearRegTrainForLambda 查找线性回归在不同的lambda下的结果

% 错误集合
errorTrain = zeros(length(lambdaVec), 1);
errorVal = zeros(length(lambdaVec), 1);

for i = 1:length(lambdaVec)
    thetaTmp = linearRegTrain(X, y, lambdaVec(i), initTheta);
    errorTrain(i) = linearRegCost(X, y, thetaTmp, 0);
    errorVal(i) = linearRegCost(XVal, yVal, thetaTmp, 0);
end

end

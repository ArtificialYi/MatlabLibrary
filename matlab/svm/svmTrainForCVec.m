function [errorTrainVec, errorValVec] = ...
    svmTrainForCVec(XTrain, YTrain, XVal, YVal, CVec, tol, maxIter)
%svmTrainForC SVM在不同C下的结果

% 错误集合
errorTrainVec = zeros(length(CVec), 1);
errorValVec = zeros(length(CVec), 1);

% 初始化alpha
m = size(XTrain, 1);
alpha = zeros(m, 1);

KTrain = svmKernelLinear(XTrain, XTrain);
KVal = svmKernelLinear(XTrain, XVal);

for i = 1:length(CVec)
    modelTmp = svmTrain(KTrain, YTrain, CVec(i), alpha, tol, maxIter);
    errorTrainVec(i) = svmCost(KTrain, YTrain, KTrain, YTrain, modelTmp.alpha, modelTmp.b, 0);
    errorValVec(i) = svmCost(KTrain, YTrain, KVal, YVal, modelTmp.alpha, modelTmp.b, 0);
end

end

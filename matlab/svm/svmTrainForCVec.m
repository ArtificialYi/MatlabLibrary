function [errorTrainVec, errorValVec] = ...
    svmTrainForCVec(XTrain, YTrain, XVal, YVal, CVec, tol, maxIter)
%svmTrainForC SVM在不同C下的结果

% 错误集合
errorTrainVec = zeros(length(CVec), 1);
errorValVec = zeros(length(CVec), 1);

for i = 1:length(CVec)
    modelTmp = svmTrain(XTrain, YTrain, CVec(i), tol, maxIter);
    errorTrainVec(i) = svmCost(XTrain, YTrain, modelTmp.w, modelTmp.b, 0);
    errorValVec(i) = svmCost(XVal, YVal, modelTmp.w, modelTmp.b, 0);
end

end

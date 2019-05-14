function [errorValVec, errorTrainVec] = svmTrainCPUForCVec(func, CVec)
%svmTrainCPUForCVec SVM-CPU训练

errorValVec = CVec;
errorTrainVec = CVec;

for i=1:length(CVec)
    [errorVal, errorTrain] = func(CVec(i));
    fprintf('当前计算C:%f,误差:%f\n', CVec(i), errorVal);
    errorValVec(i) = errorVal;
    errorTrainVec(i) = errorTrain;
end

end


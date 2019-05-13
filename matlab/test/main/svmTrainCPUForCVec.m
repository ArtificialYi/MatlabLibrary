function [errorValVec, errorTrainVec] = svmTrainCPUForCVec(func, CVec)
%svmTrainCPUForCVec SVM-CPU训练

errorValVec = CVec;
errorTrainVec = CVec;

for i=1:length(CVec)
    [errorVal, errorTrain] = func(CVec(i));
    errorValVec(i) = errorVal;
    errorTrainVec(i) = errorTrain;
end

end


function [errorTrainVecGPU, errorValVecGPU] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecGPU, tolGPU, maxIterGPU)
%svmTrainForC SVM在不同C下的结果

% 错误集合
errorTrainVecGPU = gpuArray.zeros(length(CVecGPU), 3);
errorValVecGPU = gpuArray.zeros(length(CVecGPU), 3);

% 初始化alpha
mGPU = size(KTrainGPU, 1);
alphaGPU = gpuArray.zeros(mGPU, 1);

n = 2;
for i = 1:length(CVecGPU)
    modelTmpGPU = svmTrainGPU(KTrainGPU, YTrainGPU, CVecGPU(i), alphaGPU, tolGPU, maxIterGPU);
    
    [errorTrainTmp, pointTrainTmp] = svmCost(KTrainGPU, YTrainGPU, KTrainGPU, YTrainGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
    [errorValTmp, pointValTmp] = svmCost(KTrainGPU, YTrainGPU, KValGPU, YValGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
    errorTrainVecGPU(i, 1) = errorTrainTmp;
    errorTrainVecGPU(i, 2) = 1 - pointTrainTmp + 0.01;
    errorTrainVecGPU(i, 3) = errorTrainVecGPU(i, 1).*errorTrainVecGPU(i, 2)*(n+1)./(errorTrainVecGPU(i, 1)*n+errorTrainVecGPU(i, 2));
    errorValVecGPU(i, 1) = errorValTmp;
    errorValVecGPU(i, 2) = 1 - pointValTmp + 0.01;
    errorValVecGPU(i, 3) = ...
        errorValVecGPU(i, 1).*errorValVecGPU(i, 2)*(n+1)./(errorValVecGPU(i, 1)*n+errorValVecGPU(i, 2));
    fprintf('cross-error:%f, pred:%f, comp:%f\n', errorValVecGPU(i, 1), errorValVecGPU(i, 2), errorValVecGPU(i, 3));
end
end

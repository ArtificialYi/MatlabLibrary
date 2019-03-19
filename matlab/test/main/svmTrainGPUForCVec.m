function [errorTrainVecGPU, errorValVecGPU] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecGPU, tolGPU, maxIterGPU)
%svmTrainForC SVM在不同C下的结果

% 错误集合
errorTrainVecGPU = gpuArray.zeros(length(CVecGPU), 3);
errorValVecGPU = gpuArray.zeros(length(CVecGPU), 3);

% 初始化alpha
mGPU = size(KTrainGPU, 1);
alphaGPU = gpuArray.zeros(mGPU, 1);

for i = 1:length(CVecGPU)
    modelTmpGPU = svmTrainGPU(KTrainGPU, YTrainGPU, CVecGPU(i), alphaGPU, tolGPU, maxIterGPU);
    [errorTrainVecGPU(i, 1), errorTrainVecGPU(i, 2)] = svmCost(KTrainGPU, YTrainGPU, KTrainGPU, YTrainGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
    [errorValVecGPU(i, 1), errorValVecGPU(i, 2)] = svmCost(KTrainGPU, YTrainGPU, KValGPU, YValGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
end

n = 2;
errorTrainVecGPU(:, 3) = ...
    errorTrainVecGPU(:, 1)*(1-errorTrainVecGPU(:, 2))*(n+1)/(errorTrainVecGPU(:, 1)*n+(1-errorTrainVecGPU(:, 2)));
errorValVecGPU(:, 3) = ...
    errorValVecGPU(:, 1)*(1-errorValVecGPU(:, 2))*(n+1)/(errorValVecGPU(:, 1)*n+(1-errorValVecGPU(:, 2)));

end

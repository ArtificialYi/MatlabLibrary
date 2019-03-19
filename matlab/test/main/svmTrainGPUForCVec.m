function [errorTrainVecGPU, errorValVecGPU] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecGPU, tolGPU, maxIterGPU)
%svmTrainForC SVM在不同C下的结果

% 错误集合
errorTrainVecGPU = gpuArray.zeros(length(CVecGPU), 1);
errorValVecGPU = gpuArray.zeros(length(CVecGPU), 1);

% 初始化alpha
mGPU = size(KTrainGPU, 1);
alphaGPU = gpuArray.zeros(mGPU, 1);

for i = 1:length(CVecGPU)
    modelTmpGPU = svmTrainGPU(KTrainGPU, YTrainGPU, CVecGPU(i), alphaGPU, tolGPU, maxIterGPU);
    errorTrainVecGPU(i) = svmCost(KTrainGPU, YTrainGPU, KTrainGPU, YTrainGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
    errorValVecGPU(i) = svmCost(KTrainGPU, YTrainGPU, KValGPU, YValGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
end

end

function [CCurrentGPU, errorMinCurrentGPU] = svmFindCurrentMinC(KTrainGPU, YTrainGPU, KValGPU, YValGPU, tolCurrentGPU, maxIterCurrentGPU, predCurrentGPU)
%svmFindCurrentMinC 找到最优C和最优误差
%   此处显示详细说明

splitCCurrentGPU = gpuArray(21);

%% 先用等比数列找到范围
CVecCurrentGPU = logspace(gpuArray(-5), gpuArray(5), splitCCurrentGPU);
[errorTrainCurrentTmpGPU, errorValCurrentTmpGPU] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU, tolCurrentGPU, maxIterCurrentGPU);
indexCurrentGPU = indexMinForMulti(errorValCurrentTmpGPU(:, 3));
if length(indexCurrentGPU) > 1
    indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
end
[indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
    getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);
    
% 再开始用等差数列做循环
while CRightCurrentGPU - CLeftCurrentGPU > predCurrentGPU
    CVecCurrentGPU = linspace(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU);
    [errorTrainCurrentTmpGPU, errorValCurrentTmpGPU] = ...
        svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU, tolCurrentGPU, maxIterCurrentGPU);
    indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU(:, 3));
    if length(indexCurrentGPU) > 1
        indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
    end
    [indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
        getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
    CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
    CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);
end

CCurrentGPU = CVecCurrentGPU(indexCurrentGPU);
errorMinCurrentGPU = errorValCurrentTmpGPU(indexCurrentGPU, 3);
end


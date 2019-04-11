function [CCurrentGPU, errorMinCurrentGPU] = svmFindCurrentMinC(KTrainGPU, YTrainGPU, KValGPU, YValGPU, tolCurrentGPU, maxIterCurrentGPU, predCurrentGPU)
%svmFindCurrentMinC 找到最优C和最优误差
%   此处显示详细说明

splitCCurrentGPU = gpuArray(11);

%% 先用等比数列找到范围
CVecCurrentGPU = logspace(gpuArray(-5), gpuArray(5), splitCCurrentGPU);
CLeftCurrentGPU = CVecCurrentGPU(1);
CRightCurrentGPU = CVecCurrentGPU(end);

% 初始化左右极限
[~, errorValCurrentTmpGPU] = ...
    svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU([1 end]), tolCurrentGPU, maxIterCurrentGPU);
errorLeft = errorValCurrentTmpGPU(1, 3);
errorRight = errorValCurrentTmpGPU(end, 3);

% 再开始用等差数列做循环
while CRightCurrentGPU - CLeftCurrentGPU > predCurrentGPU
    [~, errorValCurrentTmpGPU] = ...
        svmTrainGPUForCVec(KTrainGPU, YTrainGPU, KValGPU, YValGPU, CVecCurrentGPU(2:end-1), tolCurrentGPU, maxIterCurrentGPU);
    % 将左右极限拼上去
    errorValCurrentTmpGPU = [0 0 errorLeft;errorValCurrentTmpGPU;0 0 errorRight];
    indexCurrentGPU = indexMinForVec(errorValCurrentTmpGPU(:, 3));
    if length(indexCurrentGPU) > 1
        indexCurrentGPU = indexCurrentGPU(length(indexCurrentGPU));
    end
    [indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
        getLeftAndRightIndex(indexCurrentGPU, 1, splitCCurrentGPU);
    CLeftCurrentGPU = CVecCurrentGPU(indexCurrentLeftTmpGPU);
    CRightCurrentGPU = CVecCurrentGPU(indexCurrentRightTmpGPU);
    CVecCurrentGPU = linspace(CLeftCurrentGPU, CRightCurrentGPU, splitCCurrentGPU);
    % 将左右极限保留下来
    errorLeft = errorValCurrentTmpGPU(indexCurrentLeftTmpGPU, 3);
    errorRight = errorValCurrentTmpGPU(indexCurrentRightTmpGPU, 3);
end

CCurrentGPU = CVecCurrentGPU(indexCurrentGPU);
errorMinCurrentGPU = errorValCurrentTmpGPU(indexCurrentGPU, 3);
end


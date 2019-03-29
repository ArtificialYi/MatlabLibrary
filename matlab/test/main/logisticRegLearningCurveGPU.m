function [errorTrainGPU, errorValGPU, realSplitVecGPU] = ...
    logisticRegLearningCurveGPU(XTrainGPU, YTrainGPU, ...
        XValGPU, YValGPU, thetaInitGPU, maxIterGPU, splitGPU)
%logisticRegLearningCurveGPU 逻辑回归-学习曲线

mGPU = gpuArray(size(XTrainGPU, 1));

realSplitGPU = min(mGPU, splitGPU);

errorTrainGPU = gpuArray.zeros(realSplitGPU, 1);
errorValGPU = gpuArray.zeros(realSplitGPU, 1);
realSplitVecGPU = gpuArray.zeros(realSplitGPU, 1);

for i=1:realSplitGPU
    currentIndexGPU = floor(mGPU*i/realSplitGPU);
    XTrainTmpGPU = XTrainGPU(1:currentIndexGPU, :);
    YTrainTmpGPU = YTrainGPU(1:currentIndexGPU, :);
    
    thetaTmpGPU = logisticRegTrainGPU(XTrainTmpGPU, YTrainTmpGPU, thetaInitGPU, maxIterGPU);
    
    realSplitVecGPU(i) = currentIndexGPU;
    errorTrainGPU(i) = logisticRegCostFunc(XTrainTmpGPU, YTrainTmpGPU, thetaTmpGPU);
    errorValGPU(i) = logisticRegCostFunc(XValGPU, YValGPU, thetaTmpGPU);
end

end


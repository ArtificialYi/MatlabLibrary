function [errorTrainGPU, errorValGPU, realSplitVecGPU, thetaMatrixGPU] = ...
    logisticRegLearningCurveGPU(XTrainGPU, YTrainGPU, ...
        XValGPU, YValGPU, thetaInitGPU, maxIterGPU, predGPU, splitGPU)
%logisticRegLearningCurveGPU 逻辑回归-学习曲线

mGPU = gpuArray(size(XTrainGPU, 1));
nGPU = gpuArray(size(XTrainGPU, 2));

realSplitGPU = min(mGPU, splitGPU);

errorTrainGPU = gpuArray.zeros(realSplitGPU, 1);
errorValGPU = gpuArray.zeros(realSplitGPU, 1);
realSplitVecGPU = gpuArray.zeros(realSplitGPU, 1);
thetaMatrixGPU = gpuArray.zeros(nGPU, realSplitGPU);

for i=1:realSplitGPU
    currentIndexGPU = floor(mGPU*i/realSplitGPU);
    XTrainTmpGPU = XTrainGPU(1:currentIndexGPU, :);
    YTrainTmpGPU = YTrainGPU(1:currentIndexGPU, :);
    
    thetaTmpGPU = logisticRegTrainGPU(XTrainTmpGPU, YTrainTmpGPU, thetaInitGPU, maxIterGPU, predGPU);
    
    thetaMatrixGPU(:, i) = thetaTmpGPU;
    realSplitVecGPU(i) = currentIndexGPU;
    errorTrainGPU(i) = logisticRegCostFunc(XTrainTmpGPU, YTrainTmpGPU, thetaTmpGPU, predGPU);
    
    showHy(XValGPU, 'XValGPU');
    showHy(YValGPU, 'YValGPU');
    showHy(thetaTmpGPU, 'thetaTmpGPU');
    errorValGPU(i) = logisticRegCostFunc(XValGPU, YValGPU, thetaTmpGPU, predGPU);
    showHy(i, 'i');
    showHy(errorValGPU(i), 'errorValGPU(i)');
end

end


function [errorTrainGPU, errorValGPU, realSplitVecGPU, thetaMatrixGPU] = ...
    logisticRegLearningCurveGPU(XTrainGPU, YTrainGPU, XValGPU, YValGPU, ...
        thetaInitGPU, lambdaGPU, maxIterGPU, predGPU, splitGPU)
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
    realSplitVecGPU(i) = currentIndexGPU;
    
    fprintf('学习曲线:%d:%d:%d\n', mGPU, i, realSplitVecGPU(i));
    XTrainTmpGPU = XTrainGPU(1:currentIndexGPU, :);
    YTrainTmpGPU = YTrainGPU(1:currentIndexGPU, :);
    
    thetaTmpGPU = logisticRegTrainGPU(XTrainTmpGPU, YTrainTmpGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU);
    
    thetaMatrixGPU(:, i) = thetaTmpGPU;
    errorTrainGPU(i) = logisticRegCostFunc(XTrainTmpGPU, YTrainTmpGPU, thetaTmpGPU, 0, predGPU);
    errorValGPU(i) = logisticRegCostFunc(XValGPU, YValGPU, thetaTmpGPU, 0, predGPU);
end

end


function [errorTrainGPU, errorValGPU, realSplitVecGPU] = kMeansLearningCurveGPU(XTrainGPU, XValGPU, KGPU, maxIterGPU, splitGPU)
%kMeansLearningCurveGPU 均值算法-学习曲线

% 训练集大小
m = size(XTrainGPU, 1);

if m < splitGPU
    realSplitGPU = m;
else
    realSplitGPU = splitGPU;
end

% 初始化结果数组
errorTrainGPU = gpuArray.zeros(realSplitGPU, 1);
errorValGPU = gpuArray.zeros(realSplitGPU, 1);
realSplitVecGPU = gpuArray.zeros(realSplitGPU, 1);

for i=1:realSplitGPU
    currentIndexGPU = floor(m * i / realSplitGPU); 
    KTmpGPU = min(currentIndexGPU, KGPU);
    
    fprintf('学习曲线-%d:%d-%d\n', realSplitGPU, i, currentIndexGPU);
    XTmpGPU = XTrainGPU(1:currentIndexGPU, :);
    
    [centroidsTrainGPU, ~, errorTrainGPU] = kMeansTrainRandGPU(XTmpGPU, KTmpGPU, maxIterGPU);
    
    [~, ~, errorValGPU] = kMeansTrainGPU(XValGPU, centroidsTrainGPU, 1);
    
    realSplitVecGPU(i) = currentIndexGPU;
    errorTrainGPU(i) = errorTrainGPU;
    errorValGPU(i) = errorValGPU;
end

end


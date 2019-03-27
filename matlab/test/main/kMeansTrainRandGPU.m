function [centroidsMinGPU, YMinGPU, errorMinGPU] = kMeansTrainRandGPU(XGPU, KGPU, maxIterGPU)
%kMeansTrainRandGPU 为kMeans算法添加随机因子
%   此处显示详细说明

[m, n] = gpuArray(size(XGPU)); 
K = gather(KGPU);

mTrain = ceil(sqrt(m*n));
timeTrain = 0;

centroidsGPU = gpuArray.zeros(KGPU, n);
[centroidsGPU, YTmpGPU, errorTmpGPU] = kMeansTrainGPU(XGPU, centroidsGPU, 1);

centroidsMinGPU = centroidsGPU;
errorMinGPU = errorTmpGPU;
YMinGPU = YTmpGPU;
while timeTrain < mTrain
    indexVecRand = randperm(m, K);
    centroidsGPU(:) = XGPU(indexVecRand, :);
    [centroidsTmpGPU, YTmpGPU, errorTmpGPU] = kMeansTrainGPU(XGPU, centroidsGPU, maxIterGPU);
    timeTrain=timeTrain+1;
    if errorTmpGPU<errorMinGPU
        centroidsMinGPU(:) = centroidsTmpGPU;
        YMinGPU = YTmpGPU;
        errorMinGPU = errorTmpGPU;
        fprintf('%d:%d, 找到更小值!.\n', mTrain, timeTrain);
        timeTrain = 1;
    end
end
fprintf('%d个分类完毕!.\n', KGPU);
end


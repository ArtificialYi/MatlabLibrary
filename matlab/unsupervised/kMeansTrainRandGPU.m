function [centroidsMinGPU, YMinGPU, errorMinGPU] = kMeansTrainRandGPU(XGPU, KGPU, maxIterGPU)
%kMeansTrainRandGPU 为kMeans算法添加随机因子
%   此处显示详细说明

[m, n] = size(XGPU); 
K = gather(KGPU);

mTrain = ceil(sqrt(m*n*K));
timeTrain = 0;
fprintf('XGPU:%d,%d\n', size(XGPU));
centroidsGPU = findInitPointRand(XGPU, KGPU);
fprintf('XGPU:%d,%d\n', size(XGPU));
fprintf('KGPU:%d,%d,%d\n', size(XGPU), KGPU);
fprintf('centroidsGPU:%d,%d\n', size(centroidsGPU));
[centroidsGPU, YTmpGPU, errorTmpGPU] = kMeansTrainGPU(XGPU, centroidsGPU, maxIterGPU);

centroidsMinGPU = centroidsGPU;
errorMinGPU = errorTmpGPU;
YMinGPU = YTmpGPU;
while timeTrain < mTrain
    indexVecRand = randperm(m, K);
    centroidsGPU(:) = XGPU(indexVecRand, :);
    [centroidsTmpGPU, YTmpGPU, errorTmpGPU] = kMeansTrainGPU(XGPU, centroidsGPU, maxIterGPU);
    timeTrain=timeTrain+1;
    if errorTmpGPU < errorMinGPU
        centroidsMinGPU(:) = centroidsTmpGPU;
        YMinGPU = YTmpGPU;
        errorMinGPU = errorTmpGPU;
        fprintf('%d:%d, 找到更小值!.\n', mTrain, timeTrain);
        timeTrain = 1;
    end
end
fprintf('%d个分类完毕!.\n', KGPU);
end


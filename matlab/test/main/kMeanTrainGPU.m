function [centroidsGPU, idxPtr] = kMeanTrainGPU(XGPU, centroidsGPU, maxIter)
%kMeanTrainGPU K-mean算法训练函数

% 初始化参数
[mGPU, nGPU] = size(XGPU);
KGPU = size(centroidsGPU, 1);
centroidsRepeatGPU = gpuArray.zeros(mGPU*KGPU, nGPU);
matrixTmp = gpuArray.zeros(KGPU, mGPU);
idxPre = gpuArray.zeros(mGPU, 1);

XGPUMultiGPU = multiMatrix(XGPU, KGPU);

for i=1:maxIter
    % 划分数据点到聚类点
    centroidsRepeatGPU(:) = repeatMatrix(centroidsGPU, mGPU);
    idxMatrixTmp = reshape(sum((XGPUMultiGPU-centroidsRepeatGPU).^2, 2), mGPU, KGPU);
    [dist, idxPtr] = min(idxMatrixTmp, [], 2);

    % 如果没有误差变动
    if all(idxPre==idxPtr)
        break;
    end
    
    % 求新的聚类点
    matrixTmp(:) = 0;
    matrixTmp((0:mGPU-1)'*KGPU+idxPtr) = 1;
    centroidsGPU(:) = matrixTmp./sum(matrixTmp, 2)*XGPU;
    
    idxPre(:) = idxPtr;
end
fprintf('K:%d, iter:%d\n', KGPU, i);

end


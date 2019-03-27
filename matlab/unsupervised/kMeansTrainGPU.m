function [centroidsGPU, idxPtrGPU, distUnitGPU, iter] = kMeansTrainGPU(XGPU, centroidsGPU, maxIter)
%kMeanTrainGPU K-mean算法训练函数

% 初始化参数
[mGPU, nGPU] = size(XGPU);
KGPU = size(centroidsGPU, 1);
centroidsRepeatGPU = gpuArray.zeros(mGPU*KGPU, nGPU);
matrixTmpGPU = gpuArray.zeros(KGPU, mGPU);
idxPreGPU = gpuArray.zeros(mGPU, 1);

XGPUMultiGPU = multiMatrix(XGPU, KGPU);

for i=1:maxIter
    % 划分数据点到聚类点
    centroidsRepeatGPU(:) = repeatMatrix(centroidsGPU, mGPU);
    idxMatrixTmpGPU = reshape(sum((XGPUMultiGPU-centroidsRepeatGPU).^2, 2), mGPU, KGPU);
    [distGPU, idxPtrGPU] = min(idxMatrixTmpGPU, [], 2)

    % 如果没有误差变动
    if all(idxPreGPU==idxPtrGPU)
        break;
    end
    
    % 求新的聚类点
    matrixTmpGPU(:) = 0;
    matrixTmpGPU((0:mGPU-1)'*KGPU+idxPtrGPU) = 1;
    centroidsGPU(:) = matrixTmpGPU./sum(matrixTmpGPU, 2)*XGPU;
    
    idxPreGPU(:) = idxPtrGPU;
end
iter = i;

distUnitGPU = mean(distGPU);

end


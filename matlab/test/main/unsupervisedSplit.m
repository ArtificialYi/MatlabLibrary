function [centroidsGPU, YGPU, K] = unsupervisedSplit(XGPU, func, KMax)
%unsupervisedSplit 无监督分类
% 使用func找到X的最佳K分类

[m, n] = size(XGPU);

errorElbowVec = zeros(KMax, 1);

YMatrixGPU = gpuArray.zeros(m, KMax);
centroidsMatrixGPU = gpuArray.zeros((1+KMax)*KMax/2, n);
% 手肘法
indexBegin = 1;
for i=1:KMax
    [centroidsTmpGPU, YTmpGPU, errorTmpGPU, KGPU] = func(XGPU, i);
    
    indexEnd = indexBegin + i - 1;
    centroidsMatrixGPU(indexBegin:indexEnd, :) = centroidsTmpGPU;
    YMatrixGPU(:, i) = YTmpGPU;
    indexBegin = indexEnd + 1;
    
    errorElbowVec(i) = gather(errorTmpGPU);
    if KGPU < i
        errorElbowVec(i+1:KMax) = errorElbowVec(i);
        break;
    end
end

[~, rightVec] = matrixMove(errorElbowVec);
rightVec(rightVec==0) = 1e8;
disp(errorElbowVec);
disp(errorElbowVec.*((1:KMax).^2)');
vecTmp = errorElbowVec ./ rightVec
indexMin = indexMinForMulti(vecTmp);
K = indexMin(1);

% 将点集合、分布集合、集群个数返回
YGPU = YMatrixGPU(:, K);
centroidsGPU = centroidsMatrixGPU(K*(K-1)/2+1:K*(K+1)/2, :);
fprintf('预计的K值为:%d\n', K);

end


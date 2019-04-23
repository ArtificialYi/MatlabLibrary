function [YGPU, K] = unsupervisedSplit(XGPU, func)
%unsupervisedSplit 无监督分类
% 使用func找到X的最佳K分类

m = size(XGPU, 1);
KMax = floor(sqrt(m));

errorElbowVec = zeros(KMax, 1);

YMatrixGPU = gpuArray.zeros(m, KMax);
% 手肘法
for i=1:KMax
    [~, YTmpGPU, errorTmpGPU] = func(XGPU, i);
    YMatrixGPU(:, i) = YTmpGPU;
    errorElbowVec(i) = gather(errorTmpGPU);
end

[~, dV1ErrorElbowVec] = indexMinForMulti(errorElbowVec);
[~, dV2ErrorElbowVec] = indexMinForMulti(dV1ErrorElbowVec);

[~, K] = max(dV2ErrorElbowVec);
YGPU = YMatrixGPU(:, K);

end


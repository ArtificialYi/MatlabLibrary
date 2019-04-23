function [YGPU, K] = unsupervisedSplit(XGPU, func)
%unsupervisedSplit 无监督分类
% 使用func找到X的最佳K分类

m = size(XGPU, 1);
KMax = floor(sqrt(m));

errorElbowVec = zeros(KMax, 1);

YMatrixGPU = gpuArray.zeros(m, KMax);
% 手肘法
for i=1:KMax
    [~, YTmpGPU, errorTmpGPU, KGPU] = func(XGPU, i);
    YMatrixGPU(:, i) = YTmpGPU;
    errorElbowVec(i) = gather(errorTmpGPU);
    if KGPU < i
        errorElbowVec(i+1:KMax) = errorElbowVec(i);
        break;
    end
end

[~, rightVec] = matrixMove(errorElbowVec);
rightVec(rightVec==0) = 1e8;
vecTmp = errorElbowVec ./ rightVec;
indexMin = indexMinForMulti(vecTmp);
K = indexMin(1);

YGPU = YMatrixGPU(:, K);
disp(errorElbowVec);
fprintf('预计的K值为:%d\n', K);

end


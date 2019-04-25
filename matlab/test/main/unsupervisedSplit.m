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
    if KGPU < i
        errorElbowVec(i:KMax) = errorElbowVec(i);
        break;
    end
    
    indexEnd = indexBegin + i - 1;
    centroidsMatrixGPU(indexBegin:indexEnd, :) = centroidsTmpGPU;
    YMatrixGPU(:, i) = YTmpGPU;
    indexBegin = indexEnd + 1;
    
    errorElbowVec(i) = gather(errorTmpGPU);
end

% 找到最小值
[errorMin, indexMin] = min(errorElbowVec);
% 如果可以完美分割所有集合
if errorMin == 0
    K = indexMin;
else
    % 先计算每个点的斜率（导数）
    errorElbowVec2 = errorElbowVec.*((1:KMax))';
    % 先计算每个点的斜率（导数）
    [~, ~, errorDv] = indexMinForMulti(errorElbowVec2);
    % 通过斜率计算夹角，如果小于0，+pi
    thetaVec = atan(errorDv);
    % 计算夹角的斜率(导数),找到所有大于0的斜率中，最大的
    [~, ~, thetaDv] = indexMinForMulti(thetaVec);
    % 取最大值即为手肘法的K值
    [~, K] = max(thetaDv);
end

% 将点集合、分布集合、集群个数返回
YGPU = YMatrixGPU(:, K);
centroidsGPU = centroidsMatrixGPU(K*(K-1)/2+1:K*(K+1)/2, :);
fprintf('预计的K值为:%d\n', K);

end


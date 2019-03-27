function [centroids, KReal] = findInitPointRand(XGPU, KGPU)
%findPointRand 在X上寻找K个初始点，使用最远距离方案

[~, n] = size(XGPU);

centroids = gpuArray.zeros(KGPU, n);

for i=1:KGPU
    m = size(XGPU, 1);
    if m < 1
        KReal = i - 1;
        centroids = centroids(1:KReal, :);
        break;
    end
    % 从现有的XGPU中随机取出一行
    indexTmp = ceil(rand()*m);
    centroids(i, :) = XGPU(indexTmp, :);
    % 移除已经成为中心点的点
    XGPU(all(XGPU==centroids(i, :),2), :) = [];
end

end


function [centroids, KReal] = findInitPointRand(XGPU, KGPU)
%findPointRand 在X上寻找K个初始点

m = size(XGPU, 1);

tmp = zeros(KGPU, m);

% 获取一个 K * n 的缓存矩阵
centroids = tmp * XGPU;
KReal = KGPU;

for i=1:KGPU
    m = size(XGPU, 1);
    if m < 1
        KReal = i - 1;
        centroids = centroids(1:KReal, :);
        break;
    end
    % 从现有的XGPU中随机取出一行
    indexTmp = randperm(m, 1);
    centroids(i, :) = XGPU(indexTmp, :);
    % 移除已经成为中心点的点
    XGPU(all(XGPU==centroids(i, :),2), :) = [];
end

end


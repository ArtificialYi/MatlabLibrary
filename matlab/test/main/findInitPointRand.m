function [centroids] = findInitPointRand(XGPU, KGPU)
%findPointRand 在X上寻找K个初始点，使用最远距离方案

[m, n] = size(XGPU);

centroids = gpuArray.zeros(KGPU, n);
indexTmp = ceil(rand()*m);

centroids(1, :) = XGPU(indexTmp, :);
i = 1;
for i=2:KGPU
    [pointTmp, findSuccess] = findFarPoint(XGPU, centroids(1:i-1, :));
    if ~findSuccess 
        break;
    end
    centroids(i, :) = pointTmp;
end

centroids = centroids(1:i, :);

end


function [centroids] = findInitPointRand(XGPU, KGPU)
%findPointRand 在X上寻找K个初始点，使用最远距离方案

[m, n] = size(XGPU);

centroids = gpuArray.zeros(KGPU, n);
indexTmp = ceil(rand()*m);

centroids(1, :) = XGPU(indexTmp, :);

for i=1:KGPU
    [pointTmp, findSuccess] = findFarPoint(XGPU, centroids(1:i-1, :));
    if ~findSuccess 
        iTmp = i-1;
        iTmp(iTmp<1)=1;
        centroids = centroids(1:iTmp, :);
        break;
    end
    centroids(i, :) = pointTmp;
end

end


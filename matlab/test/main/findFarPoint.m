function [vec, isOK] = findFarPoint(X, centroids)
%findFarPoint 在X中寻找距离中心点最远的点

K = size(centroids, 1);

% 从X中移除所有和centroids相同的点
for i=1:K
    X(all(X==centroids(i,:),2), :) = [];
end
m = size(X, 1);
isOK = m > 0;
vec = [];

% 找到最远的点
if isOK
    XMulti = multiMatrix(X, K);
    centroidsRepeat = repeatMatrix(centroids, m);

    distMatrix = reshape(sum((XMulti-centroidsRepeat).^2, 2), m, K);
    distVec = sum(distMatrix, 2);

    [~, indexMax] = max(distVec);
    vec = X(indexMax, :);
end


end


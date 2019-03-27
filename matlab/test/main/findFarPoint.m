function [indexMax] = findFarPoint(X, centroids)
%findFarPoint 在X中寻找距离中心点最远的点

m = size(X, 1);
K = size(centroids, 1);

XMulti = multiMatrix(X, K);
centroidsRepeat = repeatMatrix(centroids, m);

distMatrix = reshape(sum((XMulti-centroidsRepeat).^2, 2), m, K);
distVec = sum(distMatrix, 2);

[~, indexMax] = max(distVec);

end


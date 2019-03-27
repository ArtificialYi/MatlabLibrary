function [centroids] = findInitPointRand(X, K)
%findPointRand 在X上寻找K个初始点，使用最远距离方案

[m, n] = size(X);

centroids = zeros(K, n);
indexTmp = ceil(rand()*m);

centroids(1, :) = X(indexTmp, :);
for i=2:K
    centroids(i, :) = findFarPoint(X, centroids(1:i-1, :));
end

end


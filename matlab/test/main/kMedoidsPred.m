function [centroids, predY] = kMedoidsPred(X, centroids)
%kMedoidsPred 中心点预测函数

m = size(X, 1);
K = size(centroids, 1);

% 计算点距离
distMatrix = zeros(m, K);
for i=1:K
    distMatrix(:, i) = sum((X-centroids(i, :)).^2, 2);
end

% 预测所属类别
[~, predY] = min(distMatrix, [], 2);

end


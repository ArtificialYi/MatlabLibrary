function [centroids, predY, errorMin, KReal] = kMedoidsTrain(X, K)
%kMedoidsTrain medoids训练函数

[predY, ~, ~, D, indexX, ~] = kmedoids(X, K);

indexXUniq = unique(indexX);
centroids = X(indexXUniq, :);
errorMin = mean(min(D, [], 2));
KReal = size(indexXUniq, 1);

% 切换KTmp
KTmp = K;
for i=1:KReal
    if isempty(find(predY==i, 1))
        predY(predY==KTmp) = i;
        KTmp = KTmp - 1;
    end
end

end


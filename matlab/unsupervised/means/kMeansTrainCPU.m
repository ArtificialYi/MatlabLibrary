function [centroids, idxPtr, distUnit, iter] = kMeansTrainCPU(X, centroids, maxIter)
%kMeansTrainCPU kMeans训练函数-CPU版

[m, n] = size(X);
K = size(centroids, 1);
centroidsRepeat = zeros(m*K, n);
matrixTmp = zeros(K, m);
idxPre = zeros(m, 1);

XMulti = multiMatrix(X, K);

for i=1:maxIter
    % 划分数据点到聚类点
    centroidsRepeat(:) = repeatMatrix(centroids, m);
    idxMatrixTmp = reshape(sum((XMulti - centroidsRepeat).^2, 2), m, K);
    
    [dist, idxPtr] = min(idxMatrixTmp, [], 2);
    
    % 如果没有误差变动
    if all(idxPre==idxPtr)
        break;
    end
    
    % 求新的聚类点
    matrixTmp(:) = 0;
    matrixTmp((0:m-1)'*K+idxPtr) = 1;
    centroids(:) = matrixTmp./sum(matrixTmp, 2)*X;
    
    idxPre(:) = idxPtr;
end

iter = i;
distUnit = mean(dist);

end

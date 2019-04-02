function out = mapFeature2polynomial(X, p)
% mapFeature 扩充多项式特征

[m, n] = size(X);
%   初始化内存空间
tmpN = zeros(1, p - 1);
for i=2:p
    tmpN(i - 1) = numOfPolynomialFeature(n, i);
end
out = [X, zeros(m, sum(tmpN))];

% 开始赋值
tmpOutCol = n;
for i=2:p
    % 获取扩充所需的数组
    powerMatrix = matrixOfSumWithNum(n, i);
    
    % 赋值
    for j=1:tmpN(i-1)
        tmpFeatureVec = prod(X .^ powerMatrix(j, :), 2);
        out(:, tmpOutCol+j) = tmpFeatureVec;
    end
    tmpOutCol = tmpOutCol + tmpN(i-1);
end

end


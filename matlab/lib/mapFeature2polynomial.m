function out = mapFeature2polynomial(X, p)
% mapFeature 多项式扩充特征
% X 原始数据
% p 多项式次数

%   获取大小
[m, n] = size(X);

%   初始化结果矩阵
tmpN = zeros(1, p - 1);
for i=2:p
    tmpN(i - 1) = numOfPolynomialFeature(n, i);
end
out = [X, zeros(m, sum(tmpN))];

%   为结果赋值
tmpOutCol = n;
for i=2:p
    % 获取扩充特征所有排列组合
    powerMatrix = matrixOfSumWithNum(n, i);
    featureNum = size(powerMatrix, 1);
    
    % 使原始数据和排列组合的行数保持一致
    repeatNoneConstX = repeatMatrix(X, featureNum);
    multiPowerMatrix = multiMatrix(powerMatrix, m);
    
    % 开始计算
    tmpFeatureX = repeatNoneConstX .^ multiPowerMatrix;
    tmpFeatureVec = prod(tmpFeatureX, 2);
    
    % 重新组合成需要的特征矩阵
    tmpX = reshape(tmpFeatureVec, featureNum, m);
    
    % 将特征融合到结果中
    out(:, (tmpOutCol + 1):(tmpOutCol + featureNum)) = tmpX';
    tmpOutCol = tmpOutCol + featureNum;
end

end


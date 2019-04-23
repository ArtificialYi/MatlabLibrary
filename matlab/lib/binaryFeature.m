function [KResMatrix, func] = binaryFeature(X, KMax, p, splitFunc, predFunc)
%binaryFeature 将所有枚举型特征扩充为2进制特征

[mNorm, nNorm] = size(X);
% 列矩阵
colIndexMatrix = vec2subMatrix(1:nNorm, p);
nBinaryNew = size(colIndexMatrix, 1);

% 点矩阵初始化
centroidsMatrix = size(nBinaryNew * KMax, p);
centroidsIndexMatrix = size(nBinaryNew, 2);
KResMatrix = size(mNorm, nBinaryNew);

indexBegin = 1;
for i=1:nBinaryNew
    % 某个需要计算集群的数据集，存储点矩阵、分布向量、
    [centroidsGPU, YGPU, K] = unsupervisedSplit(X(:, colIndexMatrix(i, :)), splitFunc, KMax);
    
    centroidsMatrix(indexBegin:indexBegin+K-1, :) = centroidsGPU;
    centroidsIndexMatrix(i, :) = [indexBegin, indexBegin+K-1];
    KResMatrix(:, i) = YGPU;
    indexBegin = indexBegin + K;
end
centroidsMatrix = centroidsMatrix(1:indexBegin-1, :);

% 扩展用的函数
func = @(paramX) data2binaryData(paramX, colIndexMatrix, centroidsIndexMatrix, centroidsMatrix, predFunc);
function XBinary = data2binaryData(XTest, colIndexMatrixTmp, centroidsIndexMatrixTmp, centroidsMatrixTmp, funcTmp)
    mXTest = size(XTest, 1);
    nBinary = size(colIndexMatrixTmp, 1);
    
    % 给XBinary赋值
    XBinary = zeros(mXTest, nBinary);
    for j=1:nBinary
        indexVec = centroidsIndexMatrixTmp(j, :);
        centroidsTmp = centroidsMatrixTmp(indexVec(1):indexVec(2), :);
        [~, KPred] = funcTmp(XTest(:, colIndexMatrixTmp(j, :)), centroidsTmp);
        XBinary(:, j) = KPred;
    end
end

end


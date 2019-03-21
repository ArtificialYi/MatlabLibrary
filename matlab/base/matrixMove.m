function [XLeft, XRight] = matrixMove(X)
%matrixMove 将矩阵按照第一个轴左右横移

% 将矩阵的每个维度数据取出
lenVec = size(X);
numAll = numel(X);

% 矩阵预分配
XRight = zeros(lenVec);
XLeft = zeros(lenVec);

% 找到横移后的无用索引
rightIndex = find(mod(1:numAll, lenVec(1))==1);
leftIndex = find(mod(1:numAll, lenVec(1))==0);
if lenVec(1) == 1
    rightIndex = leftIndex;
end

% 横移 && 替换横移后的无用数据
XRight(:) = X([1 1:numAll-1]);
XRight(rightIndex) = X(rightIndex);
XLeft(:) = X([2:numAll numAll]);
XLeft(leftIndex) = X(leftIndex);
end


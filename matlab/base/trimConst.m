function [noneConstX, noneIndex] = trimConst(X)
%trimConst 清除数据中的所有常量特征
%   X 原始数据
%   noneConstX 无常量特征数据
%   noneIndex 无常量特征索引

stdX = std(X);
noneIndex = find(stdX ~= 0);
noneConstX = X(:, noneIndex);
end


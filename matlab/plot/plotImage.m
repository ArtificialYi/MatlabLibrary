function [plotMatrix] = plotImage(X, height, weight, rowPad, colPad, imgVec)
%plotImage 画图像

n = size(X, 1);
% 获取行和列
row = ceil(sqrt(n*(weight+colPad)/(height+rowPad)));
col = ceil(n/row);

plotMatrix = zeros(row*(height+rowPad)+rowPad, col*(weight+colPad)+colPad)+imgVec(1);
for i=1:n
    % 获取行索引和列索引
    colIndex = mod(i, col);
    colIndex(colIndex==0) = col;
    rowIndex = (i-colIndex)/col+1;
    
    % 找到行列坐标
    rowBegin = rowIndex*(height+rowPad)-height+1;
    colBegin = colIndex*(weight+colPad)-weight+1;
    
    % 赋值
    plotMatrix(rowBegin:rowBegin+height-1, colBegin:colBegin+weight-1) = ...
        reshape(X(i, :), height, weight);
end

imagesc(plotMatrix, imgVec);
end


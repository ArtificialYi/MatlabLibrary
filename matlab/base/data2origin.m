function [dataOrigin, func] = data2origin(data, row, col)
%data2origin 将未处理的数据转成数据形式

% 将基本数据转成可见
dataOrigin = zeros(row, col);

indexEnd = 0;
indexMatrix = zeros(0, 3);
VUniqueVec = string([]);
for i=1:col
    dataTmp = data{i}(:);
    [dataOrigin(:, i), VUniqueTmp] = vec2double(dataTmp);
    if ~isempty(VUniqueTmp)
        indexBegin = indexEnd + 1;
        indexEnd = indexEnd + length(VUniqueTmp);
        VUniqueVec(indexBegin:indexEnd, 1) = VUniqueTmp(:);
        indexMatrix(end+1, :) = [i, indexBegin, indexEnd];
    end

end

func = @(paramXTest) dataTest2dataTestOrigin(paramXTest, VUniqueVec, indexMatrix);
function XTestOrigin = dataTest2dataTestOrigin(XTest, uniqueVec, indexMatrixTmp)
    
    n = size(XTest, 2);
    indexTmp = 1;
    indexMax = size(indexMatrixTmp, 1);
    
    % 初始化XTestOrigin
    colTmp = XTest{1}(:);
    mTest = size(colTmp, 1);
    XTestOrigin = zeros(mTest, n);
    
    for j=1:n
        % 获取数据的列
        XColTmp = XTest{j}(:);
        
        % 如果该列需要转成枚举
        if indexTmp <= indexMax && j==indexMatrixTmp(indexTmp, 1)
            XColTmp = vecStr2vecIndex(XColTmp(:), ...
                uniqueVec(indexMatrixTmp(indexTmp, 2):indexMatrixTmp(indexTmp, 3)));
            indexTmp = indexTmp + 1;
        end
        
        XTestOrigin(:, j) = XColTmp(:);
    end
end

end

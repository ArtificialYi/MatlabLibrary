function [XFeature, func] = binaryFeature(X, lenMax)
%binaryFeature 将所有枚举型特征扩充为2进制特征

[mX, nX] = size(X);

matrixFeatureTmp = zeros(mX,0);

recoverMatrix = zeros(0, 3);
recoverVec = zeros(0, 1);
lenRecoverV = 0;

for i=1:nX
    VTrainTmp = unique(X(:, i));
    lenVTrainTmp = length(VTrainTmp);
    if lenVTrainTmp > 2 && lenVTrainTmp < lenMax
        matrixTrainTmp = VTrainTmp' == X(:, i);
        showHy(matrixTrainTmp, 'matrixTrainTmp');
        matrixFeatureTmp(:, end+1:end+lenVTrainTmp-1) = matrixTrainTmp;
        
        % 恢复用数据
        recoverMatrix(end+1, :) = [i, lenRecoverV+1, lenRecoverV+lenVTrainTmp];
        recoverVec(end:end+lenVTrainTmp-1) = VTrainTmp(:);
        lenRecoverV = length(recoverVec);
    end
end

% 扩充后的特征
XFeature = [X matrixFeatureTmp];

% 扩展用的函数
func = @(paramX) data2binaryData(paramX, recoverVec, recoverMatrix);
function XBinary = data2binaryData(XOrigin, recoverVecTmp, recoverMatrixTmp)
    m = size(XOrigin, 1);
    
    dataBinaryTmp = zeros(m, 0);
    for j=1:size(recoverMatrixTmp, 1)
        lenVec = recoverMatrixTmp(j, 3) - recoverMatrixTmp(j, 2) + 1;
        dataBinaryTmp(:, lenVec) = XOrigin(:, recoverMatrixTmp(j, 1)) == ...
            recoverVecTmp(recoverMatrixTmp(j, 2):recoverMatrixTmp(j, 3))';
    end
    XBinary = [XOrigin dataBinaryTmp];
end

end


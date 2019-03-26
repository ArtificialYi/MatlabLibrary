function [XTrain, XVal, XTest] = splitData(XOrigin, indexVec, trainPoint, valPoint)
%splitData 分割数据

m = size(XOrigin, 1);
mTrainIndex = floor(m * trainPoint);
mValIndex = floor(m * (trainPoint+valPoint));

% 分割结果
XTrain = XOrigin(indexVec(1: mTrainIndex), :);

XVal = XOrigin(indexVec(mTrainIndex+1: mValIndex), :);

XTest = XOrigin(indexVec(mValIndex+1: m), :);

end


function [XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVec, trainPoint, valPoint)
%splitOriginData 将原始数据分割成 训练集、交叉验证集、训练集
% 初始化随机索引
m = size(XOrigin, 1);
mTrainIndex = floor(m * trainPoint);
mValIndex = floor(m * (trainPoint+valPoint));

% 分割结果
XTrain = XOrigin(indexVec(1: mTrainIndex), :);
YTrain = YOrigin(indexVec(1: mTrainIndex), :);

XVal = XOrigin(indexVec(mTrainIndex+1: mValIndex), :);
YVal = YOrigin(indexVec(mTrainIndex+1: mValIndex), :);

XTest = XOrigin(indexVec(mValIndex+1: m), :);
YTest = YOrigin(indexVec(mValIndex+1: m), :);
end
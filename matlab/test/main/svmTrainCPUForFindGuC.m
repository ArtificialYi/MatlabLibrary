function [guMin, CMin, errorMin] = svmTrainCPUForFindGuC(func, predGu, predC)
%svmTrainCPUForFindGuC 使用SVM查找gu和C

split = 21;
indexMin = split;

guVec = logspace(-5, 5, split);
guLeft = guVec(1);
guRight = guVec(end);

% 初始化左右极限
[errorValVecTmp, CVecTmp] = svmTrainCPUForGuVec(func, guVec([1 end]), predC);
errorLeft = errorValVecTmp(1);
CLeft = CVecTmp(1);
errorRight = errorValVecTmp(end);
CRight = CVecTmp(end);

while guRight - guLeft > predGu
    % 先计算非极限值
    [errorValVecTmp, CVecTmp] = svmTrainCPUForGuVec(func, guVec(2:end-1), predC);
    % 将左右极限拼上去
    errorValVecTmp = [errorLeft errorValVecTmp errorRight];
    CVecTmp = [CLeft CVecTmp CRight];
    
    % 获取索引值
    indexVec = indexMinForMulti(errorValVecTmp);
    indexMin = indexVec(1);
    guMin = guVec(indexMin);
    CMin = CVecTmp(indexMin);
    errorMin = errorValVecTmp(indexMin);
    
    % 获取左右索引
    [indexLeft, indexRight] = getLeftAndRightIndex(indexMin, 1, split);
    guLeft = guVec(indexLeft);
    guRight = guVec(indexRight);
    
    % 获取新的CVec
    guVec = linspace(guLeft, guRight, split);
    % 将左右极限保留下来
    errorLeft = errorValVecTmp(indexLeft);
    errorRight = errorValVecTmp(indexRight);
end

end

function [CMin, errorMin] = svmTrainCPUForFindC(func, pred)
%svmTrainCPUForFindC 用svm算法查找固定C的情况下的最优C

% 先用等比数列找到范围
split = 11;
indexMin = split;
CVec = logspace(-5, 5, split);
CLeft = CVec(1);
CRight = CVec(end);

% 初始化左右极限
errorValVecTmp = svmTrainCPUForCVec(func, CVec([1 end]));
errorLeft = errorValVecTmp(1);
errorRight = errorValVecTmp(end);

while CRight - CLeft > pred
    % 先计算非极限值
    errorValVecTmp = svmTrainCPUForCVec(func, CVec(2:end-1));
    % 将左右极限拼上去
    errorValVecTmp = [errorLeft errorValVecTmp errorRight];
    
    % 获取索引值
    indexVec = indexMinForMulti(errorValVecTmp);
    indexMin = indexVec(end);
    CMin = CVec(indexMin);
    errorMin = errorValVecTmp(indexMin);
    
    % 获取左右索引
    [indexLeft, indexRight] = getLeftAndRightIndex(indexMin, 1, split);
    CLeft = CVec(indexLeft);
    CRight = CVec(indexRight);
    
    % 获取新的CVec
    CVec = linspace(CLeft, CRight, split);
    % 将左右极限保留下来
    errorLeft = errorValVecTmp(indexLeft);
    errorRight = errorValVecTmp(indexRight);
end


end


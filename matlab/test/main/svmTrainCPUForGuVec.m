function [CMinVec, errorMinVec] = svmTrainCPUForGuVec(func, guVec, pred)
%svmTrainCPUForGuVec 返回gu向量对应的最优C和最小误差

CMinVec = guVec;
errorMinVec = guVec;

for i=1:length(guVec)
    svmFunCWithC = @(paramC) func(paramC, guVec(i));
    [CMinTmp, errorMinTmp] = svmTrainCPUForFindC(svmFunCWithC, pred);
    CMinVec(i) = CMinTmp;
    errorMinVec(i) = errorMinTmp;
end

end


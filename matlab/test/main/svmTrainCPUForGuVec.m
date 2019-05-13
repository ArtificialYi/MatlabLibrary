function [errorMinVec, CMinVec] = svmTrainCPUForGuVec(func, guVec, predC)
%svmTrainCPUForGuVec 返回gu向量对应的最优C和最小误差

CMinVec = guVec;
errorMinVec = guVec;

for i=1:length(guVec)
    fprintf('当前gu:%f\n', guVec(i));
    svmFunCWithC = @(paramC) func(paramC, guVec(i));
    [CMinTmp, errorMinTmp] = svmTrainCPUForFindC(svmFunCWithC, predC);
    fprintf('最优C:%f,最小误差:%f\n', CMinTmp, errorMinTmp);
    CMinVec(i) = CMinTmp;
    errorMinVec(i) = errorMinTmp;
end

end


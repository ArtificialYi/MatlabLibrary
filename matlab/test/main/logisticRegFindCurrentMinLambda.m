function [lambdaCurrentGPU, errorMinCurrentGPU] = ...
    logisticRegFindCurrentMinLambda(XTrainGPU, YTrainGPU, XValGPU, YValGPU, ...
    thetaInitGPU, maxIterGPU, predGPU, predLambdaGPU)
%logisticRegFindCurrentMinLambda 逻辑回归找到当前最优lambda

splitLambdaInitGPU = gpuArray(11);
splitLambdaCurrentGPU = gpuArray(21);

%% 先用等比数列找到范围
lambdaVecCurrentGPU = logspace(gpuArray(-5), gpuArray(5), splitLambdaInitGPU);
lambdaLeftCurrentGPU = lambdaVecCurrentGPU(1);
lambdaRightCurrentGPU = lambdaVecCurrentGPU(end);

% 再用等差数列做循环
while lambdaRightCurrentGPU - lambdaLeftCurrentGPU > predLambdaGPU
    showHy('等差数列里面的循环', '测试循环');
    [~, errorValCurrentTmpGPU] = ...
        logisticRegTrainForLambdaVec(XTrainGPU, YTrainGPU, XValGPU, YValGPU, ...
            thetaInitGPU, lambdaVecCurrentGPU, maxIterGPU, predGPU);
    indexCurrentGPU = indexMinForMulti(errorValCurrentTmpGPU);
    indexCurrentGPU = indexCurrentGPU(1);
    
    [indexCurrentLeftTmpGPU, indexCurrentRightTmpGPU] = ...
        getLeftAndRightIndex(indexCurrentGPU, 1, length(lambdaVecCurrentGPU));
    lambdaLeftCurrentGPU = lambdaVecCurrentGPU(indexCurrentLeftTmpGPU);
    lambdaRightCurrentGPU = lambdaVecCurrentGPU(indexCurrentRightTmpGPU);
    
    lambdaVecCurrentGPU = linspace(lambdaLeftCurrentGPU, lambdaRightCurrentGPU, splitLambdaCurrentGPU);
end

lambdaCurrentGPU = lambdaVecCurrentGPU(indexCurrentGPU);
errorMinCurrentGPU = errorValCurrentTmpGPU(indexCurrentGPU);

end


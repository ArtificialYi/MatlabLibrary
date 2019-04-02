function [errorTrainVecGPU, errorValVecGPU] = ...
    logisticRegTrainForLambdaVec(XTrainGPU, YTrainGPU, XValGPU, YValGPU, ...
    thetaInitGPU, lambdaVecGPU, maxIterGPU, predGPU)
%logisticRegTrainForLambdaVec 逻辑回归带lambda向量训练函数

errorTrainVecGPU = lambdaVecGPU;
errorValVecGPU = lambdaVecGPU;

lenLambda = length(lambdaVecGPU);

for i=1:lenLambda
    % 计算不同lambda
    [thetaTrainGPU, ~] = ...
        logisticRegTrainGPU(XTrainGPU, YTrainGPU, ...
            thetaInitGPU, lambdaVecGPU(i), maxIterGPU, predGPU);
    % 添加误差到数组
    errorTrainVecGPU(i) = logisticRegCostFunc(XTrainGPU, YTrainGPU, thetaTrainGPU, 0, predGPU);
    errorValVecGPU(i) = logisticRegCostFunc(XValGPU, YValGPU, thetaTrainGPU, 0, predGPU);
    fprintf('最优化:%d:%d:%f, %f\n', lenLambda, i, lambdaVecGPU(i), errorValVecGPU(i));
end

end


function [thetaGPU, costGPU, exitFlag] = ...
    logisticRegTrainGPU(XGPU, YGPU, thetaInitGPU, lambdaGPU, maxIterGPU, predGPU)
%logisticRegTrainGPU 逻辑回归训练函数

maxIter = gather(maxIterGPU);
X = gather(XGPU);
Y = gather(YGPU);
thetaInit = gather(thetaInitGPU);
lambda = gather(lambdaGPU);
pred = gather(predGPU);

options = optimoptions('fminunc', 'Display', 'off', 'MaxIter', maxIter);

func = @(t) logisticRegCostFunc(X, Y, t, lambda, pred);
fprintf('训练前夕');
[thetaGPU, costGPU, exitFlag] = fminunc(func, thetaInit, options);
fprintf('训练完毕');
exitFlag(exitFlag>0) = 1;
exitFlag(exitFlag<0) = -1;

switch exitFlag
    case 1
        fprintf('收敛成功:%d\n', maxIterGPU);
    case 0
        error('达到最大收敛次数:%d\n', maxIterGPU);
    case -1
        error('无法收敛\n');
end

end


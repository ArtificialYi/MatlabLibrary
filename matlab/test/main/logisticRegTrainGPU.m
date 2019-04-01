function [thetaGPU, costGPU, exitFlag] = logisticRegTrainGPU(XGPU, YGPU, thetaInitGPU, maxIterGPU)
%logisticRegTrainGPU 逻辑回归训练函数

maxIter = gather(maxIterGPU);
X = gather(XGPU);
Y = gather(YGPU);
thetaInit = gather(thetaInitGPU);

options = optimoptions('fminunc', 'Display', 'off', 'MaxIter', maxIter);

func = @(t) logisticRegCostFunc(X, Y, t);

[thetaGPU, costGPU, exitFlag] = fminunc(func, thetaInit, options);

exitFlag(exitFlag>0) = 1;
exitFlag(exitFlag<0) = -1;

switch exitFlag
    case 1
        fprintf('收敛成功:%d\n', maxIterGPU);
    case 0
        fprintf('达到最大收敛次数:%d\n', maxIterGPU);
    case -1
        fprintf('无法收敛\n');
end

end


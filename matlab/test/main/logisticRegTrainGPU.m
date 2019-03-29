function [thetaGPU, costGPU] = logisticRegTrainGPU(XGPU, YGPU, thetaInitGPU, maxIterGPU)
%logisticRegTrainGPU 逻辑回归训练函数

maxIter = gather(maxIterGPU);
X = gather(XGPU);
Y = gather(YGPU);
thetaInit = gather(thetaInitGPU);

options = optimoptions('fminunc', 'Display', 'final', 'MaxIter', maxIter);

func = @(t) logisticRegCostFunc(X, Y, t);

[thetaGPU, costGPU] = fminunc(func, thetaInit, options);

end


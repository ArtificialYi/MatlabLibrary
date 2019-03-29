function [thetaGPU, costGPU] = logisticRegTrainGPU(XGPU, YGPU, thetaInitGPU, maxIterGPU)
%logisticRegTrainGPU 逻辑回归训练函数

options = optimset('GradObj', 'on', 'MaxIter', maxIterGPU);

func = @(t)(logisticRegCostFunc(XGPU, YGPU, t));

[thetaGPU, costGPU] = fminunc(func, thetaInitGPU, options);

end


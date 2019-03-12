function [errorTrainGPU, errorValGPU, realSplitVecGPU] = ...
    svmLearningCurveGPU(XGPU, YGPU, XValGPU, YValGPU, CGPU, tolGPU, maxIterGPU, splitGPU)
%svmLearningCurve SVM的学习曲线
% X 训练集
% y 训练集结果
% XVal 交叉验证集
% yVal 交叉验证集结果
% C 正则化参数
% tol 精度
% maxIter 最大迭代次数

% 将入参转化为GPU参数

% 训练集大小
mGPU = size(XGPU, 1);

exist = existsOnGPU(mGPU);
fprintf('size也是GPU内存%d\n', exist);

if mGPU < splitGPU
    realSplitGPU = mGPU;
else
    realSplitGPU = splitGPU;
end

% 初始化结果数组
errorTrainGPU = gpuArray.zeros(realSplitGPU, 1);
errorValGPU = gpuArray.zeros(realSplitGPU, 1);
realSplitVecGPU = gpuArray.zeros(realSplitGPU, 1);

for i=1:realSplitGPU
    currentIndexGPU = floor(mGPU * i / realSplitGPU);
    XTmpGPU = XGPU(1:currentIndexGPU, :);
    YTmpGPU = YGPU(1:currentIndexGPU);
    alphaTmpGPU = gpuArray.zeros(currentIndexGPU, 1);

    modelTmpGPU = svmTrainGPU(XTmpGPU, YTmpGPU, CGPU, alphaTmpGPU, tolGPU, maxIterGPU);
    
    realSplitVecGPU(i) = currentIndexGPU;
    
    errorTrainGPU(i) = svmCost(XTmpGPU, YTmpGPU, modelTmpGPU.w, modelTmpGPU.b, 0);
    errorValGPU(i) = svmCost(XValGPU, YValGPU, modelTmpGPU.w, modelTmpGPU.b, 0);
end


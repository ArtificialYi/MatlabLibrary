function [errorTrainGPU, errorValGPU, realSplitVecGPU] = ...
    svmLearningCurveTrainGPU(XGPU, YGPU, XValGPU, YValGPU, CGPU, tolGPU, maxIterGPU, splitGPU, kernelFunc)
%svmLearningCurveGPU SVM的学习曲线
% X 训练集
% y 训练集结果
% XVal 交叉验证集
% yVal 交叉验证集结果
% C 正则化参数
% tol 精度
% maxIter 最大迭代次数

% 训练集大小
mGPU = size(XGPU, 1);

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
    KTrainTmpGPU = kernelFunc(XTmpGPU, XTmpGPU);
    KValTmpGPU = kernelFunc(XTmpGPU, XValGPU);
    
    modelTmpGPU = svmTrainGPU(KTrainTmpGPU, YTmpGPU, CGPU, alphaTmpGPU, tolGPU, maxIterGPU);
    
    realSplitVecGPU(i) = currentIndexGPU;
    
    errorTrainGPU(i) = svmCost(KTrainTmpGPU, YTmpGPU, KTrainTmpGPU, YTmpGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
    errorValGPU(i) = svmCost(KTrainTmpGPU, YTmpGPU, KValTmpGPU, YValGPU, modelTmpGPU.gpu.alpha, modelTmpGPU.gpu.b, 0);
end


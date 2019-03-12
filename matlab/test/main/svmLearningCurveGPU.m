function [errorTrain, errorVal, realSplitVec] = ...
    svmLearningCurveGPU(X, y, XVal, yVal, C, tol, maxIter, split, gpuNumArr)
%svmLearningCurve SVM的学习曲线
% X 训练集
% y 训练集结果
% XVal 交叉验证集
% yVal 交叉验证集结果
% C 正则化参数
% tol 精度
% maxIter 最大迭代次数

% 获取新的GPU资源
gpuDevice(gpuNumArr(1));

% 将入参转化为GPU参数
XGPU = gpuArray(X);
yGPU = gpuArray(y);
XValGPU = gpuArray(XVal);
yValGPU = gpuArray(yVal);
CGPU = gpuArray(C);
tolGPU = gpuArray(tol);
maxIterGPU = gpuArray(maxIter);
splitGPU = gpuArray(split);

% 训练集大小
mGPU = size(XGPU, 1);

if mGPU < splitGPU
    realSplitGPU = mGPU;
else
    realSplitGPU = splitGPU;
end

% 初始化结果数组
errorTrain = zeros(realSplitGPU, 1);
errorVal = zeros(realSplitGPU, 1);
realSplitVec = zeros(realSplitGPU, 1);

for i=1:realSplitGPU
    currentIndexGPU = floor(mGPU * i / realSplitGPU);
    XTmpGPU = XGPU(1:currentIndexGPU, :);
    yTmpGPU = yGPU(1:currentIndexGPU);
    alphaTmpGPU = gpuArray.zeros(currentIndexGPU, 1);

    modelTmp = svmTrainGPU(XTmpGPU, yTmpGPU, CGPU, alphaTmpGPU, tolGPU, maxIterGPU, gpuNumArr(2));
    
    realSplitVec(i) = gather(currentIndexGPU);
    
    errorTrain(i) = gather(svmCost(XTmpGPU, yTmpGPU, modelTmp.w, modelTmp.b, 0));
    errorVal(i) = gather(svmCost(XValGPU, yValGPU, modelTmp.w, modelTmp.b, 0));
end


function [tmp] = test6extra3multi1n(gu, C, maxIter)
%test6extra3multi1n SVM-高斯-GPU-考试成绩

% 初始化数据
gu = str2double(gu);
C = str2double(C);
maxIter = str2double(maxIter);


%% 读取数据
% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:,1:2);
YOrigin = data(:,3);
YOrigin(YOrigin==0)=-1;

m = size(XOrigin, 1);
trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

% 获取核结果
kernelFunc = @(X1, X2) svmKernelGaussian(X1, X2, gu);
KOrigin = kernelFunc(XOriginNorm, XOriginNorm);

% 边界线数据准备
minX1 = min(XOrigin(:,1));
maxX1 = max(XOrigin(:,1));
minX2 = min(XOrigin(:,2));
maxX2 = max(XOrigin(:,2));

splitTrain = 51;
vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

%% 基础训练模型
% CPU->GPU
KOriginGPU = gpuArray(KOrigin);
YOriginGPU = gpuArray(YOrigin);
CTrainGPU = gpuArray(C);
tolTrainGPU = gpuArray(1e-15);
maxIterTrainGPU = gpuArray(maxIter);
alphaTrainGPU = gpuArray.zeros(m, 1);

modelOriginGPU = ...
    svmTrainGPU(KOriginGPU, YOriginGPU, CTrainGPU, alphaTrainGPU, tolTrainGPU, maxIterTrainGPU);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
KTestTmp = kernelFunc(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOriginGPU.cpu.alpha .* YOrigin)'*KTestTmp'+modelOriginGPU.cpu.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线
%CPU->GPU
XTrainNormGPU = gpuArray(XTrainNorm);
YTrainGPU = gpuArray(YTrain);
XValNormGPU = gpuArray(XValNorm);
YValGPU = gpuArray(YVal);
CLearnGPU = gpuArray(C);
tolLearnGPU = gpuArray(1e-15);
maxIterLearnGPU = gpuArray(50000);
splitLearnGPU = gpuArray(101);

[errorTrainLearnGPU, errorValLearnGPU, realSplitVecLearnGPU] = ...
    svmLearningCurveGPU(XTrainNormGPU, YTrainGPU, ...
        XValNormGPU, YValGPU, CLearnGPU, ...
        tolLearnGPU, maxIterLearnGPU, splitLearnGPU, kernelFunc);

%% save
% 获取文件名
fileName = sprintf('data/data_test6extra3multi1n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName);
save(fileName, ...
    'XOrigin', 'YOrigin', 'vecX1', 'vecX2', 'predYTestTmp_2D', ...
    'realSplitVecLearn', 'errorTrainLearn', 'errorValLearn');
fprintf('保存完毕\n');
end


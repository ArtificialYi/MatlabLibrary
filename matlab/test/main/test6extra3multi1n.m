function [tmp] = test6extra3multi1n(gu, C, maxIter, guLeft, guRight, isTrain)
%test6extra3multi1n SVM-高斯-GPU-考试成绩

% 初始化数据
gu = str2double(gu);
C = str2double(C);
maxIter = str2double(maxIter);
guLeft = str2double(guLeft);
guRight = str2double(guRight);
isTrain = str2double(isTrain);

tol = 1e-8;

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
tolTrainGPU = gpuArray(tol);
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
tolLearnGPU = gpuArray(tol);
maxIterLearnGPU = gpuArray(maxIter);
splitLearnGPU = gpuArray(50);

[errorTrainLearnGPU, errorValLearnGPU, realSplitVecLearnGPU] = ...
    svmLearningCurveGPU(XTrainNormGPU, YTrainGPU, ...
        XValNormGPU, YValGPU, CLearnGPU, ...
        tolLearnGPU, maxIterLearnGPU, splitLearnGPU, kernelFunc);
    
% 学习曲线CPU数据
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitVecLearn = gather(realSplitVecLearnGPU);

%% 尝试找到全局最优C&gu
guVec = linspace(guLeft, guRight, 21);
guVec = guVec(2:end);
mGu = size(guVec, 2);

predCurrentGPU = gpuArray(1e-3);
tolCurrentGPU = gpuArray(tol);
maxIterCurrentGPU = gpuArray(maxIter);

errorMinVec = zeros(mGu, 1);
CMinVec = zeros(mGu, 1);

if isTrain
    for i=1:length(guVec)
        fprintf('1s后执行gu:%f\n', guVec(i));
        pause(1);
        kernelFunc = @(X1, X2) svmKernelGaussian(X1, X2, guVec(i));
        KTrainGPU = kernelFunc(XTrainNormGPU, XTrainNormGPU);
        KValGPU = kernelFunc(XTrainNormGPU, XValNormGPU);

        [CCurrentGPU, errorMinCurrentGPU] = ...
            svmFindCurrentMinC(KTrainGPU, YTrainGPU, KValGPU, YValGPU, tolCurrentGPU, maxIterCurrentGPU, predCurrentGPU);
        errorMinVec(i) = gather(errorMinCurrentGPU);
        CMinVec(i) = gather(CCurrentGPU);
    end
end

% 找到最优C&gu
indexMinVec = indexMinForMulti(errorMinVec);

guMinVec = guVec(indexMinVec);
CMinVec = CMinVec(indexMinVec);
% 直接用最大的那个就完事了
guMin = guMinVec(end);
CMin = CMinVec(end);

%% save
% 获取文件名
fileName = sprintf('data/data_test6extra3multi1n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName);
save(fileName, ...
    'XOrigin', 'YOrigin', 'vecX1', 'vecX2', 'predYTestTmp_2D', ...
    'realSplitVecLearn', 'errorTrainLearn', 'errorValLearn', ...
    'XTrain', 'YTrain', 'XVal', 'YVal', ...
    'guMin', 'CMin', 'guMinVec', 'CMinVec');
fprintf('保存完毕\n');
end


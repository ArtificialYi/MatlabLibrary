function [tmp] = test7base0n(K, maxIter)
%test7base0n 无监督初始化

K = str2double(K);
maxIter = str2double(maxIter);

%% 读取数据
% 读取数据
data = load('resource/ex7data1.mat');
XOrigin = data.X;

m = size(XOrigin, 1);
trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, XVal, XTest] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

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
indexVecRand = randperm(m, K);
XOriginNormGPU = gpuArray(XOriginNorm);
centroidsGPU = XOriginNormGPU(indexVecRand);
maxIterGPU = gpuArray(maxIter);

[centroidsOriginGPU, YOriginGPU] = kMeanTrainGPU(XOriginNormGPU, centroidsGPU, maxIterGPU);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);

XTestTmpNormGPU = gpuArray(XTestTmpNorm);

[~, YTestGPU] = kMeanTrainGPU(XTestTmpNormGPU, centroidsOriginGPU, 1);

% 输出数据准备
centroidsOrigin = gather(centroidsOriginGPU);
YTest = gather(YTestGPU);

%% save
% 获取文件名
fileName = sprintf('data/data_test7base0n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName);
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', ...
    'centroidsOrigin', 'vecX1', 'vecX2', 'YTest');
fprintf('保存完毕\n');

end


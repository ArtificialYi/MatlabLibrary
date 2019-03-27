function [pcaRes] = testPca0(K)
%testPca0 pca测试

%% str2double
K = str2double(K);

%% 读取数据
data = load('resource/ex7data1.mat');
XOrigin = data.X;

[m, n] = size(XOrigin);
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

splitTrain = 101;
vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

%% 基础训练模型
% CPU->GPU
XOriginNormGPU = gpuArray(XOriginNorm);
nGPU = gpuArray(n);
KGPU = gpuArray(K);

[UOriginGPU, SOriginGPU] = pcaTrainGPU(XOriginNormGPU);
XOriginPcaGPU = data2pca(XOriginNormGPU, UOriginGPU, nGPU);
XOriginPcaKGPU = data2pca(XOriginNormGPU, UOriginGPU, KGPU);
XOriginRecKGPU = pca2data(XOriginPcaKGPU, UOriginGPU, KGPU);

UOrigin = gather(UOriginGPU);
SOrigin = gather(SOriginGPU);
XOriginPca = gather(XOriginPcaGPU);
XOriginPcaK = gather(XOriginPcaKGPU);
XOriginRecK = gather(XOriginRecKGPU);

%% save
% 获取文件名
fileName = sprintf('data/data_testPca0_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'XOrigin', 'XTrain', 'XVal', 'vecX1', 'vecX2', 'sigma', 'mu', ...
    'UOrigin', 'SOrigin', 'XOriginPca', 'XOriginPcaK', 'XOriginRecK');
fprintf('保存完毕\n');
end

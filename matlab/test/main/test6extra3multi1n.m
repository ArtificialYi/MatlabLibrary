function [tmp] = test6extra3multi1n(p, l, s, C, isTrain)
%test6extra3multi1n SVM-多项式-GPU-芯片数据与质量

% 初始化数据
p = str2double(p);
l = str2double(l);
s = str2double(s);
C = str2double(C);
isTrain = str2double(isTrain);

%% 读取数据
% 读取数据
data = load('resource/ex2data2.txt');
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
kernelFunc = @(X1, X2) svmKernelPolynomial(X1, X2, l, s, p);
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

%% save
% 获取文件名
fileName = sprintf('data/data_test6extra3multi1n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
save(fileName, ...
    'XOrigin', 'YOrigin', 'vecX1', 'vecX2');
end


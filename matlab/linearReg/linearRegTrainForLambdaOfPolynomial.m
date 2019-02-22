function [errorTrain, errorVal, noneIndexOrigin, noneIndexNorm, mu, sigma] = ...
    linearRegTrainForLambdaOfPolynomial(X, y, XVal, yVal, p, lambdaVec)
%linearRegTrainForLambdaOfPolynomial 线性回归在某个多项式次数下不同lambda的结果
% X 原始训练集
% y 原始训练集的结果
% XVal 交叉验证集
% yVal 交叉验证集的结果
% p 多项式次数
% lambdaVec 正则化向量
addpath('./lib');

% 参数准备
m = size(X, 1);
mVal = size(XVal, 1);

% 多项式数据准备
[normNoneConstX, noneIndexOrigin, noneIndexNorm, mu, sigma] = ...
    featurePolynomialParam(X, p);

% 获取交叉验证集的多项式特征
normPolyXVal = mapFeatureWithParam(XVal, p, noneIndexOrigin, noneIndexNorm, mu, sigma);

% 获取真实的多项式训练集&交叉验证集
realPolyX = [ones(m, 1) normNoneConstX];
realPolyXVal = [ones(mVal, 1) normPolyXVal];

% 初始化theta
initThetaPoly = zeros(size(realPolyX, 2), 1);
[errorTrain, errorVal] = ...
    linearRegTrainForLambda(realPolyX, y, realPolyXVal, yVal, initThetaPoly, lambdaVec);


end
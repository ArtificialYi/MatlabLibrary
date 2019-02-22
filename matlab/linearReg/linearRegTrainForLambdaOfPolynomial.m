function [errorTrain, errorVal, noneIndexOrigin, noneIndexNorm, mu, sigma] = ...
    linearRegTrainForLambdaOfPolynomial(X, y, XVal, yVal, p, lambdaVec)
%linearRegTrainForLambdaOfPolynomial 线性回归在某个多项式次数下不同lambda的结果
% X 原始训练集
% y 原始训练集的结果
% XVal 交叉验证集
% yVal 交叉验证集的结果
% p 多项式次数
% lambdaVec 正则化向量

% 参数准备
m = size(X, 1);
mVal = size(XVal, 1);

% 清空X的常量
[noneConstX, noneIndexOrigin] = trimConst(X);

% 获取X的多项式特征
polyX = mapFeature2polynomial(noneConstX, p);
[normNoneConstX, mu, sigma, noneIndexNorm] = ...
    featureNormalize(polyX);
realPolyX = [ones(m, 1) normNoneConstX];

% 获取交叉验证集的多项式特征
polyXVal = mapFeature2polynomial(XVal(:, noneIndexOrigin), p);
muPolyXVal = bsxfun(@minus, polyXVal(:, noneIndexNorm), mu);
normPolyXVal = bsxfun(@rdivide, muPolyXVal, sigma);
realPolyXVal = [ones(mVal, 1) normPolyXVal];

% 初始化theta
initThetaPoly = zeros(size(realPolyX, 2), 1);
[errorTrain, errorVal] = ...
    linearRegTrainForLambda(realPolyX, y, realPolyXVal, yVal, initThetaPoly, lambdaVec);


end
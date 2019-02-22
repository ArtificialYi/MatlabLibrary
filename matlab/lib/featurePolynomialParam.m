function [normNoneConstX, noneIndexOrigin, noneIndexNorm, mu, sigma] = ...
    featurePolynomialParam(X, p)
%featurePolynomialParam 多项式数据初始化的准备

addpath('./base');
% 清空X的常量
[noneConstX, noneIndexOrigin] = trimConst(X);

% 获取多项式扩充后的X
polyX = mapFeature2polynomial(noneConstX, p);
% 获取数据标准化&清空常量的X
[normNoneConstX, mu, sigma, noneIndexNorm] = ...
    featureNormalize(polyX);

end
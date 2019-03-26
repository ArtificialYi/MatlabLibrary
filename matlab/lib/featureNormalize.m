function [normNoneConstX, mu, sigma, noneIndex] = featureNormalize(X)
%featureNormalize 特征缩放
% X 原始数据集

% 清除常量变量
[noneConstX, noneIndex] = trimConst(X);

% 减去平均值
mu = mean(noneConstX);
avgNoneConstX = bsxfun(@minus, noneConstX, mu);

% 除去标准差
sigma = std(avgNoneConstX);
normNoneConstX = bsxfun(@rdivide, avgNoneConstX, sigma);

end

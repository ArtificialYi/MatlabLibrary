function normPolyX = ...
    mapFeatureWithParam(X, p, noneIndexOrigin, noneIndexNorm, mu, sigma)
%mapFeatureWithParam 将数据多项式扩充&标准化
% X 待处理数据
% p 多项式次数
% noneIndexOrigin 扩充前非常量索引
% noneIndexNorm 扩充后非常量索引
% mu 扩充后均值
% sigma 扩充后标准差

% 清除常量&扩充多项式
polyX = mapFeature2polynomial(X(:, noneIndexOrigin), p);
% 清除常量&减去平均值
muPolyX = bsxfun(@minus, polyX(:, noneIndexNorm), mu);
% 除去标准差
normPolyX = bsxfun(@rdivide, muPolyX, sigma);
end
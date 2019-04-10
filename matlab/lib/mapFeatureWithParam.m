function XNormPolyNorm = ...
    mapFeatureWithParam(X, p, ...
    muNorm, sigmaNorm, indexNorm, ...
    muPoly, sigmaPoly, indexPoly)
%mapFeatureWithParam 将数据多项式扩充&标准化
% X 待处理数据
% p 多项式次数
% muNorm, sigmaNorm, indexNorm 第一次归一化所需参数
% muPoly, sigmaPoly, indexPoly 第二次归一化所需参数

% 清除常量&减去平均值
XMu = bsxfun(@minus, X(:, indexNorm), muNorm);
showHy(XMu, 'XMu');
% 除去标准差
XNorm = bsxfun(@rdivide, XMu, sigmaNorm);

% 清除常量&扩充多项式
XNormPoly = mapFeature2polynomial(XNorm, p);

% 清除常量&减去平均值
XNormPolyMu = bsxfun(@minus, XNormPoly(:, indexPoly), muPoly);
% 除去标准差
XNormPolyNorm = bsxfun(@rdivide, XNormPolyMu, sigmaPoly);

end
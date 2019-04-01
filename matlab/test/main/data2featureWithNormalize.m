function [XNormPolyNorm, data2normFunc] = data2featureWithNormalize(X, p)
%data2featureWithNormalize 扩充多项式&归一化

% 归一化
[XNorm, muNorm, sigmaNorm, indexNorm] = featureNormalize(X);
% 扩充多项式
XNormPoly = mapFeature2polynomial(XNorm, p);
% 归一化
[XNormPolyNorm, muPoly, sigmaPoly, indexPoly] = featureNormalize(XNormPoly);

data2normFunc = @(paramX) mapFeatureWithParam(paramX, p, ...
    muNorm, sigmaNorm, indexNorm, muPoly, sigmaPoly, indexPoly);

end


function [K] = svmKernelGaussian(X1, X2, gu)
%svmKernelGaussian 高斯核函数

% 取出核数据
m1 = size(X1, 1);
m2 = size(X2, 1);

% 获取高斯核准备数据
X1Multi = multiMatrix(X1, m2);
X2Repeat = repeatMatrix(X2, m1);

% 开始计算
XVec = exp(-sum((X2Repeat - X1Multi).^2, 2)./(2*gu^2));

K = reshape(XVec, m1, m2)';
end


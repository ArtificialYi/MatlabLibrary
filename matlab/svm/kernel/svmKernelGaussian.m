function [K] = svmKernelGaussian(X1, X2, gu)
%svmKernelGaussian 高斯核函数

% 取出核数据
m2 = size(X2, 1);

K = X2(:, 1) + X1(:, 1)';
for i=1:m2
    K(i, :) = exp(-sum((X2(i, :) - X1).^2,2)./(2*gu^2));  
end

end


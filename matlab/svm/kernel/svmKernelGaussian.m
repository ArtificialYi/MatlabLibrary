function [K] = svmKernelGaussian(X1, X2, gu)
%svmKernelGaussian 高斯核函数

% 取出核数据
m1 = size(X1, 1);
m2 = size(X2, 1);

K = X2(:, 1) + X1(:, 1)';

for i=1:m2
    for j=1:m1
        vecTmp = X2(i, :) - X1(j, :);
        K(i, j) = exp(-vecTmp*vecTmp'/(2*gu^2));
    end
end

end


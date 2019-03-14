function [K] = svmKernelLinear(XOrigin, XPred)
%svmKernelLinear SVM线性核
%   SVM的线性核函数
K = XOrigin * XPred';
end


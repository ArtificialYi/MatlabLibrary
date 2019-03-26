function [K] = svmKernelPolynomial(X1, X2, l, s, p)
%svmKernelPolynomial svm多项式核

K = (X2 * X1' * l + s).^p;

end


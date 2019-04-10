function [tmp] = testSvmGaus(C, gu, guLeft, guRight, maxIter, isTrain)
%testSvmGaus SVM-高斯

% 初始化数据
gu = str2double(gu);
C = str2double(C);
maxIter = str2double(maxIter);
guLeft = str2double(guLeft);
guRight = str2double(guRight);
isTrain = str2double(isTrain);

tol = 1e-8;


end
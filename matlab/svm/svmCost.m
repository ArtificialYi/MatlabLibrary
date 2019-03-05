function [J] = svmCost(X, Y, w, b, C)
%svmCost SVM代价函数
% X 原始数据
% Y 原始结果
% w theta
% b 截距
% C 正则化程度

m = size(Y,1);
Y(Y==0) = -1;

predTheta = (X*w+b).*Y;
predTheta(predTheta>1)=1;

J = (sum(1-predTheta) + w'*w/(2*C)) / m;
end


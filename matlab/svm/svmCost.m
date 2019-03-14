function [J] = svmCost(K, Y, alpha, b, lambda)
%svmCost SVM代价函数
% 只有K=m*m的时候这个代价函数生效
% K 原始数据
% Y 原始结果
% alpha alpha
% b 截距
% lambda 正则化程度

m = size(Y,1);
Y(Y==0) = -1;

predTheta = (K*(alpha.*Y)+b).*Y;
predTheta(predTheta>1)=1;

J = (sum(1-predTheta) + ((alpha.*Y)'*K*(alpha.*Y))*lambda/2) / m;

end


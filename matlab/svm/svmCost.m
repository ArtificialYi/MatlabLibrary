function [J, point] = svmCost(KOrigin, YOrigin, KPred, YPred, alpha, b, lambda)
%svmCost SVM代价函数
% 只有K=m*m的时候这个代价函数生效
% K 原始数据
% Y 原始结果
% alpha alpha
% b 截距
% lambda 正则化程度

mPred = size(YPred,1);
YOrigin(YOrigin==0) = -1;
YPred(YPred==0) = -1;

predTheta = (((alpha.*YOrigin)'*KPred)'+b).*YPred;
predTheta(predTheta>1)=1;

J = (sum(1-predTheta) + ((alpha.*YOrigin)'*KOrigin*(alpha.*YOrigin))*lambda/2) / mPred;
YOriginPred = (alpha.*YOrigin)'*KPred+b;
point = mean(YPred==YOriginPred);

end


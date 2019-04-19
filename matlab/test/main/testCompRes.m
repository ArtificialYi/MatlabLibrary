%% 用结果训练
clear; close all; clc;

%% 读取csv
data = csvread('data/comp_res.csv');

%% 转成数据
allLine = 350;

XOrigin = data(1:allLine, :)';
XOrigin(XOrigin==0) = -1;
YOrigin = data(allLine+1, :)';

distVec = sum((XOrigin(1:end-1, :) - XOrigin(end, :)).^2/4,2);

% 找到最大的Y，设置为初始theta
[~, indexMax] = max(YOrigin);
thetaInit = XOrigin(indexMax, :)'*0.5;
thetaInit = rand(allLine, 1)*2-1;



lambda = 1;
maxIter = 1e5;
options = optimset('MaxIter', maxIter, 'GradObj', 'on', 'Algorithm', 'trust-region');
costFunc = @(paramTheta) linearRegCostComp(XOrigin, YOrigin, paramTheta, lambda);

[theta, Jval, exitFlag] = ...
        fminunc(costFunc, thetaInit, options);

% 开始迭代
tol = 1;
JTmpPtr = 1;
JTmpPre = 2;
thetaTmp = thetaInit;
alpha = 1/allLine;
time = 0;
while tol > 1e-8 && JTmpPtr > 1e-8
    [JTmpPtr, gradTheta] = linearRegCostComp(XOrigin, YOrigin, thetaTmp, lambda);
    thetaTmp = thetaTmp - alpha * gradTheta;
    time = time + 1;
    tol = abs(JTmpPtr - JTmpPre);
    JTmpPre = JTmpPtr;
    
    % 开始走歪路
    if tol < 1e-8 && JTmpPtr > 1e-8
        
    end
end

thetaPred = double(thetaTmp>0);

%% 代价函数
function [J, gradTheta] = linearRegCostComp(X, Y, theta, lambda)
%linearRegCostComp 线性回归正则化代价函数
% X的枚举为[-1,1]
% Y、h的范围为[0-n]
% theta的枚举为[-0.5, 0.5]

% 数据的大小
[m, n] = size(X);

% 模型函数计算结果
h = X * theta + n / 2;

% 计算代价函数
constraint = lambda*sum((theta.*theta - 0.25).^2);
constraintGrad = 4*theta.^3-theta;

J = ((h-Y)' * (h-Y) + constraint*2)/ (m*2);
gradTheta = (((h-Y)' * X)' + constraintGrad) / m;
gradTheta = gradTheta(:);
end
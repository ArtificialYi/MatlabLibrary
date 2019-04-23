%% 用结果训练
clear; close all; clc;

%% 读取csv
data = csvread('data/comp_res.csv');

%% 转成数据
allLine = 350;

XOrigin = data(1:allLine, :)';
XOrigin(XOrigin==0) = -1;
YOrigin = data(allLine+1, :)';
m = size(XOrigin, 1);

distVec = sum((XOrigin(1:end-1, :) - XOrigin(end, :)).^2/4,2);

% 预测最后一行的结果
predYRange = zeros(m - 1, 2) - distVec;
predYRange(:, 2) = distVec;
predYRange(:, :) = predYRange + YOrigin(1:end-1);
predYLeft = max(predYRange(:, 1));
predYRight = min(predYRange(:, 2));
predY = [predYLeft predYRight];

% 找到最大的Y，设置为初始theta
[~, indexMax] = max(YOrigin);
thetaInit = XOrigin(indexMax, :)'*0.5;

lambda = 1;
maxIter = 1e6;
offset = zeros(m, 1) + allLine / 2;
options = optimset('MaxIter', maxIter, 'GradObj', 'on', 'Algorithm', 'trust-region');
costFunc = @(paramTheta) linearRegCostComp(XOrigin, YOrigin, paramTheta, lambda, offset);

%[theta, Jval, exitFlag] = ...
%        fminunc(costFunc, thetaInit, options);

% 开始迭代
tol = 1;
JTmpPtr = 1;
JTmpPre = 2;
thetaTmp = thetaInit;
alpha = 1/allLine;
time = 0;

% 最优化theta
vecIndexTrain = 1:allLine;
vecIndexOK = [];
while JTmpPtr > 1e-8
    [JTmpPtr, gradTheta] = linearRegCostComp(XOrigin(:, vecIndexTrain), YOrigin, thetaTmp(vecIndexTrain), lambda, offset);
    thetaTmp(vecIndexTrain) = thetaTmp(vecIndexTrain) - alpha * gradTheta;
    time = time + 1;
    tol = abs(JTmpPtr - JTmpPre);
    JTmpPre = JTmpPtr;
    
    % 最大迭代次数
    if time > maxIter
        break;
    end
    
    % 开始走歪路
    if tol < 1e-8 && JTmpPtr > 1e-8
        % 找到最有可能正确的theta索引
        [~, indexOKTrain] = max(abs(thetaTmp(vecIndexTrain)));
        indexOK = vecIndexTrain(indexOKTrain);
        
        % 将该索引的值修改为对应的极限
        thetaTmp(indexOK) = sign(thetaTmp(indexOK))*0.5;
        
        % 重新计算特征
        vecIndexOK = find(abs(thetaTmp)==0.5);
        vecIndexTrain = find(abs(thetaTmp)~=0.5);
        
        % 重新设置偏移
        offset = XOrigin(:, vecIndexOK)*thetaTmp(vecIndexOK) + allLine / 2;
        
        % 重新设置pre
        JTmpPre = JTmpPtr * 2;
        
        % 极限设置
        if length(vecIndexOK) == allLine
            break;
        end
    end
end

thetaPred = double(thetaTmp>0);

%% 代价函数
function [J, gradTheta] = linearRegCostComp(X, Y, theta, lambda, offset)
%linearRegCostComp 线性回归正则化代价函数
% X的枚举为[-1,1]
% Y、h的范围为[0-n]
% theta的枚举为[-0.5, 0.5]

% 数据的大小
m = size(X, 1);

% 模型函数计算结果
h = X * theta + offset;

% 计算代价函数
constraint = lambda*sum((theta.*theta - 0.25).^2);
constraintGrad = 4*theta.^3-theta;

J = ((h-Y)' * (h-Y) + constraint*2)/ (m*2);
gradTheta = (((h-Y)' * X)' + constraintGrad) / m;
gradTheta = gradTheta(:);
end
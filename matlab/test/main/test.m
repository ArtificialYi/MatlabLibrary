%% 初始化环境
clear; close all; clc;

% 28定律
scall = 0.2732;

% 基础入参
xVec = linspace(0,1, 1001);
xVec = xVec(2:end);

yVec = scall.^(log(xVec)/log(1-scall));
dVYVec = yVec*log(scall)./(xVec*log(1-scall));

% 画图图形

plot(xVec, yVec, xVec, dVYVec);
%% 初始化环境
clear; close all; clc;

% 
mius = [1 -3 4; ...
    2 0 2];

x = [-1;2];

dist = (mius-x).^2;
distSum = sum(dist);

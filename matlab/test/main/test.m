%% 初始化环境
clear; close all; clc;

%% 先读取数据
begin = 0.84;
vec = (ceil(350*begin):floor(350*(begin+0.01)))';
fprintf('%.6f %d\n', [vec/350 vec]');

plot(vec, vec/350);

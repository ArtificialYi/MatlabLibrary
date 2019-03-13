%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
data = load('../resource/ex6data1.mat');
XOrigin = data.X;
YOrigin = data.Y;

m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;
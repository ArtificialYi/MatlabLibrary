%% 清空工作区
clear; close all; clc;

%% 读取原始数据-抽象出测试集、训练集
% 读取原始数据
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;

fprintf('X的大小为:%d, %d\n', size(XOrigin, 1), size(XOrigin, 2));
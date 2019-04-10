%% 初始化环境
clear; close all; clc;

m = 1100;
n = 30;

XTrain = zeros(m, n);

% 读取基本数据
fTrainTxt = fopen('resource/pfm_train.txt', 'r');
data = textscan(fTrainTxt, '%d%d%s%s%d%d%s%d%d%s%d%d%s%d%s%d%d%s%s%d%d%d%d%d%d%d%d%d%d%d%d', ...
    'delimiter', ',');
fclose(fTrainTxt);

% 转成数值型
dataY = data(2);
dataX = data([1 3:end]);
[XOrigin, funcDataInit] = data2origin(dataX, m, n);
YOrigin = data2origin(dataY, m, 1);

% 将测试数据也转成可用格式
fTestTxt = fopen('resource/pfm_test.txt', 'r');
dataTestX = textscan(fTestTxt, '%d%s%s%d%d%s%d%d%s%d%d%s%d%s%d%d%s%s%d%d%d%d%d%d%d%d%d%d%d%d', ...
    'delimiter', ',');
fclose(fTestTxt);
XTest = funcDataInit(dataTestX);
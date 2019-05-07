%% 初始化环境
clear; close all; clc;

%%
data = load('resource/pfm_data.mat');
XOrigin = data.XOrigin;

[m, n] = size(XOrigin);

for i=1:n
    tmp = unique(XOrigin(:, i));
    len = size(tmp, 1);
    fprintf('%d:%d\n', i, len);
end
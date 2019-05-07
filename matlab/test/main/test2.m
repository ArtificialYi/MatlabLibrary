%% 初始化环境
clear; close all; clc;

%%
data = load('resource/pfm_data.mat');
XOrigin = data.XOrigin;
[m, n] = size(XOrigin);
colIndexMatrix = vec2subMatrix(1:n, 2);
len = size(colIndexMatrix, 1);

vecLen = zeros(len, 1);
for i=1:len
    tmp = unique(XOrigin(:, colIndexMatrix(i, :)));
    len = size(tmp, 1);
    vecLen(i) = len;
    fprintf('%d:%d\n', i, len);
end

hist(vecLen, 101);
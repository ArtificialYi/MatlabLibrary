%% 初始化环境
clear; close all; clc;

%%
data = load('resource/pfm_data.mat');
XOrigin = data.XOrigin;
[XOriginNorm, data2normFuncOrigin] = data2featureWithNormalize(XOrigin, 1);
[m, n] = size(XOriginNorm);
colIndexMatrix = vec2subMatrix(1:n, 2);
len = size(colIndexMatrix, 1);

vecLen = zeros(len, 1);
for i=1:len
    tmp = unique(XOriginNorm(:, colIndexMatrix(i, :)));
    len = size(tmp, 1);
    vecLen(i) = len;
    fprintf('%d:%d\n', i, len);
end

hist(vecLen, 101);
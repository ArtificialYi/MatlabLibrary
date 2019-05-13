%% 初始化环境
clear; close all; clc;

%%
funcVec = {};
funcVec1 = @(a) a + 1;
funcVec{1} = funcVec1;
funcVec{2} = funcVec1;

len = length(funcVec);

for i=1:len
    funcVec{i}(5)
end
%% 初始化环境
clear; close all; clc;

%% 先读取数据
vec = [1.2500 0.2500 0.1250 0 0 0];
[leftVec, rightVec] = matrixMove(vec');
rightVec(rightVec==0) = 1e8;
vecTmp = leftVec ./ rightVec;

[indexMin] = indexMinForMulti(vecTmp);
indexMin = indexMin(1) + 1;
vec2 = 1:6;


plot(vec2, vec);


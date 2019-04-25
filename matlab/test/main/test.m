%% 初始化环境
clear; close all; clc;

%% 计算工作日和休息日的学习时间分配
workTime = 0;
workDay = 5;

aVec = linspace(0, 11-workTime, (11-workTime)+1);

bVec = ((77*4 - 5*workTime*workDay) - 5*workDay*aVec) / (5*(7-workDay));

plot(aVec, bVec);
xlabel('工作日学习时间');
ylabel('休息日学习时间');


%% 计算曲率
errorElbowVec = [0.999 0.5 0.15 0.08 0.06 0.05 0.03 0.01 0 0];
errorElbowVec = errorElbowVec.*(1:10);
% 先计算每个点的斜率（导数）
[~, ~, errorDv] = indexMinForMulti(errorElbowVec);

% 通过斜率计算夹角，如果小于0，+pi
thetaVec = atan(errorDv);

% 计算夹角的斜率(导数),找到所有大于0的斜率中，最大的
[~, ~, thetaDv] = indexMinForMulti(thetaVec);

% 取最大值即为手肘法的K值
[~, K] = max(thetaDv);
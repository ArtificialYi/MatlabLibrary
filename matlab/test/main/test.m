%% 初始化环境
clear; close all; clc;

% 基础入参
gu = 6.1213;
year = 70;
firstWork = 20;

% 基础数据准备
xVec = linspace(1, year, year);
meanVec = zeros(1, year);

% 年龄与收入
yVec = exp(-(xVec-year/2).^2/(2*gu^2))/(gu*sqrt(2*pi));
meanVec(:) = 1/year;

% 收入与GDP比例
scallVec = yVec ./ meanVec;
[~, dVScallVec] = indexMinForMulti(scallVec);

% 画出年龄与收入曲线
figure(1);
plot(xVec, yVec, xVec, meanVec);
legend('年龄与收入占比', '年均收入');
title('年龄与收入曲线');

figure(2);
plot(xVec, scallVec, xVec, sqrt(dVScallVec)/2);
legend('收入与GDP比例', '工资涨幅');
title('收入与GDP比例');

%% 开始计算求gu
leftGu = 1e-2;
rightGu = 1e2;
pred = 1e-3;
scall = 0.22;
indexXVec = ceil(year*(0.5-scall/2)):floor(year*(0.5+scall/2));

while rightGu-leftGu>pred
    tmpGu = (leftGu+rightGu)/2;
    yVecTmp = exp(-(xVec-year/2).^2/(2*tmpGu^2))/(tmpGu*sqrt(2*pi));
    sumTmp = sum(yVecTmp(indexXVec));
    if sumTmp > 1-scall
        leftGu = tmpGu;
    elseif sumTmp < 1-scall
        rightGu = tmpGu;
    else
        break;
    end
end
guCompute = tmpGu;
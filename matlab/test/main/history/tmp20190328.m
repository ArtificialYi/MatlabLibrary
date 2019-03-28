%% 人均收入占比图
%% 初始化环境
clear; close all; clc;

% 基础入参
scall = 0.22;
gu = 10.6106;
year = 75;
firstWork = 25;
gdp = 142682.72;

% 基础数据准备
xVec = linspace(1, year, year);
meanVec = zeros(1, year);

% 年龄与收入
yVec = exp(-(xVec-year/2).^2/(2*gu^2))/(gu*sqrt(2*pi));
yVec = 1/sum(yVec)*yVec;
meanVec(:) = 1/year;

% 收入与GDP比例
scallVec = yVec ./ meanVec;
[~, dVScallVec] = indexMinForMulti(scallVec);

% 排行占比
% 28定律
scallAll = 0.32;
% 基础入参
pointSplit = 1000;
pointVec = linspace(0,1, pointSplit+1);
pointVec = pointVec(2:end);

moneyAllVec = scallAll.^(log(pointVec)/log(1-scallAll));
moneyDVVec = moneyAllVec*log(scallAll)./(pointVec*log(1-scallAll));

meneyMatrix = moneyDVVec' * scallVec;


% 画出年龄与收入曲线
figure(1);
plot(xVec, yVec, xVec, meanVec);
legend('年龄与收入占比', '年均收入');
title('年龄与收入曲线');

figure(2);
plot(xVec, scallVec, xVec, sqrt(dVScallVec)/2);
legend('收入与GDP比例', '工资涨幅');
title('收入与GDP比例');

figure(3);
plot(xVec, scallVec*gdp);
title('年龄对应收入');

figure(4);
mesh(xVec, pointVec, meneyMatrix);
xlabel('年龄');
ylabel('击败百分比');
title('年龄、收入');

%% 开始计算gu
leftGu = 1e-3;
rightGu = 1e3;
pred = 1e-3;

while rightGu - leftGu > pred
    tmpGu = (leftGu+rightGu)/2;
    yVecTmp = exp(-(xVec-year/2).^2/(2*tmpGu^2))/(tmpGu*sqrt(2*pi));
    
    sumAll = sum(yVecTmp);
    sumUp = sum(yVecTmp(firstWork:year-firstWork));
    sumTmp = sumUp /sumAll;
    
    if sumTmp > 1-scall
        leftGu = tmpGu;
    elseif sumTmp < 1-scall
        rightGu = tmpGu;
    else
        break;
    end
end
guCompute = tmpGu;

%% 开始计算scall
leftScall = 1e-3;
rightScall = 1;
pred = 1e-3;

while rightScall-leftScall>pred
    tmpScall = (leftScall+rightScall)/2;
    
    moneyAllVecTmp = tmpScall.^(log(pointVec)/log(1-tmpScall));
    moneyDVVecTmp = moneyAllVecTmp*log(tmpScall)./(pointVec*log(1-tmpScall));

    % 收入与GDP比例
    meneyMatrixTmp = moneyDVVecTmp' * scallVec;
    
    vecTmp = sort(meneyMatrixTmp(:));
    
    % 求出所有数总和
    sumAll = sum(vecTmp);
    % 求出最大的前22%的数的和
    indexTmp = floor(year*pointSplit*(1-scall));
    sumUp = sum(vecTmp(indexTmp:end));
    
    sumTmp = sumUp/sumAll;
    if sumTmp > 1-scall
        leftScall = tmpScall;
    elseif sumTmp < 1-scall
        rightScall = tmpScall;
    else
        break;
    end
end
scallCompute = tmpScall;
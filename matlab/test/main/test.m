%% 初始化环境
clear; close all; clc;

first = 2;
timeMax = 400;
time = first+(1:timeMax);
time = time';

figure(1);
hold on;

tol = 1e11;

errorTmp = zeros(21-5+1, 1);
errorTime = zeros(21-5+1, 1);
for i=5:21
    nTmp = floor((time-first)/(i-first));
    yTmp = ((i-1)/2).^nTmp;
    plot(time, yTmp);
    errorTmp(i-4) = yTmp(end);
    
    for j=1:timeMax
        if yTmp(j) > tol
            errorTime(i-4) = j;
            break;
        end
    end
end
xlabel('次数');
ylabel('达到的精度');

hold off;

figure(2);
plot(5:21, errorTmp);

figure(3);
plot(5:21, errorTime);
%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
load data/data_test6extra3_multi_0_n.mat;

%% 画出数据图
% 原始数据图
figure(1);
posOrigin = find(YOrigin == 1); 
negOrigin = find(YOrigin == -1);

plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 学习曲线
figure(2);
plot(realSplitVecLearn, errorTrainLearn, realSplitVecLearn, errorValLearn);
title('学习曲线');
legend('训练集', '交叉验证集');
xlabel('数量');
ylabel('误差');
fprintf('学习曲线\n');

%% 训练集图
figure(3);
posTrain = find(YTrain == 1); 
negTrain = find(YTrain == -1);

plot(XTrain(posTrain, 1), XTrain(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XTrain(negTrain, 1), XTrain(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('训练集图');
fprintf('训练集图\n');
hold off;

%% 交叉验证集图
figure(4);
posVal = find(YVal == 1); 
negVal = find(YVal == -1);

plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
contour(vecX1, vecX2, predYTestTmp_2D, [0 0]);
title('交叉验证集图');
fprintf('交叉验证集图\n');
hold off;

%% 测试图

figure(5);
hold on;

split = 101;
nVec = [1 3 10 100];
plotVec = ["k." "r." "g." "b."];
tmpPointVec = linspace(0, 1, split);

for i=1:length(nVec)
    n = nVec(i);
    tmpUpMatrix = (tmpPointVec.*1)'*(tmpPointVec.*1)*(n+1);
    tmpDownMatrix = (tmpPointVec.*1)'+(tmpPointVec.*n);
    tmpValueMatrix = tmpUpMatrix./tmpDownMatrix;

    for j=1:length(tmpPointVec)
        plot3(zeros(split, 1)+tmpPointVec(j), tmpPointVec, tmpValueMatrix(j, :), plotVec(i),'LineWidth', 1, 'MarkerSize', 7);
    end
end

xlabel('1-point');
ylabel('error');
hold off;

%% 
figure(6)
plot3([1 2], [2 3], [3 4], 'rx');
xlabel('x');
ylabel('y');
zlabel('z');
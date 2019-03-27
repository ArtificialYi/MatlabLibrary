%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
fileName = ['data/', 'data_test7base0n_20190327132445.mat'];
load(fileName);

mK = size(centroidsOrigin, 1);
contourVec = mean(vec2subMatrix(2.^(1:mK), 2),2);
if length(contourVec)==1
    contourVec = [contourVec contourVec];
end
m1 = size(vecX1, 1);
pred2D = reshape(2.^YTest, m1, m1);



%% 画出数据图
% 原始数据图
figure(1);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
hold on;
plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
contour(vecX1, vecX2, pred2D, contourVec);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

%% 训练集图
figure(2);

plot(XTrain(:, 1), XTrain(:, 2), 'r+','LineWidth', 1, 'MarkerSize', 7);
hold on;
contour(vecX1, vecX2, pred2D, contourVec);
title('训练集图');
hold off;

%% 交叉验证集图
figure(3);

plot(XVal(:, 1), XVal(:, 2), 'b+','LineWidth', 1, 'MarkerSize', 7);
hold on;
contour(vecX1, vecX2, pred2D, contourVec);
hold off;
title('交叉验证集图')

%% 学习曲线
figure(4);
plot(1:20, errorElbowVec, 1:20, errorElbowVec);
xlabel('K个数');
ylabel('误差');
title('学习曲线');

%% 学习曲线
figure(5);
tmpVec1 = errorElbowVec([2:end end])-errorElbowVec([1 1:end-1]);
plot(1:20, tmpVec1, 1:20, tmpVec1);
xlabel('K个数');
ylabel('一阶导数');
title('学习曲线');


%% 学习曲线
figure(6);
tmpVec2 = tmpVec1([2:end end])-tmpVec1([1 1:end-1]);
plot(1:20, tmpVec2, 1:20, tmpVec2);
xlabel('K个数');
ylabel('二阶导数');
title('学习曲线');
%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
fileName = ['data/', 'data_testLogisticReg0_20190402171206.mat'];
load(fileName);

posFlag = 1;
negFlag = 0;

markSize = 4;

% 数据准备
posOrigin = find(YOrigin == posFlag); 
negOrigin = find(YOrigin == negFlag);
posTrain = find(YTrain == posFlag); 
negTrain = find(YTrain == negFlag);
posVal = find(YVal == posFlag); 
negVal = find(YVal == negFlag);

%% 画出数据图
% 画出pca图像
figure(1);

subplot(3, 3, 1);
plot(XOriginNormPca(posOrigin, 1), XOriginNormPca(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XOriginNormPca(negOrigin, 1), XOriginNormPca(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('PCA-原始数据图');
fprintf('PCA-原始数据图\n');
hold off;

subplot(3, 3, 2);
plot(XTrainNormPca(posTrain, 1), XTrainNormPca(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XTrainNormPca(negTrain, 1), XTrainNormPca(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('PCA-训练集图');
fprintf('PCA-训练集图\n');
hold off;

subplot(3, 3, 3);
plot(XValNormPca(posVal, 1), XValNormPca(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XValNormPca(negVal, 1), XValNormPca(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('PCA-交叉验证集图');
fprintf('PCA-交叉验证集图\n');
hold off;

% 原始数据图
subplot(3, 3, 4);
plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('原始数据图');
fprintf('原始数据图\n');
hold off;

subplot(3, 3, 5);
plot(XTrain(posTrain, 1), XTrain(posTrain, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XTrain(negTrain, 1), XTrain(negTrain, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('训练集图');
fprintf('训练集图\n');
hold off;

subplot(3, 3, 6);
plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
hold on;
plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
title('交叉验证集图');
fprintf('交叉验证集图\n');
hold off;

%% 在原始数据图上画分割结果
figure(1)
for i=1:3
    subplot(3, 3, i+3);
    hold on;
    contour(vecX1, vecX2, predYDataTmp_2D, [0.5 0.5]);
    hold off;
end
for i=1:3
    subplot(3, 3, i);
    hold on;
    contour(vecX1Pca, vecX2Pca, predYPcaTmp_2D, [0.5 0.5]);
    hold off;
end

%% 学习曲线 & pca曲线
figure(2);

subplot(2, 2, 1);
plot(realSplitVec, errorTrain, realSplitVec, errorVal);
title('学习曲线');
legend('训练集', '交叉验证集');
xlabel('数量');
ylabel('误差');
fprintf('学习曲线\n');

subplot(2, 2, 2);
plot(1:length(pcaVec), pcaSumVec/pcaSumVec(end));
title('pca曲线');
xlabel('K的数量');
ylabel('数据保留率');
legend('');

%% 学习曲线的所有细节
figure(1)
for i=1:50
    subplot(3, 3, 7);
    plot(XOrigin(posOrigin, 1), XOrigin(posOrigin, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
    hold on;
    plot(XOrigin(negOrigin, 1), XOrigin(negOrigin, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
    contour(vecX1, vecX2, predYLearnDataTmp_3D(:, :, i), [0.5 0.5]);
    title('原始数据图');
    fprintf('原始数据图\n');
    hold off;
    
    posTrainLearn = find(YTrain(1:realSplitVec(i)) == posFlag); 
    negTrainLearn = find(YTrain(1:realSplitVec(i)) == negFlag);
    subplot(3, 3, 8);
    plot(XTrain(posTrainLearn, 1), XTrain(posTrainLearn, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
    hold on;
    plot(XTrain(negTrainLearn, 1), XTrain(negTrainLearn, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
    contour(vecX1, vecX2, predYLearnDataTmp_3D(:, :, i), [0.5 0.5]);
    title('训练集图');
    fprintf('训练集图\n');
    hold off;
    
    subplot(3, 3, 9);
    plot(XVal(posVal, 1), XVal(posVal, 2), 'k+','LineWidth', 1, 'MarkerSize', markSize);
    hold on;
    plot(XVal(negVal, 1), XVal(negVal, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', markSize);
    contour(vecX1, vecX2, predYLearnDataTmp_3D(:, :, i), [0.5 0.5]);
    title('交叉验证集图');
    fprintf('交叉验证集图\n');
    hold off;
    pause;
end
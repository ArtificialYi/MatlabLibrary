%% 初始化环境
clear; close all; clc;

%% 读取数据
% 读取数据
fileName = ['data/', 'data_testComp_20190410194929.mat'];
load(fileName);

posFlag = 1;
negFlag = 0;
high = (posFlag+negFlag)/2;

markSize = 4;

% 数据准备
posOrigin = find(YOrigin == posFlag); 
negOrigin = find(YOrigin == negFlag);
posTrain = find(YTrain == posFlag); 
negTrain = find(YTrain == negFlag);
posVal = find(YVal == posFlag); 
negVal = find(YVal == negFlag);


%% 训练结果
predYOrigin(predYOrigin>=0.5)=1;
predYOrigin(predYOrigin<0.5)=0;
tpTn = sum(predYOrigin==YOrigin);
fprintf('%d个对的, pred:%f\n', tpTn, tpTn/size(YOrigin, 1));
predYTest(predYTest>=0.5)=1;
predYTest(predYTest<0.5)=0;

%% 
plotInitFunc = @(paramRowNum, paramOrigin, paramTrain, paramVal, paramStr) ...
    plotInitY(paramRowNum, markSize, ...
    paramOrigin, paramTrain, paramVal, paramStr, ...
    posOrigin, negOrigin, ...
    posTrain, negTrain, ...
    posVal, negVal);
plotInitFunc2 = @(paramRowNum, ...
    paramOrigin, paramTrain, paramVal, ...
    paramStr, paramPosTrain, paramNegTrain) ...
    plotInitY(paramRowNum, markSize, ...
    paramOrigin, paramTrain, paramVal, paramStr, ...
    posOrigin, negOrigin, ...
    paramPosTrain, paramNegTrain, ...
    posVal, negVal);

%% 画出数据图
% 画出pca图像
figure(1);
plotInitFunc(1, XOriginNormPca, XTrainNormPca, XValNormPca, 'PCA');
plotInitFunc(2, XOrigin, XTrain, XVal, 'origin');
plotOn(1);
plotOn(2);
plotContourY(1, matrixXPca, predYPcaTmp_DMulti, high);
plotContourY(2, matrixX, predYDataTmp_DMulti, high);
plotOff(1);
plotOff(2);

%% 学习曲线 & pca曲线
figure(2);

subplot(2, 2, 1);
plot(realSplitLearnVec, errorTrainLearn, realSplitLearnVec, errorValLearn);
title('学习曲线');
legend('训练集', '交叉验证集');
xlabel('数量');
ylabel('误差');
fprintf('学习曲线\n');
%% 画出当前自由
subplot(2, 2, 2);
plot(1:length(pcaVec), pcaSumVec/pcaSumVec(end));
title('pca曲线');
xlabel('K的数量');
ylabel('数据保留率');
legend('');

%% 学习曲线的所有细节
numRow = 3;
for i=1:50
    figure(1);
    posTrainLearn = find(YTrain(1:realSplitLearnVec(i)) == posFlag); 
    negTrainLearn = find(YTrain(1:realSplitLearnVec(i)) == negFlag);
    plotInitFunc2(numRow, XOrigin, XTrain, XVal, '学习曲线', ...
        posTrainLearn, negTrainLearn);
    plotOn(numRow);
    plotContourY(numRow, matrixX, predYLearnDataTmp_3D(:, :, i), high);
    plotOff(numRow);
    pause(1/60);
end

%% 最优化
figure(3)
plot(1:10, pErrorVec);
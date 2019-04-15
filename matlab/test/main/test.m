%% 初始化环境
clear; close all; clc;

%% 先读取数据
data = load('resource/ex6data1.mat');

% 获取原始数据
XOrigin = data.X;
YOrigin = data.y;
YOrigin(YOrigin==0)=-1;

% 特征扩充
% 将所有枚举型特征扩充为2进制特征
lenMax = 30;
[XOriginBinary, data2binaryFunc] = binaryFeature(XOrigin, lenMax);

%% 使用SVM基础训练
SVMModel = fitcsvm(XOriginBinary, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', 10, 'KernelScale', 1);

CClassSVMModel = CompactSVMModel.Trained{1};
[a, score] = predict(SVMModel, XOriginBinary);
score(score>0.5) = 1;
score(score<0.5) = -1;
b = a==score;
c = a==YOrigin;

%% 打印数据
% 数据准备
posFlag = 1;
negFlag = -1;
markSize = 4;

posOrigin = find(YOrigin == posFlag); 
negOrigin = find(YOrigin == negFlag);
posTrain = find(YTrain == posFlag); 
negTrain = find(YTrain == negFlag);
posVal = find(YVal == posFlag); 
negVal = find(YVal == negFlag);

plotInitFunc = @(paramRowNum, paramOrigin, paramTrain, paramVal, paramStr) ...
    plotInitY(paramRowNum, markSize, ...
    paramOrigin, paramTrain, paramVal, paramStr, ...
    posOrigin, negOrigin, ...
    posTrain, negTrain, ...
    posVal, negVal);

% 准备画等高线
minX = min(XOriginBinary);
maxX = max(XOriginBinary);
split = 101;
vecX1 = linspace(minX(1),maxX(1),split)';
vecX2 = linspace(minX(2),maxX(2),split)';
[X1, X2] = meshgrid(vecX1, vecX2);
[predLine, score] = predict(CClassSVMModel, [X1(:) X2(:)]);

row = 1;
plotInitFunc(row, XOrigin, XTrain, XVal, 'origin');
plotOn(row);
plotContourY(row, [vecX1 vecX2], reshape(predLine, split, split), 0);

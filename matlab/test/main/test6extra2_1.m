%% ��չ�����
clear; close all; clc;

%% ��ȡԭʼ����-��������Լ���ѵ����
% ��ȡԭʼ����
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;
m = size(XOrigin, 1);

% ��ֵ������
classNum = 10;
maxClass = ceil(log2(classNum));
YOriginMatrix = zeros(m, maxClass);

YOriginTmp = YOrigin;
for i=1:maxClass
    YOriginMatrix(:,i) = mod(YOriginTmp, 2);
    YOriginTmp = (YOriginTmp - YOriginMatrix(:,i))/2; 
end

trainPoint = 0.7;
valPoint = 0.3;

% ��ԭʼ����ת��Ϊ�������������
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% ������֤��������
mVal = size(XVal, 1);

% ������һ��
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);
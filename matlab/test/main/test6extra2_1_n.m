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

%% ������һ��-ѵ��ģ��-40�飯2min 80�飯5min 160�飯15min 320��/30min
CTrain = 1;
tolTrain = 1e-5;
maxIterTrain = 1;
alphaTrain = zeros(m, 1);

modelOriginTmp = ...
    svmTrainGPU(XOriginNorm, YOriginMatrix(:,1), CTrain, alphaTrain, tolTrain, 1);
modelOriginMatrix = repmat(modelOriginTmp, maxClass, 1);
for i=1:maxClass
    [modelOriginMatrix(i)] = svmTrain(XOriginNorm, YOriginMatrix(:,i), CTrain, modelOriginMatrix(i).alpha, tolTrain, maxIterTrain);
    fprintf('��%d���%d��ѵ�����.\n', i, maxIterTrain);
end

%% ѵ�����չʾ
for i=1:maxClass
    fprintf('alpha�����ֵΪ:%f\n', max(modelOriginMatrix(i).alpha));
    fprintf('w�Ľ��Ϊ:\n');
    fprintf('b�Ľ��Ϊ:%f\n', modelOriginMatrix(i).b);
    fprintf('point�ĺ�Ϊ:%f\n�ܵĴ���ĵ���Ϊ:%d\n', sum(modelOriginMatrix(i).point), sum(modelOriginMatrix(i).point>tolTrain));
    fprintf('�������Ϊ��%.20f\n', modelOriginMatrix(i).error);
    fprintf('��ʵ����Ϊ:%.20f\n', modelOriginMatrix(i).tol);
    fprintf('�������Ϊ:%.20f\n', modelOriginMatrix(i).floatError);
end

%% ��һ������ݴ���
XTestTmp = [vecX1Repeat vecX2Multi];
nTestTmp = size(XTestTmp, 2);

XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, noneIndex, mu, sigma);

predYTestTmp = XTestTmpNorm*modelOrigin.w+modelOrigin.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);
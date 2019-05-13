function [outputArg1,outputArg2] = ...
    testSysSVMPtr(C, gu, maxIter, isTrain, guLeft, guRight, guSplit)
%testSysSVMPtr SVM训练数据&找到最优解

% 初始化数据
gu = str2double(gu);
C = str2double(C);
guLeft = str2double(guLeft);
guRight = str2double(guRight);
guSplit = str2double(guSplit);
maxIter = str2double(maxIter);
isTrain = str2double(isTrain);

%% 先读取数据
data0 = load('resource/pfm_data.mat');

XOrigin = data0.XOrigin;
YOrigin = data0.YOrigin;
XTest = data0.XTest;
YOrigin(YOrigin==0) = -1;

XOriginNorm = XOrigin;
XTestNorm = XTest;

%% 使用SVM基础训练
rng('shuffle');

% 基本svm训练
fprintf('训练原始特征归一化\n');
SVMModel = fitcsvm(XOrigin, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);

predY = predict(SVMModel, XOrigin);
fprintf('交叉训练原始特征归一化\n');
CVSVMModel = crossval(SVMModel, 'KFold', 5);
fprintf('kFold损失\n');
classLoss = kfoldLoss(CVSVMModel);

fprintf('原始特征预测: %f\n', classLoss);
predRes = sum(predY==YOrigin)/size(YOrigin, 1);
fprintf('准确率:%f\n', predRes);
lossRes = kfoldLoss(CVSVMModel, 'LossFun','binodeviance');
fprintf('二项异常:%f\n', lossRes);
lossRes = kfoldLoss(CVSVMModel, 'LossFun','hinge');
fprintf('铰链:%f\n', lossRes);

% medoids离散化特征
fprintf('训练离散化特征归一化\n');
SVMModel2 = fitcsvm(XOriginNorm, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);

fprintf('交叉训练离散化特征归一化\n');
[predY2, ~] = predict(SVMModel2, XOriginNorm);
fprintf('kFold损失\n');
CVSVMModel2 = crossval(SVMModel2, 'KFold', 5);
classLoss = kfoldLoss(CVSVMModel2);

fprintf('medoids-离散化特征: %f\n', classLoss);
predRes2 = sum(predY2==YOrigin)/size(YOrigin, 1);
fprintf('准确率:%f\n', predRes2);
lossRes = kfoldLoss(CVSVMModel2, 'LossFun','binodeviance');
fprintf('二项异常:%f\n', lossRes);
lossRes = kfoldLoss(CVSVMModel2, 'LossFun','hinge');
fprintf('铰链:%f\n', lossRes);

%% 查找最优值
seed = floor(rand()*1e9);
pred = 1e-3;
svmFuncWithGuC = @(paramC, paramGu) valLossSVM(XOriginNorm, YOrigin, ...
    paramC, paramGu, maxIter, seed);

[errorVal, errorTrain] = svmFuncWithGuC(C, gu);
if isTrain
    % 获取gu列表
    vecGu = logspace(guLeft, guRight, guSplit);
    
    % 找到最优gu和最优gu对应的C
    guTmp = 1;
    svmFunCWithC = @(paramC) svmFuncWithGuC(paramC, guTmp);
    [CMinTmp, errorMinTmp] = svmTrainCPUForFindC(svmFunCWithC, pred);
end

function [errorVal, errorTrain] = valLossSVM(paramX, paramY, paramC, paramGu, maxIter, seed)
    rng(seed);
    MdlSVM = fitcsvm(paramX, paramY, ...
        'Standardize', true, 'KernelFunction', 'RBF', ...
        'BoxConstraint', paramC, 'KernelScale', paramGu, 'IterationLimit', maxIter);
    MdlVal = crossval(MdlSVM, 'KFold', 5);
    errorVal = kfoldLoss(MdlVal, 'LossFun', 'hinge');
    errorTrain = loss(MdlSVM, paramX, paramY, 'LossFun', 'classiferror');
end

end


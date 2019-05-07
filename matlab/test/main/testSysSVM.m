function [outputArg1,outputArg2] = testSysSVM(C, gu, guLeft, guRight, guSplit, maxIter, isTrain)
%testSysSVM 使用系统自带的分类器

% 初始化数据
gu = str2double(gu);
C = str2double(C);
guLeft = str2double(guLeft);
guRight = str2double(guRight);
guSplit = str2double(guSplit);
maxIter = str2double(maxIter);
isTrain = str2double(isTrain);

%% 先读取数据
data = load('resource/ex6data2.mat');

% 获取原始数据
XOrigin = data.X;
YOrigin = data.Y;
YOrigin(YOrigin==0)=-1;

[mOrigin, nOrigin] = size(XOrigin);

trainPoint = 0.7;
valPoint = 0.3;
pred = 1e-16;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(mOrigin);
[XTrainSplit, XValSplit, ~] = ...
    splitData(XOrigin, indexVecRand, trainPoint, valPoint);
[YTrain, YVal, ~] = ...
    splitData(YOrigin, indexVecRand, trainPoint, valPoint);

%% 特征扩充
%% 将所有特征离散化-K-means
% 特征归一化
[XOriginNorm, data2normFuncOrigin] = data2featureWithNormalize(XOrigin, 1);

% 特征离散化
kMeansFunc = @(paramX, paramK) kMeansTrainRandGPU(paramX, paramK, maxIter);
kMeansPredFunc = @(paramX, paramCentroids) kMeansTrainGPU(paramX, paramCentroids, 1);
KMax = 30;
p = 1;
% means-离散化函数
[XOriginNormMeansP1, data2binaryP1] = binaryFeature(XOriginNorm, KMax, p, kMeansFunc, kMeansPredFunc);
% means-01化函数
XOriginNormMeansP1_01 = K201(XOriginNormMeansP1);

% medoids-离散化函数
kMedoidsTrainFunc = @(paramX, paramK) kMedoidsTrain(paramX, paramK);
kMedoidsPredFunc = @(paramX, paramCentroids) kMedoidsPred(paramX, paramCentroids);
[XOriginNormMedoidsP1, data2binaryP1] = binaryFeature(XOriginNorm, KMax, p, kMedoidsTrainFunc, kMedoidsPredFunc);
% medoids-01化函数
XOriginNormMedoidsP1_01 = K201(XOriginNormMedoidsP1);

% 最终特征归一化
XOriginMeansFinal = [XOriginNorm XOriginNormMeansP1_01];
[XOriginMeansFinalNorm, data2normFuncMeansFinal] = data2featureWithNormalize(XOriginMeansFinal, 1);
XOriginMedoidsFinal = [XOriginNorm XOriginNormMedoidsP1_01];
[XOriginMedoidsFinalNorm, data2normFuncMedoidsFinal] = data2featureWithNormalize(XOriginMedoidsFinal, 1);

% 保存离散化数据
fprintf('离散化数据开始保存\n');
save('data/pfm_data_means.mat', 'XOriginNormMeansP1', 'XOriginNormMeansP1_01', ...
    'XOriginNormMedoidsP1', 'XOriginNormMedoidsP1_01');
fprintf('保存完毕\n');

%% 使用SVM基础训练
rng('shuffle');
% 原始特征
SVMModel = fitcsvm(XOriginNorm, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);

[predY, scoreOrigin] = predict(SVMModel, XOriginNorm);
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
fprintf('原始特征: %f\n', classLoss);
predRes = sum(predY==YOrigin)/size(YOrigin, 1);
fprintf('准确率:%f\n', predRes);
lossRes = loss(SVMModel, XOriginNorm, YOrigin, 'LossFun','binodeviance');
fprintf('二项异常:%f\n', lossRes);
lossRes = loss(SVMModel, XOriginNorm, YOrigin, 'LossFun','hinge');
fprintf('铰链:%f\n', lossRes);

% means离散化特征
SVMModel = fitcsvm(XOriginMeansFinalNorm, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);

[predY, scoreOrigin] = predict(SVMModel, XOriginMeansFinalNorm);
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
fprintf('means-离散化特征: %f\n', classLoss);
predRes = sum(predY==YOrigin)/size(YOrigin, 1);
fprintf('准确率:%f\n', predRes);
lossRes = loss(SVMModel, XOriginMeansFinalNorm, YOrigin, 'LossFun','binodeviance');
fprintf('二项异常:%f\n', lossRes);
lossRes = loss(SVMModel, XOriginMeansFinalNorm, YOrigin, 'LossFun','hinge');
fprintf('铰链:%f\n', lossRes);

% medoids离散化特征
SVMModel = fitcsvm(XOriginMedoidsFinalNorm, YOrigin, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);

[predY, scoreOrigin] = predict(SVMModel, XOriginMedoidsFinalNorm);
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
fprintf('medoids-离散化特征: %f\n', classLoss);
predRes = sum(predY==YOrigin)/size(YOrigin, 1);
fprintf('准确率:%f\n', predRes);
lossRes = loss(SVMModel, XOriginMedoidsFinalNorm, YOrigin, 'LossFun','binodeviance');
fprintf('二项异常:%f\n', lossRes);
lossRes = loss(SVMModel, XOriginMedoidsFinalNorm, YOrigin, 'LossFun','hinge');
fprintf('铰链:%f\n', lossRes);

%% 找到最优解
minGu = 1e-8;
minC = 1e5;
minError = 0;

% 学习曲线X
numSplit = ceil(sqrt(mOrigin));
vecSplit = floor((1:numSplit)*mOrigin/numSplit);

% 学习曲线Y
vecErrorTrain = zeros(numSplit, 1);
vecErrorVal = zeros(numSplit, 1);
    
if isTrain
    %% 学习曲线
    % 0. 将数据随机化
    XOriginBinaryRand = XOriginFinalNorm(indexVecRand, :);
    YOriginRand = YOrigin(indexVecRand);
    
    for i=1:length(vecSplit)
        % 1. 将不同的数据集放入训练器，查看训练集的结果和交叉验证集的结果
        MdlLearn = fitcsvm(XOriginBinaryRand(1:vecSplit(i), :), YOriginRand(1:vecSplit(i)), ...
            'Standardize', true, 'KernelFunction', 'RBF', ...
            'BoxConstraint', C, 'KernelScale', gu, 'IterationLimit', maxIter);
        errorLossTrain = loss(MdlLearn, XOriginBinaryRand(1:vecSplit(i), :), YOriginRand(1:vecSplit(i)), ...
            'LossFun', 'classiferror');
        hingeLossTrain = loss(MdlLearn, XOriginBinaryRand(1:vecSplit(i), :), YOriginRand(1:vecSplit(i)), ...
            'LossFun','hinge');
        
        MdlLearnCross = crossval(MdlLearn, 'KFold', 5);
        errorLossVal = kfoldLoss(MdlLearnCross, 'LossFun', 'classiferror');
        hingeLossVal = kfoldLoss(MdlLearnCross, 'LossFun', 'hinge');
        fprintf('错误率:%f, %f\n', errorLossTrain, errorLossVal);
        fprintf('hinge:%f, %f\n', hingeLossTrain, hingeLossVal);
        
        % 2. 将训练结果存储
        vecErrorTrain(i) = hingeLossTrain;
        vecErrorVal(i) = hingeLossVal;
    end
    
    %% 查找当前最优解
    vecGu = linspace(guLeft, guRight, guSplit);
    vecGu = vecGu(2:end);
    seed = floor(rand()*1e9);
    svmFunc = @(paramC, paramGu) valLossSVM(XOriginBinaryRand, YOriginRand, ...
        paramC, paramGu, maxIter, seed);
            
    vecMinC = vecGu;
    vecMinError = vecGu;
    for i=1:length(vecGu)
        fprintf('开始gu:%f\n', vecGu(i));
        % 查找范围内C的最优解
        CPred = 1e-3;
        % 1. 先用等比数列查找范围
        split = 11;
        vecC = logspace(-5, 5, split);
        CLeft = vecC(1);
        CRight = vecC(end);
        indexCurrent = 1;

        % 2. 将左右极限保存下来
        [errorCLeft, errorCLeftTrain] = svmFunc(vecC(1), vecGu(i));
        if errorCLeftTrain > 0.09
            errorCLeft = errorCLeft + 1;
        end
        [errorCRight, errorCRightTrain] = svmFunc(vecC(end), vecGu(i));
        if errorCRightTrain > 0.09
            errorCRight = errorCRight + 1;
        end
        
        while CRight - CLeft > CPred
            % 3. 使用当前数列训练数据集
            errorCVec = vecC;
            for j=2:length(vecC)-1
                [errorCVec(j), errorTrain] = svmFunc(vecC(j), vecGu(i));
                if errorTrain > 0.09
                    errorCVec(j) = errorCVec(j) + 1;
                end
                fprintf('当前C:%f, %f, %f\n', vecC(j), errorCVec(j), errorTrain);
            end
            
            % 4. 将左右极限拼上去
            errorCVec(1) = errorCLeft;
            errorCVec(end) = errorCRight;
            
            % 5. 找到新的范围
            indexCurrent = indexMinForMulti(errorCVec);
            indexCurrent = indexCurrent(1);
            [indexLeft, indexRight] = getLeftAndRightIndex(indexCurrent, 1, split);
            minC = vecC(indexCurrent);
            CLeft = vecC(indexLeft);
            CRight = vecC(indexRight);
            vecC = linspace(CLeft, CRight, split);
            
            % 6. 将新的左右极限保存下来
            errorCLeft = errorCVec(indexLeft);
            errorCRight = errorCVec(indexRight);
        end
        
        % 7. 将当前最优C和error存入errorGu
        vecMinC(i) = minC;
        vecMinError(i) = errorCVec(indexCurrent);
    end
    
    % 8. 找到当前最优gu&C
    indexCurrent = indexMinForMulti(vecMinError);
    indexCurrent = indexCurrent(1);
    
    %
    minGu = vecGu(indexCurrent);
    minC = vecMinC(indexCurrent);
    minError = vecMinError(indexCurrent);
    fprintf('最小gu:%f\n最优C:%f\n最小error:%f\n', minGu, minC, minError);
end

% 获取文件名
fileName = sprintf('data/data_testSvmGaus_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, ...
    'predRes', 'lossRes', ...
    'vecSplit', 'vecErrorTrain', 'vecErrorVal', ...
    'minGu', 'minC', 'minError');
fprintf('保存完毕\n');

function [errorVal, errorTrain] = valLossSVM(paramX, paramY, paramC, paramGu, maxIter, seed)
    MdlSVM = fitcsvm(paramX, paramY, ...
        'Standardize', true, 'KernelFunction', 'RBF', ...
        'BoxConstraint', paramC, 'KernelScale', paramGu, 'IterationLimit', maxIter);
    MdlVal = crossval(MdlSVM, 'KFold', 5);
    errorVal = kfoldLoss(MdlVal, 'LossFun', 'hinge');
    errorTrain = loss(MdlSVM, paramX, paramY, 'LossFun', 'classiferror');
end

end


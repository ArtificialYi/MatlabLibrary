function [tmp] = test6extra3_multi_0_n(p, l, s, C, maxIter, isTrain, pVecMax)
%% 测试函数
% p 多项式的值
% l 高阶参数
% s 低阶参数
% C svm训练参数

p = str2double(p);
l = str2double(l);
s = str2double(s);
C = str2double(C);
maxIter = str2double(maxIter);
isTrain = str2double(isTrain);
pVecMax = str2double(pVecMax);

tol = 1e-8;

%% 读取数据
% 读取数据
data = load('resource/ex2data1.txt');
XOrigin = data(:,1:2);
YOrigin = data(:,3);
YOrigin(YOrigin==0)=-1;
m = size(XOrigin, 1);

trainPoint = 0.7;
valPoint = 0.3;

% 切成训练集、交叉验证集、测试集
indexVecRand = randperm(m);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    splitOriginData(XOrigin, YOrigin, indexVecRand, trainPoint, valPoint);

% 归一化数据
[XTrainNorm, mu, sigma, noneIndex] = featureNormalize(XTrain);
XOriginNorm = ...
    mapFeatureWithParam(XOrigin, 1, noneIndex, 1:length(noneIndex), mu, sigma);
XValNorm = ...
    mapFeatureWithParam(XVal, 1, noneIndex, 1:length(noneIndex), mu, sigma);

% 获取核结果
kernelFunc = @(X1, X2) svmKernelPolynomial(X1, X2, l, s, p);
KOrigin = kernelFunc(XOriginNorm, XOriginNorm);

% 边界线数据准备
minX1 = min(XOrigin(:,1));
maxX1 = max(XOrigin(:,1));
minX2 = min(XOrigin(:,2));
maxX2 = max(XOrigin(:,2));

splitTrain = 51;
vecX1 = linspace(minX1, maxX1, splitTrain)';
vecX2 = linspace(minX2, maxX2, splitTrain)';
vecX1Repeat = repeatMatrix(vecX1, splitTrain);
vecX2Multi = multiMatrix(vecX2, splitTrain);

%% 基础训练模型
% CPU->GPU
KOriginGPU = gpuArray(KOrigin);
YOriginGPU = gpuArray(YOrigin);
CTrainGPU = gpuArray(C);
tolTrainGPU = gpuArray(1e-8);
maxIterTrainGPU = gpuArray(maxIter);
alphaTrainGPU = gpuArray.zeros(m, 1);

modelOriginGPU = ...
    svmTrainGPU(KOriginGPU, YOriginGPU, CTrainGPU, alphaTrainGPU, tolTrainGPU, maxIterTrainGPU);

% 训练结果预测
XTestTmp = [vecX1Repeat vecX2Multi];
XTestTmpNorm = ...
    mapFeatureWithParam(XTestTmp, 1, noneIndex, 1:length(noneIndex), mu, sigma);
KTestTmp = kernelFunc(XOriginNorm, XTestTmpNorm);

predYTestTmp = (modelOriginGPU.cpu.alpha .* YOrigin)'*KTestTmp'+modelOriginGPU.cpu.b;
predYTestTmp_2D = reshape(predYTestTmp, splitTrain, splitTrain);

%% 学习曲线训练
%CPU->GPU
XTrainNormGPU = gpuArray(XTrainNorm);
YTrainGPU = gpuArray(YTrain);
XValNormGPU = gpuArray(XValNorm);
YValGPU = gpuArray(YVal);
CLearnGPU = gpuArray(C);
tolLearnGPU = gpuArray(tol);
maxIterLearnGPU = gpuArray(maxIter);
splitLearnGPU = gpuArray(50);

[errorTrainLearnGPU, errorValLearnGPU, realSplitVecLearnGPU] = ...
    svmLearningCurveGPU(XTrainNormGPU, YTrainGPU, ...
        XValNormGPU, YValGPU, CLearnGPU, ...
        tolLearnGPU, maxIterLearnGPU, splitLearnGPU, kernelFunc);

% 学习曲线
errorTrainLearn = gather(errorTrainLearnGPU);
errorValLearn = gather(errorValLearnGPU);
realSplitVecLearn = gather(realSplitVecLearnGPU);

%% 寻找全局最优C
%% 将当前最优各种参数打印出来
pVec = 1:pVecMax;
lVec = linspace(0.1, 1.9, 3);
sVec = linspace(0.1, 1.9, 3);

predCurrentGPU = gpuArray(1e-3);
tolCurrentGPU = gpuArray(tol);
maxIterCurrentGPU = gpuArray(maxIter);

errorMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));
CMinMatrix3 = zeros(length(pVec), length(lVec), length(sVec));

% 最小化结果集
pMin = p;
lMin = l;
sMin = s;
lMinReal = sqrt(lMin/sMin);
sMinReal = sqrt(sMin/lMin);
CMin = 0;
errorMin = 1;

if isTrain
    for i=1:length(pVec)
        for j=1:length(lVec)
            for k=1:length(sVec)
                pCurrent = pVec(i);
                lCurrent = lVec(j);
                sCurrent = sVec(k);
                fprintf('1s后开始:p-%d,l-%.2f,s-%.2f\n', pCurrent, lCurrent, sCurrent);
                pause(1);
                
                lCurrentReal = sqrt(lCurrent/sCurrent);
                sCurrentReal = sqrt(sCurrent/lCurrent);

                kernelFuncTmp = @(X1, X2) svmKernelPolynomial(X1, X2, lCurrentReal, sCurrentReal, pCurrent);
                KTrainTmpGPU = gpuArray(kernelFuncTmp(XTrainNorm, XTrainNorm));
                KValTmpGPU = gpuArray(kernelFuncTmp(XTrainNorm, XValNorm));

                [CCurrentGPU, errorMinCurrentGPU] = ...
                    svmFindCurrentMinC(KTrainTmpGPU, YTrainGPU, KValTmpGPU, YValGPU, tolCurrentGPU, maxIterCurrentGPU, predCurrentGPU);

                errorMinMatrix3(i, j, k) = gather(errorMinCurrentGPU);
                CMinMatrix3(i, j, k) = gather(CCurrentGPU);
            end
        end
    end
    
    % 找到最优的p、l、s、C
    indexMin3 = indexMinForMulti(errorMinMatrix3);
    lenP = length(pVec);
    lenL = length(lVec);
    indexPL = mod(indexMin3, lenP*lenL);
    indexPL(indexPL==0)=lenP*lenL;
    indexKMin = (indexMin3-indexPL)./(lenP*lenL)+1;
    indexJMin = mod(indexPL, lenP);
    indexJMin(indexJMin==0)=lenP;
    indexIMin = (indexPL-indexJMin)./lenP+1;

    pMinVec = pVec(indexIMin);
    lMinVec = lVec(indexJMin);
    sMinVec = sVec(indexKMin);
    lMinRealVec = sqrt(lMin./sMin);
    sMinRealVec = sqrt(sMin./lMin);
    
    % 找到最小的P的索引
    indexPMin = indexMinForMulti(pMinVec); % [1 2 3]
    % 找到最接近1的L的索引
    lMinRealPointVec = abs(log(lMinRealVec)); % [0 0.5 1]
    indexLMin = indexMinForMulti(lMinRealPointVec(indexPMin)); % [1 2 3]
    indexLMinReal = indexPMin(indexLMin); % [1 2 3]
    % 找到最接近1的S的索引
    sMinRealPointVec = abs(log(sMinRealVec)); % [0 0.5 1]
    indexSMin = indexMinForMulti(sMinRealPointVec(indexLMinReal));
    indexSMinReal = indexLMinReal(indexSMin); % [1 2 3]
    % 找到原始L中最接近1的索引
    lMinPointVec = abs(log(lMinVec));
    indexLMinOrigin = indexMinForMulti(lMinPointVec(indexSMinReal));
    indexLMinOriginReal = indexSMinReal(indexLMinOrigin); % [1 2 3]
    % 取以上4次过滤后的最大的索引值
    indexMinEnd = indexLMinOriginReal(end);
    
    % 找到真实的最优
    pMin = pMinVec(indexMinEnd);
    lMin = lMinVec(indexMinEnd);
    sMin = sMinVec(indexMinEnd);
    lMinReal = lMinRealVec(indexMinEnd);
    sMinReal = sMinRealVec(indexMinEnd);
    
    CMin = CMinMatrix3(iMin, jMin, kMin);
    errorMin = errorMinMatrix3(iMin, jMin, kMin);
end

%% 变量存储

% 获取文件名
fileName = sprintf('data/data_test6extra3_multi_0_n_%s.mat', datestr(now, 'yyyymmddHHMMss'));
save(fileName, ...
    'XOrigin', 'YOrigin', 'vecX1', 'vecX2', 'predYTestTmp_2D', ...
    'realSplitVecLearn', 'errorTrainLearn', 'errorValLearn', ...
    'XTrain', 'YTrain', 'XVal', 'YVal', 'CCurrent', 'errorMinCurrent', ...
    'pMin', 'lMin', 'lMinReal', 'sMin', 'sMinReal', 'CMin', 'errorMin');

end
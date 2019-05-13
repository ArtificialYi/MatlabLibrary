function [XMedoidsFinalNorm, func] = featureEngineer(X, KMax, p)
%featureEngineer 特征工程基本函数

[XNorm, data2normFuncOrigin] = data2featureWithNormalize(X, 1);

% medoids-离散化函数
kMedoidsTrainFunc = @(paramX, paramK) kMedoidsTrain(paramX, paramK);
kMedoidsPredFunc = @(paramX, paramCentroids) kMedoidsPred(paramX, paramCentroids);

% 自动离散化，&存储离散化结果
XMedoidsFinal = XNorm;

func2KVec = {};
func201Vec = {};

for i=1:p
    [XNormMedoidsPTmp, data2binaryPTmp] = binaryFeature(XNorm, KMax, i, kMedoidsTrainFunc, kMedoidsPredFunc);
    [XNormMedoidsP_01Tmp, data201Tmp] = K201(XNormMedoidsPTmp);
    XMedoidsFinal = [XMedoidsFinal XNormMedoidsPTmp XNormMedoidsP_01Tmp];
    func2KVec{i} = data2binaryPTmp;
    func201Vec{i} = data201Tmp;
end

% 归一化
[XMedoidsFinalNorm, data2normFuncFinal] = ... 
    data2featureWithNormalize(XMedoidsFinal, 1);

% 复现函数
func = @(paramX)featureRecover(paramX, data2normFuncOrigin, func2KVec, func201Vec, data2normFuncFinal);
function XTestFinalNorm = featureRecover(XTest, data2norm0, func2KVecTmp, func201VecTmp, data2norm1)
    XTestNorm = data2norm0(XTest);
    len = length(func2KVecTmp);
    
    XTestFinal = XTestNorm;
    for j=1:len
        XTestNormPTmp = func2KVecTmp{j}(XTestNorm);
        XTestNormP_01Tmp = func201VecTmp{j}(XTestNormPTmp);
        XTestFinal = [XTestFinal XTestNormPTmp XTestNormP_01Tmp];
    end
    
    XTestFinalNorm = data2norm1(XTestFinal);
end

end


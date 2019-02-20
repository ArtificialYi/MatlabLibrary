function num = numOfPolynomialFeature(n, p)
%numOfPolynomialFeature n个特征在p次多项式扩充后返回num个特征
%   n个特征，p次多项式扩充

% 初始化辅助数组
global numOfPolynomialFeature_maxIndex;
numOfPolynomialFeature_maxIndex = 5;
global numOfPolynomialFeature_helpArr;
if isempty(numOfPolynomialFeature_helpArr)
    numOfPolynomialFeature_helpArr = zeros(numOfPolynomialFeature_maxIndex, numOfPolynomialFeature_maxIndex) - 1;
end
num = -1;

% 如果索引没有越界&索引可用
if n < numOfPolynomialFeature_maxIndex && n > -1 ...
    && p < numOfPolynomialFeature_maxIndex && p > -1 ... 
    && numOfPolynomialFeature_helpArr(n+1, p+1) > -1
    num = numOfPolynomialFeature_helpArr(n+1, p+1);
end

% n<1的乱七八糟的情况
if num == -1 && n < 1
    num = 0;
end

% n为1的情况
if num == -1 && n == 1
    num = 1;
end

% n为2的情况
if num == -1 && n == 2
    num = p + 1;
end

% n > 2 & all为0的情况，可以删除这个if
if num == -1 && p == 0
    num = 1;
end

% n > 2的情况
if num == -1 && n > 2
    tmpSum = 0;
    for i=0:p
        tmpSum = tmpSum + numOfPolynomialFeature(n - 1, i);
    end
    num = tmpSum;
end

% 如果索引没有越界 & 索引为空
if n < numOfPolynomialFeature_maxIndex && n > -1 ...
    && p < numOfPolynomialFeature_maxIndex && p > -1 ... 
    && numOfPolynomialFeature_helpArr(n+1, p+1) == -1
    numOfPolynomialFeature_helpArr(n+1, p+1) = num;
end

end


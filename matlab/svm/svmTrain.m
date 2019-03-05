function [model] = svmTrain(X, Y, C, tol, maxIter)
%svmTrain SVM基础模型训练函数-SMO算法
% X 原始数据
% Y 结果集
% C 最大容许误差值
% tol 精准度
% maxIter 最大迭代次数

% 初始化参数
m = size(X, 1);
Y(Y==0) = -1;

% 初始化alpha
alpha = zeros(m, 1);
% 初始化核函数 m*m
K = X * X';
% 初始化数据差 m*m
eta = diag(K) + diag(K)' - K*2;
eta(eta==0) = -1;

% 获取初始b的值 1*1
b = -sum(K*(alpha.*Y)-Y) / m;

% 设置最大循环次数
timeMax = maxIter;
timeTmp = 0;

% 连续最小循环次数
tolTimeMax = floor(sqrt(m));
tolTimeTmp = 0;

% 开始循环计算
while timeTmp < timeMax && tolTimeTmp < tolTimeMax
    % 获取函数误差 m*1
    E = K*(alpha.*Y)-Y + b;
    % 获取两两误差和误差梯度 m*m
    EMinus = E - E';
    
    % 找到leftMatrix和rightMatrix
    sMatrix = Y * Y';
    % leftMatrix
    leftMatrixTmp1 = alpha + alpha' - C;
    leftMatrixTmp2 = alpha' - alpha;
    leftMatrixTmp1(sMatrix ~= 1) = 0;
    leftMatrixTmp2(sMatrix == 1) = 0;
    leftMatrix = leftMatrixTmp1+leftMatrixTmp2;
    leftMatrix(leftMatrix<0) = 0;
    
    % rightMatrix
    rightMatrixTmp1 = alpha + alpha';
    rightMatrixTmp2 = alpha' - alpha + C;
    rightMatrixTmp1(sMatrix ~= 1) = 0;
    rightMatrixTmp2(sMatrix == 1) = 0;
    rightMatrix = rightMatrixTmp1+rightMatrixTmp2;
    rightMatrix(rightMatrix>C) = C;
    
    % 未使用上下界验证前的alphaNew
    alphaNewMatrix = EMinus .* Y' ./ eta + alpha';
    
    % 使用上下界验证
    alphaNewMatrix = min(rightMatrix, alphaNewMatrix);
    alphaNewMatrix = max(leftMatrix, alphaNewMatrix);
    
    % 计算所有误差
    tolMatrix = abs(alphaNewMatrix - alpha');
    % 将边界误差设置为0
    tolMatrix = tril(tolMatrix, -1) + tril(tolMatrix', -1)';
    
    % 取出最大的一个误差，开始计算
    [indexMax] = find(tolMatrix==max(max(tolMatrix)));
    if length(indexMax) > 1
        indexMax = indexMax(1);
    end
    index2 = ceil(indexMax / m);
    index1 = indexMax - (index2-1)*m;
    
    % 最大误差已为0
    if index1 == index2
        break;
    end
    
    alphaOld1 = alpha(index1);
    alphaOld2 = alpha(index2);
    
    % 获得最新的alpha
    alpha(index2) = alphaNewMatrix(index1, index2);
    alpha(index1) = alphaOld1 + sMatrix(index1, index2)*(alphaOld2-alpha(index2));
    
    % 获取新的b
    b = -sum(K*(alpha.*Y)-Y) / m;
    
    % 获取误差
    timeTmp = timeTmp + 1;
    tolTmp = tolMatrix(index1, index2);

    % 找到theta
    w = ((alpha'.*Y') * X)';
    JError = svmCost(X, Y, w, b, 1/C);
    fprintf('Iter:%d, error:%f\n', timeTmp, JError);
    
    % 连续误差小于某个范围，确定已经收敛
    if tolTmp < tol
        tolTimeTmp = tolTimeTmp + 1;
    else
        tolTimeTmp = 0;
    end
end

% 找到theta
w = ((alpha'.*Y') * X)';

model.w = w;
model.b = b;
model.maxTime = timeTmp;
model.alpha = alpha;

end

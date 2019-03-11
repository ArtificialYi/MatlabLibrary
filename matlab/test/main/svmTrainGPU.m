function [model] = svmTrainGPU(X, Y, C, alpha, tol, maxIter)
%UNTITLED19 使用GPU训练SVM模型
% X 原始数据
% Y 结果集
% C 最大容许误差值
% tol 精准度
% maxIter 最大迭代次数

% 初始化参数
XExist = existOnGPU(X);
fprintf('查看X是否存在于GPU中:%d\n', XExist);

% 初始化参数
m = size(X, 1);
Y(Y==0) = -1;

% 初始化浮点误差和精度范围
floatErrorUnit = C*1e-14;
tol = max(floatErrorUnit, tol);
floatErrorMax = min(floatErrorUnit, tol);

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

% 点误差
E = zeros(m, 1);
EMinus = zeros(m, m);
tolMatrix = zeros(m, m);

% 点的误差积分
posPoint = zeros(m, 1);
negPoint = zeros(m, 1);
point = zeros(m, 1);
pointMatrix = zeros(m, m);

% 左右横跳
sMatrix = zeros(m, m);
leftMatrixTmp1 = zeros(m, m);
leftMatrixTmp2 = zeros(m, m);
leftMatrix = zeros(m, m);
rightMatrixTmp1 = zeros(m, m);
rightMatrixTmp2 = zeros(m, m);
rightMatrix = zeros(m, m);

% alpha相关
alphaNewMatrix = zeros(m, m);
alphaErrorVec = zeros(m, 1);
alphaError = alpha'*Y;

% 随机数
destiny = zeros(m, m);
sumY = sum(Y);

% 开始循环计算
while timeTmp < timeMax && tolTimeTmp < tolTimeMax
    % 获取函数误差 m*1
    E(:) = K*(alpha.*Y)-Y + b;
    % 获取两两误差和误差梯度 m*m
    EMinus(:) = E - E';
    
    % 寻找违反KKT条件的所有点
    % 寻找对的点调整alpha m*1
    posPoint(:) = E.*Y.*alpha;
    posPoint(posPoint<0)=0;
    % 寻找错误的点调整alpha
    negPoint(:) = E.*Y.*(C-alpha);
    negPoint(negPoint>0)=0;
    % 将有问题的点整理出来
    point(:) = abs(negPoint+posPoint);
    pointNum = sum(point>0);

    % 随机选点事件
    r = rand();
    destiny(:) = rand(m, m);
    if r > 0.4
        % 60%的概率用加法-消去没有问题的两个点之间的权重
        pointMatrix(:) = point + point';
    elseif r > 0.2 && pointNum > 2
        % 20%的概率用乘法-消去没有问题的一个点相关的所有权重  
        pointMatrix(:) = point * point';
    elseif r > 0.1
        % 10%的概率加法+随机因子
        pointMatrix(:) = (point + point').*destiny;
    elseif r > 0.01
        % 9%的概率乘法+随机因子
        pointMatrix(:) = (point * point').*destiny;
    else
        % 1%的概率完全随机
        pointMatrix(:) = destiny;
    end

    % 找到leftMatrix和rightMatrix
    sMatrix(:) = Y * Y';
    % leftMatrix
    leftMatrixTmp1(:) = alpha + alpha' - C;
    leftMatrixTmp2(:) = alpha' - alpha;
    leftMatrixTmp1(sMatrix ~= 1) = 0;
    leftMatrixTmp2(sMatrix == 1) = 0;
    leftMatrix(:) = leftMatrixTmp1+leftMatrixTmp2;
    leftMatrix(leftMatrix<0) = 0;
    
    % rightMatrix
    rightMatrixTmp1(:) = alpha + alpha';
    rightMatrixTmp2(:) = alpha' - alpha + C;
    rightMatrixTmp1(sMatrix ~= 1) = 0;
    rightMatrixTmp2(sMatrix == 1) = 0;
    rightMatrix(:) = rightMatrixTmp1+rightMatrixTmp2;
    rightMatrix(rightMatrix>C) = C;
    
    % 未使用上下界验证前的alphaNew
    alphaNewMatrix(:) = EMinus .* Y' ./ eta + alpha';
    
    % 使用上下界验证
    alphaNewMatrix(:) = min(rightMatrix, alphaNewMatrix);
    alphaNewMatrix(:) = max(leftMatrix, alphaNewMatrix);
    
    % 每多少次迭代纠正一下alpha可能存在的误差

    % 计算所有误差
    tolMatrix(:) = abs(alphaNewMatrix - alpha');
    % 将边界误差设置为0
    tolMatrix(:) = tril(tolMatrix, -1) + tril(tolMatrix', -1)';
    tolMatrix(:) = tolMatrix .* pointMatrix;
    
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

    % 每隔开方次就矫正一次alpha

    % 获取误差
    timeTmp = timeTmp + 1;
    tolTmp = tolMatrix(index1, index2);

    % 获取新的b
    b = -sum(K*(alpha.*Y)-Y) / m;

    % 如果alpha浮点误差超过误差极限了，尝试重新计算alpha
    alphaError = alpha'*Y;
    tolError = abs(alphaError);
    if abs(alphaError) > floatErrorMax
        alphaErrorVec(:) = 0;

        % 将alpha拆分成只正、只负、可正可负
        alpha1 = (alpha<tolError)-(alpha>(C-tolError));
        % 从只正、只负中获取负收益数
        alphaErrorNum = sum(alpha1.*Y*alphaError>0);
        % 如果负收益数小于一半，开始重新计算alpha
        sumAlpha = m - 2*alphaErrorNum;
        if sumAlpha > 0
            % 可正可负中的值
            alpha2 = -(alpha1==0).*Y*(alphaError/sumAlpha);
            alpha3 = alpha1*(tolError/sumAlpha) + alpha2;
            alpha = alpha + alpha3;
            b = -sum(K*(alpha.*Y)-Y) / m;
            fprintf('success:\n%.20f\n%.20f\n', alphaError, alpha'*Y);
        else
            fprintf('modify float fail:%d\n', sumAlpha);
        end
    end
    

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

% 找到theta和b
w = ((alpha'.*Y') * X)';

model.w = w;
model.b = b;
model.maxTime = timeTmp;
model.alpha = alpha;
model.point = point;
model.error = alphaError;
model.tol = tol;
model.floatError = floatErrorMax;
end


function [model] = svmTrainGPU(X, Y, C, alpha, tol, maxIter, gpuNum)
%svmTrain SVM基础模型训练函数-SMO算法，C越大，收敛概率越低
% X 原始数据
% Y 结果集
% C 最大容许误差值
% tol 精准度
% maxIter 最大迭代次数

% 获取GPU资源
gpuDevice(gpuNum);

% 将数据放入GPU
XGPU = gpuArray(X);
YGPU = gpuArray(Y);
CGPU = gpuArray(C);
alphaGPU = gpuArray(alpha);
tolGPU = gpuArray(tol);
maxIterGPU = gpuArray(maxIter);

% 初始化参数
mGPU = gpuArray(size(XGPU, 1));
YGPU(YGPU==0) = -1;


% 初始化浮点误差和精度范围
floatErrorUnit = C*1e-14;
tol = max(floatErrorUnit, tol);
floatErrorMax = min(floatErrorUnit, tol);

% 初始化核函数 m*m
KGPU = XGPU * XGPU';
% 初始化数据差 m*m
etaGPU = diag(KGPU) + diag(KGPU)' - KGPU*2;
etaGPU(etaGPU==0) = -1;

% 获取初始b的值 1*1
bGPU = -sum(KGPU*(alphaGPU.*YGPU)-YGPU) / mGPU;

% 设置最大循环次数
timeMaxGPU = maxIterGPU;
timeTmpGPU = gpuArray(0);

% 连续最小循环次数
tolTimeMaxGPU = floor(sqrt(mGPU));
tolTimeTmpGPU = gpuArray(0);

% 点误差
EGPU = gpuArray.zeros(mGPU, 1);
EMinusGPU = gpuArray.zeros(mGPU, mGPU);
tolMatrixGPU = gpuArray.zeros(mGPU, mGPU);

% 点的误差积分
posPointGPU = gpuArray.zeros(mGPU, 1);
negPointGPU = gpuArray.zeros(mGPU, 1);
pointGPU = gpuArray.zeros(mGPU, 1);
pointMatrixGPU = gpuArray.zeros(mGPU, mGPU);

% 左右横跳
sMatrixGPU = gpuArray.zeros(mGPU, mGPU);
leftMatrixTmp1GPU = gpuArray.zeros(mGPU, mGPU);
leftMatrixTmp2GPU = gpuArray.zeros(mGPU, mGPU);
leftMatrixGPU = gpuArray.zeros(mGPU, mGPU);
rightMatrixTmp1GPU = gpuArray.zeros(mGPU, mGPU);
rightMatrixTmp2GPU = gpuArray.zeros(mGPU, mGPU);
rightMatrixGPU = gpuArray.zeros(mGPU, mGPU);

% alpha相关
alphaNewMatrixGPU = gpuArray.zeros(mGPU, mGPU);
alphaErrorVecGPU = gpuArray.zeros(mGPU, 1);
alphaErrorGPU = alphaGPU'*YGPU;

% 随机数
destinyGPU = gpuArray.zeros(mGPU, mGPU);
sumY = sum(YGPU);

% 开始循环计算
while timeTmpGPU < timeMaxGPU && tolTimeTmpGPU < tolTimeMaxGPU
    % 获取函数误差 m*1
    EGPU(:) = KGPU*(alphaGPU.*YGPU)-YGPU + bGPU;
    % 获取两两误差和误差梯度 m*m
    EMinusGPU(:) = EGPU - EGPU';
    
    % 寻找违反KKT条件的所有点
    % 寻找对的点调整alpha m*1
    posPointGPU(:) = EGPU.*YGPU.*alphaGPU;
    posPointGPU(posPointGPU<0)=0;
    % 寻找错误的点调整alpha
    negPointGPU(:) = EGPU.*YGPU.*(CGPU-alphaGPU);
    negPointGPU(negPointGPU>0)=0;
    % 将有问题的点整理出来
    point(:) = abs(negPointGPU+posPointGPU);
    pointNumGPU = sum(pointGPU>0);

    % 随机选点事件
    rGPU = gpuArray.rand();
    destinyGPU(:) = gpuArray.rand(mGPU, mGPU);
    if rGPU > 0.4
        % 60%的概率用加法-消去没有问题的两个点之间的权重
        pointMatrixGPU(:) = pointGPU + pointGPU';
    elseif rGPU > 0.2 && pointNumGPU > 2
        % 20%的概率用乘法-消去没有问题的一个点相关的所有权重  
        pointMatrixGPU(:) = pointGPU * pointGPU';
    elseif rGPU > 0.1
        % 10%的概率加法+随机因子
        pointMatrixGPU(:) = (pointGPU + pointGPU').*destinyGPU;
    elseif rGPU > 0.01
        % 9%的概率乘法+随机因子
        pointMatrixGPU(:) = (pointGPU * pointGPU').*destinyGPU;
    else
        % 1%的概率完全随机
        pointMatrixGPU(:) = destinyGPU;
    end

    % 找到leftMatrix和rightMatrix
    sMatrixGPU(:) = YGPU * YGPU';
    % leftMatrixGPU
    leftMatrixTmp1GPU(:) = alphaGPU + alphaGPU' - CGPU;
    leftMatrixTmp2GPU(:) = alphaGPU' - alphaGPU;
    leftMatrixTmp1GPU(sMatrixGPU ~= 1) = 0;
    leftMatrixTmp2GPU(sMatrixGPU == 1) = 0;
    leftMatrixGPU(:) = leftMatrixTmp1GPU+leftMatrixTmp2GPU;
    leftMatrixGPU(leftMatrixGPU<0) = 0;
    
    % rightMatrix
    rightMatrixTmp1GPU(:) = alphaGPU + alphaGPU';
    rightMatrixTmp2GPU(:) = alphaGPU' - alphaGPU + CGPU;
    rightMatrixTmp1GPU(sMatrixGPU ~= 1) = 0;
    rightMatrixTmp2GPU(sMatrixGPU == 1) = 0;
    rightMatrixGPU(:) = rightMatrixTmp1GPU+rightMatrixTmp2GPU;
    rightMatrixGPU(rightMatrixGPU>C) = C;
    
    % 未使用上下界验证前的alphaNew
    alphaNewMatrixGPU(:) = EMinusGPU .* YGPU' ./ etaGPU + alphaGPU';
    
    % 使用上下界验证
    alphaNewMatrixGPU(:) = min(rightMatrixGPU, alphaNewMatrixGPU);
    alphaNewMatrixGPU(:) = max(leftMatrixGPU, alphaNewMatrixGPU);

    % 计算所有误差
    tolMatrixGPU(:) = abs(alphaNewMatrixGPU - alphaGPU');
    % 将边界误差设置为0
    tolMatrixGPU(:) = tril(tolMatrixGPU, -1) + tril(tolMatrixGPU', -1)';
    tolMatrixGPU(:) = tolMatrixGPU .* pointMatrixGPU;
    
    % 取出最大的一个误差，开始计算
    [indexMaxGPU] = find(tolMatrixGPU==max(max(tolMatrixGPU)));
    if length(indexMaxGPU) > 1
        indexMaxGPU = indexMaxGPU(1);
    end
    index2GPU = ceil(indexMaxGPU / mGPU);
    index1GPU = indexMaxGPU - (index2GPU-1)*mGPU;
    
    exist = existsOnGPU(alphaNewMatrixGPU);
    fprintf('min&max是否已经移动至GPU中:%d\n', exist);
    exist = existsOnGPU(tolMatrixGPU);
    fprintf('abs是否已经移动至GPU中:%d\n', exist);
    exist = existsOnGPU(indexMax);
    fprintf('find是否已经移动至GPU中:%d\n', exist);
    exist = existsOnGPU(index2);
    fprintf('ceil是否已经移动至GPU中:%d\n', exist);
    exist = existsOnGPU(index1);
    fprintf('compute是否已经移动至GPU中:%d\n', exist);
    
    % 最大误差已为0
    if index1 == index2
        break;
    end
    
    alphaOld1GPU = alphaGPU(index1);
    alphaOld2GPU = alphaGPU(index2);

    % 获得最新的alpha
    alphaGPU(index2GPU) = alphaNewMatrixGPU(index1GPU, index2GPU);
    alphaGPU(index1GPU) = alphaOld1GPU + sMatrixGPU(index1GPU, index2GPU)*(alphaOld2GPU-alphaGPU(index2GPU));

    % 获取误差
    timeTmpGPU = timeTmpGPU + 1;
    tolTmpGPU = tolMatrixGPU(index1GPU, index2GPU);

    % 获取新的b
    bGPU = -sum(KGPU*(alphaGPU.*YGPU)-YGPU) / mGPU;

    % 如果alpha浮点误差超过误差极限了，尝试重新计算alpha
    alphaErrorGPU = alphaGPU'*YGPU;
    tolErrorGPU = abs(alphaErrorGPU);
    if abs(alphaErrorGPU) > floatErrorMaxGPU
        alphaErrorVecGPU(:) = 0;

        % 将alpha拆分成只正、只负、可正可负
        alpha1GPU = (alphaGPU<tolErrorGPU)-(alphaGPU>(CGPU-tolErrorGPU));
        % 从只正、只负中获取负收益数
        alphaErrorNumGPU = sum(alpha1GPU.*YGPU*alphaErrorGPU>0);
        % 如果负收益数小于一半，开始重新计算alpha
        sumAlphaGPU = mGPU - 2*alphaErrorNumGPU;
        if sumAlphaGPU > 0
            % 可正可负中的值
            alpha2GPU = -(alpha1GPU==0).*YGPU*(alphaErrorGPU/sumAlphaGPU);
            alpha3GPU = alpha1GPU*(tolErrorGPU/sumAlphaGPU) + alpha2GPU;
            alphaGPU = alphaGPU + alpha3GPU;
            bGPU = -sum(KGPU*(alphaGPU.*YGPU)-YGPU) / mGPU;
            fprintf('success:\n%.20f\n%.20f\n', alphaErrorGPU, alphaGPU'*YGPU);
        else
            fprintf('modify float fail:%d\n', sumAlphaGPU);
        end
    end
    

    % 找到theta
    wGPU = ((alphaGPU'.*YGPU') * XGPU)';
    JErrorGPU = svmCost(XGPU, YGPU, wGPU, bGPU, 1/CGPU);
    fprintf('Iter:%d, error:%f\n', timeTmpGPU, JErrorGPU);
    
    % 连续误差小于某个范围，确定已经收敛
    if tolTmpGPU < tolGPU
        tolTimeTmpGPU = tolTimeTmpGPU + 1;
    else
        tolTimeTmpGPU = 0;
    end
end

% 找到theta和b
wGPU = ((alpha'.*Y') * X)';

model.w = wGPU;
model.b = bGPU;
model.maxTime = timeTmpGPU;
model.alpha = alphaGPU;
model.point = pointGPU;
model.error = alphaErrorGPU;
model.tol = tolGPU;
model.floatError = floatErrorMaxGPU;

end

function [model] = svmTrainGPU(X, Y, C, alpha, tol, maxIter)
%UNTITLED19 ʹ��GPUѵ��SVMģ��
% X ԭʼ����
% Y �����
% C ����������ֵ
% tol ��׼��
% maxIter ����������

% ��ʼ������
XExist = existOnGPU(X);
fprintf('�鿴X�Ƿ������GPU��:%d\n', XExist);

% ��ʼ������
m = size(X, 1);
Y(Y==0) = -1;

% ��ʼ���������;��ȷ�Χ
floatErrorUnit = C*1e-14;
tol = max(floatErrorUnit, tol);
floatErrorMax = min(floatErrorUnit, tol);

% ��ʼ���˺��� m*m
K = X * X';
% ��ʼ�����ݲ� m*m
eta = diag(K) + diag(K)' - K*2;
eta(eta==0) = -1;

% ��ȡ��ʼb��ֵ 1*1
b = -sum(K*(alpha.*Y)-Y) / m;

% �������ѭ������
timeMax = maxIter;
timeTmp = 0;

% ������Сѭ������
tolTimeMax = floor(sqrt(m));
tolTimeTmp = 0;

% �����
E = zeros(m, 1);
EMinus = zeros(m, m);
tolMatrix = zeros(m, m);

% ���������
posPoint = zeros(m, 1);
negPoint = zeros(m, 1);
point = zeros(m, 1);
pointMatrix = zeros(m, m);

% ���Һ���
sMatrix = zeros(m, m);
leftMatrixTmp1 = zeros(m, m);
leftMatrixTmp2 = zeros(m, m);
leftMatrix = zeros(m, m);
rightMatrixTmp1 = zeros(m, m);
rightMatrixTmp2 = zeros(m, m);
rightMatrix = zeros(m, m);

% alpha���
alphaNewMatrix = zeros(m, m);
alphaErrorVec = zeros(m, 1);
alphaError = alpha'*Y;

% �����
destiny = zeros(m, m);
sumY = sum(Y);

% ��ʼѭ������
while timeTmp < timeMax && tolTimeTmp < tolTimeMax
    % ��ȡ������� m*1
    E(:) = K*(alpha.*Y)-Y + b;
    % ��ȡ������������ݶ� m*m
    EMinus(:) = E - E';
    
    % Ѱ��Υ��KKT���������е�
    % Ѱ�ҶԵĵ����alpha m*1
    posPoint(:) = E.*Y.*alpha;
    posPoint(posPoint<0)=0;
    % Ѱ�Ҵ���ĵ����alpha
    negPoint(:) = E.*Y.*(C-alpha);
    negPoint(negPoint>0)=0;
    % ��������ĵ��������
    point(:) = abs(negPoint+posPoint);
    pointNum = sum(point>0);

    % ���ѡ���¼�
    r = rand();
    destiny(:) = rand(m, m);
    if r > 0.4
        % 60%�ĸ����üӷ�-��ȥû�������������֮���Ȩ��
        pointMatrix(:) = point + point';
    elseif r > 0.2 && pointNum > 2
        % 20%�ĸ����ó˷�-��ȥû�������һ������ص�����Ȩ��  
        pointMatrix(:) = point * point';
    elseif r > 0.1
        % 10%�ĸ��ʼӷ�+�������
        pointMatrix(:) = (point + point').*destiny;
    elseif r > 0.01
        % 9%�ĸ��ʳ˷�+�������
        pointMatrix(:) = (point * point').*destiny;
    else
        % 1%�ĸ�����ȫ���
        pointMatrix(:) = destiny;
    end

    % �ҵ�leftMatrix��rightMatrix
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
    
    % δʹ�����½���֤ǰ��alphaNew
    alphaNewMatrix(:) = EMinus .* Y' ./ eta + alpha';
    
    % ʹ�����½���֤
    alphaNewMatrix(:) = min(rightMatrix, alphaNewMatrix);
    alphaNewMatrix(:) = max(leftMatrix, alphaNewMatrix);
    
    % ÿ���ٴε�������һ��alpha���ܴ��ڵ����

    % �����������
    tolMatrix(:) = abs(alphaNewMatrix - alpha');
    % ���߽��������Ϊ0
    tolMatrix(:) = tril(tolMatrix, -1) + tril(tolMatrix', -1)';
    tolMatrix(:) = tolMatrix .* pointMatrix;
    
    % ȡ������һ������ʼ����
    [indexMax] = find(tolMatrix==max(max(tolMatrix)));
    if length(indexMax) > 1
        indexMax = indexMax(1);
    end
    index2 = ceil(indexMax / m);
    index1 = indexMax - (index2-1)*m;
    
    % ��������Ϊ0
    if index1 == index2
        break;
    end
    
    alphaOld1 = alpha(index1);
    alphaOld2 = alpha(index2);

    % ������µ�alpha
    alpha(index2) = alphaNewMatrix(index1, index2);
    alpha(index1) = alphaOld1 + sMatrix(index1, index2)*(alphaOld2-alpha(index2));

    % ÿ�������ξͽ���һ��alpha

    % ��ȡ���
    timeTmp = timeTmp + 1;
    tolTmp = tolMatrix(index1, index2);

    % ��ȡ�µ�b
    b = -sum(K*(alpha.*Y)-Y) / m;

    % ���alpha�������������ˣ��������¼���alpha
    alphaError = alpha'*Y;
    tolError = abs(alphaError);
    if abs(alphaError) > floatErrorMax
        alphaErrorVec(:) = 0;

        % ��alpha��ֳ�ֻ����ֻ���������ɸ�
        alpha1 = (alpha<tolError)-(alpha>(C-tolError));
        % ��ֻ����ֻ���л�ȡ��������
        alphaErrorNum = sum(alpha1.*Y*alphaError>0);
        % �����������С��һ�룬��ʼ���¼���alpha
        sumAlpha = m - 2*alphaErrorNum;
        if sumAlpha > 0
            % �����ɸ��е�ֵ
            alpha2 = -(alpha1==0).*Y*(alphaError/sumAlpha);
            alpha3 = alpha1*(tolError/sumAlpha) + alpha2;
            alpha = alpha + alpha3;
            b = -sum(K*(alpha.*Y)-Y) / m;
            fprintf('success:\n%.20f\n%.20f\n', alphaError, alpha'*Y);
        else
            fprintf('modify float fail:%d\n', sumAlpha);
        end
    end
    

    % �ҵ�theta
    w = ((alpha'.*Y') * X)';
    JError = svmCost(X, Y, w, b, 1/C);
    fprintf('Iter:%d, error:%f\n', timeTmp, JError);
    
    % �������С��ĳ����Χ��ȷ���Ѿ�����
    if tolTmp < tol
        tolTimeTmp = tolTimeTmp + 1;
    else
        tolTimeTmp = 0;
    end
end

% �ҵ�theta��b
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


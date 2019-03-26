function out = mapFeature2polynomial(X, p)
% mapFeature ����ʽ��������
% X ԭʼ����
% p ����ʽ����
%   ��ȡ��С

[m, n] = size(X);
%   ��ʼ���������
tmpN = zeros(1, p - 1);
for i=2:p
    tmpN(i - 1) = numOfPolynomialFeature(n, i);
end
out = [X, zeros(m, sum(tmpN))];

%   Ϊ�����ֵ
tmpOutCol = n;
for i=2:p
    % ��ȡ�������������������
    powerMatrix = matrixOfSumWithNum(n, i);
    featureNum = size(powerMatrix, 1);
    
    % ʹԭʼ���ݺ�������ϵ���������һ��
    repeatNoneConstX = repeatMatrix(X, featureNum);
    multiPowerMatrix = multiMatrix(powerMatrix, m);
    
    % ��ʼ����
    tmpFeatureX = repeatNoneConstX .^ multiPowerMatrix;
    tmpFeatureVec = prod(tmpFeatureX, 2);
    
    % ������ϳ���Ҫ����������
    tmpX = reshape(tmpFeatureVec, featureNum, m);
    
    % �������ںϵ������
    out(:, (tmpOutCol + 1):(tmpOutCol + featureNum)) = tmpX';
    tmpOutCol = tmpOutCol + featureNum;
end

end


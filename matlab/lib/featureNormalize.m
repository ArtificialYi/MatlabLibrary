function [normNoneConstX, mu, sigma, noneIndex] = featureNormalize(X)
%featureNormalize ��������
% X ԭʼ���ݼ�

% �����������
[noneConstX, noneIndex] = trimConst(X);

% ��ȥƽ��ֵ
mu = mean(noneConstX);
avgNoneConstX = bsxfun(@minus, noneConstX, mu);

% ��ȥ��׼��
sigma = std(avgNoneConstX);
normNoneConstX = bsxfun(@rdivide, avgNoneConstX, sigma);

end

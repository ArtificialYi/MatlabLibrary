function repeatX = repeatMatrix(X,repeat)
%repeatMatrix 将矩阵每行重复N次并返回
%   X 待重复矩阵
%   repeat 重复次数

%demo
%   repeatMatrix([1 2;3 4], 2)
%return
%   1   2
%   1   2
%   3   4
%   3   4

m = size(X, 1);
tmpRepeatEye = repeatEye(m, repeat);
repeatX = tmpRepeatEye * X;
end
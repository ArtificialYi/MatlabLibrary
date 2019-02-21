function multiX = multiMatrix(X, multi)
%multiMatrix 获取多个相同矩阵
%   X 待复制矩阵
%   multi 复制次数

%demo
%   multiMatrix([1 2;3 4], 2)
%return
%   1   2
%   3   4
%   1   2
%   3   4

m = size(X, 1);
tmpMultiEye = multiEye(m, multi);
multiX = tmpMultiEye * X;
end


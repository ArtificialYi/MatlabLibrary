function [indexMinVReal] = indexMinForVec(V)
%indexMinForVec 流式向量中的最小值所在的索引
% V m*1的向量
% demo: [2;2;2]
% return: [1;2;3]
% 
% demo: [2;2;3;2]
% return: [4]
% 
% demo: [2;2;2;3]
% return: [3]

len = length(V);

% V向量左右横跳
VLeft = [V(1);V(1:(len-1))];
VRight = [V(2:len);V(len)];

% 导数向量左右横跳
dV = (VRight-VLeft);
dVLeft = [dV(1);dV(1:(len-1))];
dVRight = [dV(2:len);dV(len)];
% 导数向量转换为牛顿导数向量
dVReal = abs((4*dV+dVLeft+dVRight)/6);

% 找出原生向量中的最小值-可能为向量
indexMinV = find(V==min(V));
% 找出原生向量最小值对应的导数向量的最大值-可能为向量
logicalMinDV = dVReal(indexMinV)==max(dVReal(indexMinV));
% 取出两次过滤后的值
indexMinVReal = indexMinV(logicalMinDV);

end


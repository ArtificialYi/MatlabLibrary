function [indexMinVec, dVMatrix] = indexMinForMulti(M3)
%indexMinFor3 流式多维数组中的最小值所在的索引
%   此处显示详细说明

% 多轴长度
lenVec = size(M3);
n = length(lenVec);

% 初始化结果函数
dVMatrix = M3;
dVMatrix(:) = 0;

% 所有元素沿X轴上下移动一位
for i=1:n
    M3Tmp = permute(M3, [i:n 1:i-1]);
    % 翻转第一个轴
    [M3Left, M3Right] = matrixMove(M3Tmp);
    % 求导后继续左右横跳
    M3dV = M3Right - M3Left;
    [M3dVLeft, M3dVRight] = matrixMove(M3dV);
    % 转化为牛顿导数
    M3dVReal = abs((4*M3dV+M3dVLeft+M3dVRight)/6);
    % 导数结果
    dVMatrix(:) = dVMatrix(:)+M3dVReal(:).^2;
end

% 找出原生多维数据中的最小值
indexMinOrigin = find(min(M3(:))==M3(:));
indexMinDV = dVMatrix(indexMinOrigin) == max(dVMatrix(indexMinOrigin));
indexMinVec = indexMinOrigin(indexMinDV);

end


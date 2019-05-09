function [X01, func] = K201(XK)
%K201 将枚举数据转成01数组

KVec = max(XK);
% 扩展用的函数
func = @(paramX) K201Recover(paramX, KVec);

X01 = func(XK);

function X201 = K201Recover(X, vec)
    [m, n] = size(X);
    len = length(vec);
    nX_01 = sum(vec);
    % 数据错误，无法恢复
    if n ~= len
        fprintf('数据错误，无法恢复\n');
        error('数据错误，无法恢复\n');
    end
    
    indexBegin = 1;
    X201 = zeros(m, nX_01);
    for j=1:n
        X_01Tmp = X(:, j) == (1:vec(j));
        X201(:, indexBegin:indexBegin+vec(j)-1) = X_01Tmp;
        indexBegin = indexBegin + vec(j);
    end
end

end

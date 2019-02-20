function subX = vec2subMatrix(X, n)
%vec2subMatrix 从一个特征数大于等于n的向量中获取n个特征的子向量列表
%   返回子向量列表


if n <= 0
    subX = [];
elseif size(X, 2) > n
    tmpRight = vec2subMatrix(X(2:end), n-1);
    m = size(tmpRight, 1);
    if m == 0
        m = 1;
    end
    
    tmpLeft = zeros(m, 1) + X(1);
    subX = [
        [tmpLeft, tmpRight];
        vec2subMatrix(X(2:end), n)
    ];
elseif size(X, 2) == n
    subX = X;
elseif size(X, 2) < n
    % 此处可以抛出错误
    subX = [];
end

end


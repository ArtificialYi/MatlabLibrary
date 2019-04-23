function [X01] = K201(XK)
%K201 将枚举数据转成01数组

nXK = size(XK, 2);

KVec = max(XK);
nX01 = sum(KVec);
X01 = zeros(mXOriginNorm, nX01);

indexBegin = 1;
for i=1:nXK
    XP1_01Tmp = XK(:, i) == (1:KVec(i));
    X01(:, indexBegin:indexBegin+KVec(i)-1) = XP1_01Tmp;
    indexBegin = indexBegin + KVec(i);
end

end


function [VDouble, VUnique] = vecStr2double(V)
%vecStr2double 将字符串向量转化为枚举型字符

VUnique = unique(V);
VDouble = vecStr2vecIndex(V, VUnique);

end


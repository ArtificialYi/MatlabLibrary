function [VIndex] = vecStr2vecIndex(VStr, VUnique)
%vecStr2vecIndex 字符串向量转位索引向量

lenVUnique = length(VUnique);

VIndex = mod(find(VUnique==VStr'), lenVUnique);
VIndex(VIndex==0) = lenVUnique;

end


function [num] = numOfC(m, n)
%numOfC 在m个数中挑选3个值所有可能性数量

if n >= m
    n = 0;
elseif n*2>m
    n = m - n;
end
num = prod(m-n+1:m)/prod(1:n);
end


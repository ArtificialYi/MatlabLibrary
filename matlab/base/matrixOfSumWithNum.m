function arrGroup = matrixOfSumWithNum(n,num)
%matrixOfSumWithNum 获取n个数，和为num的所有排列组合

if n == 1
    arrGroup = (num);
elseif num == 0
    arrGroup = zeros(1, n);
elseif num > 0
    arrGroup=[];
    for i=0:num
        tmpRight = matrixOfSumWithNum(n-1, num-i);
        m = size(tmpRight, 1);
        if m == 0
            m = 1;
        end
        tmpLeft = zeros(m, 1) + i;
        arrGroup(end+1:end+m,:) = [tmpLeft tmpRight];
    end
end

end


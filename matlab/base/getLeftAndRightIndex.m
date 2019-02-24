function [leftIndex, rightIndex] = ...
    getLeftAndRightIndex(currentIndex, minIndex, maxIndex)
%getLeftAndRightIndex 获取一个索引的左索引和右索引
% currentIndex 当前索引
% minIndex 最小索引
% maxIndex 最大索引

leftIndex = currentIndex - 1;
rightIndex = currentIndex + 1;
if leftIndex == minIndex-1
    leftIndex = minIndex;
end
if rightIndex == maxIndex+1
    rightIndex = maxIndex;
end

end

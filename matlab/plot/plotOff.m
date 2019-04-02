function [] = plotOff(numRow)
%plotOff hold off

for i=1:3
    subplot(3, 3, numRow*3+i-3);
    hold off;
end

end


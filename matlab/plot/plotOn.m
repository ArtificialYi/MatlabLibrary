function [] = plotOn(numRow)
%plotOn 打开hold on

for i=1:3
    subplot(3, 3, numRow*3+i-3);
    hold on;
end

end


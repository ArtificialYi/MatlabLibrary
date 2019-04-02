function [c] = ...
    plotContourY(numRow, XMatrix, YMatrix, high)
%plotContourY 画出等高线
n = size(XMatrix, 2);
for i=1:3
    subplot(3, 3, numRow*3+i-3);
    switch n
        case 1
            c = contour(XMatrix(:, 1), YMatrix, [high high]);
        case 2
            c = contour(XMatrix(:, 1), XMatrix(:, 2), YMatrix, [high high]);
        otherwise
            c = contour3(XMatrix(:, 1), XMatrix(:, 2), XMatrix(:, 3), YMatrix, [high high]);
    end
end

end


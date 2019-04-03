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
            for j=1:length(XMatrix(:, 3))
                matrixTmp0 = YMatrix(:, :, j);
                matrixTmp1 = matrixTmp0;
                matrixTmp1(matrixTmp0<high) = XMatrix(j, 3) - 1;
                matrixTmp1(matrixTmp0==high) = XMatrix(j, 3);
                matrixTmp1(matrixTmp0>high) = XMatrix(j, 3) + 1;
                c = contour3(XMatrix(:, 1), XMatrix(:, 2), matrixTmp1, [XMatrix(j, 3) XMatrix(j, 3)]);
            end
    end
end

end


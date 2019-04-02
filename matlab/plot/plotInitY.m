function [f] = ...
    plotInitY(numRow, markSize, ...
    XOrigin, XTrain, XVal, strTitle, ...
    posOrigin, negOrigin, ...
    posTrain, negTrain, ...
    posVal, negVal)
%plotOneImageY 画出一张图

% 画出pca图像
% 原始数据图
subplot(3, 3, numRow*3-2);
plotOne(XOrigin, posOrigin, 'r+', markSize);
hold on;
plotOne(XOrigin, negOrigin, 'bo', markSize);
title([strTitle '-原始数据图']);
hold off;

% 训练集图
subplot(3, 3, numRow*3-1);
plotOne(XTrain, posTrain, 'r+', markSize);
hold on;
plotOne(XTrain, negTrain, 'bo', markSize);
title([strTitle '-训练集图']);
hold off;

% 交叉验证集
subplot(3, 3, numRow*3);
plotOne(XVal, posVal, 'r+', markSize);
hold on;
plotOne(XVal, negVal, 'bo', markSize);
title([strTitle '-交叉验证集图']);
hold off;

end


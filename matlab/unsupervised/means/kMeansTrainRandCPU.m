function [centroidsMin, YMin, errorMin, KReal] = kMeansTrainRandCPU(X, K, maxIter)
%kMeansTrainRandCPU kMeansTrainRandCPU

[centroids, KReal] = findInitPointRand(X, K);
[centroids(:), YTmp, errorTmp] = kMeansTrainCPU(X, centroids, maxIter);

centroidsMin = centroids;
YMin = YTmp;
errorMin = errorTmp;

timeTrain = 0;
[m, n] = size(X);
mTrain = ceil(sqrt(m*n*K));

while timeTrain < mTrain
    timeTrain = timeTrain + 1;
    [centroids(:), KReal] = findInitPointRand(X, K);
    [centroids(:), YTmp, errorTmp] = kMeansTrainCPU(X, centroids, maxIter);
    
    if errorTmp < errorMin
        centroidsMin(:) = centroids;
        YMin = YTmp;
        errorMin = errorTmp;
        fprintf('%d:%d, 找到更小值!.\n', mTrain, timeTrain);
        timeTrain = 0;
    end
end
fprintf('%d个分类完毕-实际分类:%d!.\n', K, KReal);

end


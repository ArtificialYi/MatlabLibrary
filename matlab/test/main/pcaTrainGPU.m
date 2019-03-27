function [U, S] = pcaTrainGPU(XGPU)
%pcaTrainGPU pca获取所有成分特征

mGPU = gpuArray(size(X, 1));

sigmaGPU = XGPU'*XGPU/mGPU;
[U, S] = svd(sigmaGPU);
end


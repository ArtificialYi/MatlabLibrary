function [XGPU] = pca2data(ZGPU, UGPU, KGPU)
%pca2data 使用pca算法-升维

XGPU = ZGPU * UGPU(:, 1:KGPU)';
end


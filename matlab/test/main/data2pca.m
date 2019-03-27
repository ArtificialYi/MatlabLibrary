function [Z] = data2pca(XGPU, UGPU, KGPU)
%data2pca 将数据转成pca压缩数据

Z = XGPU * UGPU(:, 1:KGPU);

end


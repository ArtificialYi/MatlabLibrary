function [outputArg1,outputArg2] = testSysSVMPre(KMax, p)
%testSysSVM 使用系统自带的分类器

% 初始化数据
KMax = str2double(KMax);
p = str2double(p);

%% 先读取数据
data = load('resource/pfm_data.mat');

% 获取原始数据
XOrigin = data.XOrigin;
XTest = data.XTest;

%% 特征扩充
%% 将所有特征离散化-K-mean
[XOriginNorm, data2norm] = featureEngineer(XOrigin, KMax, p);
XTestNorm = data2norm(XTest);

% 保存离散化数据
fileName = sprintf('data/data_testSysSVMPre_%s.mat', datestr(now, 'yyyymmddHHMMss'));
fprintf('离散化数据开始保存\n');
fprintf('正在保存文件:%s\n', fileName(6:end));
save(fileName, 'XOriginNorm', 'XTestNorm', 'data2norm');
fprintf('保存完毕\n');

end


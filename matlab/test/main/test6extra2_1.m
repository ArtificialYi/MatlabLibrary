%% ��չ�����
clear; close all; clc;

%% ��ȡԭʼ����-��������Լ���ѵ����
% ��ȡԭʼ����
data = load('../resource/ex3data1.mat');
XOrigin = data.X;
YOrigin = data.y;

fprintf('X�Ĵ�СΪ:%d, %d\n', size(XOrigin, 1), size(XOrigin, 2));
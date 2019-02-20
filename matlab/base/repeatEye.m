function repeatEyeRes = repeatEye(row, repeat)
%repeatEye 获取重复单位矩阵
%   row 单位矩阵大小 [row * row]
%   repeat 重复次数

%demo
%   repeatEye(2, 2)
%return
%   1   0
%   1   0
%   0   1
%   0   1

eyeMul = ones(repeat, 1) * (1:row);
eyeN = eye(row);
repeatEyeRes = eyeN(eyeMul, :);
end


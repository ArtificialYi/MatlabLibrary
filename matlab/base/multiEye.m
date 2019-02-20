function eyeMul = multiEye(eyeN, multiN)
%multiEye 获取多个单位矩阵 
%   eyeN 单位矩阵大小为 [eyeN * eyeN]
%   multiN 重复次数 multiN

%   demo:
%   multiEye(2, 2)
%   return:
%   1   0
%   0   1
%   1   0
%   0   1

tmpE = mod(1:eyeN*multiN, eyeN);
tmpE(tmpE == 0) = eyeN;

eN = eye(eyeN);
eyeMul = eN(tmpE, :);
end


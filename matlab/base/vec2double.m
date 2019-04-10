function [VNew, VUnique] = vec2double(V)
%vec2double 将一个向量转成double

classV = class(V);
VUnique = [];

switch classV
    case 'cell'
        VNewStr = string(V);
        [VNew, VUnique] = vecStr2double(VNewStr);
    otherwise
        VNew = V;
end

end


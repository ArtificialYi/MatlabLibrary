function [classX] = showHy(X, str)
%showHy 将一个数据渲染出来

classX = class(X);

switch classX
    case 'double'
        switch length(X)
            case 1
                fprintf('%s:%d, %d, %f\n', str, size(X), X);
            otherwise
                fprintf('%s:%d, %d\n', str, size(X));
        end
    otherwise
        printf('%s:未知对象\n', str);
end

end


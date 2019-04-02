function [p] = plotOne(X, pos, flag, markSize)
%plotOne 画出多个点

n = size(X, 2);

switch n
    case 1
        p = plot(X(pos, 1), flag, 'MarkerSize', markSize);
    case 2
        p = plot(X(pos, 1), X(pos, 2), flag, 'MarkerSize', markSize);
    otherwise
        p = plot3(X(pos, 1), X(pos, 2), X(pos, 3), flag, 'MarkerSize', markSize);
end

end


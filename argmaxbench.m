% mecomp == 0 default
% mecomp > 0  special mode
% mecomp < 0  matlab
function [my,t] = argmaxbench(x,dim,to,mecomp)

if mecomp == 5 || mecomp == 6
    tic;
    if dim == 0
        [~,mat] = max(x(:));
    else
        [~,mat] = max(x,[],dim);
    end
    my = cast(mat,'like',to);
    t = toc;
else
    argmax(mecomp);
    tic;
    my = argmax(x,dim,to);
    t = toc;
end  

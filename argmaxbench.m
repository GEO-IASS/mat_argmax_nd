% mecomp == 0 default
% mecomp > 0  special mode
% mecomp < 0  matlab
function [my,t] = argmaxbench(x,dim,to,mecomp)

tic
if mecomp == 6
    if dim == 0
        [~,mat] = max(x(:));
    else
        [~,mat] = max(x,[],dim);
    end
    my = cast(mat,'like',to);
else
    argmax(mecomp);
    my = argmax(x,dim,to);
    % cast needed due to our implementation
    if mecomp == 5
        my = cast(my,'like',to);
    end
end  
t = toc;

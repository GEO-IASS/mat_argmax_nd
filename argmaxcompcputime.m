function [c,my,mat,dt,t_my_over_mat] = argmaxcomp(x,dim,to)
t0=cputime;
my = argmax(x,dim,to);
t1 = cputime-t0;
t0=cputime;
if dim == 0
    [~,mat] = max(x(:));
else
    [~,mat] = max(x,[],dim);
end
mat = cast(mat,'like',to);
t2=cputime-t0;
dt = t2-t1;
t_my_over_mat = t1/t2;

if ndims(mat) ~= ndims(my)
    c = 0;
    return
end
if any(size(mat) ~= size(my))
    c = 0;
    return
end
if dim == 0
    c = all(x(my) == x(mat));
else
    myv = argmax_to_max(x,my,dim);
    matv = argmax_to_max(x,mat,dim);
    c = all(myv(:) == matv(:));
end

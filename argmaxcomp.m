function [c,r1,r2,dt,t_my_over_mat,t1,t2] = argmaxcomp(x,dim,to,mecomp)

if nargin < 4
    [r1,t1] = argmaxbench(x,dim,to,5);
    [r2,t2] = argmaxbench(x,dim,to,0);    
else
    [r1,t1] = argmaxbench(x,dim,to,mecomp(1));
    [r2,t2] = argmaxbench(x,dim,to,mecomp(2));    
end  
dt = t2-t1;
t_my_over_mat = t1/t2;
if ndims(r1) ~= ndims(r2)
    c = 0;
    return
end
if any(size(r1) ~= size(r2))
    c = 0;
    return
end
if dim == 0
    c = all(x(r1) == x(r2));
else
    mr1 = argmax_to_max(x,r1,dim);
    mr2 = argmax_to_max(x,r2,dim);
    c = all(mr1(:) == mr2(:));
end

function info = argmaxsimstride(sizes,dim)

if dim == 0
    info.Kstride = 1;
    info.Ksize = prod(sizes);
    info.Astride = 1;
    info.Asize = 1;
    info.Bstride = 1;
    info.Bsize = 1;
else
    w = [1,sizes(:)',1];
    info.Astride = 1;    
    info.Asize = prod(w(1:dim));
    info.Kstride = info.Asize;
    info.Ksize = w(dim+1);
    info.Bstride = info.Kstride*info.Ksize;
    info.Bsize = prod(w(dim+2:end));
end
function [Yind,scale] = argmax_to_max_setup(sX,dim)

qx = sX;
qx(dim) = 1;

% for every dimension we scan all items taken columnwise
ox = cell(length(sX),1);
six = prod(qx);
xl = 1;
for I=1:dim-1
    xln = xl * sX(I);
    xr = six/xln;
    ox{I} = reshape(repmat(1:sX(I),xl,xr),[],1);
    xl = xln;
end
xlo = xl;
ox{dim} = ones(six,1);
for I=dim+1:length(sX)
    xln = xl * sX(I);
    xr = six/xln;
    ox{I} = reshape(repmat(1:sX(I),xl,xr),[],1);
    xl = xln;    
end

Yind = sub2ind(sX,ox{:}); % linear indices
scale = xlo;
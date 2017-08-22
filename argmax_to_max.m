% given the matrix X and the argmax along dimension dim we would like to
% recover the maximum value
%
function Y = argmax_to_max(X,IX,dim)

qx = size(X);
qx(dim) = 1;
qy = size(IX);
% Matlab autosqueeze: if dim == ndims(X) then IX has one less dimension
if length(qy) < length(qx) & dim == ndims(X)
    qy =[qy,1];
end
assert(length(qx) == length(qy),'same dim of X and IX');
assert(max(IX(:)) <= size(X,dim),'maximum in given direction');
assert(min(IX(:)) >= 1,'minimum is 1');
assert(all(qx == qy),'same size of X and IX');

% for every dimension we scan all items taken columnwise
ox = cell(ndims(X),1);
six = numel(IX);
xl = 1;
for I=1:dim-1
    xln = xl * size(X,I);
    xr = six/xln;
    ox{I} = reshape(repmat(1:size(X,I),xl,xr),[],1);
    xl = xln;
end
ox{dim} = IX(:);
for I=dim+1:ndims(X)
    xln = xl * size(X,I);
    xr = six/xln;
    ox{I} = reshape(repmat(1:size(X,I),xl,xr),[],1);
    xl = xln;    
end

ind = sub2ind(size(X),ox{:}); % linear indices
Y = reshape(X(ind),qx); % lookup and reshape
  
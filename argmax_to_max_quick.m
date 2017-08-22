function [Y] = argmax_to_max_quick(X,iX,Yind,scale)


ind = Yind + (iX(:)-1)*scale;
Y = reshape(X(ind),size(iX));
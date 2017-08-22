%% Using full processing
Y = rand([2,3,5,4]);
dim=2;
[Ym,Yi] = max(Y,[],dim);
Ymm = argmax_to_max(Y,Yi,dim);
all(Ymm(:) == Ym(:))

%% Using preparation and quick execution
Y = rand([2,3,5,4]);
dim=4;
[Ym,Yi] = max(Y,[],dim);
[Yind,scale] = argmax_to_max_setup(size(Y),dim);
Ymm = argmax_to_max_quick(Y,Yi,Yind,scale);
all(Ymm(:) == Ym(:))


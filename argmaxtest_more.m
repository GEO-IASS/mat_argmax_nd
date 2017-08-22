

%% Test simple function over 
to = double(0);
w = rand(1024,10);
tic
r = argmax1(w(:),0,to); 
tme=toc;
tic
w = rand(1024,10);
[~,x] = max(w(:));
tmat=toc;
dt = tme-tmat
r
%assert(x == r)
%%
argmaxmodesdef
nmodes = {'auto','matlabmex'};
modes = cellfun(@(x) argmaxmodes.(x), nmodes);

%a = floor(10*rand([2,4,8,16]));
a = floor(10*rand([1,128,1024]));
tt = {double(0),single(0),uint8(0),int8(0),uint16(0),int16(0),int32(0),uint32(0)};
to = {double(0)}; %double(0),single(0),int8(0),int16(0),int32(0),uint32(0),uint16(0),uint8(0)};

outputc = zeros(length(tt),length(to),ndims(a)+1);
outputdt = zeros(length(tt),length(to),ndims(a)+1);
outputt_my_over_mat = outputdt;
outputt = zeros(length(tt),length(to),ndims(a)+1,2);

for I=1:length(tt)
    for J=1:length(to)
        aa = cast(a,'like',tt{I});
        for K=0:ndims(a)
            {class(aa),K}
            [c0,b0,mb0,dt0,tx,t1,t2] = argmaxcomp(aa,K,to{J},[modes(1),modes(2)]);
            outputc(I,J,K+1) = c0;
            outputdt(I,J,K+1) = dt0;
            outputt_my_over_mat(I,J,K+1) = tx;
            outputt(I,J,K+1,:) = [t1,t2];
        end
    end
end
disp('Dim 1 = input type');
inputtypes = cellfun(@class,tt,'UniformOutput',false)
disp('Dim 2 = output type (if not squeezed)');
outputtypes = cellfun(@class,to,'UniformOutput',false)
disp('Dim 3 = dimension');
inputdims = [0,size(aa)];
inputdims = arrayfun(@(x) ['d',num2str(x)],0:ndims(aa),'UniformOutput',false);
disp('Dim 3 = stride');
inputstride = [1, 1, cumprod(size(aa))];
disp('Correcteness');
disp('My over Matlab time');
outputt_my_over_mat = squeeze(outputt_my_over_mat);
disp('Delta time > 0 if Matlab slower');
outputdt = squeeze(outputdt);
outputc = squeeze(outputc);
if length(inputtypes) == 1
    outputdt = outputdt(:)';
    outputc = outputc(:)';
    outputt_my_over_mat = outputt_my_over_mat(:)';
end
outputc_tbl  = array2table(outputc,'VariableNames',inputdims,'RowNames',inputtypes);
outputdt_tbl  = array2table(outputdt,'VariableNames',inputdims,'RowNames',inputtypes);
outputdt_tblp = array2table(outputdt > 0,'VariableNames',inputdims,'RowNames',inputtypes);
outputt_my_over_mat_tbl= array2table(outputt_my_over_mat,'VariableNames',inputdims,'RowNames',inputtypes);
outputt_my_over_mat_tbl
outputdt_tblp
if ~ all(outputc(:) == 1)
outputc_tbl
end
inputstride




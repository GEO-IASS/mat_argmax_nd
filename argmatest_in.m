%%
argmaxmodesdef

nmodes = {'matlabmex','auto','alongsimd','parsimd','alongnosimd','parnosimd','matlabreal'};
nmodes = nmodes(randperm(length(nmodes)));
modes = cellfun(@(x) argmaxmodes.(x), nmodes);

%a = floor(10*rand([2,4,8,16]));
aa = floor(10*rand([128,32,1024]));
tt = {double(0),single(0),uint8(0),int8(0),uint16(0),int16(0),int32(0),uint32(0)};
to = {double(0)}; %double(0),single(0),int8(0),int16(0),int32(0),uint32(0),uint16(0),uint8(0)};

outputtt = zeros(length(tt),length(to),ndims(aa)+1,length(modes));

% for moving data to MEX
for I=1:length(tt)
    for J=1:length(to)
        a = cast(aa,'like',tt{I});
        [m,t] = argmaxbench(a,0,'double',5);
        for K=0:ndims(a)
            {class(aa),K}
            for M=1:length(modes)
                [m,t] = argmaxbench(a,K,to{J},modes(M));
                outputt(I,J,K+1,M) = t;
            end
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
disp(sprintf('Delta time > 0 if mode %s(%d) slower than mode %s(%d)',argmaxmodes_names{modes(2)+1},modes(2),argmaxmodes_names{modes(1)+1},modes(1)));
if length(inputtypes) == 1
    outputt_my_over_mat = outputt_my_over_mat(:)';
end
for C=1:size(outputt,3)   
    disp(sprintf('Dimension %d',C-1));
    outputtC_tbl = array2table(squeeze(outputt(:,1,C,:)*10000),'Variablenames',nmodes,'RowNames',inputtypes)
end

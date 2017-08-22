
argmaxmodes_names = { 'auto','alongnosimd','parsimd','parnosimd','alongsimd','matlabmex','matlabreal'};

argmaxmodes = [];
for I=1:length(argmaxmodes_names)
    argmaxmodes.(argmaxmodes_names{I}) = I-1;
end

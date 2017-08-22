
stypes = [("SS_DOUBLE","SS_SINGLE","SS_INT8","SS_UINT8","SS_INT32","SS_UINT32","SS_INT16","SS_UINT16","SS_INT64","SS_UINT64","SS_UINT8")]
types = [("mxDOUBLE_CLASS","mxSINGLE_CLASS","mxINT8_CLASS","mxUINT8_CLASS","mxINT32_CLASS","mxUINT32_CLASS","mxINT16_CLASS","mxUINT16_CLASS","mxINT64_CLASS","mxUINT64_CLASS","mxLOGICAL_CLASS"),("double","float","int8_t","uint8_t","int32_t","uint32_t","int16_t","uint16_t","int64_t","uint64_t","uint8_t")]
itypes = types;
dtypes = types;

for i,x in zip(dtypes[1],dtypes[0]):
	for j,y in zip(itypes[1],itypes[0]):
		print "\tcase dualset(%s,%s): return argmax((%s*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(%s*)po);" % (x,y,i,j)

for i,x in zip(dtypes[1],dtypes[0]):
    print "\tcase %s: std::fill_n((%s*)p0,n);break; " % (x,i)
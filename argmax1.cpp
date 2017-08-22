/**
 Argmax by Emanuele Ruffaldi 2017
 

 - supported tensors via axis
 - supported input types int8 int32 single double
 - output index as int32/double


TODO:
- recognize SSE4 and avoid includes
- global flag for simd mode benchamakring
 */
#ifndef NOMATLAB
#include "mex.h"   
#else
#include <iostream>
#endif
#include <array>
#include <memory>
#include <algorithm>

template <class Tin, class Tout>
void argmax10(const Tin * p0, int Ksize, Tout * po)
{
    int imax = 0;
    Tin vmax = p0[0];
    for(int j = 1; j < Ksize; j++)
    {
        if(p0[j] > vmax)
        {
            vmax = p0[j];
            imax = j;
        }
    }
    *po = imax+1;
}


template <class Tin, class Tout>
void argmax1(const Tin * p0, int Ksize, Tout * po)
{
    enum Q {N=2};
    int imax[N];
    Tin vmax[N];
    if(Ksize > N)
    {
        for(int i = 0; i < N; i++)
        {
            imax[i] = i;
            vmax[i] = p0[i];
        }
        
        // N..floor(Ksize/N)%N
        int KKsize = Ksize/N*N;
        for(int j = N; j < KKsize; j += N)
        {
            for(int i = 0; i < N; i++)
            {
                if(p0[j+i] > vmax[i])
                {
                    vmax[i] = p0[j+i];
                    imax[i] = j+i;
                }
            }
        }
        
        // last < N
        for(int j = KKsize; j < Ksize; j++)
        {
            int rq = j-KKsize;
            if(p0[j] > vmax[rq])
            {
                vmax[rq] = p0[j];
                imax[rq] = j;
            }
        }
        
        // aggregate N
        Tin final = vmax[0];
        int ifinal = 0;
        for(int i = 1; i < N; i++)
        {
            if(vmax[i] > final)
            {
                final = vmax[i];
                ifinal = i;
            }
        }
        *po = imax[ifinal]+1;

    }
    else
    {
        Tin final = p0[0];
        int ifinal = 0;
        for(int i = 1; i < Ksize; i++)
        {
            if(p0[i] > final)
            {
                final = p0[i];
                ifinal = i;
            }
        }
        *po = ifinal+1;
    }
}


#ifndef NOMATLAB

// argmax(data,dim,sametypeindex)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// indices
	if(nlhs != 1)
	{
        mexErrMsgTxt("Missing left value\n");
		return;
	}   
	// X,dim,sametypeindex
	if(nrhs != 3)
	{
        mexErrMsgTxt("Syntax: output = argmax(input,dim,templatetype)\n");
		return;		
	}
    if(mxIsSparse(prhs[0]))
    {
        mexErrMsgTxt("only dense\n");
        return;
    }
	// dimension
    auto dim = (int)mxGetScalar(prhs[1]);

    // dimensions of X
	auto ndims = mxGetNumberOfDimensions(prhs[0]);
    int ondims;
	const mwSize * dimi = mxGetDimensions(prhs[0]);
	if(ndims == 1) // trivial
	{
        // all ones
		return;
	}
    {
        int asize = 1;
		for(int i = 0; i < ndims; i++)
			asize *= dimi[i];   
        double po = 0;
        argmax1((double*)mxGetData(prhs[0]),asize,&po);
        plhs[0] = mxCreateDoubleScalar(po);
    }
	return ;
}    
#endif

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
//#define HAS_SSES

#include "asimd_all.h"

#define dualsel(tin,tout) ((tin)+(tout)*20)


//argmaxmodes_names = { 'auto','alongnosimd','parsimd','parnosimd','alongsimd','matlabmex','matlabreal'};
enum RunningMode { ModeAuto,ModeAlongNoSimd,ModeParSimd,ModeParNoSimd,ModeAlongSimd,ModeMatlabMEX,ModeMatlabREAL};

static RunningMode forcemode = ModeAuto; // 

template <class T>
class strided_ptr
{
public:
    strided_ptr(T * p, int stride): p_(p), stride_(stride) {}
    strided_ptr(strided_ptr<T> other, int stride) : p_(other.p_), stride_(stride) {}

    strided_ptr & operator += (int k)
    {
        p_ += k*stride_;
        return *this;
    }

    strided_ptr & operator ++()
    {
        p_ += stride_;
        return *this;
    }

    T & operator * () 
    {
        return *p_;
    }

    T * get() 
    { 
        return p_;
    }


    T * p_;
    int stride_;
};

/*
 * Type selection mechanism
 */

#ifdef __AVX2__
template <class T, class Tout>
void argmax1(const T * p0, const int Ksize, const int Kstride, Tout * po, SimdMode)
{
	const T * p00 = p0;
    typedef typename simdgen<T>::type Q; // simd_X_size
    typedef typename Q::indextype QI;
    int iK = 0; // index first in steps of C then full
    T xmax; // max for final part
    int imax; // imax for final part

    if(Ksize >= Q::csize)
    {
	    QI pg; // gathering index: 0 kstride 2kstride ... Nkstride 
	    Q curmax; // current maximum, initially the first C
	    Q cur; // current value
	    QI icur;
	    QI icurmax; // maximum indices    
	    QI icurinc(Q::csize); // increment
    	initgather<Q>(Kstride,pg); // if supported

    	#ifdef NOMATLAB
        std::cout << "pregather " << pg << std::endl;
	    #endif

	    // take 0 as current maximum
	    icurmax.initincrement(1); // 0..C
	    icur = icurmax;
        loadstrided(curmax,p0,Kstride,pg); 

        #ifdef NOMATLAB
        std::cout << "icurinc " << icurinc << std::endl;
        #endif

        // start from 1
        p0 += Kstride*Q::csize;
	    icur += icurinc;

    	#ifdef NOMATLAB
        std::cout << "iteration iQK " << 0 << " max0  is " << curmax << std::endl;
        std::cout << "iteration iQK " << 0 << " imax0 is " << icurmax << std::endl;
        #endif
        for(int iQK = 1; iQK < Ksize/Q::csize; iQK++, p0 += Kstride*Q::csize,icur += icurinc)
        {
            loadstrided(cur,p0,Kstride,pg);
            auto cmp = curmax.cmplt(cur); // curmax >= cur 

            // maximum:
            // 		update curmax for all elements in which cmp is 1111
            // index:
            //		update icurmax for all elements in which cmp is 1111 (typed as index)
           	curmax.blend(cur,cmp); // pick other if 1 at bit
           	curmax.blendindex(icurmax,icur,cmp); // special for sizeof(T) != int32

    	#ifdef NOMATLAB
            std::cout << "iteration iQK " << iQK << " other   is " << cur << std::endl;
            std::cout << "iteration iQK " << iQK << " iother  is " << icur << std::endl;
            std::cout << "iteration iQK " << iQK << " cmp     is " << cmp << std::endl;
            std::cout << "iteration iQK " << iQK << " newmax  is " << curmax << std::endl;
            std::cout << "iteration iQK " << iQK << " inewmax is " << icurmax << std::endl;
            #endif
        }
        // p0 is already at last aligned

    	#ifdef NOMATLAB
            std::cout << "outloop  " << " curmax  is " << curmax << " and " << icurmax << std::endl;
    	#endif

        // horizontally find the maximum
        // NOTE: for large parallel entities: e.g. i8 or i16 we could employ more parallelism here

        T final[Q::csize];
        int finali[QI::csize];
        curmax.store(final);
        icurmax.store(finali);


        xmax = final[0];
        imax = finali[0];
        for(int i = 1; i < Q::csize; i++)
        {        
            if(final[i] > xmax)
            {
                xmax = final[i];
                imax = finali[i];
            }
        }
	    iK = p0-p00; // Ksize/Q::csize; // starting is truncated
    }
    else
    {
    	iK = 1;
        imax = 0;
        xmax = p0[0];
        p0 += Kstride;
    }

    // not aligned mode
    for(; iK < Ksize; iK++, p0 += Kstride)
    {
        if(*p0 > xmax)
        {
            xmax = *p0;
            imax = iK;
        }
    }

    // emit
    *po = ((Tout)imax)+1; 
}
#endif

template <class Tin, class Tout>
void argmax1(const Tin * p0, int Ksize, int Kstride, Tout * po, NoSimdMode)
{
    int imax = 0;
    Tin b = p0[0]; 
    for(int iK = 1; iK < Ksize; iK++)
    {
        Tin v = p0[Kstride*iK];        
        if(v > b)
        {
            b = v;
            imax = iK;
        }
    }
    //mexPrintf("Assigning iA=%d iB=%d K opt %d as %f first %f\n",iA,iB,imax,(float)b,(float)p000[0]);
    *po = ((Tout)imax)+1; 
}

// unrolling of size Q
template <class Tin, class Tout, int Q>
void argmax1g(const Tin * p0, int Ksize, int Kstride, Tout * po, NoSimdMode)
{    
    int imax[Q];
    Tin b[Q];
    const int m = Ksize < Q ? m : Q;
        
    for(int i = 0; i < m; i++)
    {
        imax[i] = i;
        b[i] = p0[Kstride*i];
    } 
    
    if(m == Q)
    {
        int ee = (Ksize/Q)*Q;
        // scan up to floor 
        for(int iK = Q; iK < ee; iK += Q)
        {
            // par
            for(int i = 0; i < Q; i++)
            {
                Tin v = p0[Kstride*(iK+i)];
                if(v > b[i])
                {
                    imax[i] = iK+i;
                    b[i] = v;
                }
            }            
        }   

        int left = Ksize-ee;
        for(int i = 0; i < left; i++)
        {
            Tin v = p0[Kstride*(ee+i)];
            if(v > b[i])
            {
                imax[i] = ee+i;
                b[i] = v;
            }
        }           
    }

    // aggregate up to Q
    Tin omax = b[0];
    int oimax = imax[0];
    for(int j = 1; j < m; j++)
    {
        if(b[j] > omax)
        {
            omax = b[j];
            oimax = imax[j];
        }
    }
    *po = ((Tout)oimax)+1; 
}


// Astride == 1
template <class Tin, class Tout> 
void argmax1_hor(const Tin*p00,const int Asize,const int Ksize,const int Kstride,Tout * po,NoSimdMode, const int ri)
{
    const int Astride = 1;
    const int N = 16;
    std::array<Tin,N> horbuf;
    std::array<int,N> horbufi;
    const int steps = Asize/N; // truncation
    const int remainder = Asize % N;
    
    for (int n = 0; n < steps; n++, p00 += N*Astride)
    {
        const Tin * p0 = p00;        
        for(int i = 0; i < N; i++) // along A
        {
            horbuf[i] = p0[i*Astride];
            horbufi[i] = 0; // first
        }
        p0 += Kstride; // next
        for(int iK = 1; iK < Ksize; iK++, p0 += Kstride)
        {
            for(int i = 0; i < N; i++)
            {
                auto v = p0[i*Astride];
                if(v > horbuf[i])
                {
                    horbuf[i] = v;
                    horbufi[i] = iK;
                }
            }
        }      
        for(int i = 0; i < N; i++)
            *po++ = horbufi[i];
    }
    
    // a batch < N
    if(remainder > 0)
    {
        const Tin * p0 = p00;        
        for(int i = 0; i < remainder; i++) // along A
        {
            horbuf[i] = p0[i*Astride];
            horbufi[i] = 0; // first
        }
        p0 += Kstride; // next
        for(int iK = 1; iK < Ksize; iK++, p0 += Kstride)
        {
            for(int i = 0; i < remainder; i++)
            {
                auto v = p0[i*Astride];
                if(v > horbuf[i])
                {
                    horbuf[i] = v;
                    horbufi[i] = iK;
                }
            }
        }        
        for(int i = 0; i < remainder; i++)
            *po++ = horbufi[i];
    }   
}

template <class Tin, class Tout> 
void argmax1_hor(const Tin*p00,const int Asize,const int Ksize,const int Kstride,Tout * po,SimdMode, int ri)
{
    typedef typename simdgen<Tin>::type Q;
    typedef typename simdgen<Tin>::simdmarker S;
    const int steps = Asize/Q::csize; // truncation
    const int remainder = Asize%Q::csize;
    typename Q::indextype::type tmpout[Q::indextype::csize]; // Q::indextype::csize >= Q::csize    
        
    // split Asize in slices of Q::csize
    for (int n = 0; n < steps; n++, p00 += Q::csize) // Astride*Q::csize
    {
        typename Q::indextype icurmax(1); // first 1-based
        typename Q::indextype inccur(1); // increment at iteration
        typename Q::indextype icur(2); // next 1-based
        const Tin * p0 = p00;
        Q curmax;
        curmax.load(p0); // first
        p0 += Kstride; // next

        for(int iK = 1; iK < Ksize; iK++, p0 += Kstride, icur += inccur) // Kstride
        {
            Q cur;
            cur.load(p0);
            Q cmp = curmax.cmplt(cur);
            curmax.blend(cur,cmp); 
            curmax.blendindex(icurmax,icur,cmp); 
        }               
        // index already 1-based

        // now distribute the outputs by column
        icurmax.store(tmpout); // 1-base
        for(int o = 0; o < Q::csize; o++) // only the ones needed, ignore the rest
            *po++ = (Tout)tmpout[o]; //  => if same type would be faster
    }    
    
    // handle remaining part
    if(remainder != 0)
    {
        typedef typename Q::indextype QI;
        typename QI::type pgin[QI::csize]; // Q::indextype::csize >= Q::csize
	    QI pg; // gathering index: 0 kstride 2kstride ... Nkstride 
        int i;
        for(i = 0; i < remainder; i++)
            pgin[i] = i*sizeof(typename Q::type);
        for(; i < QI::csize; i++)
            pgin[i] = 0; 
        pg.load(pgin);
        
        const Tin * p0 = p00;
        typename Q::indextype icurmax(1); // first 1-based
        typename Q::indextype inccur(1); // increment at iteration
        typename Q::indextype icur(2); // next 1-based
        Q curmax;
        //curmax.gather(p0,pg);
        curmax.load(p0);
        p0 += Kstride; // next

        for(int iK = 1; iK < Ksize; iK++, p0 += Kstride, icur += inccur) // Kstride
        {
            Q cur;
            //cur.gather(p0,pg);
            cur.load(p0); // unsafe
            Q cmp = curmax.cmplt(cur);
            curmax.blend(cur,cmp); 
            curmax.blendindex(icurmax,icur,cmp); 
        }               
        // index already 1-based

        // now distribute the outputs by column
        icurmax.store(tmpout); // 1-base
        for(int o = 0; o < remainder; o++) // only the ones needed, ignore the rest
            *po++ = (Tout)tmpout[o]; //  => if same type would be faster
        
    }
}


// scanning by column (right to left that is B to A) makes output contiguous
template <class Tin, class Tout>
void argmax(const Tin * p000, int Asize, int Ksize, int Bsize, int Astride, int Kstride, int Bstride,  Tout*po)
{
    typedef typename simdgen<Tin>::type Q;
    typedef typename simdgen<Tin>::simdmarker S;
//    #ifdef NOMATLAB
    // the following crashes on matlab
    #ifndef NOMATLAB
        //mexPrintf("argmax1: Asize=%d Ksize=%d Kstride=%d steps=%d,remainder=%d\n",Asize,Ksize,Kstride,Asize/Q::csize,Asize % Q::csize);
    #endif
    switch(forcemode)
    {
        case ModeAuto:
            if(Kstride > 1 && Asize > 1)
            {
                for(int iB = 0; iB < Bsize; iB++, p000 += Bstride, po += Asize)
                {
                    argmax1_hor(p000,Asize,Ksize,Kstride,po,S(),iB);  // empty implementation for the other
                }                    
            }
            else
            {
                for(int iB = 0; iB < Bsize; iB++, p000 += Bstride)
                {
                    const Tin * p00 = p000; // begin of the submatrix
                    for (int iA = 0; iA < Asize; iA++, p00 += Astride )
                    {
                        argmax1(p00,Ksize,Kstride,po++,S());
                    }
                }                       
            }
            break;
        case ModeAlongNoSimd:
            for(int iB = 0; iB < Bsize; iB++, p000 += Bstride)
            {
                const Tin * p00 = p000; // begin of the submatrix
                for (int iA = 0; iA < Asize; iA++, p00 += Astride )
                {
                    argmax1(p00,Ksize,Kstride,po++,NoSimdMode());
                }
            }      
            break;
        case ModeParSimd:
            for(int iB = 0; iB < Bsize; iB++, p000 += Bstride, po += Asize)
            {
                argmax1_hor(p000,Asize,Ksize,Kstride,po,S(),iB);  // empty implementation for the other
            }    
            break;
        case ModeParNoSimd:
            for(int iB = 0; iB < Bsize; iB++, p000 += Bstride, po += Asize)
            {
                argmax1_hor(p000,Asize,Ksize,Kstride,po,NoSimdMode(),iB);  // empty implementation for the other
            }    
            break;
        case ModeAlongSimd:
            for(int iB = 0; iB < Bsize; iB++, p000 += Bstride)
            {
                const Tin * p00 = p000; // begin of the submatrix
                for (int iA = 0; iA < Asize; iA++, p00 += Astride )
                {
                    argmax1(p00,Ksize,Kstride,po++,S());
                }
            }       
            break;
        default:
            break;
    }
        
}

#ifndef NOMATLAB

template <class T>
void mexinfo()
{
    typedef typename simdgen<T>::type Q;
    typedef typename simdgen<T>::simdmarker S;
    typedef typename Q::indextype QI;
    mexPrintf("Type %s WrapperSIMD %-15s TypeSIMD %-10s Size %-2d Marker %-10s Index %-20s AlignmentSIMD %-2d AlignmenntIndex %-2d\n",
            typeid(T).name(),typeid(Q).name(),typeid(typename Q::simdtype).name(),(int)Q::csize,typeid(S).name(),typeid(QI).name(),alignof(Q),alignof(QI));
}

// argmax(data,dim,sametypeindex)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs == 0 && nlhs == 0)
    {
        mexPrintf("argmax SIMD Emanuele Ruffaldi 2017\n");
        // for each type
        mexinfo<float>();
        mexinfo<double>();
        mexinfo<int8_t>();
        mexinfo<uint8_t>();
        mexinfo<int16_t>();
        mexinfo<uint16_t>();
        mexinfo<int32_t>();
        mexinfo<uint32_t>();
        return;
    }
    if(nrhs == 1 && nlhs == 0)
    {
        forcemode = (RunningMode)*(double*)mxGetPr(prhs[0]);
        mexPrintf("argmax forcemode %d (0='auto',1='alongnosimd',2='parsimd',3='parnosimd',4='alongsimd',5='matlab') \n",(int)forcemode);
        return;
    }
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
        mexErrMsgTxt("sparse not supported\n");
        return;
    }
    if(mxIsComplex(prhs[0]))
    {
        mexErrMsgTxt("complex not supported\n");
        return;
    }
	// dimension argument
    auto dim = (int)mxGetScalar(prhs[1]);

    // dimensions of X
	const auto ndims = mxGetNumberOfDimensions(prhs[0]);
    int ondims; // output dimensions
	const mwSize * dimi = mxGetDimensions(prhs[0]);
    
    if(dim < 0 || dim > ndims)
	{
        mexErrMsgTxt("invalid dimension argument: 0..ndims(x)\n");
        return;
	}
       
    if(forcemode == ModeMatlabMEX)
    {
        // [~y] = max(y,[],dim);
        if(dim != 0)
        {
            mxArray *mrhs[3], *mlhs[2];
            mrhs[0] = mxDuplicateArray(prhs[0]); // input
            mrhs[1] = mxCreateDoubleMatrix( 0, 0, mxREAL ); // []
            mrhs[2] = mxCreateDoubleScalar(dim); // dim
            mexCallMATLAB(2, mlhs, 3, mrhs, "max"); 
            plhs[0] = mxDuplicateArray(mlhs[1]);
            mxDestroyArray(mlhs[0]);
            mxDestroyArray(mlhs[1]);
            mxDestroyArray(mrhs[0]);
            mxDestroyArray(mrhs[1]);
            mxDestroyArray(mrhs[2]);
        }
        // [~,y] = max(reshape(x,1,[]))
        else
        {
            mxArray *mrhs1[3], *mlhs1[1];
            mxArray *mrhs2[1], *mlhs2[2];            
            mrhs1[0] = mxDuplicateArray(prhs[0]); // input duplication
            mrhs1[1] = mxCreateDoubleScalar(1); // 1
            mrhs1[2] = mxCreateDoubleMatrix( 0, 0, mxREAL ); // []
            mexCallMATLAB(1, mlhs1, 3, mrhs1, "reshape"); 
            mxDestroyArray(mrhs1[0]);
            mxDestroyArray(mrhs1[1]);
            mxDestroyArray(mrhs1[2]);
            mrhs2[0] = mlhs1[0]; // propagate output as input of max
            mexCallMATLAB(2, mlhs2, 1, mrhs2, "max");  
            plhs[0] = mlhs2[1]; // propagate output to final output
            mxDestroyArray(mlhs2[0]); // the output not used
            mxDestroyArray(mrhs2[0]); // the input
        }
        return;
    }
    else if(forcemode == ModeMatlabREAL)
        return;
    
	if(ndims == 1 || (dim > 0 && dim < ndims && dimi[dim] == 1)) // singular dimension
	{
        // ones(size(x))
    	plhs[0] = mxCreateUninitNumericArray(ndims,(mwSize*)dimi,mxGetClassID(prhs[2]),mxREAL);
        int asize = 1;
		for(int i = 0; i < ndims; i++)
			asize *= dimi[i];   
        void * p0 = mxGetData(prhs[0]);
        int n = asize;
        switch(mxGetClassID(prhs[2]))
        {
        	case mxDOUBLE_CLASS: std::fill_n((double*)p0,n,1);break; 
            case mxSINGLE_CLASS: std::fill_n((float*)p0,n,1);break; 
            case mxINT8_CLASS: std::fill_n((int8_t*)p0,n,1);break; 
            case mxUINT8_CLASS: std::fill_n((uint8_t*)p0,n,1);break; 
            case mxINT32_CLASS: std::fill_n((int32_t*)p0,n,1);break; 
            case mxUINT32_CLASS: std::fill_n((uint32_t*)p0,n,1);break; 
            case mxINT16_CLASS: std::fill_n((int16_t*)p0,n,1);break; 
            case mxUINT16_CLASS: std::fill_n((uint16_t*)p0,n,1);break; 
            case mxINT64_CLASS: std::fill_n((int64_t*)p0,n,1);break; 
            case mxUINT64_CLASS: std::fill_n((uint64_t*)p0,n,1);break; 
            case mxLOGICAL_CLASS: std::fill_n((uint8_t*)p0,n,1);break; 
        // mxUNKNOWN_CLASS', 'mxCELL_CLASS', 'mxSTRUCT_CLASS
            default:
                mexErrMsgTxt("Unsupported type of output\n");
                break;
        }
		return;
	}
    
       
	mwSize dimo[10];
	memset(dimo,0,sizeof(dimo));
	if(ndims > sizeof(dimo)/sizeof(dimo[0]))
	{
        mexErrMsgTxt("maximum 10 dimensions of input (recompile)\n");
		return; 
	}
	int Asize,Bsize,Ksize,Astride,Kstride,Bstride;
	
    // scalar output
	if(dim == 0 || ndims == 1)
	{
		ondims = 1;
        dimo[0] = 1;
		int os = 1; // prod(size(x))
		for(int i = 0; i < ndims; i++)
			os *= dimi[i];
		Asize = 1;
		Bsize = 1;
		Ksize = os;
		Kstride = 1;
		Astride = 1;
		Bstride = 1;
	}
	else
	{
        ondims = ndims;
        memcpy(dimo,dimi,ndims*sizeof(dimo[0]));
        dim--; // zero based
        dimo[dim] = 1;

        // A size
        int asize = 1;
		for(int i = 0; i < dim; i++)
			asize *= dimi[i];
		// B size
		int bsize = 1;
		for(int i = dim+1; i < ndims;  i++)
			bsize *= dimi[i];
		// Tensor: D1...Dk-1 Dk Dk+1...Dn by column (rightmost)
		// Output: D1...DK-1 Dk+1...Dn by column
		//
		// We simplify it as:  A Dk B and AB respectively (matrix operations)
		// Column major: 
		// - A dimension runs faster and it is the step of each element in Dk. 
		// - B dimension runs slower and it is the spa
		Ksize = dimi[dim];		
		Asize = asize;
		Bsize = bsize;
		Kstride = asize; // what is on the left of K
		Bstride = asize*Ksize; // what is on the left of B
		Astride = 1;  // always increment by 1
        
		/* 
		   Examples decompising in A K B
		   e.g. matrix M N along 1 => A={} K=M B=N => Kstride=N Bstride=1 Astride=0
		   e.g. matrix M N along 2 => A=M K=N B={} => Kstride=1 Bstride=0 Astride=M
		   e.g. tensor M N Q along N => A=M K=N B=Q => Kstride=Q Bstride=1 Astride=MN
		 */
        //mexPrintf("Scan: AKB sizes %d %d %d and strides AKB %d %d %d\n",Asize,Ksize,Bsize,Astride,Kstride,Bstride);
        //mexPrintf("\tOutput:");
        //for(int i = 0; i <ondims; i++)
        //    mexPrintf(" %d ",dimo[i]);
        //mexPrintf("\n");
	}
	plhs[0] = mxCreateUninitNumericArray(ondims,dimo,mxGetClassID(prhs[2]),mxREAL);

	// TODO is complex
	void * p0 = mxGetData(prhs[0]);
    void * po = mxGetData(plhs[0]);
    if(!p0 || !po)
    {
        mexErrMsgTxt("cannot allocate output or read input data\n");
        return;
    }
	#define offset 0    
    #define dualset(a,b) ((a)*100+b)
	switch(dualset(mxGetClassID(prhs[0]),mxGetClassID(plhs[0])))
	{
    case dualset(mxDOUBLE_CLASS,mxDOUBLE_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxDOUBLE_CLASS,mxSINGLE_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxDOUBLE_CLASS,mxINT8_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxDOUBLE_CLASS,mxUINT8_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxDOUBLE_CLASS,mxINT32_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxDOUBLE_CLASS,mxUINT32_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxDOUBLE_CLASS,mxINT16_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxDOUBLE_CLASS,mxUINT16_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxDOUBLE_CLASS,mxINT64_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxDOUBLE_CLASS,mxUINT64_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxDOUBLE_CLASS,mxLOGICAL_CLASS): return argmax((double*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxSINGLE_CLASS,mxDOUBLE_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxSINGLE_CLASS,mxSINGLE_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxSINGLE_CLASS,mxINT8_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxSINGLE_CLASS,mxUINT8_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxSINGLE_CLASS,mxINT32_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxSINGLE_CLASS,mxUINT32_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxSINGLE_CLASS,mxINT16_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxSINGLE_CLASS,mxUINT16_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxSINGLE_CLASS,mxINT64_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxSINGLE_CLASS,mxUINT64_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxSINGLE_CLASS,mxLOGICAL_CLASS): return argmax((float*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT8_CLASS,mxDOUBLE_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxINT8_CLASS,mxSINGLE_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxINT8_CLASS,mxINT8_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxINT8_CLASS,mxUINT8_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT8_CLASS,mxINT32_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxINT8_CLASS,mxUINT32_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxINT8_CLASS,mxINT16_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxINT8_CLASS,mxUINT16_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxINT8_CLASS,mxINT64_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxINT8_CLASS,mxUINT64_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxINT8_CLASS,mxLOGICAL_CLASS): return argmax((int8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT8_CLASS,mxDOUBLE_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxUINT8_CLASS,mxSINGLE_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxUINT8_CLASS,mxINT8_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxUINT8_CLASS,mxUINT8_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT8_CLASS,mxINT32_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxUINT8_CLASS,mxUINT32_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxUINT8_CLASS,mxINT16_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxUINT8_CLASS,mxUINT16_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxUINT8_CLASS,mxINT64_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxUINT8_CLASS,mxUINT64_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxUINT8_CLASS,mxLOGICAL_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT32_CLASS,mxDOUBLE_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxINT32_CLASS,mxSINGLE_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxINT32_CLASS,mxINT8_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxINT32_CLASS,mxUINT8_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT32_CLASS,mxINT32_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxINT32_CLASS,mxUINT32_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxINT32_CLASS,mxINT16_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxINT32_CLASS,mxUINT16_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxINT32_CLASS,mxINT64_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxINT32_CLASS,mxUINT64_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxINT32_CLASS,mxLOGICAL_CLASS): return argmax((int32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT32_CLASS,mxDOUBLE_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxUINT32_CLASS,mxSINGLE_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxUINT32_CLASS,mxINT8_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxUINT32_CLASS,mxUINT8_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT32_CLASS,mxINT32_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxUINT32_CLASS,mxUINT32_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxUINT32_CLASS,mxINT16_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxUINT32_CLASS,mxUINT16_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxUINT32_CLASS,mxINT64_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxUINT32_CLASS,mxUINT64_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxUINT32_CLASS,mxLOGICAL_CLASS): return argmax((uint32_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT16_CLASS,mxDOUBLE_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxINT16_CLASS,mxSINGLE_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxINT16_CLASS,mxINT8_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxINT16_CLASS,mxUINT8_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT16_CLASS,mxINT32_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxINT16_CLASS,mxUINT32_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxINT16_CLASS,mxINT16_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxINT16_CLASS,mxUINT16_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxINT16_CLASS,mxINT64_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxINT16_CLASS,mxUINT64_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxINT16_CLASS,mxLOGICAL_CLASS): return argmax((int16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT16_CLASS,mxDOUBLE_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxUINT16_CLASS,mxSINGLE_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxUINT16_CLASS,mxINT8_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxUINT16_CLASS,mxUINT8_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT16_CLASS,mxINT32_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxUINT16_CLASS,mxUINT32_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxUINT16_CLASS,mxINT16_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxUINT16_CLASS,mxUINT16_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxUINT16_CLASS,mxINT64_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxUINT16_CLASS,mxUINT64_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxUINT16_CLASS,mxLOGICAL_CLASS): return argmax((uint16_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT64_CLASS,mxDOUBLE_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxINT64_CLASS,mxSINGLE_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxINT64_CLASS,mxINT8_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxINT64_CLASS,mxUINT8_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxINT64_CLASS,mxINT32_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxINT64_CLASS,mxUINT32_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxINT64_CLASS,mxINT16_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxINT64_CLASS,mxUINT16_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxINT64_CLASS,mxINT64_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxINT64_CLASS,mxUINT64_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxINT64_CLASS,mxLOGICAL_CLASS): return argmax((int64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT64_CLASS,mxDOUBLE_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxUINT64_CLASS,mxSINGLE_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxUINT64_CLASS,mxINT8_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxUINT64_CLASS,mxUINT8_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxUINT64_CLASS,mxINT32_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxUINT64_CLASS,mxUINT32_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxUINT64_CLASS,mxINT16_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxUINT64_CLASS,mxUINT16_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxUINT64_CLASS,mxINT64_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxUINT64_CLASS,mxUINT64_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxUINT64_CLASS,mxLOGICAL_CLASS): return argmax((uint64_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxLOGICAL_CLASS,mxDOUBLE_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(double*)po);
	case dualset(mxLOGICAL_CLASS,mxSINGLE_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(float*)po);
	case dualset(mxLOGICAL_CLASS,mxINT8_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int8_t*)po);
	case dualset(mxLOGICAL_CLASS,mxUINT8_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
	case dualset(mxLOGICAL_CLASS,mxINT32_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int32_t*)po);
	case dualset(mxLOGICAL_CLASS,mxUINT32_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint32_t*)po);
	case dualset(mxLOGICAL_CLASS,mxINT16_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int16_t*)po);
	case dualset(mxLOGICAL_CLASS,mxUINT16_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint16_t*)po);
	case dualset(mxLOGICAL_CLASS,mxINT64_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(int64_t*)po);
	case dualset(mxLOGICAL_CLASS,mxUINT64_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint64_t*)po);
	case dualset(mxLOGICAL_CLASS,mxLOGICAL_CLASS): return argmax((uint8_t*)p0+offset,Asize,Ksize,Bsize,Astride,Kstride,Bstride,(uint8_t*)po);
    
    // mxUNKNOWN_CLASS', 'mxCELL_CLASS', 'mxSTRUCT_CLASS
        default:
            mexPrintf("Unknown type combination: %d = max(T %d)\n",(int)mxGetClassID(plhs[0]),(int)mxGetClassID(prhs[2]));
	}
	return ;


}    
#endif

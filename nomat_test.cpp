
#define NOMATLAB
#include "argmax.cpp"
#include <memory>
#include <iostream>

#define dualsel(tin,tout) ((tin)+(tout)*20)
#define mexPrintf printf


#if 0

template <class T, class Tout>
void argmax1(const T * p0, const int Ksize, const int Kstride, Tout * po, SimdMode)
{
    const T * p00 = p0;
        std::cout << "values "; 
        std::copy(p0,p0+Ksize,std::ostream_iterator<T>(std::cout," "));
        std::cout << std::endl;
    mexPrintf("SIMD mode\n");
    typedef typename simdgen<T>::type Q;
    int iK; // index first in steps of C then full
    Q curmax; // current maximum, initially the first C
    int icurmax[Q::csize]; // maximum indices    
    Q cur; // current value
    T xmax;
    int imax;
    if(Ksize >= Q::csize)
    {

                
        // 0..Q
        for(int i = 0; i < Q::csize; i++)
            icurmax[i] = i; 
        
        // filled in
        loadstrided(curmax,p0,Kstride); 
        T tmp[Q::csize];
        p0 += Kstride*Q::csize;
        curmax.store(tmp);
        printf("Ksize %d and steps %d\n",Ksize,Ksize/Q::csize);
        std::cout << "initial "; 
        std::copy(tmp,tmp+Q::csize,std::ostream_iterator<T>(std::cout," "));
        std::cout << std::endl;
        std::cout << "initial step "; 
        std::copy(icurmax,icurmax+Q::csize,std::ostream_iterator<int>(std::cout," "));
        std::cout << std::endl;

        // iterate Q wise
        for(iK = 1; iK < Ksize/Q::csize; iK++, p0 += Kstride*Q::csize)
        {
            loadstrided(cur,p0,Kstride);
            curmax = curmax.max(cur);
            curmax.store(tmp);

            std::cout << "newmax "; 
            std::copy(tmp,tmp+Q::csize,std::ostream_iterator<T>(std::cout," "));
            std::cout << std::endl;

            //printf("newmax  %d %d %d %d\n",tmp[0],tmp[1],tmp[2],tmp[3]);
            cur.store(tmp);
            //printf("other   %d %d %d %d\n",tmp[0],tmp[1],tmp[2],tmp[3]);
            //printf("istep %d %d %d %d\n",icurmax[0],icurmax[1],icurmax[2],icurmax[3]);
            Q cmp = cur.cmplt(curmax); // curmax >= cur 
            cmp.store(tmp);
            //printf("icmp  %d %d %d %d\n",tmp[0],tmp[1],tmp[2],tmp[3]);
            // update icurmax
            for(int i = 0; i < Q::csize; i++)
            {
                if(cmp[i] == 0)
                {
                    //printf("\tReplace rel %d @ %d with new rel at %d\n",i,icurmax[i],iK*Q::csize + i);
                    icurmax[i] = iK*Q::csize + i;
                }
            }
        }
        T final[Q::csize];
        curmax.store(final);
        xmax = final[0];
        imax = icurmax[0];

        std::cout << "final      : "; 
        std::copy(final,final+Q::csize,std::ostream_iterator<T>(std::cout," "));
        std::cout << std::endl;
        std::cout << "final index: "; 
        std::copy(icurmax,icurmax+Q::csize,std::ostream_iterator<int>(std::cout," "));
        std::cout << std::endl;
        for(int i = 1; i < Q::csize; i++)
        {        
            if(final[i] > xmax)
            {
                xmax = final[i];
                imax = icurmax[i];
            }
        }
        std::cout << "output of shrink is " << imax << std::endl;
        // remainder
        iK *= Q::csize;
    }
    else
    {
        std::cout << "shortmode \n";
        // first 
        iK = 1;
        imax = 0;
        xmax = p0[0];
        p0 += Kstride;
    }

    std::cout << "remainder " << Ksize-iK <<  " for imax " << imax << " and max " << xmax << " first is " << iK << " but " << (p0-p00) << " as " << *p0 << std::endl;
    for(; iK < Ksize; iK++, p0+=Kstride)
    {
        if(*p0 > xmax)
        {
            xmax = *p0;
            imax = iK;
        }
    }
    *po = ((Tout)imax)+1; 
}

template <class Tin, class Tout>
void argmax1(Tin * p0, int Ksize, int Kstride, Tout * po, NoSimdMode)
{
    int imax = 0;
    Tin b = p0[0]; // first
    p0 += Kstride; // move to second
    for(int iK = 1; iK < Ksize; iK++, p0 += Kstride)
    {
        if(*p0 > b)
        {
            b = *p0;
            imax = iK;
        }
    }
    //mexPrintf("Assigning iA=%d iB=%d K opt %d as %f first %f\n",iA,iB,imax,(float)b,(float)p000[0]);
    *po = ((Tout)imax)+1; 
}
#endif

int main(int argc, char * argv[])
{
   // double wowd[] = {1.2,4.5960   , 2.1800  ,  7.3530, 4.5782, 7.2, 8.2}; //[128];
    float wowf[52];
    // 0..14
    // 1..15
    for (int i = 0; i < sizeof(wowf)/sizeof(wowf[0]); i ++)
       wowf[i] = i;

    double wowd[10*4];
    // 0..14
    // 1..15
    for (int i = 0; i < sizeof(wowd)/sizeof(wowd[0]); i ++)
       wowd[i] = i;

    int32_t wowi[32];
    // 0..14
    // 1..15
    for (int i = 0; i < sizeof(wowi)/sizeof(wowi[0]); i ++)
       wowi[i] = i;

    int16_t wowu[32];
    // 0..14
    // 1..15
    for (int i = 0; i < sizeof(wowu)/sizeof(wowu[0]); i ++)
       wowu[i] = i;

   int8_t wow8[128];
    // 0..14
    // 1..15
    for (int i = 0; i < sizeof(wow8)/sizeof(wow8[0]); i ++)
       wow8[i] = i;

    int stride = 1;
    //double wow[128];
    
    #define wow wowf
    typedef int32_t outtype;
    int n = (sizeof(wow)/sizeof(wow[0]));
    int en = (n+stride-1)/stride;
    int32_t po[10] = {-1};
    std::cout << "eff: ";
        for(int i = 0; i < en; i++)
            std::cout << " " << (outtype) wow[i*stride];
        std::cout << std::endl;

    std::cout << "Ksize " << en << " using stride " << stride << std::endl;
#ifdef X__AVX2__
    //argmax1(wow,en,stride,po,SimdMode());
    int Asize = 10;
    int Kstride = Asize;
    int Astride = 1;
    int Ksize = sizeof(wow)/sizeof(wow[0])/Asize;
    for(int i = 0; i < Asize; i++)
    {
        std::cout << "row " << i << ":";
        for(int k = 0; k < Ksize; k++)
        {
            std::cout << " " << wow[Astride*i+Kstride*k] ;
        }
        std::cout << "\n";
    }
    argmax1_hor(wow,Asize,Ksize,Kstride,po,SimdMode());
    for(int i = 0; i < Asize; i++)
        std::cout << "row " << i << " index " << po[i]-1 << " value is " << (outtype) wow[Astride*i + Kstride*(po[i]-1)] << std::endl;
#else
    argmax1(wow,en,stride,po,SimdMode());
#endif
    std::cout << "index " << po[0]-1 << " value is " << (outtype) wow[stride*(po[0]-1)] << std::endl;
    return 0;
}
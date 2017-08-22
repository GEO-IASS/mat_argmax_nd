# TODO #
https://db.in.tum.de/~finis/x86-intrin-cheatsheet-v2.1.pdf
http://www.pmx.it/sse4.html
https://software.intel.com/sites/landingpage/IntrinsicsGuide/


Performance is not exception and it could be improved by using AVX/AVX2/SSE4: 

They work by taking two groups of N values (2..8) and compute the pairwise 
maximum value. 

Then we need to collect these maxima 

In pratice linearly scanning our data in groups of 2N we extract the N pairwise 
maxima. We need then to mark the group from which each originates and finally
build the resulting index. 

T[M] data => T[N][Q=M/N] data;

T[N] curmax = data[0]; // register
int[N] ii = {0,1,2,3};
int[N] icurmax = ii; 

for(int i = 1; i < Q; i++, ii+=N)
{
    T[N] newmax = mm_max(curmax,data[i]);
    int[N] icmp = mm_cmplt(newmax,data[i]); // otherwise use cmp and then gather
    if(icmp[j] == 0) // for all N
        icurmax[j] = ii[j]; // that is:   (i << N)+j
}

// reduce curmax: manually hierarchically or using or via loop
int im01 = curmax[0] < curmax[1]; // 1 if #1 is maximum else 0
int im23 = (curmax[2] < curmax[3])+2; // 3 if #3 is maximum else 2
int im = curmax[im01] >= curmax[im23];
int imi = im ? im01 : im23; // final

// shifting
T[N] curmax = ABCD
T[N] curmaxr = rotateleft(curmax) = CDAB
compute max and comp AB vs CD 

// handle M%N remanining linearly

// using register 256bit that is: 16 8 4 2 for int8..double

/*
 * List all required functions per type

_mm256_max_ep[iu][8 16 32]

Special epu16: minpos_epu16 SSE4.1

max_X

    SSE ps
    SSE2 epu8 epi16 pd
    SSE4.1 epi8-32 epu8-32

cmp[op]_[T] op is all: lt gt eq and n prefix
    SSE2 ps pd ss sd epi8-32 
    SSE4.1 epi64
    AXV is special

    cmpeq_

64bit integer are not supported

Types: __m256, __m256i __m256d
 */ 
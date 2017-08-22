/**
 * Requires: AVX2, allows for AVX512F
 *
 - missing: 64bit integer, unsigned integers except uint8
 - AVX2: double 4, float/int32 8, int16 16, int8 32

 */
#pragma once
#include "asimd_32.h"

#ifdef __AVX2__

/**
 16 x uint8_t
 */
class simd_i8_32
{
public:
    typedef int8_t type;
    typedef __m256i simdtype;
    typedef simd_i32_8_a<4> indextype; 
    typedef NoGather gathermode;
    typedef simd_i8_32 self;

    enum { csize = 32 };
    
    inline simd_i8_32() {}
    inline simd_i8_32(type v) : x(_mm256_set1_epi8(v)) {}
    inline simd_i8_32(simdtype v) : x(v) {}
    inline void load(const type * ptr) { x = _mm256_loadu_ps((const float*)ptr); }
    inline void store(type * ptr) const { _mm256_storeu_ps((float*)ptr,x); }
    inline simd_i8_32 max(simd_i8_32 & y) const { return simd_i8_32(_mm256_max_epi8(x,y.x)); }
    inline simd_i8_32 cmplt(simd_i8_32 & y) const { return  simd_i8_32(_mm256_cmpgt_epi8(y.x,x)); }
    inline unsigned int size() const { return csize; }    

    void initincrement(type x)
    {
        type a[csize];
        for(int i = 0; i < csize; i++)
            a[i] = i*x;
        load(a);
    }

    /*
    inline type operator[] (unsigned int idx) const
    {
        type temp[csize];
        store(temp);
        return temp[idx];
    } 
    */   

    inline void blend(self & other, self mask)
    {
        x = _mm256_blendv_epi8(x,other.x,mask.x);
    }

    inline void blendindex(indextype & oindex, indextype other, self mask);

    simdtype x;
};



/**
 32 x uint8_t
 */
class simd_u8_32
{
public:
    typedef uint8_t type;
    typedef __m256i simdtype;
    typedef simd_i32_8_a<4> indextype; 
    typedef NoGather gathermode;
    typedef simd_u8_32 self;

    enum { csize = 32 };
    
    inline simd_u8_32() {}
    inline simd_u8_32(type v) : x(_mm256_set1_epi8(v)) {}
    inline simd_u8_32(simdtype v) : x(v) {}
    inline void load(const type * ptr) { x = _mm256_loadu_ps((const float*)ptr); }
    inline void store(type * ptr) const { _mm256_storeu_ps((float*)ptr,x); }
    inline simd_u8_32 max(simd_u8_32 & y) const { return simd_u8_32(_mm256_max_epu8(x,y.x)); }
    inline simd_u8_32 cmplt(simd_u8_32 & y) const { return  simd_u8_32(_mm256_cmpgt_epu8(y.x,x)); }
    inline unsigned int size() const { return csize; }    

    void initincrement(type x)
    {
        type a[csize];
        for(int i = 0; i < csize; i++)
            a[i] = i*x;
        load(a);
    }

    /*
    inline type operator[] (unsigned int idx) const
    {
        type temp[csize];
        store(temp);
        return temp[idx];
    } 
    */   

    inline void blend(self & other, self mask)
    {
        x = _mm256_blendv_epi8(x,other.x,mask.x);
    }

    inline void blendindex(indextype & oindex, indextype other, self mask);

    simdtype x;
};



// from 1 _mm256 8bit to 4 _mm256 32bit
inline void simd_u8_32::blendindex(indextype & oindex, indextype other, self mask)
{
    // mask = 8 items 32bit = A B C D E F G H each expresses the status of 4 variables
    // Objective: blendv: most sigificant bit: 31 63 .. 255
    //
    // ABCD can be taken from 128bit of the input, and in particular the A is just what needed
    __m128i cur = _mm256_castsi256_si128(mask.x);
    __m256i mask0 = mergelowhigh(
            _mm_cvtepi8_epi32(cur), // just the first 32bits
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,4)) // the next 32 bits
            );
    __m256i mask1 = mergelowhigh(
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,8)),
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,12))
            );
    cur = gethigh(mask.x);

    __m256i mask2 = mergelowhigh(
            _mm_cvtepi8_epi32(cur), // just the first 32bits
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,4))
            );
    __m256i mask3 = mergelowhigh(
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,8)),
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,12))
            );

    oindex.x[0] = _mm256_blendv_ps(oindex.x[0],other.x[0],mask0);
    oindex.x[1] = _mm256_blendv_ps(oindex.x[1],other.x[1],mask1);
    oindex.x[2] = _mm256_blendv_ps(oindex.x[2],other.x[2],mask2);
    oindex.x[3] = _mm256_blendv_ps(oindex.x[3],other.x[3],mask3);
}

// from 1 _mm256 8bit to 4 _mm256 32bit
inline void simd_i8_32::blendindex(indextype & oindex, indextype other, self mask)
{
    // mask = 8 items 32bit = A B C D E F G H each expresses the status of 4 variables
    // Objective: blendv: most sigificant bit: 31 63 .. 255
    //
    // ABCD can be taken from 128bit of the input, and in particular the A is just what needed
    __m128i cur = _mm256_castsi256_si128(mask.x);
    __m256i mask0 = mergelowhigh(
            _mm_cvtepi8_epi32(cur), // just the first 32bits
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,4)) // the next 32 bits
            );
    __m256i mask1 = mergelowhigh(
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,8)),
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,12))
            );
    cur = gethigh(mask.x);

    __m256i mask2 = mergelowhigh(
            _mm_cvtepi8_epi32(cur), // just the first 32bits
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,4))
            );
    __m256i mask3 = mergelowhigh(
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,8)),
            _mm_cvtepi8_epi32(_mm_srli_si128(cur,12))
            );

    oindex.x[0] = _mm256_blendv_ps(oindex.x[0],other.x[0],mask0);
    oindex.x[1] = _mm256_blendv_ps(oindex.x[1],other.x[1],mask1);
    oindex.x[2] = _mm256_blendv_ps(oindex.x[2],other.x[2],mask2);
    oindex.x[3] = _mm256_blendv_ps(oindex.x[3],other.x[3],mask3);
}




#endif

// due to _mm_blendv_epi8
#ifdef __AVX2__

/**
 16 x uint8_t
 */
class simd_u8_16
{
public:
    typedef uint8_t type;
    typedef __m128i simdtype;
    typedef simd_i32_8_a<2> indextype;
    typedef NoGather gathermode;
    typedef simd_u8_16 self;
    enum { csize = 16 };
    
    inline simd_u8_16() {}
    inline simd_u8_16(type v) : x(_mm_set1_epi8(v)) {}
    inline simd_u8_16(simdtype v) : x(v) {}
    inline void load(const type * ptr) { x = _mm_loadu_si128((const simdtype*)ptr); }
    inline void store(type * ptr) const { _mm_storeu_si128((simdtype*)ptr,x); }
    inline simd_u8_16 max(simd_u8_16 & y) const { return simd_u8_16(_mm_max_epu8(x,y.x)); }
    inline simd_u8_16 cmplt(simd_u8_16 & y) const { return  simd_u8_16(_mm_cmplt_epu8(x,y.x)); }
    inline unsigned int size() const { return csize; }    

    void initincrement(type x)
    {
        type a[csize];
        for(int i = 0; i < csize; i++)
            a[i] = i*x;
        load(a);
    }

    /*
    inline type operator[] (unsigned int idx) const
    {
        type temp[csize];
        store(temp);
        return temp[idx];
    } 
    */   

    inline void blend(self & other, self mask)
    {
        x = _mm_blendv_epi8(x,other.x,mask.x);
    }

    inline void blendindex(indextype & oindex, indextype other, self mask);

    simdtype x;
};


/**
 16 x uint8_t
 */
class simd_i8_16
{
public:
    typedef int8_t type;
    typedef __m128i simdtype;
    typedef simd_i32_8_a<2> indextype;
    typedef NoGather gathermode;
    typedef simd_i8_16 self;
    enum { csize = 16 };
    
    inline simd_i8_16() {}
    inline simd_i8_16(type v) : x(_mm_set1_epi8(v)) {}
    inline simd_i8_16(simdtype v) : x(v) {}
    inline void load(const type * ptr) { x = _mm_loadu_si128((const simdtype*)ptr); }
    inline void store(type * ptr) const { _mm_storeu_si128((simdtype*)ptr,x); }
    inline simd_i8_16 max(simd_i8_16 & y) const { return simd_i8_16(_mm_max_epi8(x,y.x)); }
    inline simd_i8_16 cmplt(simd_i8_16 & y) const { return  simd_i8_16(_mm_cmplt_epi8(x,y.x)); }
    inline unsigned int size() const { return csize; }    

    void initincrement(type x)
    {
        type a[csize];
        for(int i = 0; i < csize; i++)
            a[i] = i*x;
        load(a);
    }

    /*
    inline type operator[] (unsigned int idx) const
    {
        type temp[csize];
        store(temp);
        return temp[idx];
    } 
    */   

    inline void blend(self & other, self mask)
    {
        x = _mm_blendv_epi8(x,other.x,mask.x);
    }

    inline void blendindex(indextype & oindex, indextype other, self mask);

    simdtype x;
};
#endif




#ifdef __AVX2__
DECLAREOSTREAMT(simd_i8_16,"i8_16",int32_t)
DECLAREOSTREAMT(simd_u8_16,"u8_16",uint32_t)
DECLAREOSTREAMT(simd_i8_32,"i8_32",int32_t)
DECLAREOSTREAMT(simd_u8_32,"u8_32",uint32_t)

template <>
struct simdgen<int8_t>
{
    typedef simd_i8_32 type;
    typedef SimdMode simdmarker;
};

template <>
struct simdgen<uint8_t>
{
    typedef simd_u8_32 type;
    typedef SimdMode simdmarker;
};

template <int n>
struct simdgenn<int8_t,n>
{
    using type = typename std::conditional<n <= 16, simd_i8_16, simd_i8_32>::type;
    typedef SimdMode simdmarker;
};

template <int n>
struct simdgenn<uint8_t,n>
{
    using type = typename std::conditional<n <= 16, simd_u8_16, simd_u8_32>::type;
    typedef SimdMode simdmarker;
};


#else

/*
template <>
struct simdgen<int8_t>
{
    typedef simd_i8_16 type;
    typedef SimdMode simdmarker;
};

template <>
struct simdgen<uint8_t>
{
    typedef simd_u8_16 type;
    typedef SimdMode simdmarker;
};
*/

#endif



# Argmax for MATLAB for tensors and user specified output type using Intel SIMD #

Emanuele Ruffaldi 2017 Scuola Superiore Sant'Anna Pisa

Exercise over Intel SIMD started from the need of a mex function with the following signature:

    Y = argmax(X,dim,T)

If dim = 0 makes it allover
If dim > 0 behaves like [~,Y] = max(X,[],dim)

The output size is not squeezed 
T is a variable whose type is used to define the output type of the max e.g. int8(0) or single(0)

Depending on the compilation flags the MEX function uses AVX512/AVX2

## Status

* int8 and int16 have issues in the collection
* AVX512 works
* with large sizes the MEX seems much slower

# Implementation #

C++ using templates and Python for generating all the combination between (type(X),type(T))

Tensors are supported by using the fact that if we have an input tensor with sizes: D1..Dn and target dimension k 
the problem can be reduced to the 3 dimensional case Da Dk Db.
Where Da=D1 x .. Dk-1 and Db = Dk+1 x ... x Dn.

To remove corner cases we assume to work in the case: [1 D1 ... Dn 1] in this way Da and Db are always defined.

In the case k=0 we are working such that: [1 Dk 1] where Dk=numel(X).

SIMD operations support efficient parallel operations for M elements, depending on the type and the specific Intel
extension (e.g. for double we can have 2,4 or 8, for int8 we can have M up to 64). In general we can also consider 
to aggregate multiple SIMD registers for simulating parallel operations over M elements, e.g using arrays of registers. 
SIMD data gather can be performed in hardware with 32bit indices but the effectiveness of this operation depends on cache access patterns.

Given the reduced tensor [Da Dk Db] and the parallel operations over M elements there are two possible approaches:
parallelising along the k dimension or across the other dimensions. In the former we decompose the Dk elements in
pieces of size M and propagate max and its index in parallel along the dimension, then, when all the Dk elements
have been processed the final max is computed by reduction. In the latter the a or b dimensions are decomposed by M, scanning
the k dimension in parallel M times at once. At the end of each scan there is no need to reduce the results.

The selection of which strategy to adopt depends on the size and stride of each dimension. In column major order such as Matlab the
strides for a 3d tensors are the product of the dimensions on the left of the given dimension:

- stride_a = 1
- stride_k = size_a
- stride_c = size_a*size_k

The parallelism along k is cache friendly only if stride_k is each step (sizeof(T)*stride_k*M) can fit in a cache line, 
otherwise step will be not efficient. The parallelism along the a dimension is cache favorable because we pick all the elements 
in sequence. In the case of small size_k there could be a reduction in efficency.

We have to consider the case of M larger than the scanning size (size_k for former, size_a for latter): this can be addressed repeating
the elements inside the block of M, and then ignoring some of the results at the end. Intel SIMD works with three sizes of registers (128,256 and 512)
depending on the available extensions: a smaller M can be used before using the repetition.

## Heuristics ##

We have to decide the algorithm based on the task (Da Dk Db), the type T, and the available size M, with the option of expanding the basic M as an array.

* Da=1 favor along
* Da>1 favor parallel

TODO: plus 8bit and 16bit gather missing at SIMD instruction level.

## Results against Matlab ##

We report as: stride length pre post

Example: 1 128 1024
* Dimension 0 is OK (1/131072)
* Dimension 1 is ones(size(in)) (to be optimized using SIMD because std::fill_n is slow)
* Dimension 2 alongsimd and parsimd seems similar (stride 1/128)
* Dimension 3 alongsimd wins (stride 128/1024)

Example: 128 32 1024
* Dimension 0 is (1/4194304/1/1)
* Dimension 1 is (1/128/1/32768)
* Dimension 2 is (128/32/128/1024)
* Dimension 3 is (4096/1024/4096/1)

# Building #

mex -v CFLAGS='$CFLAGS -march=native -O3' argmax.cpp

# Optimization #

SSE...AVX2 for using SIMD in particular:

- double: simd_d_2 simd_d_4
- float: simd_f_4 simd_f_8
- int32: simd_i32_4 simd_i32_8
- int16: simd_i16_8 simd_i16_16
- int8:  simd_i8_16 simd_i8_32

I have optimized:

- gathering
- index propagation

Support for AVX-512 (AVX-512F) will bring another doubling of parallelism.

# TODO Partially outdated # 

## Horizontal Opt ##
(1) In progress: Astride=1 Kstride > something then we can proceed along A in blocks of size C moving along K: same number of operations more cache friendly

simd_x_a provides what is needed for this horizontal behavior. 

## Array of double ##

simd_d_4_a<N> is not complete (e.g. gather)

## AVX512 ##
(3) Test for AVX512 (Skylake and X)

https://software.intel.com/en-us/node/524490
g++ --std=c++11 -mavx2 -mfma -mf16c -mavx512vl -mavx512f  -DNOMATLAB nomat_test.cpp 

Added simd_d_8 as an experiment. Intel changed the API again, in particular mmask8 is a new result for cmp

### Testing ###

Testing of AVX512 results can be performed with the Intel SDE Emulator: https://software.intel.com/en-us/articles/intel-software-development-emulator#BASIC

Tested on macOS running Emulator 8.5.0, and Apple LLVM 8.1.0

Histogram of opcodes:
	
	sed -mix -- ./a.out 

## Picking the Correct Type ##
(2)
Currently for a given type only the maximum size is picked for the compiled hardware (e.g. int32 8 items) downgrading to scalar for smaller sizes. This means that we are not covering some optimizations. E.g. a dimension of 6 can be managed as simg_i32_4 plus 2 leftovers.

# Issue with Gather epi8 and epi16 #

There is no gether such as the other types but there is masked load in AVX512VL and AVX512BW.

# Dynamic Dispatch #

https://github.com/Mysticial/FeatureDetector/ provides a code based on CPUID for detecting the presence of AVX2/AVX512. This tool
works also in the simulated environment provided by Intel SDE. All AVX512 is supported except for AVX512-PF AVX512-ER XOP and FMA4.
# Strassen_CUDA
An implementation of Strassen's algorithm for matrix multiplication on CUDA. It computes the matrix multiplication of two $2^n\times2^n$ matrices. 

It takes two arguments, $k$ and $k'$. $k$ is the exponent of 2 for the size of the matix. $k'$ is a number of levels to recurse before using the matrix multiplication kernel (so the matrix multiplication kernel will run on matrices
of size $2^{k-k'}\times2^{k-k'}$). For my machine, choosing $k'$ such that this size is $2^8$ was most efficient.

It includes some optimizations for speed (utilizes shared memory, warp primitives, avoids bank conflicts), but there are probably better optimized MM kernels out there.

It does not optimize memory so there are probably significantly more memory efficient implementations. This one utilizes about 3 times the sum of memory used for the two input matrices and the output matrix.

# Compiling and Running It
```bash
nvcc -o mm strassen.cu
./mm 12 4
```
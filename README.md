# Strassen_CUDA
An implementation of Strassen's algorithm for matrix multiplication on CUDA. It computes the matrix multiplication of two $2^n\times2^n$ matrices. 

It takes two arguments, $k$ and $k'$. $k$ is the exponent of 2 for the size of the matix. $k'$ is a number of levels to recurse before using the matrix multiplication kernel (so the matrix multiplication kernel will run on matrices
of size $2^{k-k'}\times2^{k-k'}$). Generally, choosing a small $k'$ was most efficient, say $k=12$ and $k'=4$.

It includes some optimizations for speed (utilizes shared memory, warp primitives, avoids bank conflicts), but there are probably better optimized MM kernels out there. 
The bank conflict optimization (allocating an additional column in the ```col``` shared memory array) yields slightly more than a 4x speedup compared to not including it.

It does not optimize memory so there are probably significantly more memory efficient implementations. This one utilizes about 3 times the sum of memory used for the two input matrices and the output matrix.

NOTE: it doesn't utilize tensor cores or lower precision operations so there's a lot of opportunity for improved speed there.

# Compiling and Running It
```bash
nvcc -o mm strassen.cu
./mm 12 4
```

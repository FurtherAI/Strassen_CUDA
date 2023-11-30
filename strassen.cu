#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32
#define MAX_GRID 32
#define FULL_MASK 0xffffffff
#define RANGE_MAX 1000

int k, kprime;

__global__ void matrix_mul(int N, float *a, float *b, float *c) {
    if ((threadIdx.x >= N) || (threadIdx.y >= N)) {
        return;  // threads past the boundaries don't participate
    }

    // N here is the width/height of the matrices A and B
    __shared__ float row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float col[BLOCK_SIZE][BLOCK_SIZE + 1];  // offest by 1 to avoid bank conflicts when reading
    __shared__ float out[BLOCK_SIZE][BLOCK_SIZE];

    out[threadIdx.y][threadIdx.x] = 0;
    
    int yidx = threadIdx.y;
    int xidx = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = BLOCK_SIZE < N ? BLOCK_SIZE : N;
    for (int n = 0; n < (((N - 1) / BLOCK_SIZE) + 1); n++) {
        // load sub blocks into shared memory
        row[threadIdx.y][threadIdx.x] = a[i * N + xidx];  // each of these is fully coalesced access
        col[threadIdx.y][threadIdx.x] = b[yidx * N + j];
        __syncthreads();

        float thread_val = row[threadIdx.y][threadIdx.x];  // consistent, so keep in register

        for (int k = 0; k < cols; k++) {
            float dot = thread_val * col[threadIdx.x][k];  // since column is offset by one, warp can read whole column in one instruction
            for (int offset = 16; offset > 0; offset = offset >> 1) {  // reduction operation using warp primitives
                dot += __shfl_down_sync(FULL_MASK, dot, offset);
            }
            if (threadIdx.x == 0) {
                out[threadIdx.y][k] += dot;
            }
        }

        xidx += blockDim.x;
        yidx += blockDim.y;
        __syncthreads();  // so nothing new is read into row/col while other threads are using them
    }

    c[(blockIdx.y * blockDim.y + threadIdx.y) * N + blockIdx.x * blockDim.x + threadIdx.x] = out[threadIdx.y][threadIdx.x];
}

// grid-strided loop addition, flexible, reduces time spent on block/thread creation
__global__ void matrix_add(int n, int quarter_size, float k, float *a, float *b, float *c) {
    if (threadIdx.x >= quarter_size || threadIdx.y >= quarter_size) {
        return;
    }
    int N = quarter_size;
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < N; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += blockDim.x * gridDim.x) {
            c[i * N + j] = a[i * N + j] + k * b[i * N + j];  // constant multiple here for dual function as a subtraction kernel
        }
    }
}

__global__ void write_quadrant(int n, int quarter_size, int yoffset, int xoffset, float *src, float *dst) {
    if (threadIdx.x >= quarter_size || threadIdx.y >= quarter_size) {
        return;
    }
    int N = quarter_size;
    int M = N << 1;
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < N; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += blockDim.x * gridDim.x) {
            dst[(i + yoffset) * M + j + xoffset] = src[i * N + j];
        }
    }
}

__global__ void quarter(int n, size_t quarter_size, float *in, float *q1, float *q2, float *q3, float *q4) {
    if (threadIdx.x >= quarter_size || threadIdx.y >= quarter_size) {
        return;
    }
    int N = quarter_size;  // size of each quarter along each side
    int M = N << 1;  // size for the in array

    int blocks_per_quarter = quarter_size / BLOCK_SIZE;
    blocks_per_quarter = blocks_per_quarter > 0 ? blocks_per_quarter : 1;
    int offset = BLOCK_SIZE > quarter_size ? BLOCK_SIZE - quarter_size : 0;

    int quad_offset = blocks_per_quarter * BLOCK_SIZE;
    int ct = 0;
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < M || ct < 1; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < M || ct < 1; j += blockDim.x * gridDim.x) {
            if (j < quad_offset && i < quad_offset) {
                q1[i * N + j] = in[i * M + j];
            }
            else if (j >= quad_offset && i < quad_offset) {
                q2[i * N + j - quad_offset] = in[i * M + j - offset];
            }
            else if (j < quad_offset && i >= quad_offset) {
                q3[(i - quad_offset) * N + j] = in[(i - offset) * M + j];
            }
            else if (j >= quad_offset && i >= quad_offset) {
                q4[(i - quad_offset) * N + j - quad_offset] = in[(i - offset) * M + j - offset];
            }
            ct++;
        }
    }
}

void strassen(float *d_A, float *d_B, float *d_C, int k, int kterm) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s: %s...aborting\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
    if (k == kterm) {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        size_t grid_size = (1 << kterm) / BLOCK_SIZE;
        grid_size = grid_size > 0 ? grid_size : 1;
        dim3 grid(grid_size, grid_size);
        matrix_mul<<<grid, block>>>(1 << kterm, d_A, d_B, d_C);
        return;
    }
    else {
        // allocate device memory for each quarter, to then perform reads from for the following additions and multiplications
        size_t quarter_size = 1 << (k - 1);  // size of quarters along one side
        size_t matrix_size = 1 << (2 * (k - 1) + 2);  // size of quarters total

        float *A00, *A01, *A10, *A11;
        cudaMalloc(&A00, matrix_size);cudaMalloc(&A01, matrix_size);cudaMalloc(&A10, matrix_size);cudaMalloc(&A11, matrix_size);
        float *B00, *B01, *B10, *B11;
        cudaMalloc(&B00, matrix_size);cudaMalloc(&B01, matrix_size);cudaMalloc(&B10, matrix_size);cudaMalloc(&B11, matrix_size);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        size_t grid_size = (quarter_size << 1) / BLOCK_SIZE;
        grid_size = grid_size > 2 ? grid_size : 2;

        // grid-strided loop stuff
        int n = 1;
        if (grid_size > MAX_GRID) {
            n = grid_size / MAX_GRID;
            grid_size = MAX_GRID;
        }
        dim3 grid(grid_size, grid_size);
        quarter<<<grid, block>>>(n, quarter_size, d_A, A00, A01, A10, A11);
        quarter<<<grid, block>>>(n, quarter_size, d_B, B00, B01, B10, B11);

        float *M1, *M2, *M3, *M4, *M5, *M6, *M7;
        // only need to store up to two sums at a time, will be used to accumulate M's also
        float *S1, *S2;
        cudaMalloc(&M1, matrix_size);cudaMalloc(&M2, matrix_size);cudaMalloc(&M3, matrix_size);cudaMalloc(&M4, matrix_size);
        cudaMalloc(&M5, matrix_size);cudaMalloc(&M6, matrix_size);cudaMalloc(&M7, matrix_size);

        cudaMalloc(&S1, matrix_size);cudaMalloc(&S2, matrix_size);

        grid_size = quarter_size / BLOCK_SIZE;
        grid_size = grid_size > 1 ? grid_size : 1;
        n = 1;
        if (grid_size > MAX_GRID) {
            n = grid_size / MAX_GRID;
            grid_size = MAX_GRID;
        }
        grid.x = grid.y = grid_size;

        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, A00, A11, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, B00, B11, S2);
        strassen(S1, S2, M1, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, A10, A11, S1);
        strassen(S1, B00, M2, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, B01, B11, S1);
        strassen(A00, S1, M3, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, B10, B00, S1);
        strassen(A11, S1, M4, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, A00, A01, S1);
        strassen(S1, B11, M5, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, A10, A00, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, B00, B01, S2);
        strassen(S1, S2, M6, k - 1, kterm);

        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, A01, A11, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, B10, B11, S2);
        strassen(S1, S2, M7, k - 1, kterm);

        // make sure matrix multiplication is happening on the right sides
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, M1, M4, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, S1, M5, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, S1, M7, S1);
        write_quadrant<<<grid, block>>>(n, quarter_size, 0, 0, S1, d_C);

        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, M3, M5, S1);
        write_quadrant<<<grid, block>>>(n, quarter_size, 0, quarter_size, S1, d_C);
        
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, M2, M4, S1);
        write_quadrant<<<grid, block>>>(n, quarter_size, quarter_size, 0, S1, d_C);

        matrix_add<<<grid, block>>>(n, quarter_size, -1.0, M1, M2, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, S1, M3, S1);
        matrix_add<<<grid, block>>>(n, quarter_size, 1.0, S1, M6, S1);
        write_quadrant<<<grid, block>>>(n, quarter_size, quarter_size, quarter_size, S1, d_C);

        cudaFree(A00);cudaFree(A01);cudaFree(A10);cudaFree(A11);
        cudaFree(B00);cudaFree(B01);cudaFree(B10);cudaFree(B11);
        cudaFree(M1);cudaFree(M2);cudaFree(M3);cudaFree(M4);cudaFree(M5);cudaFree(M6);cudaFree(M7);
        cudaFree(S1);cudaFree(S2);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Incorrect number of arguments given.\n");
        printf("Usage: ./executable k (2^k x 2^k matrices) k' (k' levels of recursion before using standard matrix multiplication)\n");
        exit(1);
    }
    k = atoi(argv[1]);
    kprime = atoi(argv[2]);
    if (kprime >= k || kprime <= 1) {
        printf("k' must be less than k and greater than 1 (k > k' > 1).\n");
        exit(1);
    }

    int n = 1 << k;
    size_t n_squared = 1 << (2 * k + 2);  // each side is 2^k, so 2^2k and float is 4 bytes so 2^(2k + 2)
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    h_A = (float *)malloc(n_squared);
    h_B = (float *)malloc(n_squared);
    h_C = (float *)malloc(n_squared);

    // could make these allocations page locked if you wanted to
    cudaMalloc(&d_A, n_squared);
    cudaMalloc(&d_B, n_squared);
    cudaMalloc(&d_C, n_squared);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // random float between 0 and RANGE_MAX
            // h_A[i * n + j] = (float)((double)rand()/(double)(RAND_MAX/RANGE_MAX));
            // h_B[i * n + j] = (float)((double)rand()/(double)(RAND_MAX/RANGE_MAX));
            h_A[i * n + j] = 1;
            h_B[i * n + j] = 1;
        }
    }

    cudaMemcpy(d_A, h_A, n_squared, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_squared, cudaMemcpyHostToDevice);

    strassen(d_A, d_B, d_C, k, k - kprime);
    cudaMemcpy(h_C, d_C, n_squared, cudaMemcpyDeviceToHost);
    bool incorrect = false;
    for (int i = 0; (i < (1 << k)) && !incorrect; i++) {
        for (int j = 0; (j < (1 << k)) && !incorrect; j++) {
            if (h_C[i * n + j] != (1 << k)) {
                printf("incorrect value\n");
                incorrect = true;
            }
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


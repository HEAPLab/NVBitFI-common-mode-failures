#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include "cuspis/CUSPIS.cuh"

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int col = threadIdx.x % N;
    int row = blockIdx.x + N * (int)(threadIdx.x / N);
    float sum = 0.0f;

    //if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[(row * N) + k] * B[(k * N) + col];
        }
        C[row * N + col] = sum;
    // }
}
static uint32_t z1, z2, z3, z4;
void random_set_seed(uint32_t seed)
{
    z1 = 1+seed;
    z2 = 7+seed;
    z3 = 15+seed;
    z4 = 127+seed;
}

uint32_t random_get_int(void) {
    uint32_t b;
    b  = ((z1 << 6) ^ z1) >> 13;
    z1 = ((z1 & 4294967294U) << 18) ^ b;
    b  = ((z2 << 2) ^ z2) >> 27;
    z2 = ((z2 & 4294967288U) << 2) ^ b;
    b  = ((z3 << 13) ^ z3) >> 21;
    z3 = ((z3 & 4294967280U) << 7) ^ b;
    b  = ((z4 << 3) ^ z4) >> 12;
    z4 = ((z4 & 4294967168U) << 13) ^ b;
    return (z1 ^ z2 ^ z3 ^ z4);
}

double random_get(void)
{

    return random_get_int() * 2.3283064365386963e-10;
}

int main(int argc, char* argv[]) {
    int N = 16; // default size

    random_set_seed(0xabcd);

    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_C_ref;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = random_get() * 10;
        h_B[i] = random_get() * 10;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CUSPIS::cuspisMalloc(&d_A, size);
    CUSPIS::cuspisMalloc(&d_B, size);
    CUSPIS::cuspisMalloc(&d_C, size);

    CUSPIS::cuspisMemcpyToDevice(d_A, h_A, size);
    CUSPIS::cuspisMemcpyToDevice(d_B, h_B, size);

    // Kernel launch configuration
    CUSPIS::Kernel<float*, float*, float*, int> 
				kernel_b(N, N, matMulKernel, CUSPIS::cuspisRedundantThreads);
    kernel_b.launch(d_A, d_B, d_C, N);

    // matMulKernel<<<N,N>>>(d_A, d_B, d_C, N);
    CUSPIS::cuspisMemcpyToHost(h_C, d_C, size);

    for (int i=0; i<N*N; i++) {
        printf("%f ", h_C[i]);
    }

    // Cleanup
    CUSPIS::cuspisFree(&d_A);
    CUSPIS::cuspisFree(&d_B);
    CUSPIS::cuspisFree(&d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    // free(h_C_ref);

    return 0;
}

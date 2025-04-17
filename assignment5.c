%%writefile vector_add.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024  // Size of vectors

// Declare vectors in unified memory (accessible from host and device)
_device_ _managed_ float A[N], B[N], C[N];

// CUDA kernel for vector addition
_global_ void vectorAdd() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID
    if (i < N)
        C[i] = A[i] + B[i];  // Perform element-wise addition
}

// CUDA error checker utility
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize vectors A and B on the host
    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    float time_ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);  // Start timing

    // Launch vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();  // Launch GPU kernel
    check(cudaGetLastError(), "Kernel launch");

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);  // Calculate elapsed time in ms

    // Display first 5 elements of the result vector
    printf("C = [");
    for (int i = 0; i < 5; ++i) printf("%.1f ", C[i]);
    printf("...]\n");

    // Query device properties for theoretical bandwidth calculation
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float memClock = prop.memoryClockRate * 1e3;   // Convert to Hz
    float busWidth = prop.memoryBusWidth;          // In bits
    float theoreticalBW = 2 * memClock * busWidth / 8 / 1e9; // GB/s (DDR hence x2)
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

    // Calculate actual bandwidth used by the kernel
    float totalBytes = 2 * N * sizeof(float) + N * sizeof(float); // A and B read, C written
    float measuredBW = totalBytes / (time_ms / 1000.0f) / 1e9;     // GB/s
    printf("Measured Bandwidth: %.2f GB/s\n", measuredBW);
    printf("Execution Time: %.4f ms\n", time_ms);

    return 0;
}


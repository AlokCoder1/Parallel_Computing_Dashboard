%%writefile sumofn.cu
#include <stdio.h>
#include <cuda_runtime.h>


#define N 1024




__global__ void sumKernel(int *input, int *output) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Task A: Iterative Sum
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
    else if (tid == 1) {


        output[1] = (N * (N - 1)) / 2;
    }
}


int main() {
    int h_input[N], h_output[2] = {0};


    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }


    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, 2 * sizeof(int));


    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);




    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);


    sumKernel<<<1, 2>>>(d_input, d_output);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);




    printf("Sum using Iteration (Thread 0): %d\n", h_output[0]);
    printf("Sum using Formula   (Thread 1): %d\n", h_output[1]);
    printf("GPU Kernel Execution Time: %.6f ms\n", milliseconds);




    cudaFree(d_input);
    cudaFree(d_output);


    return 0;
}

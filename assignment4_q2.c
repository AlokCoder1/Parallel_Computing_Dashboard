%%writefile merge.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ void merge(int *arr, int l, int m, int r, int *temp) {
    int i = l, j = m + 1, k = 0;

    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    for (i = l, k = 0; i <= r; i++, k++) {
        arr[i] = temp[k];
    }
}

__global__ void mergeSortKernel(int *arr, int n, int step) {
    extern __shared__ int temp[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l = idx * step * 2;
    int m = l + step - 1;
    int r = min(l + step * 2 - 1, n - 1);

    if (l < n && m < n && r < n) {
        merge(arr, l, m, r, temp);
    }
}

void cudaMergeSort(int *h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Perform iterative merge sort
    for (int step = 1; step < n; step *= 2) {
        mergeSortKernel<<<blocksPerGrid, threadsPerBlock, BLOCK_SIZE * sizeof(int)>>>(d_arr, n, step);
        cudaDeviceSynchronize();
    }

    // Copy sorted array back to host
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
}

int main() {
    int n = 1000;
    int *arr = (int*)malloc(n * sizeof(int));

    // Initialize array with random values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Unsorted Array:\n");
    for (int i = 0; i < 10; i++) { // Print first 10 elements
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Time CUDA Merge Sort
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMergeSort(arr, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Sorted Array:\n");
    for (int i = 0; i < 10; i++) { // Print first 10 elements
        printf("%d ", arr[i]);
    }
    
    printf("\nCUDA Merge Sort Time: %.4f ms\n", milliseconds);

    free(arr);
    
    return 0;
}

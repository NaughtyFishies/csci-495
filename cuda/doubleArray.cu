#include <iostream>


__global__ void doubleArray(int *array, int N) {
    // Unique thread index
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        array[index] *= 2;
    }
}

int main() {
    int *cpuMem, *gpuMem;
    int size = 100 * sizeof(int);

    // Intialize same size array on CPU and GPU memory
    cpuMem = (int*)malloc(size);
    cudaMalloc((void**)&gpuMem, size);

    // Fill CPU memory
    for (int i = 0; i < 100; i++){
        cpuMem[i] = i;
    }

    // Copy array to the GPU memory
    cudaMemcpy(gpuMem, cpuMem, size, cudaMemcpyHostToDevice);

    // Perform task
    doubleArray<<<1,100>>>(gpuMem, 100);
    cudaDeviceSynchronize();

    // Copy array back to CPU memory
    cudaMemcpy(cpuMem, gpuMem, size, cudaMemcpyDeviceToHost);

    // Print array
    for (int i = 0; i < 10; i++){
        int tens = i * 10;
        for (int j = 0; j < 10; j++){
            std::cout << cpuMem[tens + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(gpuMem);
    free(cpuMem);
}
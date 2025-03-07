#include <iostream>

__global__ void helloCUDA(int numHellos) {
    int hello = blockIdx.x * blockDim.x + threadIdx.x;
    if (hello < numHellos) {
        printf("Hello # %d\n", hello);
    }
}

int main() {
    dim3 threadsPerBlock(10);
    const int numHellos = 10;
    dim3 numBlocks((numHellos+threadsPerBlock.x-1)/threadsPerBlock.x)

    helloCUDA<<<numBlocks,threadsPerBlock>>>(numHellos); // 1 block, 10 threads
    cudaDeviceSynchronize();
    return 0;
}

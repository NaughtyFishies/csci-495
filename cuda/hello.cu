#include <iostream>

__global__ void helloCUDA() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    helloCUDA<<<1, 10>>>(); // 1 block, 10 threads
    cudaDeviceSynchronize();
    return 0;
}
